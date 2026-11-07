"""Episodic memory: an event log in SQLite, searchable by embedding similarity.

store()  - junk filter -> embed the text (and the screenshot or audio) -> insert row
search() - embed the query -> cosine similarity against every row -> top k

Each row has a level (Session 2): "raw" rows come from the senses and tasks, consolidate.py turns
them into "summary" rows and "fact" rows. A raw row that a summary covers has `covered_by` set, and
a fact that a newer fact replaces has `superseded_by` set.

Multimodal memory: EmbeddingGemma-2 puts text, images and audio in one 768-d space. A row can keep
a second vector of its screenshot or audio clip, so a text query can find a picture directly.
Brute-force search over a numpy matrix is fine for thousands of memories.
"""

import json
import sqlite3
import time

import numpy as np

from agent import config, events, serving, sql

FILLER = {"um", "uh", "ok", "okay", "yeah", "hmm", "thanks", "thank", "you", "so", "like", "right", "yes", "no"}
COLUMNS = "id, ts, kind, text, meta, media, level, covered_by, superseded_by"
MODES = ("text", "media", "both")  # which vector a search compares: the text, the image/audio, or the best


def _init(db: sqlite3.Connection) -> None:
    """Create the table, and give a Session 1 database the Session 2 columns."""
    db.execute("""CREATE TABLE IF NOT EXISTS memories (
        id INTEGER PRIMARY KEY, ts REAL, kind TEXT, text TEXT, meta TEXT, media TEXT, embedding BLOB)""")
    have = {r[1] for r in db.execute("PRAGMA table_info(memories)")}
    for col, spec in (
        ("level", "TEXT DEFAULT 'raw'"),
        ("covered_by", "INTEGER"),
        ("superseded_by", "INTEGER"),
        ("media_embedding", "BLOB"),
    ):
        if col not in have:
            try:
                db.execute(f"ALTER TABLE memories ADD COLUMN {col} {spec}")
            except sqlite3.OperationalError as e:  # another process added it a moment ago
                if "duplicate column" not in str(e):
                    raise


def _db():
    return sql.connect(config.MEMORY_DB, _init)


def _doc(kind: str, text: str) -> str:  # EmbeddingGemma document format
    return f"title: {kind} | text: {text}"


def _query(text: str) -> str:  # EmbeddingGemma query format
    return f"task: search result | query: {text}"


def junk_reason(text: str, vec: np.ndarray, existing: np.ndarray) -> str | None:
    """Why this should NOT be stored, or None if it is worth keeping.

    Args:
        text (str): The text to store.
        vec (np.ndarray): Its unit-length embedding.
        existing (np.ndarray): Embeddings of the stored rows of the same level.

    Returns:
        str | None: The reason, or None.
    """
    words = [w.strip(".,!?").lower() for w in text.split()]
    content = [w for w in words if w and w not in FILLER]
    if len(content) < config.MIN_MEMORY_WORDS:
        return f"too little content ({len(content)} meaningful words)"
    if len(existing) and float((existing @ vec).max()) >= config.DUPLICATE_SIMILARITY:
        return "near-duplicate of an existing memory"
    return None


def embed_media(kind: str, value) -> np.ndarray:
    """Embed a screenshot or an audio clip into the same space as text.

    Args:
        kind (str): "image" (PIL image or path) or "audio" (16 kHz float32 array).
        value: The image or audio.

    Returns:
        np.ndarray: A unit-length 768-d vector.
    """
    from agent import brain

    part = {k: v for k, v in brain.part(kind, value).items() if not k.startswith("_")}
    return np.array(serving.embed([{"content": [part]}])[0], np.float32)


def store(
    text: str,
    kind: str = "note",
    meta: dict | None = None,
    media: str | tuple | None = None,
    ts: float | None = None,
    level: str = "raw",
) -> int | None:
    """Store one memory. Returns its id, or None if the junk filter rejected it.

    Args:
        text (str): What to remember.
        kind (str, optional): What made it, e.g. "voice_note", "transcript", "summary". Defaults to "note".
        meta (dict, optional): Extra fields. Defaults to {}.
        media (str | tuple, optional): A file path to show with the memory, or ("image"|"audio", value) to also
            store a media vector. A path to a PNG also stores the image vector. Defaults to None.
        ts (float, optional): When it happened, in epoch seconds. Defaults to now.
        level (str, optional): "raw", "summary" or "fact". Defaults to "raw".

    Returns:
        int | None: The row id.
    """
    vec = np.array(serving.embed([_doc(kind, text)])[0], np.float32)
    _, matrix = _matrix(levels=(level,))
    if reason := junk_reason(text, vec, matrix):
        events.emit("memory.rejected", text=text, mem_kind=kind, reason=reason)
        return None
    path, media_vec = _media(media)
    with _db() as db:
        cur = db.execute(
            "INSERT INTO memories (ts, kind, text, meta, media, embedding, level, media_embedding) VALUES (?,?,?,?,?,?,?,?)",
            (ts or time.time(), kind, text, json.dumps(meta or {}), path, vec.tobytes(), level, media_vec),
        )
    events.emit(
        "memory.stored", memory_id=cur.lastrowid, text=text, mem_kind=kind, level=level, media=media_vec is not None
    )
    return cur.lastrowid


def _media(media: str | tuple | None) -> tuple[str | None, bytes | None]:
    """Split the `media` argument of `store` into a path to show and the bytes of a media vector."""
    if isinstance(media, tuple):
        return None, embed_media(*media).tobytes()
    if isinstance(media, str) and media.lower().endswith(".png"):
        return media, embed_media("image", media).tobytes()
    return media, None


def search(
    query: str,
    k: int = config.MEMORY_TOP_K,
    levels: tuple[str, ...] = config.MEMORY_LEVELS,
    mode: str = "both",
    since: float | None = None,
    until: float | None = None,
) -> list[dict]:
    """Find the memories closest to a query. Facts that a newer fact replaced are left out.

    Args:
        query (str): Text to search for.
        k (int, optional): Number of memories to return. Defaults to `config.MEMORY_TOP_K`.
        levels (tuple[str, ...], optional): Levels to search. Defaults to all.
        mode (str, optional): "text", "media" or "both" (the better of the two scores). Defaults to "both".
        since (float, optional): Only rows at or after this epoch time. Defaults to None.
        until (float, optional): Only rows before this epoch time. Defaults to None.

    Returns:
        list[dict]: Memories, best first, each with a `score` (cosine similarity).
    """
    ids, text_m, media_m = _vectors(levels, since, until)
    if not ids:
        return []
    q = np.array(serving.embed([_query(query)])[0], np.float32)
    scores = _scores(q, text_m, media_m, mode)  # vectors are unit length, so dot product = cosine
    top = [i for i in np.argsort(-scores)[:k] if scores[i] > -1]
    rows = {r["id"]: r for r in all_memories(ids=[ids[i] for i in top])}
    hits = [{**rows[ids[i]], "score": round(float(scores[i]), 3)} for i in top]
    events.emit("memory.search", query=query, mode=mode, hits=[(h["id"], h["score"], h["text"][:60]) for h in hits])
    return hits


def _scores(q: np.ndarray, text_m: np.ndarray, media_m: np.ndarray, mode: str) -> np.ndarray:
    """Score every row for one query. A row without a media vector scores -2 on media (never found by it)."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    text_s, media_s = text_m @ q, media_m @ q
    media_s[~media_m.any(axis=1)] = -2.0
    return {"text": text_s, "media": media_s, "both": np.maximum(text_s, media_s)}[mode]


def all_memories(limit: int = 200, ids: list[int] | None = None, levels: tuple[str, ...] | None = None) -> list[dict]:
    """Read memories from the database.

    Args:
        limit (int, optional): Maximum number of rows when `ids` is None, newest first. Defaults to 200.
        ids (list[int], optional): Return exactly these memories instead. Defaults to None.
        levels (tuple[str, ...], optional): Only these levels. Defaults to all.

    Returns:
        list[dict]: Memories with `id`, `ts`, `when`, `kind`, `text`, `meta`, `media`, `level`, `covered_by`
            and `superseded_by`.
    """
    with _db() as db:
        if ids is not None:
            marks = ",".join("?" * len(ids))
            rows = db.execute(f"SELECT {COLUMNS} FROM memories WHERE id IN ({marks})", ids).fetchall()
        else:
            lv = levels or config.MEMORY_LEVELS
            marks = ",".join("?" * len(lv))
            sql = f"SELECT {COLUMNS} FROM memories WHERE level IN ({marks}) ORDER BY ts DESC, id DESC LIMIT ?"
            rows = db.execute(sql, (*lv, limit)).fetchall()
    return [_row(r) for r in rows]


def _row(r: tuple) -> dict:
    i, ts, kind, text, meta, media, level, covered_by, superseded_by = r
    when = time.strftime("%a %H:%M", time.localtime(ts))
    return {
        "id": i,
        "ts": ts,
        "when": when,
        "kind": kind,
        "text": text,
        "meta": json.loads(meta),
        "media": media,
        "level": level,
        "covered_by": covered_by,
        "superseded_by": superseded_by,
    }


def mark_covered(ids: list[int], summary_id: int) -> None:
    """Record that a summary covers some raw rows, so consolidation does not summarize them again.

    Args:
        ids (list[int]): The raw rows.
        summary_id (int): The summary row.
    """
    with _db() as db:
        db.executemany("UPDATE memories SET covered_by = ? WHERE id = ?", [(summary_id, i) for i in ids])


def supersede(old_ids: list[int], new_id: int) -> None:
    """Mark facts as replaced by a newer fact. They stay in the database but leave search.

    Args:
        old_ids (list[int]): The facts that are no longer true.
        new_id (int): The fact that replaces them.
    """
    with _db() as db:
        db.executemany(
            "UPDATE memories SET superseded_by = ? WHERE id = ? AND level = 'fact'", [(new_id, i) for i in old_ids]
        )
    events.emit("memory.superseded", old=old_ids, new=new_id)


def delete(ids: list[int]) -> int:
    """Delete rows for good.

    Args:
        ids (list[int]): The rows.

    Returns:
        int: Rows deleted.
    """
    with _db() as db:
        cur = db.executemany("DELETE FROM memories WHERE id = ?", [(i,) for i in ids])
    return cur.rowcount


def query_rows(where: str, args: tuple = ()) -> list[dict]:
    """Read rows that match an SQL condition, oldest first. For consolidate.py and the evals.

    Args:
        where (str): The condition after WHERE, with ? placeholders.
        args (tuple, optional): Values for the placeholders. Defaults to ().

    Returns:
        list[dict]: Memories as `all_memories` returns them.
    """
    with _db() as db:
        rows = db.execute(f"SELECT {COLUMNS} FROM memories WHERE {where} ORDER BY ts, id", args).fetchall()
    return [_row(r) for r in rows]


def clear() -> None:
    """Delete every memory."""
    with _db() as db:
        db.execute("DELETE FROM memories")
    events.emit("memory.cleared")


def _vectors(
    levels: tuple[str, ...], since: float | None = None, until: float | None = None
) -> tuple[list[int], np.ndarray, np.ndarray]:
    """Load the ids, text vectors and media vectors (zeros when a row has none) of the searchable rows."""
    marks = ",".join("?" * len(levels))
    sql = f"SELECT id, embedding, media_embedding FROM memories WHERE level IN ({marks}) AND superseded_by IS NULL"
    args: list = list(levels)
    if since is not None:
        sql, args = sql + " AND ts >= ?", [*args, since]
    if until is not None:
        sql, args = sql + " AND ts < ?", [*args, until]
    with _db() as db:
        rows = db.execute(sql, args).fetchall()
    zeros = np.zeros((0, config.EMBED_DIM), np.float32)
    if not rows:
        return [], zeros, zeros
    text_m = np.stack([np.frombuffer(r[1], np.float32) for r in rows])
    blank = np.zeros(config.EMBED_DIM, np.float32)
    media_m = np.stack([np.frombuffer(r[2], np.float32) if r[2] else blank for r in rows])
    return [r[0] for r in rows], text_m, media_m


def _matrix(levels: tuple[str, ...] = config.MEMORY_LEVELS) -> tuple[list[int], np.ndarray]:
    ids, text_m, _ = _vectors(levels)
    return ids, text_m
