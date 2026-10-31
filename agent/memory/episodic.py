"""Episodic memory v1: an event log in SQLite, searchable by embedding similarity.

store()  - junk filter -> embed with EmbeddingGemma-2 -> insert row
search() - embed the query -> cosine similarity against every row -> top k

Brute-force search over a numpy matrix is fine for thousands of memories.
"""

import json
import sqlite3
import time

import numpy as np

from agent import config, events, serving

FILLER = {"um", "uh", "ok", "okay", "yeah", "hmm", "thanks", "thank", "you", "so", "like", "right", "yes", "no"}


def _db() -> sqlite3.Connection:
    config.MEMORY_DB.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(config.MEMORY_DB)
    db.execute("""CREATE TABLE IF NOT EXISTS memories (
        id INTEGER PRIMARY KEY, ts REAL, kind TEXT, text TEXT, meta TEXT, media TEXT, embedding BLOB)""")
    return db


def _doc(kind: str, text: str) -> str:  # EmbeddingGemma document format
    return f"title: {kind} | text: {text}"


def _query(text: str) -> str:  # EmbeddingGemma query format
    return f"task: search result | query: {text}"


def junk_reason(text: str, vec: np.ndarray, existing: np.ndarray) -> str | None:
    """Why this should NOT be stored, or None if it is worth keeping."""
    words = [w.strip(".,!?").lower() for w in text.split()]
    content = [w for w in words if w and w not in FILLER]
    if len(content) < config.MIN_MEMORY_WORDS:
        return f"too little content ({len(content)} meaningful words)"
    if len(existing) and float((existing @ vec).max()) >= config.DUPLICATE_SIMILARITY:
        return "near-duplicate of an existing memory"
    return None


def store(text: str, kind: str = "note", meta: dict | None = None, media: str | None = None) -> int | None:
    """Store one memory. Returns its id, or None if the junk filter rejected it."""
    vec = np.array(serving.embed([_doc(kind, text)])[0], np.float32)
    ids, matrix = _matrix()
    if reason := junk_reason(text, vec, matrix):
        events.emit("memory.rejected", text=text, mem_kind=kind, reason=reason)
        return None
    with _db() as db:
        cur = db.execute(
            "INSERT INTO memories (ts, kind, text, meta, media, embedding) VALUES (?,?,?,?,?,?)",
            (time.time(), kind, text, json.dumps(meta or {}), media, vec.tobytes()),
        )
    events.emit("memory.stored", memory_id=cur.lastrowid, text=text, mem_kind=kind)
    return cur.lastrowid


def search(query: str, k: int = config.MEMORY_TOP_K) -> list[dict]:
    """Find the memories closest to a query.

    Args:
        query (str): Text to search for.
        k (int, optional): Number of memories to return. Defaults to `config.MEMORY_TOP_K`.

    Returns:
        list[dict]: Memories, best first, each with a `score` (cosine similarity).
    """
    ids, matrix = _matrix()
    if not ids:
        return []
    q = np.array(serving.embed([_query(query)])[0], np.float32)
    scores = matrix @ q  # vectors are unit length, so dot product = cosine
    top = np.argsort(-scores)[:k]
    rows = {r["id"]: r for r in all_memories(ids=[ids[i] for i in top])}
    hits = [{**rows[ids[i]], "score": round(float(scores[i]), 3)} for i in top]
    events.emit("memory.search", query=query, hits=[(h["id"], h["score"], h["text"][:60]) for h in hits])
    return hits


def all_memories(limit: int = 200, ids: list[int] | None = None) -> list[dict]:
    """Read memories from the database.

    Args:
        limit (int, optional): Maximum number of rows when `ids` is None, newest first. Defaults to 200.
        ids (list[int], optional): Return exactly these memories instead. Defaults to None.

    Returns:
        list[dict]: Memories with `id`, `ts`, `when`, `kind`, `text`, `meta` and `media`.
    """
    with _db() as db:
        if ids is not None:
            marks = ",".join("?" * len(ids))
            rows = db.execute(f"SELECT id, ts, kind, text, meta, media FROM memories WHERE id IN ({marks})", ids)
        else:
            rows = db.execute("SELECT id, ts, kind, text, meta, media FROM memories ORDER BY id DESC LIMIT ?", (limit,))
        return [
            {
                "id": i,
                "ts": ts,
                "when": time.strftime("%a %H:%M", time.localtime(ts)),
                "kind": kind,
                "text": text,
                "meta": json.loads(meta),
                "media": media,
            }
            for i, ts, kind, text, meta, media in rows
        ]


def clear() -> None:
    """Delete every memory."""
    with _db() as db:
        db.execute("DELETE FROM memories")
    events.emit("memory.cleared")


def _matrix() -> tuple[list[int], np.ndarray]:
    with _db() as db:
        rows = db.execute("SELECT id, embedding FROM memories").fetchall()
    if not rows:
        return [], np.zeros((0, config.EMBED_DIM), np.float32)
    return [r[0] for r in rows], np.stack([np.frombuffer(r[1], np.float32) for r in rows])
