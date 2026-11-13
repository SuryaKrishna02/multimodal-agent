"""Consolidation: raw rows -> summaries -> facts, with a citation for every claim. Then forgetting.

1. window     group raw rows that are close in time (config.SUMMARY_WINDOW_S)
2. summarize  one request per window returns claims and facts, and each one cites the raw ids it comes from
3. check      code drops cited ids that are not in the window, and every claim that cites nothing
4. store      facts first, then the summary. The summary is the commit marker: once it exists,
              a rerun after a crash finds it by `window_key` and does not ask the model again
5. replace    a new fact can name older facts it replaces; they are marked superseded, not deleted

forget() deletes raw rows that a summary covers once they are older than RAW_RETENTION_DAYS.
judge() is the faithfulness check of the eval: a separate request reads one claim and only the rows it cites.
"""

import re
import time
from pathlib import Path

from agent import brain, config, events
from agent.memory import episodic

CLAIM = {
    "type": "object",
    "properties": {"text": {"type": "string"}, "sources": {"type": "array", "items": {"type": "integer"}}},
    "required": ["text", "sources"],
}
FACT = {
    "type": "object",
    "properties": {
        "text": {"type": "string"},
        "sources": {"type": "array", "items": {"type": "integer"}},
        "replaces": {"type": "array", "items": {"type": "integer"}},
    },
    "required": ["text", "sources", "replaces"],
}
SCHEMA = {
    "type": "object",
    "properties": {"claims": {"type": "array", "items": CLAIM}, "facts": {"type": "array", "items": FACT}},
    "required": ["claims", "facts"],
}

PROMPT = """Above are raw memories from one stretch of time. Each starts with its [id].
Summarize them in at most {n} claims: merge related memories into one claim.
Write each claim in note style, shorter than the memories it merges: keep names, dates, times, numbers and
decisions, and drop filler words such as "is scheduled to" or "it was decided that".
Each claim cites the [id]s of the raw memories that support it in "sources".
Then list facts worth keeping after today, one short sentence each: decisions, dates, deadlines, names.
Each fact also cites raw memory [id]s. The "Known facts" are listed as [F<number>]: never cite them as sources.
If a new fact makes a known fact no longer true, put that known fact's number in "replaces".
Do not add anything that the memories do not say."""

# The model sometimes copies its citation into the text ("... Monday. Sources: [1]"). The ids belong in "sources".
CITED_SUFFIX = re.compile(r"\s*\(?\b(?:sources?|ids?)\s*:\s*\[?[\d,\s\[\]]*\]?\)?\.?\s*$", re.I)

JUDGE = {
    "type": "object",
    "properties": {"supported": {"type": "boolean"}, "reason": {"type": "string"}},
    "required": ["supported", "reason"],
}
JUDGE_PROMPT = """Sources:
{sources}

Claim: {claim}

Is every part of the claim stated in the sources above? Answer supported=false if the claim adds,
changes or guesses anything that the sources do not say."""


def windows(rows: list[dict]) -> list[list[dict]]:
    """Group rows (oldest first) into windows of at most SUMMARY_WINDOW_S and SUMMARY_MAX_ROWS.

    Args:
        rows (list[dict]): Raw memories, oldest first.

    Returns:
        list[list[dict]]: The windows, oldest first.
    """
    out: list[list[dict]] = []
    for r in rows:
        if out and r["ts"] - out[-1][0]["ts"] < config.SUMMARY_WINDOW_S and len(out[-1]) < config.SUMMARY_MAX_ROWS:
            out[-1].append(r)
        else:
            out.append([r])
    return out


def pending(force: bool = False, now: float | None = None) -> list[list[dict]]:
    """Return the windows of raw rows that no summary covers yet and that are ready to summarize.

    A window is ready once it is complete (its start is SUMMARY_WINDOW_S ago) and has SUMMARY_MIN_ROWS rows.

    Args:
        force (bool, optional): Summarize open and small windows too. Defaults to False.
        now (float, optional): The current time. Defaults to `time.time()`.

    Returns:
        list[list[dict]]: Windows, oldest first.
    """
    now = now or time.time()
    rows = episodic.query_rows("level = 'raw' AND covered_by IS NULL")
    ready = []
    for w in windows(rows):
        complete = now - w[0]["ts"] >= config.SUMMARY_WINDOW_S
        if force or (complete and len(w) >= config.SUMMARY_MIN_ROWS):
            ready.append(w)
    return ready


def window_key(rows: list[dict]) -> str:
    """Name a window by its first and last row id, so a rerun finds the summary it already made."""
    return f"{rows[0]['id']}-{rows[-1]['id']}"


def _line(r: dict) -> str:
    return f"[{r['id']}] ({time.strftime('%H:%M', time.localtime(r['ts']))}, {r['kind']}) {r['text']}"


def _known_facts(rows: list[dict]) -> list[dict]:
    query = " ".join(r["text"] for r in rows)[:2000]
    return episodic.search(query, k=5, levels=("fact",))


def _cited(items: list[dict], allowed: set[int]) -> tuple[list[dict], int]:
    """Keep only the cited ids that exist in `allowed`, and drop items that cite nothing valid.

    Returns:
        tuple[list[dict], int]: The kept items, and how many cited ids were invalid.
    """
    kept, bad = [], 0
    for it in items:
        good = [i for i in it["sources"] if i in allowed]
        bad += len(it["sources"]) - len(good)
        text = CITED_SUFFIX.sub("", it["text"]).strip()
        if good and text:
            kept.append({**it, "text": text, "sources": good})
    return kept, bad


def summarize_window(rows: list[dict]) -> dict:
    """Summarize one window into a summary row and fact rows, and mark the raw rows as covered.

    Safe to run twice: a summary with the same `window_key` is reused.

    Args:
        rows (list[dict]): Raw memories of one window, oldest first.

    Returns:
        dict: `summary_id`, `fact_ids`, `claims`, `invalid_citations` and `reused`.
    """
    key, ids = window_key(rows), [r["id"] for r in rows]
    done = episodic.query_rows("level = 'summary' AND json_extract(meta, '$.window_key') = ?", (key,))
    if done:
        episodic.mark_covered(ids, done[0]["id"])
        return {"summary_id": done[0]["id"], "fact_ids": [], "claims": done[0]["meta"]["claims"], "reused": True}
    known = _known_facts(rows)
    text = "Raw memories:\n" + "\n".join(map(_line, rows))
    if known:
        text += "\n\nKnown facts:\n" + "\n".join(f"[F{f['id']}] {f['text']}" for f in known)
    prompt = PROMPT.format(n=max(1, -(-len(rows) // 3)))  # about one claim per three raw rows
    data, _ = brain.ask_json([brain.part("text", text + "\n\n" + prompt)], SCHEMA)
    if data is None:  # no valid JSON: leave the rows uncovered, so forget() cannot delete them unsummarized
        events.emit("memory.consolidate_failed", window=key, rows=len(rows))
        return {
            "summary_id": None,
            "fact_ids": [],
            "claims": [],
            "invalid_citations": 0,
            "reused": False,
            "failed": True,
        }
    claims, bad_c = _cited(data["claims"], set(ids))
    facts, bad_f = _cited(data["facts"], set(ids))
    fact_ids = _store_facts(facts, {f["id"] for f in known}, rows[-1]["ts"])
    cited = sorted({i for c in claims for i in c["sources"]})
    meta = {
        "window_key": key,
        "covers": ids,
        "sources": cited,
        "claims": claims,
        "window": [rows[0]["ts"], rows[-1]["ts"]],
    }
    summary_id = None
    if claims:
        summary_id = episodic.store(
            " ".join(c["text"] for c in claims), "summary", meta, ts=rows[0]["ts"], level="summary"
        )
    # 0 = looked at, nothing worth a summary (or the junk filter refused it): do not summarize these rows again
    episodic.mark_covered(ids, summary_id or 0)
    out = {"summary_id": summary_id, "fact_ids": fact_ids, "claims": claims, "invalid_citations": bad_c + bad_f}
    events.emit("memory.consolidated", window=key, rows=len(rows), **out)
    return {**out, "reused": False}


def _store_facts(facts: list[dict], known_ids: set[int], ts: float) -> list[int]:
    """Store new facts and mark the known facts they replace."""
    new_ids = []
    for f in facts:
        fid = episodic.store(f["text"], "fact", {"sources": f["sources"]}, ts=ts, level="fact")
        if fid is None:
            continue
        new_ids.append(fid)
        if old := [i for i in f.get("replaces", []) if i in known_ids]:
            episodic.supersede(old, fid)
    return new_ids


def consolidate(force: bool = False, step=None) -> list[dict]:
    """Summarize every ready window.

    Args:
        force (bool, optional): Also summarize open and small windows. Defaults to False.
        step (Callable, optional): `checkpoint.step` bound to a job, so each window is one checkpointed step.
            Defaults to running each window directly.

    Returns:
        list[dict]: The result of `summarize_window` for each window.
    """
    out = []
    for w in pending(force):
        key = window_key(w)
        run = lambda w=w: summarize_window(w)  # noqa: E731
        out.append(step(f"window:{key}", run, {"rows": len(w)}) if step else run())
    return out


def judge(claim: str, source_ids: list[int]) -> dict:
    """Ask a separate request whether one claim is supported by only the rows it cites.

    Args:
        claim (str): The claim text.
        source_ids (list[int]): The ids it cites.

    Returns:
        dict: `supported` (bool) and `reason`. Unsupported when no cited row exists any more.
    """
    rows = episodic.all_memories(ids=source_ids)
    if not rows:
        return {"supported": False, "reason": "no cited row exists"}
    prompt = JUDGE_PROMPT.format(sources="\n".join(map(_line, rows)), claim=claim)
    data, _ = brain.ask_json([brain.part("text", prompt)], JUDGE)
    return data or {"supported": False, "reason": "judge returned no valid JSON"}


def forget(now: float | None = None) -> dict:
    """Delete raw rows that a summary covers and that are older than RAW_RETENTION_DAYS, with their capture files.

    Args:
        now (float, optional): The current time. Defaults to `time.time()`.

    Returns:
        dict: `deleted` (count) and `cutoff` (epoch seconds).
    """
    cutoff = (now or time.time()) - config.RAW_RETENTION_DAYS * 86400
    rows = episodic.query_rows("level = 'raw' AND covered_by IS NOT NULL AND ts < ?", (cutoff,))
    for r in rows:  # a screenshot is personal data too: delete the file the row points to
        if r["media"] and Path(r["media"]).resolve().is_relative_to(config.RUNTIME.resolve()):
            Path(r["media"]).unlink(missing_ok=True)
    deleted = episodic.delete([r["id"] for r in rows])
    events.emit("memory.forgotten", deleted=deleted, cutoff=cutoff, ids=[r["id"] for r in rows][:50])
    return {"deleted": deleted, "cutoff": cutoff}


def stats() -> dict:
    """Count rows per level and the characters they hold, for the dashboard and the compression ratio.

    Returns:
        dict: For each level, `rows` and `chars`.
    """
    rows = episodic.query_rows("1 = 1")
    out = {lv: {"rows": 0, "chars": 0} for lv in config.MEMORY_LEVELS}
    for r in rows:
        out[r["level"]]["rows"] += 1
        out[r["level"]]["chars"] += len(r["text"])
    return out
