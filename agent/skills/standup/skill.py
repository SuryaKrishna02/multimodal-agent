"""The standup skill: the last day's action items and summaries, grouped for a stand-up."""

import time

from agent.memory import episodic


def run(hours: float = 24) -> dict:
    """Collect what the memories of the last hours say, for the model to phrase as a stand-up.

    Args:
        hours (float, optional): How far back to look. Defaults to 24.

    Returns:
        dict: `done` (summaries and notes), `next` (open action items) and `hours`.
    """
    since = time.time() - hours * 3600
    rows = [m for m in episodic.all_memories(limit=200) if m["ts"] >= since and m["superseded_by"] is None]
    items = [i for m in rows for i in m["meta"].get("action_items", [])]
    done = [f"({m['when']}) {m['text']}" for m in rows if m["level"] != "raw" or m["kind"] in ("note", "screen_help")]
    return {"hours": hours, "done": done[:12], "next": items[:12]}
