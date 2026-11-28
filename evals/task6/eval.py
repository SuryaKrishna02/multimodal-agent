"""Task 6 end-to-end eval: a spoken request becomes a reminder through the reminders MCP server.

Cases: data/evals/{synthetic,user}/task6/
    r01.wav    the spoken request, without the wake word (the wake eval measures that part)
    r01.json   {"text": "Remind me to call Mike in two minutes.", "mention": "Mike", "minutes": 2}
               `mention`: a word the reminder must hold; `minutes`: when it is due, from the start of the turn

Each request takes the real path in a throwaway runtime folder: Whisper (agent/senses/stt.py), then `brain.respond()`
with the reminders server connected over stdio and its add_reminder tool at tier safe (as in the live demo), then the
reminder is read back from the server.

Metrics:
    success_rate   the reminder exists, holds the word, and is due within 30 s of the asked time
    text_rate      the reminder holds the word
    time_rate      the reminder is due at the right time
    turn_s         median seconds from the end of the request to the reply (speech to text + model)
"""

import json
import statistics
import time
from pathlib import Path
from unittest import mock

from agent import brain, config, mcp_client
from agent.senses import audio, stt
from evals.common import NoCases

LEVEL, USES_LLM = "end-to-end", True
NEEDS_EMBED_SERVER = True
TOLERANCE_S = 30.0


def load_cases(cases_dir: Path):
    """Load each spoken request with its label.

    Args:
        cases_dir (Path): The case folder of this eval.

    Returns:
        list[dict]: Items with `wav`, `text`, `mention` and `minutes`.

    Raises:
        NoCases: If the folder has no labeled request.
    """
    items = []
    for label in sorted(cases_dir.glob("*.json")) if cases_dir.is_dir() else []:
        if label.with_suffix(".wav").exists():
            items.append({"id": label.stem, "wav": str(label.with_suffix(".wav")), **json.loads(label.read_text())})
    if not items:
        raise NoCases(cases_dir)
    return items


def _turn(item: dict) -> tuple[str, dict, list[dict], float, float]:
    """Run one spoken turn with the reminders server; return the transcript, the reply, the reminders and times."""
    t0 = time.time()
    heard = stt.transcribe(audio.load_wav(item["wav"]))
    out = brain.respond([brain.part("text", heard)], use_memory=False, spoken=True)  # as in a dashboard voice turn
    turn_s = time.time() - t0
    raw = mcp_client.hub.call(config.REMINDERS_SERVER, "list_reminders", {})
    return heard, out, json.loads(raw) if isinstance(raw, str) else list(raw), t0, turn_s


def run_item(item):
    """Run one request in a throwaway runtime folder and check the reminder it left.

    Args:
        item (dict): An item from `load_cases`.

    Returns:
        dict: `heard`, `reply`, `tools`, `reminder`, `text_ok`, `time_ok` and `turn_s`.
    """
    folder = config.RESULTS / "task6-db" / item["id"]
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "reminders.sqlite").unlink(missing_ok=True)
    with mock.patch.object(config, "RUNTIME", folder), mock.patch.object(config, "MCP_CONFIG", folder / "mcp.json"):
        mcp_client.save({"mcpServers": mcp_client.default_servers(), "tools": {}, "enabled": {}, "tiers": {}})
        mcp_client.hub.connect(config.REMINDERS_SERVER)
        mcp_client.enable(config.REMINDERS_SERVER, ["add_reminder", "list_reminders"])
        mcp_client.set_tier(config.REMINDERS_SERVER, "add_reminder", "safe")
        try:
            heard, out, reminders, t0, turn_s = _turn(item)
        finally:
            mcp_client.hub.disconnect(config.REMINDERS_SERVER)
    newest = reminders[-1] if reminders else None
    due = (newest["due_ts"] - t0) if newest else None
    return {
        "id": item["id"],
        "heard": heard,
        "reply": out["text"],
        "tools": [c.get("tool") for c in out["tool_calls"]],
        "reminder": newest,
        "text_ok": bool(newest) and item["mention"].lower() in newest["text"].lower(),
        "time_ok": due is not None and abs(due - item["minutes"] * 60) <= TOLERANCE_S + turn_s,
        "turn_s": round(turn_s, 2),
    }


def score(records):
    """Compute the success, text and time rates and the median turn time.

    Args:
        records (list[dict]): The records from `run_item`.

    Returns:
        dict: Metric name to value.
    """
    n = max(len(records), 1)
    return {
        "success_rate": round(sum(r["text_ok"] and r["time_ok"] for r in records) / n, 3),
        "text_rate": round(sum(r["text_ok"] for r in records) / n, 3),
        "time_rate": round(sum(r["time_ok"] for r in records) / n, 3),
        "turn_s": round(statistics.median(r["turn_s"] for r in records), 2) if records else None,
    }
