"""Task 5 end-to-end eval: watch a screen session and speak up once, when a test run fails.

Cases: data/evals/{synthetic,user}/task5/<session>/
    00.png, 01.png, ...   the screenshots in order, as the watcher sees them
    sequence.json         {"fail_at": 3, "mention": "test_add"}   index of the first failing frame (null if none)
                          and a word the spoken alert must hold

The session runs through the real `proactive.ScreenWatcher`: the sampler skips an unchanged screen, Gemma checks a
changed one, and `alert()` applies the cool-down. Speech and notifications are recorded, not played.

Metrics:
    detect_rate        sessions with a failure where the alert came at the failing frame or the next one
    false_alert_rate   sessions where an alert came before the failure, or with no failure at all
    repeat_rate        sessions that alerted more than once (the cool-down should stop a repeat)
    mention_rate       detected alerts that name the failing test
    requests_per_frame model requests per frame (the sampler skips unchanged screens)
"""

import json
from pathlib import Path
from unittest import mock

from PIL import Image

from agent import proactive
from evals.common import NoCases

LEVEL, USES_LLM = "end-to-end", True


def load_cases(cases_dir: Path):
    """Load each screen session.

    Args:
        cases_dir (Path): The case folder of this eval.

    Returns:
        list[dict]: Items with `frames` (paths in order), `fail_at` and `mention`.

    Raises:
        NoCases: If the folder has no session.
    """
    items = []
    for label in sorted(cases_dir.glob("*/sequence.json")) if cases_dir.is_dir() else []:
        frames = sorted(str(p) for p in label.parent.glob("*.png"))
        items.append({"id": label.parent.name, "frames": frames, **json.loads(label.read_text())})
    if not items:
        raise NoCases(cases_dir)
    return items


def run_item(item):
    """Feed one session to a fresh watcher and record each alert that gets through the cool-down.

    Args:
        item (dict): An item from `load_cases`.

    Returns:
        dict: `alerts` (frame index and the spoken text of each), `requests` (model calls) and the labels.
    """
    alerts, requests, now = [], 0, {"frame": 0}
    real = proactive.alert

    def record(key, title, message, reason, spoken=None, speak=True):
        event = real(key, title, message, reason, spoken, speak=False)  # the cool-down still applies; no sound
        if event is not None:
            alerts.append({"frame": now["frame"], "said": spoken or message})
        return event

    watcher = proactive.ScreenWatcher()
    with (
        mock.patch.dict(proactive._last, clear=True),
        mock.patch.object(proactive.platforms, "current", return_value=mock.Mock()),
        mock.patch.object(proactive, "alert", side_effect=record),
    ):
        for i, frame in enumerate(item["frames"]):
            now["frame"] = i
            requests += watcher.check(Image.open(frame).convert("RGB")) is not None
    return {
        "id": item["id"],
        "fail_at": item["fail_at"],
        "mention": item["mention"],
        "alerts": alerts,
        "requests": requests,
        "frames": len(item["frames"]),
    }


def _on_time(r: dict) -> bool:
    """The session has a failure and an alert came at the failing frame or the next one."""
    return r["fail_at"] is not None and any(r["fail_at"] <= a["frame"] <= r["fail_at"] + 1 for a in r["alerts"])


def _early(r: dict) -> bool:
    """An alert came before the failure, or in a session without one."""
    return any(r["fail_at"] is None or a["frame"] < r["fail_at"] for a in r["alerts"])


def score(records):
    """Compute detection, false alerts, repeats and mentions.

    Args:
        records (list[dict]): The records from `run_item`.

    Returns:
        dict: Metric name to value.
    """
    failing = [r for r in records if r["fail_at"] is not None]
    detected = [r for r in failing if _on_time(r)]
    named = [r for r in detected if r["mention"] and r["mention"].lower() in r["alerts"][0]["said"].lower()]
    return {
        "detect_rate": round(len(detected) / len(failing), 3) if failing else None,
        "false_alert_rate": round(sum(_early(r) for r in records) / len(records), 3),
        "repeat_rate": round(sum(len(r["alerts"]) > 1 for r in records) / len(records), 3),
        "mention_rate": round(len(named) / len(detected), 3) if detected else None,
        "requests_per_frame": round(sum(r["requests"] for r in records) / sum(r["frames"] for r in records), 3),
    }
