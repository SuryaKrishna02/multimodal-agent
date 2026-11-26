"""Proactive alert eval: does the agent speak up when it should, and stay quiet when it should not?

Cases: data/evals/{synthetic,user}/proactive/
    shot.png    a screenshot
    shot.json   {"alert": true, "mention": "test_add"}   whether it shows the watched condition
                (config.WATCH_CONDITION: a failed test run), and a word a correct alert names (or null)

Each screenshot goes through `tasks.check_screen()`, the same request as the Task 5 watcher.

Metrics:
    false_alert_rate    alerts on screens that do not show the condition (the agent cried wolf)
    missed_alert_rate   no alert on screens that do show it
    mention_rate        alerts that name what the label says (the failing test)
"""

import json
from pathlib import Path

from PIL import Image

from agent import tasks
from evals.common import NoCases

LEVEL, USES_LLM = "component", True


def load_cases(cases_dir: Path):
    """Load each screenshot with its label.

    Args:
        cases_dir (Path): The case folder of this eval.

    Returns:
        list[dict]: Items with `image`, `alert` and `mention`.

    Raises:
        NoCases: If the folder has no labeled screenshot.
    """
    items = []
    for label in sorted(cases_dir.glob("*.json")) if cases_dir.is_dir() else []:
        image = next((p for p in label.parent.glob(f"{label.stem}.*") if p.suffix != ".json"), None)
        if image:
            items.append({"id": label.stem, "image": str(image), **json.loads(label.read_text())})
    if not items:
        raise NoCases(cases_dir)
    return items


def run_item(item):
    """Ask the model about one screenshot.

    Args:
        item (dict): An item from `load_cases`.

    Returns:
        dict: `expected`, `alert`, `what`, `say` and `named` (the alert names the label's word).
    """
    out = tasks.check_screen(Image.open(item["image"]).convert("RGB"))
    said = f"{out['what']} {out['say']}".lower()
    return {
        "id": item["id"],
        "expected": item["alert"],
        "alert": bool(out["alert"]),
        "what": out["what"],
        "say": out["say"],
        "named": bool(out["alert"]) and bool(item.get("mention")) and item["mention"].lower() in said,
    }


def score(records):
    """Compute false and missed alerts.

    Args:
        records (list[dict]): The records from `run_item`.

    Returns:
        dict: Metric name to value.
    """
    neg = [r for r in records if not r["expected"]]
    pos = [r for r in records if r["expected"]]
    hits = [r for r in pos if r["alert"]]
    return {
        "false_alert_rate": round(sum(r["alert"] for r in neg) / len(neg), 3) if neg else None,
        "missed_alert_rate": round(1 - len(hits) / len(pos), 3) if pos else None,
        "mention_rate": round(sum(r["named"] for r in hits) / len(hits), 3) if hits else None,
    }
