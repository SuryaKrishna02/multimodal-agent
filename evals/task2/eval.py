"""Task 2 end-to-end eval: is the screen explanation + fix correct?

Gold set: data/gold/task2/
    shot01.png    a real screenshot (error message, failing test, confusing dialog, or a normal screen)
    shot01.json   {"has_problem": true,
                   "must_mention": [["torchvision"], ["pip install", "conda install"]]}
"must_mention" is a list of groups; the answer must contain at least one phrase from EVERY group
(case-insensitive). For a screen with no problem use {"has_problem": false, "must_mention": []}.

Metric: correct_rate = answers where has_problem matches AND every group is mentioned.
Keyword checks are simple on purpose: you can read exactly why an answer passed or failed.
"""
import json
from pathlib import Path

from PIL import Image

from agent import config, tasks
from evals.common import NoGold

LEVEL, USES_LLM = "end-to-end", True


def load_gold(gold_dir: Path):
    items = [{"png": p, **json.loads(p.with_suffix(".json").read_text())}
             for p in sorted(gold_dir.glob("*.png")) if p.with_suffix(".json").exists()]
    if not items:
        raise NoGold(gold_dir)
    return items


def run_item(item):
    img = Image.open(item["png"]).convert("RGB")
    img.thumbnail((config.SCREEN_MAX_SIDE, config.SCREEN_MAX_SIDE))
    r = tasks.screen_help(img, store=False)
    answer = " ".join(str(r[k] or "") for k in ("what_i_see", "problem", "suggested_fix")).lower()
    missing = [g for g in item["must_mention"] if not any(phrase.lower() in answer for phrase in g)]
    problem_ok = (r["problem"] is not None) == item["has_problem"]
    return {"file": item["png"].name, "image": str(item["png"]), "problem": r["problem"], "fix": r["suggested_fix"],
            "missing": missing, "problem_flag_ok": problem_ok, "correct": problem_ok and not missing}


def score(records):
    return {"correct_rate": round(sum(r["correct"] for r in records) / max(len(records), 1), 3)}
