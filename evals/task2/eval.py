"""Task 2 end-to-end eval: is the screen explanation + fix correct?

Cases: data/evals/{synthetic,user}/task2/
    shot01.png    a real screenshot (error message, failing test, confusing dialog, or a normal screen)
                  or shot01.mp4 / .mov: a short screen recording, sent as one video part
    shot01.json   {"has_problem": true,
                   "must_mention": [["torchvision"], ["pip install", "conda install"]]}
"must_mention" is a list of groups; the answer must contain at least one phrase from EVERY group
(case-insensitive). For a screen with no problem use {"has_problem": false, "must_mention": []}.

Metrics: correct_rate = answers where has_problem matches AND every group is mentioned,
also split into correct_rate_image and correct_rate_video when both kinds are present.
Keyword checks are simple on purpose: you can read exactly why an answer passed or failed.
"""

import json
from pathlib import Path

from PIL import Image

from agent import config, tasks
from agent.senses import video
from evals.common import NoCases

LEVEL, USES_LLM = "end-to-end", True
MEDIA = (".png", ".mp4", ".mov")


def load_cases(cases_dir: Path):
    """Load the cases: each screenshot or recording with the .json file beside it.

    Args:
        cases_dir (Path): The case folder of this eval.

    Returns:
        list[dict]: One item per case.

    Raises:
        NoCases: If there is no labeled screenshot or recording.
    """
    items = [
        {"media": p, **json.loads(p.with_suffix(".json").read_text())}
        for p in sorted(cases_dir.glob("*"))
        if p.suffix in MEDIA and p.with_suffix(".json").exists()
    ]
    if not items:
        raise NoCases(cases_dir)
    return items


def run_item(item):
    """Run one case: run Task 2 on one image or video and check the keywords.

    Args:
        item (dict): An item from `load_cases`.

    Returns:
        dict: A record with `correct`, `missing` and `problem_flag_ok`.
    """
    kind = "image" if item["media"].suffix == ".png" else "video"
    if kind == "image":
        img = Image.open(item["media"]).convert("RGB")
        img.thumbnail((config.SCREEN_MAX_SIDE, config.SCREEN_MAX_SIDE))
        r = tasks.screen_help(img, store=False)
    else:
        small = config.RESULTS / "task2-videos" / (item["media"].stem + ".mp4")
        small.parent.mkdir(parents=True, exist_ok=True)
        r = tasks.screen_help(video_path=video.shrink(item["media"], small), store=False)
    answer = " ".join(str(r[k] or "") for k in ("what_i_see", "problem", "suggested_fix")).lower()
    missing = [g for g in item["must_mention"] if not any(phrase.lower() in answer for phrase in g)]
    problem_ok = (r["problem"] is not None) == item["has_problem"]
    return {
        "file": item["media"].name,
        kind: str(item["media"]),
        "kind": kind,
        "problem": r["problem"],
        "fix": r["suggested_fix"],
        "missing": missing,
        "problem_flag_ok": problem_ok,
        "correct": problem_ok and not missing,
    }


def score(records):
    """Compute the metrics: the share of correct answers, split by image and video when both are present.

    Args:
        records (list[dict]): The records from `run_item`.

    Returns:
        dict: Metric name to value.
    """
    rate = lambda rs: round(sum(r["correct"] for r in rs) / max(len(rs), 1), 3)  # noqa: E731
    out = {"correct_rate": rate(records)}
    for kind in ("image", "video"):
        if any(r["kind"] == kind for r in records) and len({r["kind"] for r in records}) > 1:
            out[f"correct_rate_{kind}"] = rate([r for r in records if r["kind"] == kind])
    return out
