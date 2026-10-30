"""Screen sampling eval: did the sampler keep a frame when something important happened?

Cases: data/evals/{synthetic,user}/screen/<sequence>/   (record with: python cli.py record-screen <sequence> --seconds 60)
    00000.png, 00001.png, ...   raw frames at 2 fps
    frames.json                 [{"frame": "00000.png", "t": 0.0}, ...]   (written by the recorder)
    key_moments.json            [{"t": 12.5, "label": "error dialog appears"}, ...]   (you write this)

Metrics:
    key_moment_recall   share of key moments with a kept frame within TOLERANCE_S after them
    kept_ratio          share of frames kept (lower = fewer image tokens sent)
"""

import json
from pathlib import Path

from PIL import Image

from agent import config
from agent.senses import screen
from evals.common import NoCases

LEVEL, USES_LLM = "component", False
TOLERANCE_S = 2.0


def load_cases(cases_dir: Path):
    """Load the cases: each recorded sequence that has `key_moments.json`.

    Args:
        cases_dir (Path): The case folder of this eval.

    Returns:
        list[dict]: One item per case.

    Raises:
        NoCases: If there is no labeled sequence.
    """
    items = (
        [{"dir": d} for d in sorted(cases_dir.iterdir()) if (d / "key_moments.json").exists()]
        if cases_dir.exists()
        else []
    )
    if not items:
        raise NoCases(cases_dir)
    return items


def run_item(item):
    """Run one case: replay one recorded sequence through the sampler.

    Args:
        item (dict): An item from `load_cases`.

    Returns:
        dict: A record with `frames`, `kept`, `moments`, `caught` and `missed`.
    """
    frames = json.loads((item["dir"] / "frames.json").read_text())
    moments = json.loads((item["dir"] / "key_moments.json").read_text())
    sampler, kept, next_look = screen.Sampler(), [], 0.0
    for f in frames:  # replay at the sampler's own rate
        if f["t"] < next_look:
            continue
        next_look = f["t"] + config.SCREEN_SAMPLE_EVERY_S
        if sampler.offer(Image.open(item["dir"] / f["frame"])):
            kept.append(f)
    caught = [m for m in moments if any(m["t"] <= k["t"] <= m["t"] + TOLERANCE_S for k in kept)]
    return {
        "sequence": item["dir"].name,
        "frames": len(frames),
        "kept": [k["t"] for k in kept],
        "moments": len(moments),
        "caught": len(caught),
        "missed": [m["label"] for m in moments if m not in caught],
    }


def score(records):
    """Compute the metrics: key moment recall and kept ratio.

    Args:
        records (list[dict]): The records from `run_item`.

    Returns:
        dict: Metric name to value.
    """
    return {
        "key_moment_recall": round(sum(r["caught"] for r in records) / max(sum(r["moments"] for r in records), 1), 3),
        "kept_ratio": round(sum(len(r["kept"]) for r in records) / max(sum(r["frames"] for r in records), 1), 3),
    }
