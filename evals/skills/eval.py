"""Native skill eval: does a skill fire when it should, and stay quiet when it should not?

Cases: data/evals/{synthetic,user}/skills/prompts.jsonl, one JSON object per line:
    {"id": "k01", "prompt": "How many kilometers is a 10 mile run?", "skill": "unit_convert"}
    {"id": "k10", "prompt": "What is 15 percent of 80?", "skill": null}
`null` means no skill should fire (another tool, or no tool, is the right answer).

Each prompt takes the real path: `tools.select()` offers the top tools (built-in tools and skills compete), then one
model step (no execution), as in the tool_calls eval.

Metrics:
    fire_recall       prompts for a skill where the model called that skill
    false_fire_rate   prompts for no skill where the model called a skill anyway
    selection_recall  prompts for a skill where tool selection offered it
"""

import json
from pathlib import Path

from agent import config, tools
from evals.common import NoCases
from evals.tool_calls.eval import first_tool_call

LEVEL, USES_LLM = "component", True
NEEDS_EMBED_SERVER = True


def load_cases(cases_dir: Path):
    """Load the prompts.

    Args:
        cases_dir (Path): The case folder of this eval.

    Returns:
        list[dict]: One item per prompt.

    Raises:
        NoCases: If prompts.jsonl is missing.
    """
    f = cases_dir / "prompts.jsonl"
    if not f.exists():
        raise NoCases(cases_dir)
    return [json.loads(line) for line in f.read_text().splitlines() if line.strip()]


def run_item(item):
    """Offer the selected tools for one prompt and see which one the model calls.

    Args:
        item (dict): `id`, `prompt` and `skill`.

    Returns:
        dict: `expected`, `called`, `fired` (the skill called, if any) and `offered`.
    """
    skills = {n for n, t in tools.REGISTRY.items() if getattr(t, "skill", None)}
    offered = tools.select(item["prompt"], config.MAX_TOOLS_PER_REQUEST)
    called, _ = first_tool_call(item["prompt"], offered)
    return {
        "id": item["id"],
        "prompt": item["prompt"],
        "expected": item["skill"],
        "called": called,
        "fired": called if called in skills else None,
        "offered": offered,
    }


def score(records):
    """Compute fire recall, false fires and selection recall.

    Args:
        records (list[dict]): The records from `run_item`.

    Returns:
        dict: Metric name to value.
    """
    pos = [r for r in records if r["expected"]]
    neg = [r for r in records if not r["expected"]]
    return {
        "fire_recall": round(sum(r["fired"] == r["expected"] for r in pos) / len(pos), 3) if pos else None,
        "false_fire_rate": round(sum(r["fired"] is not None for r in neg) / len(neg), 3) if neg else None,
        "selection_recall": round(sum(r["expected"] in r["offered"] for r in pos) / len(pos), 3) if pos else None,
    }
