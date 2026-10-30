"""Tool-call eval: does the model pick the right tool with the right arguments?

Uses the real agent path: select a few tools for the prompt, then one model step (no execution).

Cases: data/evals/{synthetic,user}/tool_calls/prompts.jsonl, one JSON object per line:
    {"id": "t01", "prompt": "What's 15% of 80?", "tool": "calculator", "args": {"expression": "80"}}
    {"id": "t02", "prompt": "Tell me a joke", "tool": null, "args": {}}
"tool": null means the model should answer without a tool.
Argument matching: each expected string must appear inside the actual value (case-insensitive);
numbers and booleans must be equal. Arguments you leave out are not checked.

Metrics:
    tool_accuracy     right tool (or correctly no tool)
    exact_rate        right tool AND every expected argument matches
    selection_recall  the expected tool was among the tools offered to the model
"""

import json
from pathlib import Path

from agent import brain, config, tools
from evals.common import NoCases

LEVEL, USES_LLM = "component", True
NEEDS_EMBED_SERVER = True  # tool selection uses embeddings


def load_cases(cases_dir: Path):
    """Load the cases: each line of `prompts.jsonl`.

    Args:
        cases_dir (Path): The case folder of this eval.

    Returns:
        list[dict]: One item per case.

    Raises:
        NoCases: If `prompts.jsonl` is missing.
    """
    f = cases_dir / "prompts.jsonl"
    if not f.exists():
        raise NoCases(cases_dir)
    return [json.loads(line) for line in f.read_text().splitlines() if line.strip()]


def first_tool_call(prompt: str, tool_names: list[str]) -> tuple[str | None, dict | None]:
    """One model step, no execution: which tool and arguments does the model pick?"""
    msgs, _ = brain.fit_budget(brain.system_prompt(), [], [], [brain.part("text", prompt)], tools.schemas(tool_names))
    msg = brain.send(msgs, tools=tools.schemas(tool_names))["choices"][0]["message"]
    calls = msg.get("tool_calls") or []
    if not calls:
        return None, None
    try:
        return calls[0]["function"]["name"], json.loads(calls[0]["function"]["arguments"] or "{}")
    except json.JSONDecodeError:
        return calls[0]["function"]["name"], None


def args_match(expected: dict, actual: dict | None) -> bool:
    """Check the arguments of a tool call against the expected ones.

    Args:
        expected (dict): Expected arguments. Strings must appear in the actual value, other values must be equal.
        actual (dict | None): Arguments of the model, or None if they were not valid JSON.

    Returns:
        bool: True if every expected argument matches. Extra arguments are ignored.
    """
    if actual is None:
        return False
    for k, v in expected.items():
        if k not in actual:
            return False
        if isinstance(v, str):
            if v.lower().strip() not in str(actual[k]).lower():
                return False
        elif v != actual[k]:
            return False
    return True


def run_item(item):
    """Run one case: select tools for one prompt and take one model step.

    Args:
        item (dict): An item from `load_cases`.

    Returns:
        dict: A record with `right_tool`, `exact` and `selected`.
    """
    offered = tools.select(item["prompt"], config.MAX_TOOLS_PER_REQUEST)
    name, args = first_tool_call(item["prompt"], offered)
    right_tool = name == item["tool"]
    return {
        "id": item["id"],
        "prompt": item["prompt"],
        "expected": [item["tool"], item.get("args", {})],
        "got": [name, args],
        "offered": offered,
        "right_tool": right_tool,
        "exact": right_tool and (name is None or args_match(item.get("args", {}), args)),
        "selected": item["tool"] is None or item["tool"] in offered,
    }


def score(records):
    """Compute the metrics: tool accuracy, exact rate and selection recall.

    Args:
        records (list[dict]): The records from `run_item`.

    Returns:
        dict: Metric name to value.
    """
    n = max(len(records), 1)
    return {
        "tool_accuracy": round(sum(r["right_tool"] for r in records) / n, 3),
        "exact_rate": round(sum(r["exact"] for r in records) / n, 3),
        "selection_recall": round(sum(r["selected"] for r in records) / n, 3),
    }
