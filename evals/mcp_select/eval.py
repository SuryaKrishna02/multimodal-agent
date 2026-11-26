"""MCP tool selection eval: does the model still pick the right tool among 5, 20 and 40?

Cases: data/evals/{synthetic,user}/mcp_select/
    catalog.jsonl   the tools, one per line: {"name": "weather__forecast", "description": ..., "input_schema": {...}}
    prompts.jsonl   {"id": "m01", "prompt": "Will it rain in Hyderabad tomorrow?", "tool": "weather__forecast"}

For each prompt and each size N (5, 20, 40), the registry holds exactly N catalog tools: the right one plus N - 1
others in a fixed order per prompt. Then two ways to offer them:
    all     every one of the N schemas goes into the request
    top-5   `tools.select()` offers the 5 that match the request best (what the agent does)
and one model step (no execution) shows which tool the model calls.

Metrics:
    acc_5, acc_20, acc_40           right tool with all N offered (how a small model degrades with more tools)
    sel_acc_20, sel_acc_40          right tool with the top-5 selection over N
    sel_recall_40                   the right tool was among the 5 selected from 40
"""

import hashlib
import json
from pathlib import Path
from unittest import mock

from agent import config, tools
from evals.common import NoCases
from evals.tool_calls.eval import first_tool_call

LEVEL, USES_LLM = "component", True
NEEDS_EMBED_SERVER = True
SIZES = (5, 20, 40)
_catalog: dict[str, tools.Tool] = {}  # built once: each Tool caches its embedding


def load_cases(cases_dir: Path):
    """Load the catalog and the prompts.

    Args:
        cases_dir (Path): The case folder of this eval.

    Returns:
        list[dict]: One item per prompt.

    Raises:
        NoCases: If a file is missing or the catalog has fewer than 40 tools.
    """
    cat, prompts = cases_dir / "catalog.jsonl", cases_dir / "prompts.jsonl"
    if not (cat.exists() and prompts.exists()):
        raise NoCases(cases_dir)
    for row in (json.loads(line) for line in cat.read_text().splitlines() if line.strip()):
        if row["name"] not in _catalog:
            _catalog[row["name"]] = tools.Tool(row["name"], row["description"], row["input_schema"], tools.SAFE, dict)
    if len(_catalog) < max(SIZES):
        raise NoCases(f"the catalog needs {max(SIZES)} tools")
    return [json.loads(line) for line in prompts.read_text().splitlines() if line.strip()]


def subset(item: dict, size: int) -> list[str]:
    """The right tool plus `size - 1` others, in an order that is fixed for each prompt.

    Args:
        item (dict): A prompt with its `id` and `tool`.
        size (int): How many tools.

    Returns:
        list[str]: Tool names.
    """
    others = sorted(
        (n for n in _catalog if n != item["tool"]), key=lambda n: hashlib.sha256(f"{item['id']}:{n}".encode()).digest()
    )
    return [item["tool"], *others[: size - 1]]


def _ask(item: dict, names: list[str], select: bool) -> tuple[bool, bool]:
    """One model step with only these tools registered. Return (right tool, right tool was offered)."""
    with mock.patch.dict(tools.REGISTRY, {n: _catalog[n] for n in names}, clear=True):
        offered = tools.select(item["prompt"], config.MAX_TOOLS_PER_REQUEST) if select else names
        called, _ = first_tool_call(item["prompt"], offered)
    return called == item["tool"], item["tool"] in offered


def run_item(item):
    """Ask the model with 5, 20 and 40 tools, offered all at once and through the top-5 selection.

    Args:
        item (dict): `id`, `prompt` and `tool`.

    Returns:
        dict: `all_<N>` and `sel_<N>` (right tool), and `sel_offered_<N>`.
    """
    rec: dict = {"id": item["id"], "prompt": item["prompt"], "tool": item["tool"]}
    for size in SIZES:
        names = subset(item, size)
        rec[f"all_{size}"], _ = _ask(item, names, select=False)
        if size > config.MAX_TOOLS_PER_REQUEST:
            rec[f"sel_{size}"], rec[f"sel_offered_{size}"] = _ask(item, names, select=True)
    return rec


def score(records):
    """Compute accuracy per size and offering.

    Args:
        records (list[dict]): The records from `run_item`.

    Returns:
        dict: Metric name to value.
    """
    n = max(len(records), 1)
    keys = {k for r in records for k in r if k.startswith(("all_", "sel_"))}  # the True/False fields
    shares = {k: round(sum(r.get(k, False) for r in records) / n, 3) for k in keys}
    out = {f"acc_{s}": shares.get(f"all_{s}", 0.0) for s in SIZES}
    out |= {f"sel_acc_{s}": shares.get(f"sel_{s}", 0.0) for s in SIZES if s > config.MAX_TOOLS_PER_REQUEST}
    out["sel_recall_40"] = shares.get("sel_offered_40", 0.0)
    return out
