"""Tools: a registry of local Python functions, each with a permission tier.

    safe     - runs without asking
    confirm  - waits for your approval (dashboard button or CLI prompt)
    blocked  - never runs, whatever the model asks

A guard can also block one specific call (e.g. reading outside the workspace).
The model never decides permissions: every call passes through check() first.
"""

import itertools
from collections.abc import Callable
from dataclasses import dataclass, field

import jsonschema
import numpy as np

from agent import events

SAFE, CONFIRM, BLOCKED = "safe", "confirm", "blocked"


@dataclass
class Tool:
    """One registered tool.

    Attributes:
        name (str): Name the model calls.
        description (str): What the tool does, as the model reads it.
        parameters (dict): JSON schema of the arguments.
        tier (str): SAFE, CONFIRM or BLOCKED.
        fn (Callable): The Python function to run.
        guard (Callable | None): Returns a reason to block one specific call, or None to allow it.
        examples (list[str]): Sample requests, used only for tool selection.
        embedding (list[float] | None): Cached embedding of the description and examples.
    """

    name: str
    description: str
    parameters: dict
    tier: str
    fn: Callable
    guard: Callable[[dict], str | None] | None = None  # returns a reason to block, or None
    examples: list[str] = field(default_factory=list)  # sample requests, used only for tool selection
    embedding: list[float] | None = field(default=None, repr=False)


REGISTRY: dict[str, Tool] = {}
PENDING: dict[int, tuple[str, dict]] = {}  # calls waiting for approval
_ids = itertools.count(1)


def tool(name: str, description: str, parameters: dict, tier: str = SAFE, guard=None, examples=()):
    """Register a function as a tool.

    Args:
        name (str): Name the model calls.
        description (str): What the tool does, as the model reads it.
        parameters (dict): JSON schema of the arguments, without the outer `"type": "object"`.
        tier (str, optional): SAFE, CONFIRM or BLOCKED. Defaults to SAFE.
        guard (Callable, optional): Returns a reason to block one specific call, or None. Defaults to None.
        examples (Sequence[str], optional): Sample requests for tool selection. Defaults to ().

    Returns:
        Callable: A decorator that registers the function and returns it unchanged.
    """

    def register(fn):
        REGISTRY[name] = Tool(name, description, {"type": "object", **parameters}, tier, fn, guard, list(examples))
        return fn

    return register


def schemas(names: list[str]) -> list[dict]:
    """OpenAI-style tool definitions for the request."""
    return [
        {
            "type": "function",
            "function": {"name": n, "description": REGISTRY[n].description, "parameters": REGISTRY[n].parameters},
        }
        for n in names
    ]


def select(query: str, k: int) -> list[str]:
    """Pick the k tools closest to the request (small list per request).

    Each tool is embedded as description + example requests: descriptions alone rank badly
    (e.g. "Remember that the demo is Wednesday at 4pm" picked get_time and write_file).
    """
    from agent import serving

    names = list(REGISTRY)
    if len(names) <= k:
        return names
    try:
        missing = [n for n in names if REGISTRY[n].embedding is None]
        if missing:
            vecs = serving.embed(
                [
                    f"title: {n} | text: {REGISTRY[n].description} Examples: " + "; ".join(REGISTRY[n].examples)
                    for n in missing
                ]
            )
            for n, v in zip(missing, vecs, strict=True):
                REGISTRY[n].embedding = v
        q = np.array(serving.embed([f"task: search result | query: {query}"])[0])
        scores = {n: float(q @ np.array(REGISTRY[n].embedding)) for n in names}
        return sorted(names, key=lambda n: scores[n], reverse=True)[:k]
    except Exception:  # noqa: BLE001 - embedding server down: fall back to registry order
        return names[:k]


def check(name: str, args: dict) -> tuple[str, str]:
    """Decide what happens to a call. Returns (status, reason).

    status is one of: ok, invalid (model should retry), blocked, confirm.
    """
    if name not in REGISTRY:
        return "invalid", f"unknown tool '{name}'"
    t = REGISTRY[name]
    try:
        jsonschema.validate(args, t.parameters)
    except jsonschema.ValidationError as e:
        return "invalid", f"bad arguments: {e.message}"
    if t.tier == BLOCKED:
        return "blocked", f"'{name}' is in the blocked tier"
    if t.guard and (reason := t.guard(args)):
        return "blocked", reason
    if t.tier == CONFIRM:
        return "confirm", f"'{name}' needs your approval"
    return "ok", ""


def run(name: str, args: dict, approved: bool = False) -> dict:
    """Check, then execute. Returns {"status", "result" or "error"}; never raises."""
    status, reason = check(name, args)
    if status == "confirm" and approved:
        status = "ok"
    if status == "confirm":
        call_id = next(_ids)
        PENDING[call_id] = (name, args)
        out = {
            "status": "pending",
            "call_id": call_id,
            "error": f"NOT done yet: {reason}. Tell the user it is waiting for their approval.",
        }
    elif status != "ok":
        out = {"status": status, "error": reason}
    else:
        try:
            out = {"status": "ok", "result": REGISTRY[name].fn(**args)}
        except Exception as e:  # noqa: BLE001 - tool errors go back to the model as text
            out = {"status": "error", "error": f"{type(e).__name__}: {e}"}
    events.emit("tools.call", tool=name, args=args, tier=REGISTRY[name].tier if name in REGISTRY else None, **out)
    return out


def approve(call_id: int, yes: bool) -> dict:
    """Resolve a call that waits for the user.

    Args:
        call_id (int): Id from the `pending` result of `run`.
        yes (bool): True to run the call, False to deny it.

    Returns:
        dict: The result of the call, or `{"status": "denied"}`.
    """
    name, args = PENDING.pop(call_id)
    if not yes:
        events.emit("tools.call", tool=name, args=args, status="denied", error="denied by user")
        return {"status": "denied"}
    return run(name, args, approved=True)


from agent.tools import builtin  # noqa: E402,F401  (registers the built-in tools)
