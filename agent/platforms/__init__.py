"""OS plug-ins. Drop a file in this folder and it is found: no registry to edit.

    from agent import platforms
    platforms.current().record_screen(path, 10)

Choice: AGENT_PLATFORM=<name> if set, else the first plug-in whose matches() is true,
else the generic fallback (portable hooks work, OS hooks raise Unsupported).
"""

from __future__ import annotations

import importlib
import inspect
import os
import pkgutil
from functools import lru_cache

from agent.platforms.base import HOOKS, Platform, Unsupported

__all__ = ["HOOKS", "Platform", "Unsupported", "available", "current", "load"]


def available() -> list[str]:
    """Names of every plug-in file here (files starting with _ are templates)."""
    return sorted(m.name for m in pkgutil.iter_modules(__path__) if not m.name.startswith("_") and m.name != "base")


def load(name: str) -> Platform:
    """Load one platform plug-in by name.

    Args:
        name (str): File name of the plug-in without `.py`.

    Returns:
        Platform: An instance of the only Platform subclass in the file.

    Raises:
        ValueError: If no plug-in has that name.
        TypeError: If the file does not define exactly one Platform subclass.
    """
    if name not in available():
        raise ValueError(
            f"No platform plug-in '{name}'. Available: {', '.join(available())}. "
            "To add one: copy agent/platforms/_template.py (see CONTRIBUTING.md)."
        )
    mod = importlib.import_module(f"agent.platforms.{name}")
    classes = [
        c
        for _, c in inspect.getmembers(mod, inspect.isclass)
        if issubclass(c, Platform) and c.__module__ == mod.__name__
    ]
    if len(classes) != 1:
        raise TypeError(f"agent/platforms/{name}.py must define exactly one Platform subclass, found {len(classes)}")
    return classes[0]()


@lru_cache(maxsize=1)
def current() -> Platform:
    """Return the platform plug-in for this machine.

    Uses `AGENT_PLATFORM` if set, else the first plug-in whose `matches()` is true, else the generic base.

    Returns:
        Platform: The chosen plug-in.
    """
    forced = os.environ.get("AGENT_PLATFORM")
    if forced:
        return load(forced)
    for name in available():
        p = load(name)
        if p.matches():
            return p
    return Platform()
