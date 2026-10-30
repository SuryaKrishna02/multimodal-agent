"""Accelerator plug-ins. Drop a file in this folder and it is found: no registry to edit.

    from agent import accel
    accel.current().server_args()

Choice: AGENT_ACCEL=<name> if set, else the matching plug-in with the highest priority.
"""

from __future__ import annotations

import importlib
import inspect
import os
import pkgutil
from functools import lru_cache

from agent.accel.base import Accelerator, Download

__all__ = ["Accelerator", "Download", "available", "current", "load", "llama_server"]


def available() -> list[str]:
    """List the accelerator plug-ins.

    Returns:
        list[str]: Plug-in names, which are the file names in this folder.
    """
    return sorted(m.name for m in pkgutil.iter_modules(__path__) if not m.name.startswith("_") and m.name != "base")


def load(name: str) -> Accelerator:
    """Load one accelerator plug-in by name.

    Args:
        name (str): File name of the plug-in without `.py`.

    Returns:
        Accelerator: An instance of the only Accelerator subclass in the file.

    Raises:
        ValueError: If no plug-in has that name.
        TypeError: If the file does not define exactly one Accelerator subclass.
    """
    if name not in available():
        raise ValueError(
            f"No accelerator plug-in '{name}'. Available: {', '.join(available())}. "
            "To add one: copy agent/accel/_template.py (see CONTRIBUTING.md)."
        )
    mod = importlib.import_module(f"agent.accel.{name}")
    classes = [
        c
        for _, c in inspect.getmembers(mod, inspect.isclass)
        if issubclass(c, Accelerator) and c.__module__ == mod.__name__
    ]
    if len(classes) != 1:
        raise TypeError(f"agent/accel/{name}.py must define exactly one Accelerator subclass, found {len(classes)}")
    return classes[0]()


@lru_cache(maxsize=1)
def current() -> Accelerator:
    """Return the accelerator plug-in for this machine.

    Uses `AGENT_ACCEL` if set, else the matching plug-in with the highest priority.

    Returns:
        Accelerator: The chosen plug-in.

    Raises:
        RuntimeError: If no plug-in matches.
    """
    forced = os.environ.get("AGENT_ACCEL")
    if forced:
        return load(forced)
    found = sorted((load(n) for n in available()), key=lambda a: -a.priority)
    for a in found:
        if a.matches():
            return a
    raise RuntimeError(
        "No accelerator plug-in matches this machine "
        f"(have: {', '.join(available())}). Set AGENT_ACCEL=<name> to force one, or add "
        "a plug-in: see CONTRIBUTING.md."
    )


def llama_server():
    """Path to the llama-server executable for the current OS + accelerator."""
    from agent import platforms

    p = platforms.current()
    return current().bin_dir(p.name) / p.exe("llama-server")
