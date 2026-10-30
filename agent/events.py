"""The event log: every component reports what it did here.

The dashboard polls it, so each panel can show what just happened.
Events also go to data/runtime/events.jsonl so you can read them after a run.
"""

import json
import threading
import time
from collections import deque

from agent import config

_lock = threading.Lock()
_events = deque(maxlen=2000)
_next_id = 1


def emit(kind: str, **data) -> dict:
    """Record one event. `kind` is a dotted name, e.g. "senses.audio" or "brain.request"."""
    global _next_id
    with _lock:
        event = {"id": _next_id, "ts": time.time(), "kind": kind, **data}
        _next_id += 1
        _events.append(event)
    config.RUNTIME.mkdir(parents=True, exist_ok=True)
    with open(config.RUNTIME / "events.jsonl", "a") as f:
        f.write(json.dumps(event, default=str) + "\n")
    return event


def since(after_id: int = 0, kind_prefix: str = "") -> list[dict]:
    """Return the events that match a prefix and are newer than an id.

    Args:
        after_id (int, optional): Return only events with a larger id. Defaults to 0.
        kind_prefix (str, optional): Return only kinds that start with this text. Defaults to "".

    Returns:
        list[dict]: Matching events, oldest first.
    """
    with _lock:
        return [e for e in _events if e["id"] > after_id and e["kind"].startswith(kind_prefix)]


def last(kind_prefix: str) -> dict | None:
    """Return the newest event whose kind starts with a prefix.

    Args:
        kind_prefix (str): Start of the event kind.

    Returns:
        dict | None: The event, or None if there is none.
    """
    with _lock:
        for e in reversed(_events):
            if e["kind"].startswith(kind_prefix):
                return e
    return None
