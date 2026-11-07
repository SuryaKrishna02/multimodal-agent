"""The event log: every component reports what it did here.

The log is a SQLite table, so the daemon, the dashboard and the CLI write into one log even though
they are separate processes. SQLite gives each event an id that only grows, and the dashboard pages
through the log by that id. Events also go to data/runtime/events.jsonl so you can read them after a run.
"""

import json
import os
import time
from pathlib import Path

from agent import config, sql

_source = "cli"  # which process wrote the event: "cli", "dashboard" or "daemon"


def set_source(name: str) -> None:
    """Name the process that writes the next events, so the dashboard can tell the daemon's actions apart.

    Args:
        name (str): For example "daemon" or "dashboard".
    """
    global _source
    _source = name


def _path() -> Path:
    return config.RUNTIME / "events.sqlite"


SCHEMA = """CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, kind TEXT, source TEXT, pid INTEGER, data TEXT)"""


def _db():
    return sql.connect(_path(), SCHEMA)


def emit(kind: str, **data) -> dict:
    """Record one event. `kind` is a dotted name, e.g. "senses.audio" or "brain.request".

    Args:
        kind (str): Dotted event name.
        **data: The payload. Keys `id`, `ts`, `kind` and `source` belong to the log and are dropped.

    Returns:
        dict: The event as stored, with its `id`.
    """
    payload = {k: v for k, v in data.items() if k not in ("id", "ts", "kind", "source")}
    ts = time.time()
    with _db() as db:
        cur = db.execute(
            "INSERT INTO events (ts, kind, source, pid, data) VALUES (?,?,?,?,?)",
            (ts, kind, _source, os.getpid(), json.dumps(payload, default=str)),
        )
    # The log's own fields go last: pollers page by `id`, so a payload must never replace it.
    event = {**payload, "id": cur.lastrowid, "ts": ts, "kind": kind, "source": _source}
    with open(config.RUNTIME / "events.jsonl", "a") as f:
        f.write(json.dumps(event, default=str) + "\n")
    return event


def _rows(sql: str, args: tuple) -> list[dict]:
    with _db() as db:
        rows = db.execute(sql, args).fetchall()
    return [{**json.loads(d), "id": i, "ts": ts, "kind": k, "source": s} for i, ts, k, s, d in rows]


def since(after_id: int = 0, kind_prefix: str = "") -> list[dict]:
    """Return the events that match a prefix and are newer than an id.

    Args:
        after_id (int, optional): Return only events with a larger id. Defaults to 0.
        kind_prefix (str, optional): Return only kinds that start with this text. Defaults to "".

    Returns:
        list[dict]: Matching events, oldest first. At most `config.EVENTS_PAGE`, the newest ones.
    """
    rows = _rows(
        "SELECT id, ts, kind, source, data FROM events WHERE id > ? AND substr(kind, 1, ?) = ? "
        "ORDER BY id DESC LIMIT ?",
        (after_id, len(kind_prefix), kind_prefix, config.EVENTS_PAGE),
    )
    return rows[::-1]


def last(kind_prefix: str) -> dict | None:
    """Return the newest event whose kind starts with a prefix.

    Args:
        kind_prefix (str): Start of the event kind.

    Returns:
        dict | None: The event, or None if there is none.
    """
    rows = _rows(
        "SELECT id, ts, kind, source, data FROM events WHERE substr(kind, 1, ?) = ? ORDER BY id DESC LIMIT 1",
        (len(kind_prefix), kind_prefix),
    )
    return rows[0] if rows else None


def prune(keep: int = config.EVENTS_KEEP) -> int:
    """Delete all but the newest events, so the log cannot grow without limit on an always-on machine.

    Args:
        keep (int, optional): Events to keep. Defaults to `config.EVENTS_KEEP`.

    Returns:
        int: Events deleted.
    """
    with _db() as db:
        cur = db.execute("DELETE FROM events WHERE id <= (SELECT MAX(id) FROM events) - ?", (keep,))
    return cur.rowcount
