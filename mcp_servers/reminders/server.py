"""A tiny reminders MCP server: the other side of the protocol from agent/mcp_client.py.

    python mcp_servers/reminders/server.py --db data/runtime/reminders.sqlite   # an MCP client starts it

It knows nothing about the agent. It speaks MCP over stdin and stdout, keeps reminders in one SQLite file (`--db`),
and offers five tools. The path is an argument, not an environment variable: a stdio server gets only a small, safe
part of its client's environment (HOME, PATH...), so a variable the agent sets would not reach it. A client lists them, shows their JSON schemas to its model, and
calls them. Each tool has a one-line description for the model (its docstring is for people). The annotations
(read only, destructive) are hints for the client: our client shows them, but the user
sets each tool's permission tier.

Times: `in_minutes` counts from now, `at` is "HH:MM" in local time (today, or tomorrow if that time has passed).
A small model gets a relative time right more often than a full timestamp.
"""

import argparse
import sqlite3
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

DB = Path(__file__).resolve().parents[2] / "data" / "runtime" / "reminders.sqlite"  # replaced by --db
SCHEMA = """CREATE TABLE IF NOT EXISTS reminders (
    id INTEGER PRIMARY KEY AUTOINCREMENT, text TEXT NOT NULL, due REAL NOT NULL, created REAL NOT NULL,
    state TEXT NOT NULL DEFAULT 'open')"""  # state: open, done or cancelled

mcp = MCPServer("reminders", instructions="Reminders that the user sets by voice or text. Times are local.")


@contextmanager
def _db() -> Iterator[sqlite3.Connection]:
    """Open the reminders file, and always close it (sqlite3's own `with` commits but does not close)."""
    DB.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(DB, isolation_level=None)  # autocommit: each statement is its own transaction
    db.row_factory = sqlite3.Row
    try:
        db.execute(SCHEMA)
        yield db
    finally:
        db.close()


def _row(r: sqlite3.Row) -> dict:
    due = datetime.fromtimestamp(r["due"])
    return {
        "id": r["id"],
        "text": r["text"],
        "due": due.strftime("%Y-%m-%d %H:%M:%S"),
        "due_ts": r["due"],
        "state": r["state"],
    }


def due_time(in_minutes: float | None, at: str | None, now: float | None = None) -> float:
    """Turn `in_minutes` or `at` into a Unix time.

    Args:
        in_minutes (float | None): Minutes from now.
        at (str | None): "HH:MM", local time; tomorrow if it has passed today.
        now (float | None, optional): The current time, for tests. Defaults to the clock.

    Returns:
        float: The due time.

    Raises:
        ToolError: Neither or both are given, or `at` is not HH:MM. The message goes back to the model.
    """
    now = time.time() if now is None else now
    if (in_minutes is None) == (at is None):
        raise ToolError("give exactly one of in_minutes or at")
    if in_minutes is not None:
        if in_minutes <= 0:
            raise ToolError("in_minutes must be more than 0")
        return now + in_minutes * 60
    try:
        hour, minute = (int(x) for x in str(at).split(":"))
    except ValueError as e:
        raise ToolError(f"at must be HH:MM, got {at!r}") from e
    base = datetime.fromtimestamp(now)
    when = base.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if when.timestamp() <= now:
        when += timedelta(days=1)
    return when.timestamp()


@mcp.tool(
    description='Set a reminder. Give in_minutes (from now) or at ("HH:MM", local time), not both.',
    annotations=ToolAnnotations(read_only_hint=False, destructive_hint=False),
)
def add_reminder(text: str, in_minutes: float | None = None, at: str | None = None) -> dict:
    """Set a reminder. Give in_minutes (from now) or at ("HH:MM", local time), not both.

    Args:
        text (str): What to remind the user of, for example "call Mike".
        in_minutes (float | None): Minutes from now.
        at (str | None): Local time "HH:MM".

    Returns:
        dict: The new reminder with its id and due time.
    """
    with _db() as db:
        cur = db.execute(
            "INSERT INTO reminders (text, due, created) VALUES (?,?,?)",
            (text.strip(), due_time(in_minutes, at), time.time()),
        )
        return _row(db.execute("SELECT * FROM reminders WHERE id = ?", (cur.lastrowid,)).fetchone())


@mcp.tool(description="List the open reminders, soonest first.", annotations=ToolAnnotations(read_only_hint=True))
def list_reminders(include_done: bool = False) -> list[dict]:
    """List the reminders, soonest first.

    Args:
        include_done (bool): Also list done and cancelled reminders.

    Returns:
        list[dict]: The reminders.
    """
    query = "SELECT * FROM reminders ORDER BY due"
    if not include_done:
        query = "SELECT * FROM reminders WHERE state = 'open' ORDER BY due"
    with _db() as db:
        return [_row(r) for r in db.execute(query)]


@mcp.tool(description="List the open reminders whose time has come.", annotations=ToolAnnotations(read_only_hint=True))
def due_reminders() -> list[dict]:
    """List the open reminders whose time has come. The caller marks each one done after it tells the user.

    Returns:
        list[dict]: The due reminders, oldest first.
    """
    with _db() as db:
        rows = db.execute("SELECT * FROM reminders WHERE state = 'open' AND due <= ? ORDER BY due", (time.time(),))
        return [_row(r) for r in rows]


def _set_state(reminder_id: int, state: str) -> dict:
    with _db() as db:
        db.execute("UPDATE reminders SET state = ? WHERE id = ?", (state, reminder_id))
        row = db.execute("SELECT * FROM reminders WHERE id = ?", (reminder_id,)).fetchone()
    if row is None:
        raise ToolError(f"no reminder {reminder_id}")
    return _row(row)


@mcp.tool(
    description="Mark a reminder done after the user was told.",
    annotations=ToolAnnotations(read_only_hint=False, destructive_hint=False, idempotent_hint=True),
)
def complete_reminder(reminder_id: int) -> dict:
    """Mark a reminder done, after the user was told.

    Args:
        reminder_id (int): The id from add_reminder or list_reminders.

    Returns:
        dict: The reminder.
    """
    return _set_state(reminder_id, "done")


@mcp.tool(
    description="Cancel a reminder so it never fires.",
    annotations=ToolAnnotations(read_only_hint=False, destructive_hint=True, idempotent_hint=True),
)
def cancel_reminder(reminder_id: int) -> dict:
    """Cancel a reminder, so it never fires.

    Args:
        reminder_id (int): The id from add_reminder or list_reminders.

    Returns:
        dict: The reminder.
    """
    return _set_state(reminder_id, "cancelled")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Reminders MCP server (stdio).")
    ap.add_argument("--db", type=Path, default=DB, help="the SQLite file (default: data/runtime/reminders.sqlite)")
    DB = ap.parse_args().db
    mcp.run()  # stdio
