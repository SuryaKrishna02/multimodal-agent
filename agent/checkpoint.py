"""Checkpoints: the daemon writes down every step of every job in SQLite, so a crash loses nothing.

A job is a queued piece of work ("summarize what was heard", "write the brief"). Each step of a job
is recorded in three phases:

    intent     before anything happens: what the step is about to do, with its inputs
    execution  right before the code with side effects runs
    result     after it returns, with its output

After a crash, `recover()` puts every job that was running back in the queue. When the job runs again,
`step()` returns the stored output of each step that has a result, without running it again. A step
with only intent or execution ran partly or not at all, so it runs again. Steps must therefore be safe
to run twice: memory.store() has a duplicate filter, and a summary marks its rows as covered.
"""

import json
import time
from collections.abc import Callable

from agent import config, events, sql

PHASES = ("intent", "execution", "result")


SCHEMA = """
    CREATE TABLE IF NOT EXISTS jobs (id INTEGER PRIMARY KEY, kind TEXT, args TEXT, reason TEXT,
        status TEXT, attempts INTEGER DEFAULT 0, error TEXT, created REAL, updated REAL);
    CREATE TABLE IF NOT EXISTS steps (id INTEGER PRIMARY KEY, job_id INTEGER, name TEXT, phase TEXT,
        data TEXT, ts REAL);
    CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT);"""


def _db():
    return sql.connect(config.CHECKPOINT_DB, SCHEMA)


def submit(kind: str, args: dict | None = None, reason: str = "") -> int:
    """Put a job in the queue. The daemon runs it, even if it is started later.

    Args:
        kind (str): Job kind, a key of `daemon.JOBS`.
        args (dict, optional): Job inputs. Defaults to {}.
        reason (str, optional): Why the job exists, shown in the dashboard feed. Defaults to "".

    Returns:
        int: The job id.
    """
    now = time.time()
    with _db() as db:
        cur = db.execute(
            "INSERT INTO jobs (kind, args, reason, status, created, updated) VALUES (?,?,?,?,?,?)",
            (kind, json.dumps(args or {}), reason, "queued", now, now),
        )
    events.emit("checkpoint.queued", job_id=cur.lastrowid, job=kind, reason=reason)
    return int(cur.lastrowid or 0)


def claim_next() -> dict | None:
    """Mark the oldest queued job as running and return it.

    Returns:
        dict | None: The job with `id`, `kind`, `args`, `reason` and `attempts`, or None if the queue is empty.
    """
    with _db() as db:
        row = db.execute(
            "SELECT id, kind, args, reason, attempts FROM jobs WHERE status = 'queued' ORDER BY id LIMIT 1"
        ).fetchone()
        if not row:
            return None
        db.execute(
            "UPDATE jobs SET status = 'running', attempts = attempts + 1, updated = ? WHERE id = ?",
            (time.time(), row[0]),
        )
    return {"id": row[0], "kind": row[1], "args": json.loads(row[2]), "reason": row[3], "attempts": row[4] + 1}


def recover() -> list[int]:
    """Put the jobs that were running when the process died back in the queue.

    Returns:
        list[int]: Ids of the recovered jobs.
    """
    with _db() as db:
        ids = [r[0] for r in db.execute("SELECT id FROM jobs WHERE status = 'running'")]
        db.execute("UPDATE jobs SET status = 'queued', updated = ? WHERE status = 'running'", (time.time(),))
    for i in ids:
        done = [s["name"] for s in steps(i) if s["phase"] == "result"]
        events.emit("checkpoint.recovered", job_id=i, steps_done=done)
    return ids


def _record(job_id: int, name: str, phase: str, data) -> None:
    with _db() as db:
        db.execute(
            "INSERT INTO steps (job_id, name, phase, data, ts) VALUES (?,?,?,?,?)",
            (job_id, name, phase, json.dumps(data, default=str), time.time()),
        )
    events.emit(f"checkpoint.{phase}", job_id=job_id, step=name)


def stored_result(job_id: int, name: str):
    """Return the stored output of a step that finished.

    Args:
        job_id (int): The job.
        name (str): The step name.

    Returns:
        tuple[bool, Any]: (True, output) if the step has a result, else (False, None).
    """
    with _db() as db:
        row = db.execute(
            "SELECT data FROM steps WHERE job_id = ? AND name = ? AND phase = 'result' ORDER BY id DESC LIMIT 1",
            (job_id, name),
        ).fetchone()
    return (True, json.loads(row[0])) if row else (False, None)


def step(job_id: int, name: str, fn: Callable, intent: dict | None = None):
    """Run one step of a job, or reuse its result if an earlier attempt finished it.

    Args:
        job_id (int): The job.
        name (str): A step name that is unique in the job, for example "summarize:3".
        fn (Callable): Runs the step with no arguments. Its return value must be JSON serializable.
        intent (dict, optional): What the step will do, recorded before it runs. Defaults to {}.

    Returns:
        Any: The output of `fn`, or the stored output.
    """
    done, value = stored_result(job_id, name)
    if done:
        events.emit("checkpoint.skipped", job_id=job_id, step=name)
        return value
    _record(job_id, name, "intent", intent or {})
    _record(job_id, name, "execution", {})
    value = fn()
    _record(job_id, name, "result", value)
    return value


def finish(job_id: int, status: str = "done", error: str | None = None) -> None:
    """Close a job.

    Args:
        job_id (int): The job.
        status (str, optional): "done" or "failed". Defaults to "done".
        error (str, optional): Why it failed. Defaults to None.
    """
    with _db() as db:
        db.execute(
            "UPDATE jobs SET status = ?, error = ?, updated = ? WHERE id = ?", (status, error, time.time(), job_id)
        )
    events.emit("checkpoint.finished", job_id=job_id, status=status, error=error)


def jobs(limit: int = 50) -> list[dict]:
    """List the newest jobs.

    Args:
        limit (int, optional): How many. Defaults to 50.

    Returns:
        list[dict]: Jobs, newest first.
    """
    with _db() as db:
        rows = db.execute(
            "SELECT id, kind, args, reason, status, attempts, error, created, updated FROM jobs ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    keys = ("id", "kind", "args", "reason", "status", "attempts", "error", "created", "updated")
    return [{**dict(zip(keys, r, strict=True)), "args": json.loads(r[2])} for r in rows]


def get_job(job_id: int) -> dict | None:
    """Return one job with its steps.

    Args:
        job_id (int): The job.

    Returns:
        dict | None: The job and its `steps`, or None if there is no such job.
    """
    with _db() as db:
        row = db.execute("SELECT status, kind, attempts FROM jobs WHERE id = ?", (job_id,)).fetchone()
    if not row:
        return None
    return {"id": job_id, "status": row[0], "kind": row[1], "attempts": row[2], "steps": steps(job_id)}


def steps(job_id: int) -> list[dict]:
    """List every recorded phase of every step of a job, in order.

    Args:
        job_id (int): The job.

    Returns:
        list[dict]: `name`, `phase`, `data` and `ts` for each record.
    """
    with _db() as db:
        rows = db.execute("SELECT name, phase, data, ts FROM steps WHERE job_id = ? ORDER BY id", (job_id,)).fetchall()
    return [{"name": n, "phase": p, "data": json.loads(d), "ts": ts} for n, p, d, ts in rows]


def get(key: str, default=None):
    """Read a small value that must survive a restart, for example when a schedule last ran.

    Args:
        key (str): The name.
        default (Any, optional): Returned when the key is missing. Defaults to None.

    Returns:
        Any: The stored JSON value, or `default`.
    """
    with _db() as db:
        row = db.execute("SELECT value FROM kv WHERE key = ?", (key,)).fetchone()
    return json.loads(row[0]) if row else default


def put_unless(key: str, value, taken: Callable) -> object | None:
    """Store a value only if the stored one is not taken, in one write transaction.

    Two processes that call this at the same time cannot both win: SQLite lets one writer in at a time.

    Args:
        key (str): The name.
        value (Any): A JSON-serializable value.
        taken (Callable): Gets the stored value (or None) and returns True if it must not be replaced.

    Returns:
        object | None: None if `value` was stored, or the stored value that was taken.
    """
    with _db() as db:
        db.execute("BEGIN IMMEDIATE")  # take the write lock before the read
        row = db.execute("SELECT value FROM kv WHERE key = ?", (key,)).fetchone()
        old = json.loads(row[0]) if row else None
        if taken(old):
            return old
        db.execute("INSERT OR REPLACE INTO kv (key, value) VALUES (?, ?)", (key, json.dumps(value)))
    return None


def put(key: str, value) -> None:
    """Store a small value that must survive a restart.

    Args:
        key (str): The name.
        value (Any): A JSON-serializable value.
    """
    with _db() as db:
        db.execute("INSERT OR REPLACE INTO kv (key, value) VALUES (?, ?)", (key, json.dumps(value)))
