"""Session 2 endpoints of the dashboard: the daemon, consolidation, self-evolution, Task 3 and Task 4.

The daemon is its own process. The dashboard never runs its jobs while it is alive: it queues them
in the checkpoint database, and the daemon picks them up. Without a daemon, the dashboard runs
the job itself, through the same job handlers, so the feed and the checkpoints look the same.
"""

import sys
import time

import psutil
from fastapi import APIRouter, HTTPException

from agent import checkpoint, config, daemon, events, evolve, platforms, tasks
from agent.memory import consolidate, episodic

router = APIRouter(prefix="/api")
REPLAY = config.EVAL_SYNTHETIC / "task3" / "conversation.wav"  # the synthetic morning, for the live demo


# --- Daemon -----------------------------------------------------------------------------


@router.get("/daemon")
def daemon_state():
    """Report the daemon, its schedule and the newest jobs.

    Returns:
        dict: `status`, `schedule` (name, job, when, last slot) and `jobs`.
    """
    schedule = [
        {"name": n, "job": kind, "when": spec, "last": checkpoint.get(f"schedule:{n}")}
        for n, (kind, spec) in config.SCHEDULE.items()
    ]
    return {"status": daemon.status(), "schedule": schedule, "jobs": checkpoint.jobs(25)}


@router.get("/daemon/job/{job_id}")
def daemon_job(job_id: int):
    """Return one job with every recorded phase of its steps.

    Args:
        job_id (int): The job.

    Returns:
        dict: The job.

    Raises:
        HTTPException: 404 if there is no such job.
    """
    j = checkpoint.get_job(job_id)
    if not j:
        raise HTTPException(404, f"no job {job_id}")
    return j


@router.post("/daemon/listen")
def daemon_listen(body: dict):
    """Ask the daemon to listen all the time, or to stop. The microphone pause switch still wins.

    Args:
        body (dict): JSON with `on` (bool).

    Returns:
        dict: The daemon state.
    """
    daemon.set_listening(bool(body["on"]))
    return daemon_state()


@router.post("/daemon/start")
def daemon_start():
    """Start the daemon in the background, as `python cli.py daemon start` does.

    Returns:
        dict: The daemon state.
    """
    if not daemon.status()["alive"]:
        cmd = [sys.executable, str(config.ROOT / "cli.py"), "daemon", "run"]
        platforms.current().spawn_background(cmd, config.ROOT, config.DAEMON_LOG)
        time.sleep(1.0)  # let it write its first heartbeat
    return daemon_state()


@router.post("/daemon/stop")
def daemon_stop(body: dict):
    """Stop the daemon. With `kill`, end it with SIGKILL, as a crash would, for the recovery demo.

    Args:
        body (dict): JSON with optional `kill` (bool).

    Returns:
        dict: The daemon state.

    Raises:
        HTTPException: 409 if no daemon runs.
    """
    st = daemon.status()
    if not st["alive"]:
        raise HTTPException(409, "The daemon is not running.")
    proc = psutil.Process(st["pid"])
    if body.get("kill"):
        proc.kill()
        events.emit("daemon.killed", pid=st["pid"], by="dashboard")
    else:
        proc.terminate()
    return daemon_state()


@router.post("/jobs")
def submit_job(body: dict):
    """Queue (or run) one job by hand: consolidate, brief, reflect, forget.

    Args:
        body (dict): JSON with `kind` and optional `args`.

    Returns:
        dict: The result of `run_or_queue`.

    Raises:
        HTTPException: 400 for a kind that is not allowed from the page.
    """
    if body["kind"] not in ("consolidate", "brief", "reflect", "forget"):
        raise HTTPException(400, f"cannot start '{body['kind']}' from the dashboard")
    return daemon.run_or_queue(body["kind"], body.get("args", {}), "started from the dashboard")


# --- Memory consolidation -------------------------------------------------------------------


@router.get("/memory/stats")
def memory_stats():
    """Count memories per level, and the raw rows that wait for a summary.

    Returns:
        dict: `levels` from `consolidate.stats()`, `waiting` and `retention_days`.
    """
    waiting = len(episodic.query_rows("level = 'raw' AND covered_by IS NULL"))
    return {"levels": consolidate.stats(), "waiting": waiting, "retention_days": config.RAW_RETENTION_DAYS}


# --- Self-evolution -----------------------------------------------------------------------


@router.get("/evolve")
def evolve_state():
    """Report every prompt version, the active prompt, the locked-block check and the feedback.

    Returns:
        dict: `versions`, `active`, `prompt`, `locked_broken`, `feedback` and `pending` (feedback not used yet).
    """
    active = evolve.get_version()
    prompt = evolve.render(active)
    return {
        "versions": evolve.versions(),
        "active": active,
        "prompt": prompt,
        "locked": list(evolve.LOCKED),
        "locked_broken": evolve.check_locked(prompt),
        "feedback": evolve.feedback_log(),
        "pending": len(evolve.unused_feedback()),
    }


@router.get("/evolve/diff")
def evolve_diff(a: int, b: int):
    """Show the diff between two prompt versions.

    Args:
        a (int): The older version.
        b (int): The newer version.

    Returns:
        dict: `a`, `b` and `diff` (unified diff text).
    """
    return {"a": a, "b": b, "diff": evolve.diff(a, b)}


@router.post("/evolve/rollback")
def evolve_rollback(body: dict):
    """Make an older prompt version active.

    Args:
        body (dict): JSON with `version`.

    Returns:
        dict: The new evolution state.
    """
    evolve.rollback(int(body["version"]))
    return evolve_state()


# --- Task 3 and Task 4 --------------------------------------------------------------------


@router.post("/task3/replay")
def task3_replay(body: dict):
    """Replay a recording through the ambient listener (default: the synthetic morning at 09:00).

    Args:
        body (dict): JSON with optional `path`, `at` ("HH:MM") and `speed` (0 = no waiting).

    Returns:
        dict: The result of `run_or_queue`.

    Raises:
        HTTPException: 404 if the recording does not exist.
    """
    path = body.get("path") or str(REPLAY)
    if not (config.ROOT / path).exists():
        raise HTTPException(404, f"No recording at {path}. Build it: python cli.py evals build task3")
    h, m = map(int, body.get("at", "09:00").split(":"))
    start = time.mktime(time.localtime()[:3] + (h, m, 0, 0, 0, -1))
    args = {"path": str(config.ROOT / path), "start_ts": start, "speed": float(body.get("speed", 0))}
    return daemon.run_or_queue("replay", args, "replay started from the dashboard")


@router.post("/task3/ask")
def task3_ask(body: dict):
    """Answer a question about earlier events from memory.

    Args:
        body (dict): JSON with `question`.

    Returns:
        dict: The result of `tasks.recall`.
    """
    return tasks.recall(body["question"])


@router.post("/task4/brief")
def task4_brief():
    """Write today's brief with the active prompt version.

    Returns:
        dict: The result of `tasks.daily_brief`.
    """
    return tasks.daily_brief()


@router.post("/task4/feedback")
def task4_feedback(body: dict):
    """Record feedback on a brief.

    Args:
        body (dict): JSON with `brief_id`, `rating` (+1 or -1) and `text`.

    Returns:
        dict: `feedback_id` and `pending` (feedback that waits for a reflection).
    """
    fid = evolve.add_feedback(body.get("brief_id"), int(body["rating"]), body.get("text", ""))
    return {"feedback_id": fid, "pending": len(evolve.unused_feedback())}


@router.post("/evolve/reflect")
def evolve_reflect():
    """Reflect on the unused feedback now, in this process, and write a new version if the patch is valid.

    Returns:
        dict: The result of `evolve.reflect`, plus `diff` when a new version was written.
    """
    out = evolve.reflect()
    if out["version"]:
        out["diff"] = evolve.diff(evolve.get_version(out["version"])["parent"], out["version"])
    return out
