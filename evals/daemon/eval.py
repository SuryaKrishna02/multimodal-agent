"""Daemon eval: does a killed daemon resume where it stopped, how fast, and does it stay small over hours?

Built in, no case files. Each recovery case starts a real daemon process on a throwaway runtime
folder (AGENT_RUNTIME), queues a `sleep` job of N one-second steps, kills the process with SIGKILL
after K steps have a result, starts a new daemon and waits for the job to finish.

The soak numbers come from the newest `scripts/soak.py` run (evals/results/soak/*.json). A soak runs
for hours, so you run it beforehand; this eval only reads it.

Metrics:
    recovery_s            seconds from the restart until the next unfinished step starts (worst case)
    redone_steps          finished steps that ran again after the restart (must be 0)
    lost_jobs             jobs that never finished after the restart
    soak_hours            length of the newest soak run, wall clock
    soak_awake_hours      the part of it the machine was awake (under SOAK_MIN_HOURS the record warns)
    rss_growth_mb_per_h   daemon memory growth over the soak, from a straight-line fit (None: too short to fit)
    missed_events         schedule slots and queued jobs that the soak saw go missing
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from agent import checkpoint, config

LEVEL, USES_LLM, BUILTIN = "component", False, True
NEEDS_LLM_SERVER = False
CASES = [
    {"op": "recovery", "steps": 6, "kill_after": 2},
    {"op": "recovery", "steps": 6, "kill_after": 4},
    {"op": "soak"},
]
TIMEOUT_S = 90
SOAK_MIN_HOURS = 1.0  # shorter soaks prove the script works, not that the daemon stays small


def load_cases(_cases_dir: Path):
    """Return the built-in cases.

    Args:
        _cases_dir (Path): Not used: the cases are built in.

    Returns:
        list[dict]: Two recovery cases and the soak reader.
    """
    return list(CASES)


def spawn(runtime: Path) -> subprocess.Popen:
    """Start a daemon without model servers on a throwaway runtime folder.

    Args:
        runtime (Path): The folder that AGENT_RUNTIME points at.

    Returns:
        subprocess.Popen: The daemon process.
    """
    env = {**os.environ, "AGENT_RUNTIME": str(runtime)}
    cmd = [sys.executable, "cli.py", "daemon", "run", "--no-models"]
    with open(runtime / "daemon-eval.log", "a") as log:  # the child keeps its own copy of the file handle
        return subprocess.Popen(cmd, cwd=config.ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)


def wait_for(check, timeout: float = TIMEOUT_S) -> bool:
    """Poll a condition every 0.1 s.

    Args:
        check (Callable[[], bool]): The condition.
        timeout (float, optional): Seconds to wait. Defaults to TIMEOUT_S.

    Returns:
        bool: True if the condition held before the timeout.
    """
    end = time.time() + timeout
    while time.time() < end:
        if check():
            return True
        time.sleep(0.1)
    return False


def _phases(job_id: int, phase: str) -> list[dict]:
    return [s for s in checkpoint.steps(job_id) if s["phase"] == phase]


def recovery(item: dict) -> dict:
    """Kill a daemon in the middle of a job, restart it, and measure the resume.

    Args:
        item (dict): `steps` of the sleep job and `kill_after` (results before the kill).

    Returns:
        dict: The record.
    """
    runtime = config.RESULTS / "daemon-eval" / f"run-{time.time_ns()}"
    runtime.mkdir(parents=True)
    real = (config.RUNTIME, config.CHECKPOINT_DB)
    config.RUNTIME, config.CHECKPOINT_DB = runtime, runtime / "checkpoints.sqlite"
    try:
        job_id = checkpoint.submit("sleep", {"steps": item["steps"], "seconds": 1}, reason="daemon eval")
        first = spawn(runtime)
        reached = wait_for(lambda: len(_phases(job_id, "result")) >= item["kill_after"])
        first.kill()  # SIGKILL: no cleanup code runs, as in a crash
        first.wait()
        done_before = {s["name"] for s in _phases(job_id, "result")}
        t_restart = time.time()
        second = spawn(runtime)
        finished = wait_for(lambda: (checkpoint.get_job(job_id) or {}).get("status") == "done")
        second.terminate()
        second.wait(timeout=15)
        starts = [s["ts"] for s in _phases(job_id, "execution") if s["ts"] > t_restart]
        redone = [s["name"] for s in _phases(job_id, "execution") if s["ts"] > t_restart and s["name"] in done_before]
    finally:
        config.RUNTIME, config.CHECKPOINT_DB = real
    return {
        "op": "recovery",
        "kill_after": item["kill_after"],
        "reached_kill_point": reached,
        "done_before_kill": sorted(done_before),
        "finished": finished,
        "recovery_s": round(min(starts) - t_restart, 2) if starts else None,
        "redone": redone,
    }


def soak(_item: dict) -> dict:
    """Read the newest soak result.

    Returns:
        dict: The record, with `found` False when no soak has run yet.
    """
    files = sorted((config.RESULTS / "soak").glob("*.json"))
    if not files:
        return {"op": "soak", "found": False, "hint": "run: python scripts/soak.py --hours 3"}
    rec = {"op": "soak", "found": True, "file": str(files[-1]), **json.loads(files[-1].read_text())["summary"]}
    awake = rec.get("awake_hours", rec["soak_hours"])
    if awake < SOAK_MIN_HOURS:
        rec["warning"] = f"only {awake} h awake: run python scripts/soak.py --hours 3, lid open, for a growth number"
    return rec


def run_item(item):
    """Run one case.

    Args:
        item (dict): A case from `load_cases`.

    Returns:
        dict: The record.
    """
    return recovery(item) if item["op"] == "recovery" else soak(item)


def score(records):
    """Compute the metrics.

    Args:
        records (list[dict]): The records from `run_item`.

    Returns:
        dict: Metric name to value. Soak metrics are left out until a soak has run.
    """
    rec = [r for r in records if r["op"] == "recovery"]
    times = [r["recovery_s"] for r in rec if r["recovery_s"] is not None]
    out = {
        "recovery_s": max(times) if times else None,
        "redone_steps": sum(len(r["redone"]) for r in rec),
        "lost_jobs": sum(not r["finished"] for r in rec),
    }
    for r in records:
        if r["op"] == "soak" and r.get("found"):
            out |= {k: r[k] for k in ("soak_hours", "rss_growth_mb_per_h", "missed_events")}
            out["soak_awake_hours"] = r.get("awake_hours", r["soak_hours"])
    return out
