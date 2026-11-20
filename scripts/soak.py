"""Soak test: run a daemon for hours under a steady load, and record its memory and any missed events.

    python scripts/soak.py --hours 3            # record it before the session; the daemon eval reads the result
    python scripts/soak.py --hours 0.1          # a 6-minute check that the script works

It starts its own daemon on a throwaway runtime folder (AGENT_RUNTIME), so your memory stays clean. The
model servers are shared, so start them first if they are not running (`python cli.py serve` in another terminal,
or let this script start them). Every `--every` seconds it queues a replay of the synthetic morning conversation
(Whisper -> memory), and the daemon's own schedule and triggers summarize it. Once a minute it samples the
memory of the daemon and the servers.

Missed events = queued jobs that never finished + schedule slots the daemon reported as missed
+ heartbeats that came more than 3 beats late + samples where the daemon was not running at all.
Memory growth is a straight-line fit of the daemon's resident memory after the first 5 minutes
(loading the models and Whisper is not growth), so a soak shorter than about 10 minutes has no growth
number. A leak of a few MB per hour only shows over hours: 3 hours is the run the eval expects.

A sleeping machine runs nothing. Two samples more than SLEEP_GAP_S apart mean the machine slept in between
(keep-awake stops idle sleep, not a closed lid). Schedule slots that fell in such a gap are counted as
`missed_while_asleep`, not as missed events, and `awake_hours` leaves the gaps out. Keep the lid open and the
power connected for a soak.
Results: evals/results/soak/<time>.json
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUN = time.strftime("%Y%m%d-%H%M%S")
os.environ["AGENT_RUNTIME"] = str(ROOT / "evals" / "results" / "soak" / f"runtime-{RUN}")
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import psutil  # noqa: E402

from agent import checkpoint, config, daemon, events, platforms, serving  # noqa: E402
from evals import synthetic  # noqa: E402

WARMUP_S = 300  # model and Whisper loading is not growth
SLEEP_GAP_S = 180  # samples come every 60 s; a longer gap means the machine was asleep


def rss_mb(pid: int | None) -> float | None:
    """Return the resident memory of a process in MB, or None if it is gone."""
    try:
        return psutil.Process(pid).memory_info().rss / 1e6 if pid else None
    except psutil.NoSuchProcess:
        return None


def sample(t0: float) -> dict:
    """Measure the daemon, the servers and the machine once."""
    st = daemon.status()
    return {
        "t_s": round(time.time() - t0, 1),
        "alive": st["alive"],
        "heartbeat_age_s": st["age_s"],
        "daemon_mb": rss_mb(st.get("pid")),
        "llm_gb": round(serving.llm_server().rss_gb(), 2),
        "embed_gb": round(serving.embed_server().rss_gb(), 2),
        "system_used_gb": round(psutil.virtual_memory().used / 1e9, 2),
        "queued": st["queued"],
    }


def growth_mb_per_h(samples: list[dict]) -> float | None:
    """Fit a straight line to the daemon memory after the warm-up. None if too few samples."""
    pts = [(s["t_s"], s["daemon_mb"]) for s in samples if s["t_s"] >= WARMUP_S and s["daemon_mb"]]
    if len(pts) < 3:
        return None
    t, mb = np.array(pts).T
    return round(float(np.polyfit(t / 3600, mb, 1)[0]), 2)


def sleep_gaps(samples: list[dict], end_s: float) -> list[tuple[float, float]]:
    """Find the stretches (seconds after the start) when the machine slept: no sample for SLEEP_GAP_S or more."""
    times = [s["t_s"] for s in samples] + [end_s]
    return [(a, b) for a, b in zip(times, times[1:], strict=False) if b - a > SLEEP_GAP_S]


def missed_slots_split(t0: float, gaps: list[tuple[float, float]]) -> tuple[list[dict], list[dict]]:
    """Split the schedule slots that the daemon reported as missed into (while awake, while asleep)."""
    slots = [e for e in events.since(0, "daemon.missed") if e["ts"] >= t0]
    asleep = [e for e in slots if any(a <= e["slot"] - t0 <= b for a, b in gaps)]
    return [e for e in slots if e not in asleep], asleep


def summarize(samples: list[dict], submitted: list[int], t0: float, end: float | None = None) -> dict:
    """Count missed events and compute the growth. Slots that passed while the machine slept are counted apart."""
    end = end or time.time()
    jobs = {j["id"]: j for j in checkpoint.jobs(10_000)}
    unfinished = [i for i in submitted if jobs.get(i, {}).get("status") != "done"]
    gaps = sleep_gaps(samples, end - t0)
    missed_slots, asleep = missed_slots_split(t0, gaps)
    late_beats = [s for s in samples if s["alive"] and s["heartbeat_age_s"] > 3 * config.DAEMON_HEARTBEAT_S]
    down = [s for s in samples if not s["alive"]]
    return {
        "soak_hours": round((end - t0) / 3600, 2),
        "awake_hours": round((end - t0 - sum(b - a for a, b in gaps)) / 3600, 2),
        "asleep_gaps": len(gaps),
        "missed_while_asleep": len(asleep),
        "rss_growth_mb_per_h": growth_mb_per_h(samples),
        "missed_events": len(unfinished) + len(missed_slots) + len(late_beats) + len(down),
        "unfinished_jobs": unfinished,
        "missed_slots": len(missed_slots),
        "late_heartbeats": len(late_beats),
        "down_samples": len(down),
        "jobs_submitted": len(submitted),
        "jobs_failed": sum(1 for j in jobs.values() if j["status"] == "failed"),
        "peak_daemon_mb": max((s["daemon_mb"] or 0) for s in samples),
        "peak_system_gb": max(s["system_used_gb"] for s in samples),
    }


def start_daemon() -> int:
    """Start the soak's own daemon and wait for its first heartbeat."""
    cmd = [sys.executable, str(ROOT / "cli.py"), "daemon", "run"]
    pid = platforms.current().spawn_background(cmd, ROOT, config.RUNTIME / "daemon.log")
    for _ in range(300):
        if daemon.status()["alive"]:
            return pid
        time.sleep(1)
    raise SystemExit(f"the soak daemon did not start; see {config.RUNTIME / 'daemon.log'}")


def main() -> None:
    """Run the soak and write the result file."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hours", type=float, default=3.0)
    ap.add_argument("--every", type=float, default=300, help="seconds between replays of the conversation")
    a = ap.parse_args()
    synthetic.ensure("task3")
    print(f"soak runtime: {config.RUNTIME}")
    pid = start_daemon()
    t0, samples, submitted, next_replay = time.time(), [], [], 0.0
    wav = str(config.EVAL_SYNTHETIC / "task3" / "conversation.wav")
    load_end = t0 + a.hours * 3600
    end = load_end + min(300, a.every)  # let the last job finish; keep sampling, or the wait reads as a sleep gap
    try:
        while time.time() < end:
            if time.time() < load_end and time.time() >= next_replay:
                args = {"path": wav, "start_ts": time.time(), "speed": 0}
                submitted.append(checkpoint.submit("replay", args, reason="soak load"))
                next_replay = time.time() + a.every
            samples.append(sample(t0))
            s = samples[-1]
            print(f"{s['t_s'] / 60:6.1f} min  daemon {s['daemon_mb']} MB  queued {s['queued']}  alive {s['alive']}")
            time.sleep(60)
    finally:
        if psutil.pid_exists(pid):  # a daemon that died is the finding: still write the result
            psutil.Process(pid).terminate()
    out = config.RESULTS / "soak" / f"{RUN}.json"
    out.write_text(json.dumps({"summary": summarize(samples, submitted, t0), "samples": samples}, indent=1))
    print(json.dumps(json.loads(out.read_text())["summary"], indent=1))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
