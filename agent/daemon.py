"""The always-on daemon: one process that runs jobs from a queue, on a schedule and on triggers.

    python cli.py daemon run        # foreground (Ctrl-C stops it)
    python cli.py daemon start      # background; `status`, `stop`, `kill` (for the crash demo)

Main loop, once per DAEMON_TICK_S:
    1. schedule  config.SCHEDULE ("every 900", "at 18:00"): queue each job that is due.
                 A slot that passed while the daemon was down runs late and is reported as missed.
    2. triggers  e.g. many new transcripts -> summarize now, without waiting for the schedule
    3. one job   claim the oldest queued job and run it; every step is checkpointed (checkpoint.py)

A heartbeat thread writes "alive" every DAEMON_HEARTBEAT_S, even while a long job runs. On start,
`checkpoint.recover()` re-queues the job that was running when the process died, and its finished
steps are not run again. Every job the daemon finishes goes to the dashboard feed with its reason.
"""

import os
import signal
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import cast

import psutil

from agent import checkpoint, config, events, platforms, serving

JOBS: dict[str, Callable[[dict, Callable], dict]] = {}


def job(kind: str):
    """Register a job handler. A handler takes (job, step) and returns a JSON-serializable summary.

    Args:
        kind (str): The job kind that `checkpoint.submit` uses.

    Returns:
        Callable: A decorator that registers the function and returns it unchanged.
    """

    def register(fn):
        JOBS[kind] = fn
        return fn

    return register


# --- Jobs ---------------------------------------------------------------------------


@job("consolidate")
def _consolidate(j: dict, step: Callable) -> dict:
    from agent.memory import consolidate

    done = consolidate.consolidate(force=bool(j["args"].get("force")), step=step)
    return {"windows": len(done), "facts": sum(len(d.get("fact_ids", [])) for d in done)}


@job("brief")
def _brief(j: dict, step: Callable) -> dict:
    from agent import tasks

    out = step("write", lambda: {k: v for k, v in tasks.daily_brief().items() if k != "memories"})
    return {"brief_id": out["brief_id"], "version": out["version"], "headline": out["data"]["headline"]}


@job("reflect")
def _reflect(j: dict, step: Callable) -> dict:
    from agent import evolve

    out = step("reflect", lambda: evolve.reflect())
    return {"version": out["version"], "rejected": out["rejected"]}


@job("forget")
def _forget(j: dict, step: Callable) -> dict:
    from agent.memory import consolidate

    return step("forget", consolidate.forget)


@job("prune_events")
def _prune(j: dict, step: Callable) -> dict:
    return {"deleted": step("prune", events.prune)}


@job("replay")
def _replay(j: dict, step: Callable) -> dict:
    """Replay a recording through the ambient listener, one checkpointed step per utterance."""
    from agent.senses import ambient

    a, t0 = j["args"], time.time() if j.get("attempts", 1) == 1 else None
    source = "replay:" + Path(a["path"]).name
    heard = 0
    for i, u in enumerate(ambient.utterances(ambient.load_replay(Path(a["path"])), a["start_ts"])):
        name = f"utterance:{i}"
        if a.get("speed") and not checkpoint.stored_result(j["id"], name)[0]:
            t0 = t0 if t0 is not None else time.time() - u["offset_s"] / a["speed"]  # resumed: go on from here
            time.sleep(max(0.0, t0 + u["offset_s"] / a["speed"] - time.time()))  # play at `speed` x real time
        step(name, lambda u=u: ambient.hear(u, source, a.get("engine")), {"offset_s": round(u["offset_s"], 1)})
        heard += 1
    return {"utterances": heard}


@job("sleep")
def _sleep(j: dict, step: Callable) -> dict:
    """A job that only waits, one step at a time. The daemon eval kills the process in the middle of it."""
    n, seconds = int(j["args"].get("steps", 5)), float(j["args"].get("seconds", 1))
    for i in range(n):
        step(f"sleep:{i}", lambda i=i: (time.sleep(seconds), i)[1], {"seconds": seconds})
    return {"steps": n}


# --- Schedule and triggers -----------------------------------------------------------


def _slot(spec: str, now: float) -> float:
    """Return the start of the newest slot of a schedule at `now` (epoch seconds)."""
    how, value = spec.split()
    if how == "every":
        return now - now % float(value)
    hour, minute = map(int, value.split(":"))
    today = time.mktime(time.localtime(now)[:3] + (hour, minute, 0, 0, 0, -1))
    return today if today <= now else today - 86400


def _skipped(spec: str, slot: float, last: float) -> int:
    """Count the slots between the last run and this one that never ran (they are not made up)."""
    how, value = spec.split()
    step = float(value) if how == "every" else 86400.0
    return max(0, round((slot - last) / step) - 1)


def due(now: float | None = None) -> list[tuple[str, str, str]]:
    """Find the scheduled jobs whose newest slot has not run yet, and remember that they ran.

    The first time the daemon sees a schedule it only records the slot, so a first start at 20:00
    does not write a brief for 18:00.

    Args:
        now (float, optional): The current time. Defaults to `time.time()`.

    Returns:
        list[tuple[str, str, str]]: (schedule name, job kind, reason) for each job to queue.
    """
    now = now or time.time()
    out = []
    for name, (kind, spec) in config.SCHEDULE.items():
        slot, last = _slot(spec, now), checkpoint.get(f"schedule:{name}")
        if last is None or slot <= last:
            checkpoint.put(f"schedule:{name}", max(slot, last or 0))
            continue
        late = now - slot
        reason = f"schedule '{spec}'"
        if late > max(60.0, 3 * config.DAEMON_TICK_S):  # the daemon was down (or busy) at the slot time
            reason += f", {late / 60:.0f} min late"
            events.emit(
                "daemon.missed", schedule=name, slot=slot, late_s=round(late), skipped=_skipped(spec, slot, last)
            )
        checkpoint.put(f"schedule:{name}", slot)
        out.append((name, kind, reason))
    return out


def open_jobs(kind: str) -> int:
    """Count the jobs of one kind that are queued or running.

    Args:
        kind (str): Job kind.

    Returns:
        int: How many.
    """
    return sum(1 for j in checkpoint.jobs(200) if j["kind"] == kind and j["status"] in ("queued", "running"))


def triggers(now: float | None = None) -> list[tuple[str, dict, str]]:
    """Check the conditions that queue a job before its scheduled time.

    Many new transcripts queue a summary as soon as a window closes, not at the next schedule slot. The job
    summarizes only closed windows: a conversation that goes on is not cut into two summaries.

    Args:
        now (float, optional): The current time. Defaults to `time.time()`.

    Returns:
        list[tuple[str, dict, str]]: (job kind, args, reason) for each job to queue.
    """
    from agent.memory import consolidate, episodic

    waiting = len(episodic.query_rows("level = 'raw' AND kind = 'transcript' AND covered_by IS NULL"))
    if waiting < config.TRANSCRIPTS_PER_SUMMARY or open_jobs("consolidate") or not consolidate.pending(now=now):
        return []
    return [("consolidate", {}, f"trigger: {waiting} new transcripts")]


# --- The process -------------------------------------------------------------------


def status() -> dict:
    """Report whether a daemon runs, from its last heartbeat.

    Returns:
        dict: `alive`, `pid`, `age_s` (seconds since the heartbeat), `started`, `listening`, `mic_warning`
            (the silent-microphone warning, or None), `job`, `queued`, `want_listen` (listen now) and `listen`
            (the switch: `on`, `until`, `window` and `note`, why it waits).
    """
    beat = checkpoint.get("daemon") or {}
    queued = sum(1 for j in checkpoint.jobs(200) if j["status"] == "queued")
    age = round(time.time() - beat.get("ts", 0), 1)
    listen = {**listen_state(), "note": listen_note()}
    return {
        **beat,
        "alive": _alive(beat),
        "age_s": age,
        "queued": queued,
        "want_listen": wants_listening(),
        "listen": listen,
    }


def _alive(beat: dict | None) -> bool:
    """Return whether a heartbeat comes from a daemon that still runs (3 missed beats = not running)."""
    beat = beat or {}
    pid, age = beat.get("pid"), time.time() - beat.get("ts", 0)
    return bool(pid) and age < 3 * config.DAEMON_HEARTBEAT_S and psutil.pid_exists(pid)


# --- Always-on listening: the switch, an end time and a daily window ------------------


def _minutes(hhmm: str) -> int:
    """Turn "HH:MM" into minutes after midnight."""
    try:
        hour, minute = map(int, hhmm.split(":"))
    except ValueError:
        hour = minute = -1
    if not (0 <= hour < 24 and 0 <= minute < 60):
        raise ValueError(f"not a time of day: {hhmm!r} (use HH:MM, for example 09:00-17:00)")
    return hour * 60 + minute


def in_window(window: list[str] | tuple[str, str] | None, now: float) -> bool:
    """Return whether `now` falls inside a daily window of local time.

    Args:
        window (list[str] | tuple[str, str] | None): ("HH:MM", "HH:MM"), start included, end not. An end before the
            start crosses midnight. None or empty = any time.
        now (float): Epoch seconds.

    Returns:
        bool: True inside the window.
    """
    if not window:
        return True
    start, end = (_minutes(t) for t in window)
    t = time.localtime(now)
    m = t.tm_hour * 60 + t.tm_min
    return start <= m < end if start <= end else m >= start or m < end


def listen_state() -> dict:
    """Read the listening switch as the user set it.

    Returns:
        dict: `on` (bool), `until` (epoch seconds or None) and `window` (the daily window that applies, or None).
    """
    raw = checkpoint.get("listen", False)
    st = raw if isinstance(raw, dict) else {"on": bool(raw)}  # older versions stored only True or False
    window = st.get("window")
    return {
        "on": bool(st.get("on")),
        "until": st.get("until"),
        "window": config.LISTEN_WINDOW if window is None else window or None,
    }


def listen_note(now: float | None = None) -> str | None:
    """Say why the switch is on but the daemon does not listen.

    Args:
        now (float, optional): The current time. Defaults to `time.time()`.

    Returns:
        str | None: "time up", "outside HH:MM-HH:MM", or None.
    """
    now, st = now or time.time(), listen_state()
    if not st["on"]:
        return None
    if st["until"] and now >= st["until"]:
        return "time up"
    if not in_window(st["window"], now):
        return "outside " + "-".join(st["window"])
    return None


def wants_listening(now: float | None = None) -> bool:
    """Return whether the daemon must listen now: the switch is on, the end time has not passed, inside the window.

    The microphone pause switch (`audio.paused()`) is not read here: the listener checks it and it always wins.

    Args:
        now (float, optional): The current time. Defaults to `time.time()`.

    Returns:
        bool: True to listen.
    """
    return listen_state()["on"] and listen_note(now) is None


def set_listening(on: bool, for_s: float | None = None, window: list[str] | None = None, reason: str = "") -> None:
    """Ask the running daemon to start or stop always-on listening. The pause switch still wins.

    Args:
        on (bool): True to listen.
        for_s (float, optional): Stop after this many seconds. None = until switched off. Defaults to None.
        window (list[str], optional): A daily window ["HH:MM", "HH:MM"] for this switch. None = `config.LISTEN_WINDOW`,
            [] = any time of day. Defaults to None.
        reason (str, optional): Why it changed, for the feed. Defaults to "".

    Raises:
        ValueError: If `window` does not hold two "HH:MM" times, or `for_s` is not positive.
    """
    if window:
        if len(window) != 2:
            raise ValueError("a window is two times: start and end")
        for t in window:
            _minutes(t)
    if for_s is not None and for_s <= 0:
        raise ValueError("the duration must be positive")
    until = time.time() + for_s if on and for_s else None
    # "off" stays a plain False, which a daemon of an older version also reads as off
    checkpoint.put("listen", {"on": True, "until": until, "window": window} if on else False)
    events.emit("daemon.listen", want=bool(on), until=until, window=listen_state()["window"], reason=reason)


def run_or_queue(kind: str, args: dict, reason: str) -> dict:
    """Queue a job for the running daemon, or run the queue here when no daemon runs.

    Without a daemon, the older queued jobs run first, as the daemon would run them, and every job goes
    through the same handlers and checkpoints.

    Args:
        kind (str): Job kind, a key of `JOBS`.
        args (dict): Job inputs.
        reason (str): Why, for the feed.

    Returns:
        dict: `job_id`, `queued` (True if the daemon will run it) and `job` (with steps, when it ran here).
    """
    job_id = checkpoint.submit(kind, args, reason)
    if status()["alive"]:
        return {"job_id": job_id, "queued": True}
    runner = Daemon()
    while (j := checkpoint.claim_next()) is not None:
        runner.run_job(j)
        if j["id"] == job_id:
            break
    return {"job_id": job_id, "queued": False, "job": checkpoint.get_job(job_id)}


class Daemon:
    """The daemon process: heartbeat thread, main loop and the optional ambient listener."""

    def __init__(self, models: bool = True):
        """Prepare the daemon without starting anything.

        Args:
            models (bool, optional): Start and watch the model servers. False for the recovery eval. Defaults to True.
        """
        self.models, self.stop_flag = models, threading.Event()
        self.current: dict | None = None
        self.listener = None
        self.started = time.time()

    def heartbeat(self) -> dict:
        """Build the heartbeat row that `status()` reads.

        Returns:
            dict: `pid`, `ts`, `started`, `job`, `listening`, `mic_warning` and `rss_mb`.
        """
        j = self.current
        return {
            "pid": os.getpid(),
            "ts": time.time(),
            "started": self.started,
            "job": {"id": j["id"], "kind": j["kind"]} if j else None,
            "listening": self.listener is not None,
            "mic_warning": self.listener.warning if self.listener else None,
            "rss_mb": round(psutil.Process().memory_info().rss / 1e6, 1),
        }

    def beat(self) -> None:
        """Write the heartbeat until the daemon stops."""
        while not self.stop_flag.is_set():
            checkpoint.put("daemon", self.heartbeat())
            self.stop_flag.wait(config.DAEMON_HEARTBEAT_S)

    def claim(self) -> int | None:
        """Write the first heartbeat, unless another daemon still beats. Two daemons would share one runtime folder.

        Returns:
            int | None: None if this daemon may run, or the pid of the daemon that already runs.
        """
        other = checkpoint.put_unless("daemon", self.heartbeat(), lambda b: _alive(b) and b["pid"] != os.getpid())
        return cast(dict, other)["pid"] if other else None

    def _boot(self) -> None:
        if not self.models:
            return
        from agent import mcp_client, proactive

        threading.Thread(target=mcp_client.start, daemon=True).start()  # the enabled MCP tools, for the model
        threading.Thread(target=proactive.watch_reminders, args=(self.stop_flag,), daemon=True).start()
        llm, emb = serving.llm_server(), serving.embed_server()
        llm.start()
        emb.start()
        for server in (llm, emb):
            threading.Thread(target=serving.watch, args=(lambda s=server: s,), daemon=True).start()

    def _sync_listener(self) -> None:
        """Start or stop the ambient listener to match the user's switch. A switch whose time is up turns off."""
        if listen_note() == "time up":
            set_listening(False, reason="time up")
        want = self.models and wants_listening()
        if want and self.listener is None:
            from agent.senses import ambient

            self.listener = ambient.Listener()
            self.listener.start()
        elif not want and self.listener is not None:
            self.listener.stop()
            self.listener = None

    def run_job(self, j: dict) -> None:
        """Run one claimed job, checkpoint every step, and report it in the feed.

        Args:
            j (dict): A job from `checkpoint.claim_next`.
        """
        self.current = j
        step = lambda name, fn, intent=None: checkpoint.step(j["id"], name, fn, intent)  # noqa: E731
        try:
            summary = JOBS[j["kind"]](j, step)
            checkpoint.finish(j["id"])
            status = "done"
        except Exception as e:  # noqa: BLE001 - a failed job must not stop the daemon
            summary, status = {"error": f"{type(e).__name__}: {e}"}, "failed"
            checkpoint.finish(j["id"], "failed", summary["error"])
        finally:
            self.current = None
        events.emit("daemon.action", job_id=j["id"], job=j["kind"], reason=j["reason"], status=status, summary=summary)
        if status == "done" and j["kind"] in config.NOTIFY_JOBS:
            self._notify(j["kind"], summary)

    def _notify(self, kind: str, summary: dict) -> None:
        try:
            platforms.current().notify("multimodal-agent", f"{kind}: {summary.get('headline') or 'done'}")
        except Exception as e:  # noqa: BLE001 - no notification hook: the dashboard feed still has it
            events.emit("daemon.warning", message=f"notification failed: {e}")

    def tick(self) -> None:
        """Queue what the schedule and the triggers ask for, then run one job."""
        for _name, kind, reason in due():
            checkpoint.submit(kind, {}, reason)
        if self.models:
            for kind, args, reason in triggers():
                checkpoint.submit(kind, args, reason)
        self._sync_listener()
        j = checkpoint.claim_next()
        if j and j["kind"] in JOBS:
            self.run_job(j)
        elif j:
            checkpoint.finish(j["id"], "failed", f"unknown job kind '{j['kind']}'")

    def run(self) -> int | None:
        """Run until SIGTERM or Ctrl-C. A kill -9 is the crash case that checkpoints cover.

        Returns:
            int | None: None after a normal run, or the pid of a daemon that already runs (this one did not start).
        """
        events.set_source("daemon")
        if (other := self.claim()) is not None:
            events.emit("daemon.refused", pid=os.getpid(), running_pid=other)
            return other
        signal.signal(signal.SIGTERM, lambda *_: self.stop_flag.set())
        threading.Thread(target=self.beat, daemon=True).start()
        recovered = checkpoint.recover()
        events.emit("daemon.started", pid=os.getpid(), recovered=recovered, models=self.models)
        awake = self._keep_awake()
        try:
            self._boot()
            while not self.stop_flag.is_set():
                self.tick()
                self.stop_flag.wait(config.DAEMON_TICK_S)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop_flag.set()
            if self.listener:
                self.listener.stop()
            if awake:
                awake.stop()
            checkpoint.put("daemon", {"pid": None, "ts": 0, "stopped": time.time()})
            events.emit("daemon.stopped", pid=os.getpid())
        return None

    def _keep_awake(self):
        try:
            return platforms.current().keep_awake()
        except Exception as e:  # noqa: BLE001 - no keep-awake hook: run anyway, and say so
            events.emit("daemon.warning", message=f"keep-awake not available: {e}")
            return None
