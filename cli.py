"""Command line for the agent. Run `python cli.py -h` for the list.

    python cli.py serve                       start both model servers and wait
    python cli.py dashboard [--no-daemon]     the dashboard on http://127.0.0.1:8765, with the daemon (Ctrl-C stops both)
    python cli.py note                        Task 1: record a voice note (Enter to stop)
    python cli.py note --file note.wav        Task 1 from a file
    python cli.py look                        Task 2: screenshot now -> explanation + fix
    python cli.py look --watch 10             Task 2 on a 10 s screen recording (needs ffmpeg)
    python cli.py chat "what did I say about John Doe?"
    python cli.py record-clip clip.wav        record a clip from the mic (then: evals add asr clip.wav --text "...")
    python cli.py record-screen demo1         record frames for the screen eval (then: evals add screen demo1 --moment ...)
    python cli.py evals --run                 run every eval on the synthetic and your own cases
    python cli.py evals status                how many cases each eval has
    python cli.py evals add -h                add your own eval case
    python cli.py doctor [--exercise]         which OS + accelerator plug-ins were detected, and do they work

Session 2 (always on and learning):
    python cli.py daemon start|run|stop|kill|status    the always-on process (kill = crash it, for the demo)
    python cli.py daemon install|uninstall             start it at login (and after a crash)
    python cli.py listen on|off                        always-on listening in the daemon
    python cli.py listen on --for 45m --window 09:00-17:00   stop after 45 min; listen only in that window
    python cli.py replay morning.wav --at 09:00        Task 3: replay a recording as if it was heard live
    python cli.py ask "what did I decide about the venue this morning?"     Task 3: answer from memory
    python cli.py consolidate [--forget]               summarize raw memories now (and apply retention)
    python cli.py brief                                Task 4: write today's brief
    python cli.py feedback 3 --bad "too long"          Task 4: feedback on brief 3
    python cli.py evolve reflect|versions|diff A B|rollback N|simulate --rounds 3

Session 3 (voice, bolt-ons):
    python cli.py say "Hello, I am Sanjaya."          speak through the dashboard tab if Voice is on there, else here
    python cli.py say --local "..." | --voices         force this machine's speakers | list the Kokoro voices
    python cli.py wake record --count 40               record your own "Hey Sanjaya" clips for training
    python cli.py wake test                            listen and print each wake with its score

Everything OS- or GPU-specific lives in agent/platforms/ and agent/accel/. Commands here call
those plug-ins and never branch on the OS. Force a choice with AGENT_PLATFORM=<name> / AGENT_ACCEL=<name>.
"""

import argparse
import contextlib
import json
import signal
import subprocess
import sys
import time
from pathlib import Path

from agent import config, serving


def start_servers(llm=True, embed=True):
    """Start the model servers.

    Args:
        llm (bool, optional): Start the language model server. Defaults to True.
        embed (bool, optional): Start the embedding server. Defaults to True.
    """
    if llm:
        serving.llm_server().start()
    if embed:
        serving.embed_server().start()


def show(obj):
    """Print an object as indented JSON.

    Args:
        obj: Any JSON-serializable object. Other values print as strings.
    """
    print(json.dumps(obj, indent=1, default=str))


def cmd_serve(a):
    """`serve`: start both servers and keep them up until Enter or Ctrl-C.

    Args:
        a (argparse.Namespace): Parsed arguments, not used.
    """
    start_servers()
    print(f"llm ({config.TIER}) on :{config.LLM_PORT}, embeddings on :{config.EMBED_PORT}. Ctrl-C stops both.")
    with contextlib.suppress(KeyboardInterrupt, EOFError):
        input()
    serving.llm_server().stop()
    serving.embed_server().stop()


def cmd_stop(_a):
    """`stop`: stop both model servers."""
    serving.llm_server().stop()
    serving.embed_server().stop()


def cmd_dashboard(a):
    """`dashboard`: run the web dashboard on localhost, and the daemon with it unless `--no-daemon`.

    If no daemon runs, the command starts one. Ctrl-C then stops what this command started: the dashboard, the
    daemon and the model servers. A daemon that already ran (the login item, or `daemon start`) keeps running, and
    so do the model servers that it uses. A model server that already ran (another dashboard, `serve`) stays up.

    Args:
        a (argparse.Namespace): Parsed arguments with `no_daemon`.
    """
    import uvicorn

    from agent import daemon
    from dashboard import server

    st, own = daemon.status(), None
    up = {s.name for s in (serving.llm_server(), serving.embed_server()) if s.running_model()}
    if st["alive"]:
        print(f"The daemon already runs (pid {st['pid']}). It keeps running after the dashboard stops.")
    elif not a.no_daemon:
        own = start_daemon()
        print(f"Daemon started (pid {own}); log: {config.DAEMON_LOG}" if own else f"No daemon; see {config.DAEMON_LOG}")
    server.state["daemon_starts_models"] = own is not None
    print(f"Dashboard: http://127.0.0.1:{config.DASHBOARD_PORT}. Ctrl-C stops it and what it started.")
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))  # uvicorn sends SIGTERM again after its shutdown
    try:
        with contextlib.suppress(KeyboardInterrupt):  # uvicorn raises the Ctrl-C again after its shutdown
            uvicorn.run(server.app, host="127.0.0.1", port=config.DASHBOARD_PORT, log_level="warning")
    finally:
        stop_started(own, up)


def start_daemon() -> int | None:
    """Start the daemon in the background and wait for its first heartbeat.

    Returns:
        int | None: The process id of the daemon, or None if it exited or did not beat in time (its log says why).
    """
    from agent import daemon, platforms

    cmd = [sys.executable, str(config.ROOT / "cli.py"), "daemon", "run"]
    pid = platforms.current().spawn_background(cmd, config.ROOT, config.DAEMON_LOG)
    deadline = time.monotonic() + config.DAEMON_START_WAIT_S
    while time.monotonic() < deadline and running(pid):
        if daemon.status().get("pid") == pid:
            return pid
        time.sleep(0.2)
    return None


def running(pid: int) -> bool:
    """Return whether a process runs. A child that exited and was not reaped yet does not run.

    Args:
        pid (int): The process id.

    Returns:
        bool: True if the process runs.
    """
    import psutil

    try:
        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def stop_started(pid: int | None, up: set[str]) -> None:
    """Stop what `dashboard` started: its daemon, then the model servers that no daemon or other process uses.

    Args:
        pid (int | None): The daemon that `dashboard` started, or None.
        up (set[str]): Names of the model servers that ran before `dashboard` started. They stay up.
    """
    import psutil

    from agent import daemon

    if pid and running(pid):
        print(f"Stopping the daemon (pid {pid}). It finishes its current job first; Ctrl-C again kills it.")
        proc = psutil.Process(pid)
        proc.terminate()
        try:
            proc.wait()
        except KeyboardInterrupt:
            proc.kill()
            proc.wait()
    if (st := daemon.status())["alive"]:
        return print(f"The daemon (pid {st['pid']}) still runs, so the model servers stay up.")
    for s in (serving.llm_server(), serving.embed_server()):
        if s.name in up:
            print(f"The {s.name} server ran before the dashboard, so it stays up.")
        else:
            s.stop()
            print(f"The {s.name} server stopped.")


def record_until_enter():
    """Record from the microphone until the user presses Enter.

    Returns:
        np.ndarray: 16 kHz mono float32 audio.
    """
    from agent.senses import audio

    mic = audio.Mic()
    mic.start()
    input("Recording... press Enter to stop. ")
    return mic.stop()


def cmd_note(a):
    """`note`: Task 1 on a recording from the microphone or from `--file`.

    Args:
        a (argparse.Namespace): Parsed arguments with `file`.
    """
    from agent import tasks
    from agent.senses import audio

    start_servers()
    samples = audio.load_wav(a.file) if a.file else record_until_enter()
    r = tasks.voice_note(samples)
    show({k: r[k] for k in ("chunks", "transcript", "action_items")})


def cmd_look(a):
    """`look`: Task 2 on the live screen, an image, a video or a fresh recording.

    Args:
        a (argparse.Namespace): Parsed arguments with `image`, `video` and `watch`.
    """
    from PIL import Image

    from agent import tasks
    from agent.senses import video

    start_servers()
    if a.watch:
        r = tasks.watch_and_help(a.watch)
    elif a.video:
        r = tasks.screen_help(video_path=video.shrink(Path(a.video)))
    else:
        r = tasks.screen_help(Image.open(a.image).convert("RGB") if a.image else None)
    show({k: v for k, v in r.items() if k != "tokens"})


def cmd_chat(a):
    """`chat`: ask the agent one question, with tools and memory.

    Args:
        a (argparse.Namespace): Parsed arguments with `text` and `screen`.
    """
    from agent import brain
    from agent.senses import screen

    start_servers()
    parts = [brain.part("image", screen.capture())] if a.screen else []
    r = brain.respond(parts + [brain.part("text", a.text)])
    for c in r["tool_calls"]:
        print(f"[tool] {c['tool']}({c['args']}) -> {c['status']}")
    print(r["text"])


def cmd_hotkey(_a):
    """`hotkey`: run Task 2 each time the global hotkey is pressed, until Enter."""
    from agent import events, hotkey, tasks

    start_servers()
    hotkey.listen(lambda: show({k: v for k, v in tasks.screen_help().items() if k != "tokens"}))
    for e in events.since(0, "hotkey.warning"):
        print("WARNING:", e["message"])
    input(f"Listening for {config.HOTKEY}. Press Enter here to quit.\n")


def cmd_memory(a):
    """`memory`: list, search or clear the memories.

    Args:
        a (argparse.Namespace): Parsed arguments with `action` and `query`.
    """
    from agent.memory import episodic

    start_servers(llm=False)
    if a.action == "clear":
        episodic.clear()
    elif a.action == "search":
        show(episodic.search(" ".join(a.query)))
    else:
        for m in episodic.all_memories():
            print(f"{m['id']:4d} {m['when']} {m['kind']:12s} {m['text']}")


def cmd_record_clip(a):
    """`record-clip`: record an audio clip from the microphone, to use as an eval case.

    Args:
        a (argparse.Namespace): Parsed arguments with `path`.
    """
    import soundfile as sf

    samples = record_until_enter()
    Path(a.path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(a.path, samples, config.SAMPLE_RATE, subtype="PCM_16")
    print(f"saved {a.path} ({len(samples) / config.SAMPLE_RATE:.1f} s, 16 kHz mono)")


def cmd_record_screen(a):
    """`record-screen`: record frames of the screen as a screen eval case.

    Args:
        a (argparse.Namespace): Parsed arguments with `name`, `seconds` and `fps`.
    """
    from agent.senses import screen

    out = config.EVAL_USER / "screen" / a.name
    print(f"Recording {a.seconds:.0f} s of screen at {a.fps} fps into {out} ...")
    n = screen.record_frames(out, a.seconds, a.fps)
    print(f"saved {n} frames. Now mark when something important happens, for example:")
    print(f'  python cli.py evals add screen {a.name} --moment 12.5 "error dialog appears"')


# --- Session 2 -------------------------------------------------------------------


def cmd_daemon(a):
    """`daemon`: run, start, stop, kill, status, install or uninstall the always-on process.

    Args:
        a (argparse.Namespace): Parsed arguments with `action` and `no_models`.
    """
    from agent import daemon, platforms

    plat, st = platforms.current(), daemon.status()
    cmd = [sys.executable, str(config.ROOT / "cli.py"), "daemon", "run"]
    if a.action == "run":
        if (other := daemon.Daemon(models=not a.no_models).run()) is not None:
            print(f"another daemon already runs (pid {other}), so this one exits")
    elif a.action == "start":
        daemon_start(st)
    elif a.action in ("stop", "kill"):
        stop_daemon(st, kill=a.action == "kill")
    elif a.action == "install":
        print(f"installed: {plat.install_autostart(config.AUTOSTART_LABEL, cmd, config.ROOT, config.DAEMON_LOG)}")
        if st["alive"] and daemon.status().get("pid") == st["pid"]:  # the installer stopped its own old copy, not this
            print(f"a daemon that `daemon start` started still runs (pid {st['pid']}), so the login item's copy exits.")
            print("Stop it (python cli.py daemon stop), then run python cli.py daemon install again.")
    elif a.action == "uninstall":
        print("removed" if plat.uninstall_autostart(config.AUTOSTART_LABEL) else "no login item was installed")
    else:
        show(st)


def daemon_start(st: dict) -> None:
    """`daemon start`: start the daemon in the background, unless one runs.

    Args:
        st (dict): The result of `daemon.status()`.
    """
    if st["alive"]:
        return print(f"already running (pid {st['pid']})")
    pid = start_daemon()
    print(f"started (pid {pid}); log: {config.DAEMON_LOG}" if pid else f"did not start; see {config.DAEMON_LOG}")


def stop_daemon(st: dict, kill: bool) -> None:
    """Stop the daemon cleanly (SIGTERM, it finishes the current job) or kill it (SIGKILL, a crash).

    Args:
        st (dict): The result of `daemon.status()`.
        kill (bool): True to kill without cleanup, for the crash recovery demo.
    """
    import psutil

    if not st["alive"]:
        return print("not running")
    proc = psutil.Process(st["pid"])
    proc.kill() if kill else proc.terminate()
    print(f"{'killed' if kill else 'stopping'} pid {st['pid']}")


def duration_s(text: str) -> float:
    """Turn a duration such as "45m", "2h", "1h30m" or "90s" into seconds.

    Args:
        text (str): The duration. A bare number is minutes.

    Returns:
        float: Seconds.

    Raises:
        argparse.ArgumentTypeError: If the text is not a duration.
    """
    import re

    text = text.strip()
    if text.replace(".", "", 1).isdigit():
        return float(text) * 60
    parts = re.fullmatch(r"(?:(\d+(?:\.\d+)?)h)?(?:(\d+(?:\.\d+)?)m)?(?:(\d+(?:\.\d+)?)s)?", text)
    if not parts or not any(parts.groups()):
        raise argparse.ArgumentTypeError(f"not a duration: {text!r} (examples: 45m, 2h, 1h30m)")
    h, m, sec = (float(g or 0) for g in parts.groups())
    return h * 3600 + m * 60 + sec


def cmd_listen(a):
    """`listen`: ask the daemon to start or stop always-on listening, for a time or inside a daily window.

    Args:
        a (argparse.Namespace): Parsed arguments with `state`, `for_s` and `window`.
    """
    import time

    from agent import daemon

    window = None if a.window is None else [] if a.window == "any" else a.window.split("-")
    try:
        daemon.set_listening(a.state == "on", a.for_s, window, reason="cli")
    except ValueError as e:
        sys.exit(f"listen: {e}")
    st = daemon.status()
    ls, extra = st["listen"], []
    if ls["on"] and ls["until"]:
        extra.append("until " + time.strftime("%H:%M", time.localtime(ls["until"])))
    if ls["on"] and ls["window"]:
        extra.append("only " + "-".join(ls["window"]) + (f", {ls['note']} now" if ls["note"] else ""))
    if not st["alive"]:
        extra.append("the daemon is not running: python cli.py daemon start")
    print(f"listening {a.state}" + (f" ({'; '.join(extra)})" if extra else ""))


def at_today(hhmm: str | None) -> float:
    """Turn "HH:MM" into today's epoch time, or return now.

    Args:
        hhmm (str | None): A local time such as "09:00".

    Returns:
        float: Epoch seconds.
    """
    import time

    if not hhmm:
        return time.time()
    h, m = map(int, hhmm.split(":"))
    return time.mktime(time.localtime()[:3] + (h, m, 0, 0, 0, -1))


def cmd_replay(a):
    """`replay`: Task 3, feed a recording to the ambient listener as if it was heard live.

    Args:
        a (argparse.Namespace): Parsed arguments with `path`, `at` and `speed`.
    """
    from agent import daemon

    args = {"path": str(Path(a.path).resolve()), "start_ts": at_today(a.at), "speed": a.speed}
    if not daemon.status()["alive"]:
        start_servers()
    out = daemon.run_or_queue("replay", args, f"replay of {Path(a.path).name} from the CLI")
    if out["queued"]:
        return print(f"queued job {out['job_id']} in the daemon; watch the dashboard feed")
    print(f"job {out['job_id']}: {out['job']['status']}")


def cmd_ask(a):
    """`ask`: Task 3, answer a question about earlier events from memory.

    Args:
        a (argparse.Namespace): Parsed arguments with `question`.
    """
    from agent import tasks

    start_servers()
    r = tasks.recall(a.question)
    print(r["answer"])
    for m in r["memories"]:
        print(f"  {'*' if m['id'] in r['memory_ids'] else ' '} [{m['id']}] {m['text'][:100]}")


def cmd_consolidate(a):
    """`consolidate`: summarize the raw memories now, and with `--forget` apply the retention rule.

    Args:
        a (argparse.Namespace): Parsed arguments with `forget`.
    """
    from agent.memory import consolidate

    start_servers()
    for r in consolidate.consolidate(force=True):
        print(f"summary {r['summary_id']}: {len(r['claims'])} claims, facts {r['fact_ids']}")
    if a.forget:
        show(consolidate.forget())
    show(consolidate.stats())


def cmd_brief(_a):
    """`brief`: Task 4, write today's brief with the active prompt version."""
    from agent import tasks

    start_servers()
    r = tasks.daily_brief()
    print(f"brief {r['brief_id']} (prompt v{r['version']})\n\n{r['markdown']}")
    print(f'\nGive feedback: python cli.py feedback {r["brief_id"]} --bad "what to change"')


def cmd_feedback(a):
    """`feedback`: Task 4, record your feedback on a brief.

    Args:
        a (argparse.Namespace): Parsed arguments with `brief_id`, `good` and `bad`.
    """
    from agent import evolve

    rating, text = (1, a.good) if a.good is not None else (-1, a.bad)
    fid = evolve.add_feedback(a.brief_id, rating, text or "")
    print(f"feedback {fid} saved. Reflect on it: python cli.py evolve reflect")


def cmd_evolve(a):
    """`evolve`: reflect on feedback, list versions, show a diff, roll back, or run simulated rounds.

    Args:
        a (argparse.Namespace): Parsed arguments with `action`, `ids` and `rounds`.
    """
    from agent import evolve

    if a.action == "versions":
        for v in evolve.versions():
            print(f"{'*' if v['active'] else ' '} v{v['id']} from v{v['parent']} [{v['source']}] {v['reason'][:80]}")
    elif a.action == "diff":
        print(evolve.diff(*(a.ids or [get_parent(evolve), evolve.get_version()["id"]])))
    elif a.action == "rollback":
        print(f"active: v{evolve.rollback(a.ids[0])['id']}")
    elif a.action == "simulate":
        from evals.task4 import simulate

        start_servers()
        show(simulate.rounds_into_real_db(a.rounds))
    else:
        start_servers()
        show(evolve.reflect())


def get_parent(evolve) -> int:
    """Return the parent of the active version, or the active version itself when it has none."""
    v = evolve.get_version()
    return v["parent"] or v["id"]


def cmd_say(a):
    """`say`: speak a text with Kokoro, through the dashboard tab when Voice is on there, else on this machine.

    Args:
        a (argparse.Namespace): Parsed arguments with `text`, `voice`, `local` and `voices`.
    """
    from agent.voice import speaker, tts

    if a.voices:
        print("\n".join(tts.voices()))
        return
    if not a.text:
        sys.exit('nothing to say: python cli.py say "Hello"')
    show(speaker.speak(" ".join(a.text), reason="cli.py say", voice=a.voice, route="local" if a.local else None))


def _record_wake_clips(count: int, seconds: float) -> None:
    """Record `count` short clips of the wake phrase into `config.WAKE_USER_CLIPS`."""
    import time

    import numpy as np
    import sounddevice as sd
    import soundfile as sf

    config.WAKE_USER_CLIPS.mkdir(parents=True, exist_ok=True)
    print('Say "Hey Sanjaya" (SUN-juh-yuh) once per clip, when it says "now". Vary your distance and volume.')
    for i in range(count):
        input(f"[{i + 1}/{count}] press Enter, then speak...")
        print("  now")
        clip = sd.rec(int(seconds * config.SAMPLE_RATE), samplerate=config.SAMPLE_RATE, channels=1, dtype="float32")
        sd.wait()
        pcm = (np.clip(clip[:, 0], -1, 1) * 32767).astype(np.int16)
        path = config.WAKE_USER_CLIPS / f"{time.strftime('%Y%m%d-%H%M%S')}-{i:02d}.wav"
        sf.write(path, pcm, config.SAMPLE_RATE, subtype="PCM_16")
        print(f"  saved {path.name} (peak {np.abs(clip).max():.2f}; under 0.05 is too quiet)")
    print(f"{count} clips in {config.WAKE_USER_CLIPS}. Train again: python scripts/train_wake_word.py train")


def _test_wake(seconds: float) -> None:
    """Listen on the microphone and print each wake with its score."""
    import time

    import sounddevice as sd

    from agent.voice import wake

    detector, end = wake.Detector(), time.monotonic() + seconds
    print(f'Listening for {seconds:.0f} s. Say "Hey Sanjaya". Ctrl-C stops.')
    with sd.InputStream(samplerate=config.SAMPLE_RATE, channels=1, dtype="float32", blocksize=wake.FRAME) as stream:
        while time.monotonic() < end:
            block, _ = stream.read(wake.FRAME)
            for score in detector.push(block[:, 0]):
                if score >= detector.threshold:
                    print(f"  wake  score {score:.2f}  at {time.strftime('%H:%M:%S')}")
                elif score >= 0.1:
                    print(f"  near  score {score:.2f}")


def cmd_wake(a):
    """`wake`: record your own clips of the wake phrase, or test the detector live.

    Args:
        a (argparse.Namespace): Parsed arguments with `action`, `count` and `seconds`.
    """
    from agent.senses import audio

    if audio.paused():
        sys.exit("The microphone is paused: python cli.py listen ... or turn on Mic in the dashboard.")
    if a.action == "record":
        _record_wake_clips(a.count, a.seconds or 2.0)
    else:
        _test_wake(a.seconds or 60.0)


def _session3_parsers(sub) -> None:
    """Register the Session 3 commands."""
    p = sub.add_parser("say", help="speak a text with Kokoro")
    p.add_argument("text", nargs="*")
    p.add_argument("--voice", help=f"a Kokoro voice (default {config.TTS_VOICE}); see --voices")
    p.add_argument("--local", action="store_true", help="play on this machine even if a dashboard tab listens")
    p.add_argument("--voices", action="store_true", help="list the voices and exit")
    p.set_defaults(fn=cmd_say)
    p = sub.add_parser("wake", help="the wake word: record your own clips, or test it live")
    p.add_argument("action", choices=["record", "test"])
    p.add_argument("--count", type=int, default=40, help="clips to record (default 40)")
    p.add_argument("--seconds", type=float, help="seconds per clip (record, default 2) or to listen (test, default 60)")
    p.set_defaults(fn=cmd_wake)


def _session2_parsers(sub) -> None:
    """Register the Session 2 commands."""
    p = sub.add_parser("daemon", help="the always-on process")
    p.add_argument("action", choices=["run", "start", "stop", "kill", "status", "install", "uninstall"])
    p.add_argument("--no-models", action="store_true", help="do not start model servers (the daemon eval uses it)")
    p.set_defaults(fn=cmd_daemon)
    p = sub.add_parser("listen", help="always-on listening in the daemon")
    p.add_argument("state", choices=["on", "off"])
    p.add_argument("--for", dest="for_s", type=duration_s, help="stop after this long, e.g. 45m, 2h, 1h30m")
    p.add_argument("--window", help="listen only between these local times each day, e.g. 09:00-17:00; any = all day")
    p.set_defaults(fn=cmd_listen)
    p = sub.add_parser("replay", help="Task 3: replay a recording through the ambient listener")
    p.add_argument("path")
    p.add_argument("--at", help="local time the recording started, e.g. 09:00 (default: now)")
    p.add_argument("--speed", type=float, default=0, help="1 = real time, 4 = four times faster, 0 = no waiting")
    p.set_defaults(fn=cmd_replay)
    p = sub.add_parser("ask", help="Task 3: answer a question about earlier events")
    p.add_argument("question")
    p.set_defaults(fn=cmd_ask)
    p = sub.add_parser("consolidate", help="summarize raw memories now")
    p.add_argument("--forget", action="store_true", help="also delete covered raw rows past the retention time")
    p.set_defaults(fn=cmd_consolidate)
    sub.add_parser("brief", help="Task 4: write today's brief").set_defaults(fn=cmd_brief)
    p = sub.add_parser("feedback", help="Task 4: feedback on a brief")
    p.add_argument("brief_id", type=int)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--good", nargs="?", const="", help="what was good (optional text)")
    g.add_argument("--bad", help="what to change")
    p.set_defaults(fn=cmd_feedback)
    p = sub.add_parser("evolve", help="Task 4: self-evolution of the brief prompt")
    p.add_argument("action", choices=["reflect", "versions", "diff", "rollback", "simulate"])
    p.add_argument("ids", nargs="*", type=int, help="diff: two versions; rollback: one version")
    p.add_argument("--rounds", type=int, default=3, help="simulate: feedback rounds from the simulated user")
    p.set_defaults(fn=cmd_evolve)


def main():
    """Parse the command line and run the chosen command."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("serve").set_defaults(fn=cmd_serve)
    sub.add_parser("stop").set_defaults(fn=cmd_stop)
    p = sub.add_parser("dashboard")
    p.add_argument("--no-daemon", action="store_true", help="do not start the daemon with the dashboard")
    p.set_defaults(fn=cmd_dashboard)
    p = sub.add_parser("note")
    p.add_argument("--file")
    p.set_defaults(fn=cmd_note)
    p = sub.add_parser("look")
    p.add_argument("--image", help="explain this screenshot instead of the live screen")
    p.add_argument("--video", help="explain this screen recording (any format ffmpeg reads)")
    p.add_argument("--watch", type=float, metavar="SECONDS", help="record the screen this long, then explain")
    p.set_defaults(fn=cmd_look)
    p = sub.add_parser("chat")
    p.add_argument("text")
    p.add_argument("--screen", action="store_true", help="attach a screenshot")
    p.set_defaults(fn=cmd_chat)
    sub.add_parser("hotkey").set_defaults(fn=cmd_hotkey)
    p = sub.add_parser("memory")
    p.add_argument("action", nargs="?", default="list", choices=["list", "search", "clear"])
    p.add_argument("query", nargs="*")
    p.set_defaults(fn=cmd_memory)
    p = sub.add_parser("record-clip")
    p.add_argument("path")
    p.set_defaults(fn=cmd_record_clip)
    p = sub.add_parser("record-screen")
    p.add_argument("name")
    p.add_argument("--seconds", type=float, default=60)
    p.add_argument("--fps", type=float, default=2)
    p.set_defaults(fn=cmd_record_screen)
    _session2_parsers(sub)
    _session3_parsers(sub)
    sub.add_parser("smoke", add_help=False).set_defaults(
        fn=lambda a: sys.exit(subprocess.call([sys.executable, "scripts/smoke_test.py", *a.rest]))
    )
    sub.add_parser("doctor", add_help=False).set_defaults(
        fn=lambda a: sys.exit(subprocess.call([sys.executable, "scripts/doctor.py", *a.rest]))
    )
    sub.add_parser("evals", add_help=False).set_defaults(
        fn=lambda a: sys.exit(subprocess.call([sys.executable, "evals/run_all.py", *a.rest]))
    )
    a, rest = ap.parse_known_args()
    a.rest = rest
    from agent.platforms import Unsupported

    try:
        a.fn(a)
    except Unsupported as e:  # a hook this machine's plug-in lacks: say which file to add, not a traceback
        sys.exit(str(e))


if __name__ == "__main__":
    main()
