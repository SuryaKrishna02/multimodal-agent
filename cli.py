"""Command line for the agent. Run `python cli.py -h` for the list.

    python cli.py serve                       start both model servers and wait
    python cli.py dashboard                   start the dashboard on http://127.0.0.1:8765
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
    python cli.py replay morning.wav --at 09:00        Task 3: replay a recording as if it was heard live
    python cli.py ask "what did I decide about the venue this morning?"     Task 3: answer from memory
    python cli.py consolidate [--forget]               summarize raw memories now (and apply retention)
    python cli.py brief                                Task 4: write today's brief
    python cli.py feedback 3 --bad "too long"          Task 4: feedback on brief 3
    python cli.py evolve reflect|versions|diff A B|rollback N|simulate --rounds 3

Everything OS- or GPU-specific lives in agent/platforms/ and agent/accel/. Commands here call
those plug-ins and never branch on the OS. Force a choice with AGENT_PLATFORM=<name> / AGENT_ACCEL=<name>.
"""

import argparse
import contextlib
import json
import subprocess
import sys
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


def cmd_dashboard(_a):
    """`dashboard`: run the web dashboard on localhost."""
    import uvicorn

    print(f"Dashboard: http://127.0.0.1:{config.DASHBOARD_PORT}")
    uvicorn.run("dashboard.server:app", host="127.0.0.1", port=config.DASHBOARD_PORT, log_level="warning")


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
        daemon.Daemon(models=not a.no_models).run()
    elif a.action == "start":
        if st["alive"]:
            return print(f"already running (pid {st['pid']})")
        pid = plat.spawn_background(cmd, config.ROOT, config.DAEMON_LOG)
        print(f"started (pid {pid}); log: {config.DAEMON_LOG}")
    elif a.action in ("stop", "kill"):
        stop_daemon(st, kill=a.action == "kill")
    elif a.action == "install":
        print(f"installed: {plat.install_autostart(config.AUTOSTART_LABEL, cmd, config.ROOT, config.DAEMON_LOG)}")
    elif a.action == "uninstall":
        print("removed" if plat.uninstall_autostart(config.AUTOSTART_LABEL) else "no login item was installed")
    else:
        show(st)


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


def cmd_listen(a):
    """`listen`: ask the daemon to start or stop always-on listening.

    Args:
        a (argparse.Namespace): Parsed arguments with `state`.
    """
    from agent import daemon

    daemon.set_listening(a.state == "on")
    st = daemon.status()
    print(f"listening {a.state}" + ("" if st["alive"] else " (the daemon is not running: python cli.py daemon start)"))


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


def _session2_parsers(sub) -> None:
    """Register the Session 2 commands."""
    p = sub.add_parser("daemon", help="the always-on process")
    p.add_argument("action", choices=["run", "start", "stop", "kill", "status", "install", "uninstall"])
    p.add_argument("--no-models", action="store_true", help="do not start model servers (the daemon eval uses it)")
    p.set_defaults(fn=cmd_daemon)
    p = sub.add_parser("listen", help="always-on listening in the daemon")
    p.add_argument("state", choices=["on", "off"])
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
    sub.add_parser("dashboard").set_defaults(fn=cmd_dashboard)
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
