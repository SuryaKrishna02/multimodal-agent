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
    a.fn(a)


if __name__ == "__main__":
    main()
