"""Doctor: what platform and accelerator did we detect, and does everything they need work?

    python scripts/doctor.py              # report only (instant)
    python scripts/doctor.py --exercise   # also run the hooks: 2 s screen recording, speech, memory probe

Paste this output into your PR when you add a platform or accelerator.
Standard library only, so it works on a machine where setup has not finished.
"""

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from agent import accel, config, pins, platforms  # noqa: E402
from agent.platforms.base import Unsupported  # noqa: E402

failures = 0


def row(status: str, label: str, detail: str = "") -> None:
    """Print one report line and count failures.

    Args:
        status (str): "ok", "WARN", "MISS", "MISSING" or "FAIL". Only "FAIL" counts as a failure.
        label (str): What the line is about.
        detail (str, optional): Extra text. Defaults to "".
    """
    global failures
    failures += status == "FAIL"
    print(f"  {status:5s} {label:24s} {detail}")


def implemented(plat, hook: str) -> bool:
    """Check whether a platform plug-in implements a hook.

    Args:
        plat (Platform): The platform plug-in.
        hook (str): Hook name.

    Returns:
        bool: False if the hook still has the base `needs_override` marker.
    """
    return not getattr(getattr(plat, hook), "_needs_override", False)


def report_detection() -> tuple:
    """Report which platform and accelerator plug-ins were detected.

    Returns:
        tuple: (platform, accelerator), where the accelerator is None if none matched.
    """
    print("Detected")
    plat = platforms.current()
    forced = f" (forced by AGENT_PLATFORM={os.environ['AGENT_PLATFORM']})" if os.environ.get("AGENT_PLATFORM") else ""
    row("ok" if plat.name != "generic" else "WARN", f"platform: {plat.name}", plat.description + forced)
    print(f"        available platform plug-ins: {', '.join(platforms.available())}")
    try:
        a = accel.current()
        forced = f" (forced by AGENT_ACCEL={os.environ['AGENT_ACCEL']})" if os.environ.get("AGENT_ACCEL") else ""
        row("ok", f"accelerator: {a.name}", a.description + forced)
    except RuntimeError as e:
        row("FAIL", "accelerator", str(e))
        a = None
    print(f"        available accelerator plug-ins: {', '.join(accel.available())}")
    return plat, a


def report_hooks(plat) -> None:
    """Report which OS hooks the platform plug-in implements, and its permission notes.

    Args:
        plat (Platform): The platform plug-in.
    """
    print("\nOS hooks")
    for hook in platforms.HOOKS:
        if implemented(plat, hook):
            row("ok", hook)
        else:
            row("MISSING", hook, f"not implemented for '{plat.name}': see CONTRIBUTING.md")
    for name, why in plat.permission_notes():
        print(f"        permission: {name} (for {why})")


def _report_server() -> None:
    """Report whether the llama-server binary for the detected accelerator is installed."""
    try:
        server = accel.llama_server()
    except Unsupported as e:
        row("FAIL", "llama-server", str(e))
        return
    if server.exists():
        row("ok", "llama-server", str(server.relative_to(ROOT)))
    else:
        row("FAIL", "llama-server", "missing: run scripts/setup.py")


def _report_tools(plat) -> None:
    """Report the optional video tools and the virtual environment."""
    for tool in ("ffmpeg", "ffprobe"):
        found = shutil.which(tool)
        row("ok" if found else "WARN", tool, found or f"video input disabled: {plat.install_hint('ffmpeg')}")
    py = plat.venv_python(ROOT / ".venv")
    row(
        "ok" if py.exists() else "WARN",
        ".venv python",
        str(py.relative_to(ROOT)) if py.exists() else "run scripts/setup.py",
    )


def report_files(plat, a) -> None:
    """Report which binaries, model files and tools are installed."""
    print("\nInstalled")
    if a:
        _report_server()
    for pin in pins.PINS:
        p = pins.path(pin)
        row("ok" if p.exists() else "MISS", pin.group + " " + p.name, "" if p.exists() else "run scripts/setup.py")
    _report_tools(plat)


def try_hook(label: str, fn) -> None:
    """Run one hook and report: ok with its result, MISSING if unimplemented, FAIL on any other error."""
    try:
        status, detail = fn()
    except Unsupported as e:
        status, detail = "MISSING", str(e)
    except ImportError as e:
        status, detail = "WARN", f"{e} (run this with the .venv python once scripts/setup.py has finished)"
    except Exception as e:  # noqa: BLE001 - doctor reports every failure the same way
        status, detail = "FAIL", repr(e)
    row(status, label, detail)


def exercise(plat, a) -> None:
    """Run the hooks for real: record the screen, speak, check the hotkey and read the memory probe.

    Args:
        plat (Platform): The platform plug-in.
        a (Accelerator | None): The accelerator plug-in, if any.
    """
    print("\nExercise (runs the hooks for real)")
    with tempfile.TemporaryDirectory() as d:

        def record():
            out = Path(d) / f"rec{plat.raw_video_suffix}"
            plat.record_screen(out, 2)
            return ("ok" if out.exists() and out.stat().st_size else "FAIL"), f"{out.stat().st_size} bytes"

        def speak():
            out = plat.synthesize_speech("Testing one two three.", Path(d) / "speech")
            return ("ok" if out.exists() and out.stat().st_size else "FAIL"), out.name

        try_hook("record_screen", record)
        try_hook("synthesize_speech", speak)
    try_hook(
        "hotkey_warning",
        lambda: (lambda w: ("WARN", w) if w else ("ok", "no permission problem detected"))(plat.hotkey_warning()),
    )
    if a:
        import psutil

        try_hook(
            "server_memory_gb",
            lambda: ("ok", f"{a.server_memory_gb([psutil.Process()]):.2f} GB for this Python process"),
        )
        row("ok", "server_args", " ".join(a.server_args()) or "(none)")


def main() -> int:
    """Run the doctor.

    Returns:
        int: 1 if any check failed, else 0.
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exercise", action="store_true", help="also run the OS hooks for real")
    args = ap.parse_args()
    print(f"Python {sys.version.split()[0]}   llama.cpp {config.LLAMA_BUILD}   tier {config.TIER}\n")
    plat, a = report_detection()
    report_hooks(plat)
    report_files(plat, a)
    if args.exercise:
        exercise(plat, a)
    print("\nAll good." if not failures else f"\n{failures} problem(s) above.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
