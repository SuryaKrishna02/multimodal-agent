"""macOS (Apple Silicon and Intel): the reference plug-in. Verified on macOS 27 / M4."""

import importlib
import os
import plistlib
import subprocess
import sys
from pathlib import Path

from agent.platforms.base import Platform


class _Background:
    """A helper process that ends when `stop()` is called."""

    def __init__(self, proc: subprocess.Popen):
        """Keep the process.

        Args:
            proc (subprocess.Popen): The helper process.
        """
        self.proc = proc

    def stop(self) -> None:
        """End the helper process."""
        self.proc.terminate()


def _agents_dir() -> Path:
    return Path.home() / "Library" / "LaunchAgents"


class MacOS(Platform):
    """macOS plug-in: screencapture, say, caffeinate, notifications, launchd and Accessibility-gated hotkeys."""

    name = "macos"
    description = "macOS: screencapture, say, caffeinate, osascript notifications, launchd, Accessibility hotkeys"
    raw_video_suffix = ".mov"
    monospace_font = "/System/Library/Fonts/Menlo.ttc"
    screen_blank_hint = (
        "Screenshot is blank: grant Screen Recording to your terminal app "
        "(System Settings > Privacy & Security > Screen Recording)."
    )

    @classmethod
    def matches(cls) -> bool:
        """Detect macOS.

        Returns:
            bool: True when running on macOS.
        """
        return sys.platform == "darwin"

    def install_hint(self, tool: str) -> str:
        """Tell the user how to install a tool with Homebrew.

        Args:
            tool (str): Tool name.

        Returns:
            str: A `brew install` command.
        """
        return f"brew install {tool}"

    def record_screen(self, out: Path, seconds: float) -> None:
        """Record the main display with `screencapture`.

        Args:
            out (Path): Video file to write.
            seconds (float): Length to record, truncated to whole seconds.
        """
        subprocess.run(["screencapture", "-v", "-x", f"-V{int(seconds)}", str(out)], check=True)

    voices = (None, "Daniel", "Karen", "Rishi")  # default, then en_GB, en_AU, en_IN voices that ship with macOS

    def synthesize_speech(self, text: str, out: Path, voice: int = 0) -> Path:
        """Speak text into an AIFF file with `say`.

        Args:
            text (str): Text to speak.
            out (Path): Output path. The suffix becomes ".aiff".
            voice (int, optional): Index into `voices`, wrapping around. Defaults to 0 (the system voice).

        Returns:
            Path: The AIFF file.
        """
        out = out.with_suffix(".aiff")
        name = self.voices[voice % len(self.voices)]
        subprocess.run(["say", *(["-v", name] if name else []), "-o", str(out), text], check=True)
        return out

    def keep_awake(self) -> _Background:
        """Prevent idle sleep with `caffeinate` until this process exits.

        `-i` stops idle sleep only: closing the lid of a laptop still sleeps it. `-w` ends caffeinate when
        this process dies, even after a kill -9.

        Returns:
            _Background: Call `.stop()` to allow sleep again.
        """
        return _Background(subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())]))

    def notify(self, title: str, message: str) -> None:
        """Show a notification with `osascript`. The strings go in as arguments, never as script text.

        Args:
            title (str): Notification title.
            message (str): Notification text.
        """
        script = ["-e", "on run argv", "-e", "display notification (item 2 of argv) with title (item 1 of argv)"]
        subprocess.run(["osascript", *script, "-e", "end run", title, message], check=True, capture_output=True)

    def install_autostart(self, label: str, cmd: list[str], cwd: Path, log: Path) -> Path:
        """Write a LaunchAgent and load it, so launchd starts `cmd` at login and after a crash.

        `KeepAlive.SuccessfulExit = False` restarts the process only when it exits with an error or is killed.

        Args:
            label (str): The LaunchAgent label, also the file name.
            cmd (list[str]): The command to run.
            cwd (Path): Working folder.
            log (Path): File for stdout and stderr.

        Returns:
            Path: The plist file.
        """
        plist = _agents_dir() / f"{label}.plist"
        plist.parent.mkdir(parents=True, exist_ok=True)
        spec = {
            "Label": label,
            "ProgramArguments": cmd,
            "WorkingDirectory": str(cwd),
            "RunAtLoad": True,
            "KeepAlive": {"SuccessfulExit": False},
            "StandardOutPath": str(log),
            "StandardErrorPath": str(log),
        }
        plist.write_bytes(plistlib.dumps(spec))
        domain = f"gui/{os.getuid()}"
        subprocess.run(["launchctl", "bootout", f"{domain}/{label}"], capture_output=True)  # an old copy, if any
        subprocess.run(["launchctl", "bootstrap", domain, str(plist)], check=True, capture_output=True)
        return plist

    def uninstall_autostart(self, label: str) -> bool:
        """Unload the LaunchAgent and delete its plist.

        Args:
            label (str): The LaunchAgent label.

        Returns:
            bool: False if no plist existed.
        """
        plist = _agents_dir() / f"{label}.plist"
        subprocess.run(["launchctl", "bootout", f"gui/{os.getuid()}/{label}"], capture_output=True)
        if not plist.exists():
            return False
        plist.unlink()
        return True

    def hotkey_warning(self) -> str | None:
        """Check that macOS trusts this process for global key events.

        Returns:
            str | None: A message about the missing permission, or None if the process is trusted.
        """
        hi_services = importlib.import_module("HIServices")  # installed with pynput (pyobjc); no type stubs
        if hi_services.AXIsProcessTrusted():
            return None
        return (
            "Hotkey not trusted by macOS: grant Accessibility and Input Monitoring "
            "to your terminal app, then restart it."
        )

    def permission_notes(self) -> list[tuple[str, str]]:
        """List the permissions that the agent needs on macOS.

        Returns:
            list[tuple[str, str]]: (permission, what it is for) pairs.
        """
        return [
            ("Microphone", "recording voice notes"),
            ("Screen Recording", "Task 2, screen videos, screen evals"),
            ("Accessibility + Input Monitoring", "the global hotkey"),
            ("Notifications", "daemon notifications through osascript; allow the app that Settings lists"),
        ]
