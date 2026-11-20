"""macOS (Apple Silicon and Intel): the reference plug-in. Verified on macOS 27 / M4."""

import hashlib
import importlib
import os
import plistlib
import shutil
import subprocess
import sys
import time
from pathlib import Path

from agent import config
from agent.platforms.base import Platform

APP_NAME = "multimodal-agent"  # the name that System Settings > Privacy & Security > Microphone shows
LAUNCHER_SOURCE = Path(__file__).with_name("macos_launcher.swift")


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


def _bootout(target: str) -> None:
    """Unload a LaunchAgent and wait until its process is gone, or `bootstrap` of a new copy fails (code 5).

    launchd sends SIGTERM, and SIGKILL after its exit timeout (20 s by default).
    """
    subprocess.run(["launchctl", "bootout", target], capture_output=True)
    deadline = time.monotonic() + config.AUTOSTART_STOP_WAIT_S
    while time.monotonic() < deadline:
        if subprocess.run(["launchctl", "print", target], capture_output=True).returncode != 0:
            return
        time.sleep(0.5)


def _app_info(bundle_id: str) -> dict:
    return {
        "CFBundleExecutable": APP_NAME,
        "CFBundleIdentifier": bundle_id,
        "CFBundleName": APP_NAME,
        "CFBundlePackageType": "APPL",
        "CFBundleVersion": "1",
        "LSUIElement": True,  # no Dock icon and no menu bar
        "NSMicrophoneUsageDescription": "Always-on listening: the daemon transcribes what it hears into local memory.",
    }


def build_launcher(bundle_id: str, folder: Path = config.AUTOSTART_DIR) -> Path:
    """Build the small app that launchd runs, so the daemon holds the Microphone permission of the app.

    macOS gives a process without the permission silent audio, and a Python that launchd starts has none.
    The app has `NSMicrophoneUsageDescription`, asks for the permission once, and runs the daemon as its child.
    It is ad-hoc signed, so macOS knows it by its code hash: the app is rebuilt only when its source or
    Info.plist changes, because a rebuilt app must ask for the permission again.

    Args:
        bundle_id (str): The bundle identifier, the same as the LaunchAgent label.
        folder (Path, optional): Where the app goes. Defaults to `config.AUTOSTART_DIR`.

    Returns:
        Path: The executable inside the app.

    Raises:
        RuntimeError: `swiftc` is missing (it comes with the Xcode Command Line Tools).
    """
    app = folder / f"{APP_NAME}.app"
    exe, stamp = app / "Contents" / "MacOS" / APP_NAME, app / "Contents" / "Resources" / "source.sha256"
    info = plistlib.dumps(_app_info(bundle_id))
    digest = hashlib.sha256(LAUNCHER_SOURCE.read_bytes() + info).hexdigest()
    if exe.exists() and stamp.exists() and stamp.read_text() == digest:
        return exe
    if not shutil.which("swiftc"):
        raise RuntimeError("swiftc not found: install the Xcode Command Line Tools with `xcode-select --install`.")
    shutil.rmtree(app, ignore_errors=True)
    exe.parent.mkdir(parents=True)
    stamp.parent.mkdir(parents=True)
    (app / "Contents" / "Info.plist").write_bytes(info)
    subprocess.run(["swiftc", "-O", "-o", str(exe), str(LAUNCHER_SOURCE)], check=True, capture_output=True)
    subprocess.run(["codesign", "--force", "--sign", "-", str(app)], check=True, capture_output=True)
    stamp.write_text(digest)
    return exe


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
    mic_silent_hint = (
        "The microphone delivers only silence. macOS gives silent audio to a process without the Microphone "
        "permission. A daemon from `daemon start` uses the permission of your terminal app. A daemon that launchd "
        f"starts (`daemon install`) uses the '{APP_NAME}' app: allow it in System Settings > Privacy & Security > "
        "Microphone, then run `python cli.py daemon install` again."
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
        launchd runs `cmd` through the app of `build_launcher`, so the daemon can hold the Microphone permission.
        The first start asks for the permission in a macOS dialog.
        The daemon gets the PATH of the shell that installs it, so it finds the tools that this shell finds.

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
            "ProgramArguments": [str(build_launcher(label)), *cmd],
            "AssociatedBundleIdentifiers": label,  # System Settings > Login Items shows the app, not "python"
            # launchd's own PATH is /usr/bin:/bin:/usr/sbin:/sbin, so llama-server would not find a Homebrew ffprobe
            "EnvironmentVariables": {"PATH": os.environ.get("PATH", "/usr/bin:/bin:/usr/sbin:/sbin")},
            "WorkingDirectory": str(cwd),
            "RunAtLoad": True,
            "KeepAlive": {"SuccessfulExit": False},
            "StandardOutPath": str(log),
            "StandardErrorPath": str(log),
        }
        plist.write_bytes(plistlib.dumps(spec))
        domain = f"gui/{os.getuid()}"
        _bootout(f"{domain}/{label}")  # an old copy, if any
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
        _bootout(f"gui/{os.getuid()}/{label}")
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
            ("Microphone", "voice notes; the daemon that launchd starts asks for it as the 'multimodal-agent' app"),
            ("Screen Recording", "Task 2, screen videos, screen evals"),
            ("Accessibility + Input Monitoring", "the global hotkey"),
            ("Notifications", "daemon notifications through osascript; allow the app that Settings lists"),
        ]
