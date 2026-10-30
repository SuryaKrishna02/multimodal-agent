"""macOS (Apple Silicon and Intel): the reference plug-in. Verified on macOS 27 / M4."""

import importlib
import subprocess
import sys
from pathlib import Path

from agent.platforms.base import Platform


class MacOS(Platform):
    """macOS plug-in: screencapture, say and Accessibility-gated hotkeys."""

    name = "macos"
    description = "macOS: screencapture, say, Accessibility-gated hotkeys"
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

    def synthesize_speech(self, text: str, out: Path) -> Path:
        """Speak text into an AIFF file with `say`.

        Args:
            text (str): Text to speak.
            out (Path): Output path. The suffix becomes ".aiff".

        Returns:
            Path: The AIFF file.
        """
        out = out.with_suffix(".aiff")
        subprocess.run(["say", "-o", str(out), text], check=True)
        return out

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
        ]
