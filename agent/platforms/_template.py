"""TEMPLATE for a new OS plug-in. Copy to agent/platforms/<name>.py, e.g. linux.py or windows.py.

Files starting with _ are ignored by the loader. See CONTRIBUTING.md ("Add a platform").
Delete every hook you do not implement: the base class then raises Unsupported, which is the
honest answer. Do not leave placeholder code behind.
"""

import subprocess
import sys
from pathlib import Path

from agent.platforms.base import Platform


class MyOS(Platform):  # one Platform subclass per file
    name = "myos"  # must equal the file name
    description = "one line: what is special about this OS"
    # exe_suffix = ".exe"                      # Windows only
    # raw_video_suffix = ".mp4"                # container record_screen() writes
    # monospace_font = "/path/to/font.ttf"     # for rendered test images (smoke test, synthetic evals)
    # screen_blank_hint = "..."                # shown when a screenshot comes back blank

    @classmethod
    def matches(cls) -> bool:
        return sys.platform.startswith("myos")  # TODO: sys.platform value for this OS

    # def venv_python(self, venv: Path) -> Path:        # Windows: venv / "Scripts" / "python.exe"
    # def install_hint(self, tool: str) -> str:         # e.g. "sudo apt install ffmpeg"

    def record_screen(self, out: Path, seconds: float) -> None:
        """Record the main display for `seconds` into `out`. ffmpeg is a fine choice if present."""
        raise NotImplementedError

    def synthesize_speech(self, text: str, out: Path) -> Path:
        """Write spoken `text` to an audio file soundfile can read; return the path."""
        raise NotImplementedError

    # def start_hotkey(self, combo, on_press): ...      # only if pynput does not work here
    # def hotkey_warning(self) -> str | None: ...       # permissions that silently block hotkeys
    # def permission_notes(self) -> list[tuple[str, str]]: ...
