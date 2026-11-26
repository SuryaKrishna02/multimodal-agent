"""The contract every OS plug-in implements. One file per OS in this folder.

Core code (senses, serving, hotkey, cli, dashboard) calls `platforms.current().<hook>()`
and never checks sys.platform or runs an OS command itself. To support a new OS, copy
_template.py to <name>.py and fill in the hooks; nothing else changes.

Rules for a plug-in file:
- Top-level imports: standard library and `agent` only. Import OS-specific packages
  (pyobjc, pywin32, ...) inside the method that needs them, so the file imports on any OS.
- A hook you cannot implement keeps the base behaviour: it raises Unsupported with a message
  that names the file to edit. Never fake a hook.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol


class Handle(Protocol):
    """Something a hook started that the caller can end, such as the keep-awake block."""

    def stop(self) -> None:
        """End it."""


class Playback(Handle, Protocol):
    """Sound that a hook plays, which the caller can stop early."""

    def done(self) -> bool:
        """Return True once the sound has ended or was stopped."""
        ...


class Unsupported(RuntimeError):
    """This OS plug-in does not implement the requested hook (yet)."""


class _StreamPlayback:
    """Plays one decoded sound through a sounddevice output stream (the default `play_audio`)."""

    def __init__(self, sd, data, rate: int):
        """Open the stream and start it.

        Args:
            sd (module): The sounddevice module, imported by the caller so this file imports on any machine.
            data (numpy.ndarray): Float32 samples, shape (frames, channels).
            rate (int): Sample rate in Hz.
        """
        self.sd, self.data, self.pos, self.stopping = sd, data, 0, False
        self.stream = sd.OutputStream(
            samplerate=rate, channels=data.shape[1], dtype="float32", latency="low", callback=self._fill
        )
        self.stream.start()

    def _fill(self, out, frames, _time, _status) -> None:
        """Copy the next samples to the sound card; stop the stream after the last one, or when asked to."""
        n = 0 if self.stopping else min(frames, len(self.data) - self.pos)
        out[:n] = self.data[self.pos : self.pos + n]
        out[n:] = 0
        self.pos += n
        if self.stopping or self.pos >= len(self.data):
            raise self.sd.CallbackStop

    def stop(self) -> None:
        """Cut the sound off at the next audio buffer (a few ms). `abort()` would block for about 130 ms."""
        self.stopping = True

    def done(self) -> bool:
        """Tell whether the sound has ended, and close the stream once it has.

        Returns:
            bool: True once the last sample played or the stream stopped after `stop()`.
        """
        if self.stream.closed:
            return True
        if self.stream.active:
            return False
        self.stream.close()
        return True


def needs_override(fn):
    """Mark a base hook that has no portable default. `cli.py doctor` lists these as missing."""
    fn._needs_override = True
    return fn


class Platform:
    """Base class of every OS plug-in, with the portable defaults.

    Attributes:
        name (str): Must equal the file name.
        description (str): One line shown by `cli.py doctor`.
        exe_suffix (str): Suffix of executables, ".exe" on Windows.
        raw_video_suffix (str): Container that `record_screen` writes.
        monospace_font (str | None): Path of a font for the rendered test images of the smoke test and the synthetic evals, or None for the Pillow default.
        screen_blank_hint (str): Advice shown when a screenshot comes back blank.
        mic_silent_hint (str): Advice shown when the microphone stream delivers only silence after it opens.
    """

    name = "generic"  # must equal the file name, e.g. macos.py -> "macos"
    description = "No OS plug-in matched: portable defaults only."

    # --- Facts -----------------------------------------------------------------
    exe_suffix = ""  # ".exe" on Windows: llama-server -> llama-server.exe
    raw_video_suffix = ".mp4"  # container that record_screen() writes
    monospace_font: str | None = None  # path to a TTF/TTC for rendered test images; None = Pillow default
    screen_blank_hint = "The screenshot is blank. Your OS may be blocking screen capture for this terminal app."
    mic_silent_hint = (
        "The microphone delivers only silence. Your OS may be blocking the Microphone permission for this process, "
        "or the input is muted."
    )

    @classmethod
    def matches(cls) -> bool:
        """True when this plug-in is the right one for the machine we are running on."""
        return False

    # --- Setup and install -----------------------------------------------------
    def venv_python(self, venv: Path) -> Path:
        """Return the Python executable inside a virtual environment.

        Args:
            venv (Path): The virtual environment folder.

        Returns:
            Path: The executable.
        """
        return venv / "bin" / "python"

    def install_hint(self, tool: str) -> str:
        """One line telling the user how to install an external tool such as ffmpeg."""
        return f"install {tool} with your system package manager"

    def exe(self, name: str) -> str:
        """Add the executable suffix of this OS to a program name.

        Args:
            name (str): Program name, for example "llama-server".

        Returns:
            str: The file name to run.
        """
        return name + self.exe_suffix

    # --- Senses ----------------------------------------------------------------
    @needs_override
    def record_screen(self, out: Path, seconds: float) -> None:
        """Record the main display for `seconds` into the video file `out` (no audio needed).

        Any container ffmpeg can read is fine; senses/video.py shrinks it afterwards.
        """
        raise self._missing("record_screen")

    @needs_override
    def synthesize_speech(self, text: str, out: Path, voice: int = 0) -> Path:
        """Speak `text` into an audio file soundfile can read (WAV/AIFF/FLAC); return its path.

        Used by the smoke test and by the synthetic eval cases (evals/synthetic/), so neither needs recorded audio.
        `voice` picks the n-th distinct voice (0 = the default), so a synthetic conversation has several speakers.
        A plug-in with fewer voices wraps around.
        """
        raise self._missing("synthesize_speech")

    # --- Hotkey ----------------------------------------------------------------
    def start_hotkey(self, combo: str, on_press):
        """Listen for a global hotkey in a background thread. `combo` is pynput syntax.

        Default: pynput, which covers macOS, Windows and X11. Override for Wayland etc.
        Return an object with .stop().
        """
        from pynput import keyboard

        listener = keyboard.GlobalHotKeys({combo: on_press})
        listener.start()
        return listener

    def hotkey_warning(self) -> str | None:
        """A message when the OS will silently drop global key events (permissions), else None."""
        return None

    # --- Always on (Session 2) ------------------------------------------------
    @needs_override
    def keep_awake(self) -> Handle:
        """Stop the computer from sleeping while this process runs. Return an object with .stop().

        The block must end by itself when this process dies, so a crash cannot keep the machine awake.
        """
        raise self._missing("keep_awake")

    @needs_override
    def notify(self, title: str, message: str) -> None:
        """Show a desktop notification. Treat both strings as data: never let them run as a command."""
        raise self._missing("notify")

    @needs_override
    def install_autostart(self, label: str, cmd: list[str], cwd: Path, log: Path) -> Path:
        """Start `cmd` in `cwd` when the user logs in, and again if it crashes. Return the file that does it.

        A clean exit (code 0, for example after `cli.py daemon stop`) must not restart it.
        """
        raise self._missing("install_autostart")

    @needs_override
    def uninstall_autostart(self, label: str) -> bool:
        """Remove the login item that `install_autostart` made. Return False if there was none."""
        raise self._missing("uninstall_autostart")

    def spawn_background(self, cmd: list[str], cwd: Path, log: Path) -> int:
        """Start `cmd` detached from this terminal, with output in `log`. Return its process id.

        Default: a new session, which detaches on macOS and Linux. Override where that does not detach.
        """
        import subprocess

        log.parent.mkdir(parents=True, exist_ok=True)
        with open(log, "a") as out:
            proc = subprocess.Popen(
                cmd, cwd=cwd, stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True
            )
        return proc.pid

    # --- Voice (Session 3) ------------------------------------------------------
    def play_audio(self, path: Path) -> Playback:
        """Start playing an audio file on this machine's speakers, and return at once.

        Speech normally plays in the dashboard tab, because Google Meet shares only the audio of the presented tab.
        This hook speaks when no tab listens, for example an alert while nobody presents. `stop()` must end the
        sound within about 50 ms, because stopping speech is part of the voice eval.

        Default: one PortAudio output stream through sounddevice, which covers macOS, Windows and Linux. It starts in
        about 50 ms on an M4; macOS `afplay` needs about 1 s per file, which leaves gaps between sentences.
        """
        import sounddevice as sd
        import soundfile as sf

        data, rate = sf.read(str(path), dtype="float32", always_2d=True)
        return _StreamPlayback(sd, data, rate)

    # --- Documentation helpers -------------------------------------------------
    def permission_notes(self) -> list[tuple[str, str]]:
        """(permission, what it is needed for) pairs shown by `cli.py doctor`."""
        return []

    # --- Internals -------------------------------------------------------------
    def _missing(self, hook: str) -> Unsupported:
        where = f"agent/platforms/{self.name}.py" if self.name != "generic" else "a new file in agent/platforms/"
        return Unsupported(
            f"Platform '{self.name}' does not implement {hook}(). Add it in {where}; see CONTRIBUTING.md."
        )


# The hooks `doctor` checks and the tests exercise. Add new hooks here when you add them above.
HOOKS = (
    "record_screen",
    "synthesize_speech",
    "start_hotkey",
    "keep_awake",
    "notify",
    "install_autostart",
    "uninstall_autostart",
    "spawn_background",
    "play_audio",
)
