"""Speak a reply: write it sentence by sentence, and send each chunk to where it can be heard.

Route: the dashboard tab when one listens, because Google Meet shares only the audio of the presented tab. Without a
tab, the speakers of this machine play it through the OS plug-in (`play_audio`). The tab writes a heartbeat file
(`config.VOICE_TAB_FILE`) while it listens, so a daemon in another process can choose the route.

Stopping is an event (`voice.stop`), so the dashboard's Stop button, the wake word or the hotkey can stop speech that
another process writes. The writer checks for it before each chunk. The tab stops its own playback when the event
reaches it, and reports `voice.stopped`. These events also hold the times for the voice eval:

    voice.say      the text arrived          voice.chunk    one sentence is written (synth_s)
    voice.played   the first chunk is heard  voice.stop     someone asked for silence
    voice.stopped  playback ended early (after_stop_s)
"""

import json
import re
import shutil
import threading
import time
import uuid

import soundfile as sf

from agent import config, events, platforms
from agent.platforms.base import Unsupported
from agent.voice import tts

UTTERANCE_ID = re.compile(r"^[0-9]{8}-[0-9]{6}-[0-9a-f]{6}$")  # the folder names that speak() makes


def mark_tab() -> None:
    """Record that a dashboard tab listens right now (the tab's event stream calls this about once a second)."""
    config.VOICE_TAB_FILE.parent.mkdir(parents=True, exist_ok=True)
    # Each writer has its own temporary file: two streams (two tabs, or a tab that reconnects) write at once.
    part = config.VOICE_TAB_FILE.with_name(f"{config.VOICE_TAB_FILE.name}.{uuid.uuid4().hex}.part")
    part.write_text(json.dumps({"ts": time.time()}))
    part.replace(config.VOICE_TAB_FILE)  # one rename, so a reader never sees half a file


def tab_listening() -> bool:
    """Tell whether a dashboard tab with audio turned on is connected.

    Returns:
        bool: True if its heartbeat is newer than `config.VOICE_TAB_ALIVE_S`.
    """
    try:
        ts = json.loads(config.VOICE_TAB_FILE.read_text())["ts"]
    except (OSError, ValueError, KeyError):
        return False
    return time.time() - ts < config.VOICE_TAB_ALIVE_S


def stop(reason: str) -> dict:
    """Ask every process that speaks to stop now.

    Args:
        reason (str): What asked, for example "button", "wake word" or "hotkey".

    Returns:
        dict: The `voice.stop` event.
    """
    return events.emit("voice.stop", reason=reason)


def _stopped_after(event_id: int) -> dict | None:
    """Return the newest stop request if it came after the event `event_id`, else None."""
    # Runs every VOICE_POLL_S while speech plays. `since` reads only the rows after `event_id` (the primary key);
    # `events.last` would scan the whole log back to the last stop, and a stop is rare.
    newer = events.since(event_id, "voice.stop")
    return newer[-1] if newer else None


def speak(
    text: str,
    reason: str = "",
    voice: str | None = None,
    route: str | None = None,
    cancel: threading.Event | None = None,
) -> dict:
    """Speak text: write each sentence to a WAV chunk, and play it in the tab or on this machine.

    Args:
        text (str): What to say.
        reason (str, optional): Why the agent speaks, shown in the dashboard feed. Defaults to "".
        voice (str | None, optional): Kokoro voice. Defaults to `config.TTS_VOICE`.
        route (str | None, optional): "tab" or "local". Defaults to the tab if one listens, else local.
        cancel (threading.Event | None, optional): Set by a newer voice turn; no further chunk is written after it.
            It closes the gap where the turn could start speaking just after the stop event. Defaults to None.

    Returns:
        dict: `utt` (the utterance id), `route`, `chunks` written and `stopped` (True if a stop cut it short).
    """
    utt = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    folder = config.VOICE_DIR / utt
    folder.mkdir(parents=True, exist_ok=True)
    route = route or ("tab" if tab_listening() else "local")
    start = events.emit("voice.say", utt=utt, text=text, reason=reason, route=route)
    pieces = tts.sentences(tts.speakable(text))  # voice.say keeps the text as written, for the dashboard
    local = _LocalPlayer(utt, start) if route == "local" else None
    written = 0
    for n, piece in enumerate(pieces):
        if _stopped_after(start["id"]) or (cancel is not None and cancel.is_set()):
            break
        began = time.monotonic()
        audio = tts.synthesize(piece, voice)
        path = folder / f"{n}.wav"
        sf.write(path, audio, config.TTS_SAMPLE_RATE)
        written += 1
        events.emit(
            "voice.chunk",
            utt=utt,
            n=n,
            last=n == len(pieces) - 1,
            text=piece,
            route=route,
            synth_s=round(time.monotonic() - began, 3),
            audio_s=round(len(audio) / config.TTS_SAMPLE_RATE, 2),
        )
        if local and not local.play(path):
            break
    if local:
        local.finish()
    _prune()
    return {"utt": utt, "route": route, "chunks": written, "stopped": bool(_stopped_after(start["id"]))}


class _LocalPlayer:
    """Plays the chunks of one utterance on this machine, in order, and stops when a stop request comes."""

    def __init__(self, utt: str, start: dict):
        """Remember which utterance this is.

        Args:
            utt (str): The utterance id.
            start (dict): Its `voice.say` event; a stop request newer than this one stops it.
        """
        self.utt, self.start, self.current, self.started = utt, start, None, False

    def _wait(self) -> bool:
        """Wait until the chunk that plays has ended. Return False if a stop request cut it off."""
        while self.current and not self.current.done():
            request = _stopped_after(self.start["id"])
            if request:
                self.current.stop()
                events.emit("voice.stopped", utt=self.utt, route="local", after_stop_s=time.time() - request["ts"])
                return False
            time.sleep(config.VOICE_POLL_S)
        return True

    def play(self, path) -> bool:
        """Play one chunk after the one before it ends.

        Args:
            path (Path): The WAV chunk.

        Returns:
            bool: False if speech was stopped, or this OS cannot play audio.
        """
        if not self._wait():
            return False
        try:
            self.current = platforms.current().play_audio(path)
        except Unsupported as e:
            events.emit("voice.error", utt=self.utt, error=f"No dashboard tab listens, and {e}")
            return False
        if not self.started:
            self.started = True
            events.emit("voice.played", utt=self.utt, n=0, route="local", since_say_s=time.time() - self.start["ts"])
        return True

    def finish(self) -> None:
        """Wait for the last chunk to end (or be stopped)."""
        self._wait()


def _prune(keep: int = config.VOICE_KEEP) -> None:
    """Delete the oldest utterance folders, so the voice folder cannot grow without limit."""
    folders = sorted(p for p in config.VOICE_DIR.iterdir() if p.is_dir() and UTTERANCE_ID.match(p.name))
    for old in folders[:-keep] if len(folders) > keep else []:
        shutil.rmtree(old, ignore_errors=True)
