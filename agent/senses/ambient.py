"""Always-on hearing: a stream of audio -> finished utterances -> transcripts in memory.

Session 1 recorded one voice note and cut it afterwards. An always-on listener never gets an
"afterwards", so `Segmenter` cuts while the audio arrives: an utterance is finished when the VAD
hears VAD_MIN_SILENCE_S of silence after it, or when it reaches MAX_CHUNK_S.

The live microphone (`Listener`) and the replay of a recording (`replay`) feed the same Segmenter,
so a replayed recording produces exactly the utterances that live listening would.
The pause switch (`audio.paused()`) closes the microphone itself, not only the processing.
"""

import queue
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np
import sounddevice as sd

from agent import config, events
from agent.senses import audio, stt

SR = config.SAMPLE_RATE


class Segmenter:
    """Cut a stream of audio into finished utterances, with wall-clock times.

    Attributes:
        buf (np.ndarray): Audio received but not yet part of a finished utterance.
        t0 (float): Wall-clock time (epoch seconds) of the first sample in `buf`.
    """

    def __init__(self, vad: audio.VAD, start_ts: float):
        """Start with an empty buffer.

        Args:
            vad (audio.VAD): The voice activity detector.
            start_ts (float): Wall-clock time of the first sample that will be fed.
        """
        self.vad, self.buf, self.t0 = vad, np.zeros(0, np.float32), start_ts

    def feed(self, samples: np.ndarray, final: bool = False) -> list[dict]:
        """Add audio and return the utterances that are now finished.

        Args:
            samples (np.ndarray): The next 16 kHz mono float32 samples.
            final (bool, optional): The stream ends here, so open speech counts as finished. Defaults to False.

        Returns:
            list[dict]: Utterances with `samples`, `start_ts` and `end_ts`.
        """
        self.buf = np.concatenate([self.buf, samples.astype(np.float32)])
        duration = len(self.buf) / SR
        segments = self.vad.segments(self.buf)
        if not segments:  # nothing said: keep a little audio, so the start of the next word is not lost
            self._drop(max(0.0, duration - config.AMBIENT_IDLE_KEEP_S))
            return []
        out, used = [], 0.0
        for a, b in segments:
            finished = final or duration - b >= config.VAD_MIN_SILENCE_S
            if not finished and b - a < config.MAX_CHUNK_S:
                break  # still talking: wait for more audio
            b = b if finished else a + config.MAX_CHUNK_S  # non-stop speech: cut at the limit
            out.append(self._utterance(a, b))
            used = b
        self._drop(used)
        return out

    def _utterance(self, a: float, b: float) -> dict:
        return {"samples": self.buf[int(a * SR) : int(b * SR)], "start_ts": self.t0 + a, "end_ts": self.t0 + b}

    def _drop(self, seconds: float) -> None:
        self.buf = self.buf[int(seconds * SR) :]
        self.t0 += seconds


def utterances(samples: np.ndarray, start_ts: float, step_s: float = config.AMBIENT_STEP_S) -> Iterator[dict]:
    """Feed a recording to a Segmenter in live-sized steps and yield each utterance as it finishes.

    Args:
        samples (np.ndarray): The whole recording, 16 kHz mono float32.
        start_ts (float): Wall-clock time that the recording starts at.
        step_s (float, optional): Seconds of audio per step, as the live listener uses. Defaults to AMBIENT_STEP_S.

    Yields:
        dict: Utterances with `samples`, `start_ts`, `end_ts` and `offset_s` (end time in the recording).
    """
    seg, step = Segmenter(audio.VAD(), start_ts), int(step_s * SR)
    for i in range(0, len(samples), step):
        final = i + step >= len(samples)
        for u in seg.feed(samples[i : i + step], final=final):
            yield {**u, "offset_s": u["end_ts"] - start_ts}


def hear(u: dict, source: str, engine: str | None = None) -> dict:
    """Transcribe one utterance and store it in memory as a raw `transcript` row with its audio vector.

    Args:
        u (dict): An utterance from `Segmenter.feed` or `utterances`.
        source (str): Where it came from, "mic" or "replay:<file>".
        engine (str, optional): STT engine. Defaults to `config.STT_ENGINE`.

    Returns:
        dict: `text`, `memory_id` (None if the junk filter refused it), `start_ts` and `end_ts`.
    """
    from agent.memory import episodic

    text = stt.transcribe(u["samples"], engine)
    meta = {"start_ts": u["start_ts"], "end_ts": u["end_ts"], "source": source}
    mem_id = episodic.store(text, kind="transcript", meta=meta, ts=u["start_ts"], media=("audio", u["samples"]))
    events.emit(
        "senses.ambient", text=text, memory_id=mem_id, source=source, seconds=round(u["end_ts"] - u["start_ts"], 1)
    )
    return {"text": text, "memory_id": mem_id, "start_ts": u["start_ts"], "end_ts": u["end_ts"]}


def load_replay(path: Path) -> np.ndarray:
    """Load a recording to replay.

    Args:
        path (Path): A WAV, AIFF or FLAC file.

    Returns:
        np.ndarray: 16 kHz mono float32 audio.
    """
    return audio.load_wav(path)


class Listener:
    """The live microphone, always on unless paused. Each finished utterance goes to `on_utterance`."""

    def __init__(self, on_utterance: Callable[[dict], object] | None = None):
        """Prepare the listener without opening the microphone.

        Args:
            on_utterance (Callable, optional): Called with each utterance. Defaults to `hear(u, "mic")`.
        """
        self.on_utterance = on_utterance or (lambda u: hear(u, "mic"))
        self._q: queue.Queue[np.ndarray] = queue.Queue()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the worker thread. It opens and closes the microphone as the pause switch changes."""
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="ambient")
        self._thread.start()

    def stop(self) -> None:
        """Stop listening and close the microphone."""
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=10)

    def _open(self) -> sd.InputStream:
        stream = sd.InputStream(samplerate=SR, channels=1, dtype="float32", callback=self._on_audio)
        stream.start()
        events.emit("senses.ambient.state", state="listening", device=sd.query_devices(kind="input")["name"])
        return stream

    def _on_audio(self, block, _frames, _time, _status):
        self._q.put(block[:, 0].copy())

    def _drain(self) -> np.ndarray:
        blocks = []
        while not self._q.empty():
            blocks.append(self._q.get())
        return np.concatenate(blocks) if blocks else np.zeros(0, np.float32)

    def _run(self) -> None:
        stream, seg = None, Segmenter(audio.VAD(), time.time())
        while not self._stop.wait(config.AMBIENT_STEP_S):
            if audio.paused():
                if stream:  # really close the microphone
                    stream.close()
                    stream = None
                    events.emit("senses.ambient.state", state="paused")
                continue
            if stream is None:
                stream, seg = self._open(), Segmenter(audio.VAD(), time.time())
            for u in seg.feed(self._drain()):
                self.on_utterance(u)
        if stream:
            stream.close()
        events.emit("senses.ambient.state", state="stopped")
