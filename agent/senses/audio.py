"""Hearing: mic capture -> voice activity detection -> chunks under 30 s.

Why chunk ourselves? Gemma 4 takes at most 30 s per audio clip. llama.cpp cuts
longer clips every 30 s, even in the middle of a word. We cut at pauses instead.
"""

import io
import threading

import numpy as np
import onnxruntime as ort
import sounddevice as sd
import soundfile as sf

from agent import config, events

SR = config.SAMPLE_RATE
FRAME = 512  # Silero VAD frame: 512 samples = 32 ms at 16 kHz


# --- Audio I/O ---------------------------------------------------------------


def load_wav(source) -> np.ndarray:
    """A path or file object (WAV/FLAC/AIFF) -> float32 mono at 16 kHz."""
    source = source if hasattr(source, "read") else str(source)
    samples, sr = sf.read(source, dtype="float32", always_2d=True)
    samples = samples.mean(axis=1)
    if sr != SR:  # simple linear resampling is fine for speech
        t_new = np.arange(int(len(samples) * SR / sr)) / SR
        samples = np.interp(t_new, np.arange(len(samples)) / sr, samples).astype(np.float32)
    return samples


def to_wav_bytes(samples: np.ndarray) -> bytes:
    """Encode audio as 16-bit PCM WAV bytes.

    Args:
        samples (np.ndarray): 16 kHz mono float32 audio.

    Returns:
        bytes: The WAV file contents.
    """
    buf = io.BytesIO()
    sf.write(buf, samples, SR, format="WAV", subtype="PCM_16")
    return buf.getvalue()


class Mic:
    """Record from the default input device until stop(). Respects the pause switch."""

    paused = False  # the global "ambient listening" switch (dashboard + CLI)

    def __init__(self):
        """Create an idle microphone recorder."""
        self._frames: list[np.ndarray] = []
        self._lock = threading.Lock()
        self._stream: sd.InputStream | None = None

    def start(self) -> None:
        """Open the default input device and start recording.

        Raises:
            RuntimeError: If the global pause switch (`Mic.paused`) is on.
        """
        if Mic.paused:
            raise RuntimeError("Microphone is paused")
        self._frames = []
        self._stream = sd.InputStream(samplerate=SR, channels=1, dtype="float32", callback=self._on_audio)
        self._stream.start()
        events.emit("senses.mic", state="recording", device=sd.query_devices(kind="input")["name"])

    def _on_audio(self, block, _frames, _time, _status):
        with self._lock:
            self._frames.append(block[:, 0].copy())

    def stop(self) -> np.ndarray:
        """Stop recording and return what was recorded.

        Returns:
            np.ndarray: 16 kHz mono float32 audio, empty if nothing was recorded.
        """
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
        with self._lock:
            samples = np.concatenate(self._frames) if self._frames else np.zeros(0, np.float32)
        events.emit("senses.mic", state="stopped", seconds=round(len(samples) / SR, 1))
        return samples


# --- Voice activity detection ------------------------------------------------


class VAD:
    """Silero VAD (ONNX). Gives the probability of speech for each 32 ms frame."""

    def __init__(self):
        """Load the Silero VAD ONNX model from `config.VAD_MODEL`."""
        self.session = ort.InferenceSession(str(config.VAD_MODEL))

    def frame_probs(self, samples: np.ndarray) -> np.ndarray:
        """Run the model over the audio frame by frame.

        Args:
            samples (np.ndarray): 16 kHz mono float32 audio.

        Returns:
            np.ndarray: Speech probability for each 32 ms frame.
        """
        state = np.zeros((2, 1, 128), np.float32)
        context = np.zeros((1, 64), np.float32)  # model sees the last 64 samples too
        probs = []
        for i in range(0, len(samples) - FRAME + 1, FRAME):
            frame = samples[i : i + FRAME][None, :]
            out, state = self.session.run(
                None, {"input": np.concatenate([context, frame], axis=1), "state": state, "sr": np.array(SR, np.int64)}
            )
            context = frame[:, -64:]
            probs.append(float(np.asarray(out)[0][0]))
        return np.array(probs)

    def segments(self, samples: np.ndarray, probs: np.ndarray | None = None) -> list[tuple[float, float]]:
        """Speech regions as (start_s, end_s), after merging short pauses and dropping blips."""
        is_speech = (self.frame_probs(samples) if probs is None else probs) > config.VAD_THRESHOLD
        frame_s = FRAME / SR
        raw, start = [], None
        for i, s in enumerate(np.append(is_speech, False)):
            if s and start is None:
                start = i
            elif not s and start is not None:
                raw.append([start * frame_s, i * frame_s])
                start = None
        merged = []
        for seg in raw:  # a pause shorter than VAD_MIN_SILENCE_S does not end speech
            if merged and seg[0] - merged[-1][1] < config.VAD_MIN_SILENCE_S:
                merged[-1][1] = seg[1]
            else:
                merged.append(seg)
        duration = len(samples) / SR
        return [
            (max(0.0, a - config.VAD_PAD_S), min(duration, b + config.VAD_PAD_S))
            for a, b in merged
            if b - a >= config.VAD_MIN_SPEECH_S
        ]


# --- Chunking ----------------------------------------------------------------


def chunk(samples: np.ndarray, vad: VAD) -> list[dict]:
    """Group speech segments into clips of at most MAX_CHUNK_S, cutting only at pauses.

    A segment longer than the limit (someone talking non-stop) is cut at its quietest
    frame in the last 5 s before the limit, so we rarely cut through a word.
    """
    limit, frame_s = config.MAX_CHUNK_S, FRAME / SR
    probs = vad.frame_probs(samples)
    pieces = []
    for a, b in vad.segments(samples, probs):
        while b - a > limit:
            lo, hi = int((a + limit - 5) / frame_s), int((a + limit) / frame_s)
            cut = (lo + int(np.argmin(probs[lo:hi]))) * frame_s
            pieces.append((a, cut))
            a = cut
        pieces.append((a, b))

    chunks: list[list[float]] = []
    for a, b in pieces:
        if chunks and b - chunks[-1][0] <= limit:
            chunks[-1][1] = b  # still fits: extend the current chunk (keeps the pause inside)
        else:
            chunks.append([a, b])

    out = [
        {
            "start_s": round(a, 2),
            "end_s": round(b, 2),
            "samples": samples[int(a * SR) : int(b * SR)],
            "est_tokens": int((b - a) * config.AUDIO_TOKENS_PER_S),
        }
        for a, b in chunks
    ]
    events.emit(
        "senses.audio",
        seconds=round(len(samples) / SR, 1),
        speech=[(round(a, 2), round(b, 2)) for a, b in pieces],
        chunks=[(c["start_s"], c["end_s"]) for c in out],
        est_tokens=sum(c["est_tokens"] for c in out),
    )
    return out
