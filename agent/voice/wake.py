"""Wake word "Hey Jarvis": three small ONNX models, one 80 ms frame at a time.

    audio (16 kHz)  --mel spectrogram-->  32 mel bands per 10 ms
                    --speech embedding--> one 96-number vector per 80 ms (from the last 76 mel frames)
                    --classifier------->  a score from 0 to 1 (from the last 16 vectors = 1.28 s)

The first two models are openWakeWord's frozen feature extractor (Google's speech embedding), the same for every wake
word. The classifier is openWakeWord's pretrained "hey jarvis" model. All three come from openWakeWord release v0.5.1
(agent/pins.py). This file repeats the streaming steps of openWakeWord's `AudioFeatures` (openwakeword/utils.py) in
plain numpy, so the agent does not need the openwakeword package, its TensorFlow Lite runtime or scikit-learn.

Two stages decide (`Gate`). A score of `config.WAKE_THRESHOLD` or more is a wake at once. A lower score, from
`config.WAKE_CHECK_THRESHOLD`, is a "maybe": Whisper reads the last `config.WAKE_CHECK_S` of audio (`said_name`), and a
word like "Jarvis" makes it a wake. The model learned from synthetic voices, and real voices often score 0.1 to 0.5
(docs/VERIFICATION.md). Whisper runs only on a "maybe", so it costs nothing while the room is quiet. After a wake, the
detector ignores the phrase for `config.WAKE_REFRACTORY_S`, so one "Hey Jarvis" wakes once.
"""

import difflib
import re
from collections.abc import Callable
from pathlib import Path

import numpy as np

from agent import config, events
from agent.senses import stt

FRAME = 1280  # samples per step: 80 ms at 16 kHz
MEL_CONTEXT = 480  # extra samples before each frame, so the mel model sees whole windows (openWakeWord: 160 * 3)
MEL_WINDOW, EMBED_FRAMES = 76, 16  # mel frames per embedding, embeddings per classifier input
WARMUP_FRAMES = 5  # openWakeWord reports 0 for the first 5 frames, while the buffers still hold start-up data


def said_name(samples: np.ndarray) -> bool:
    """Tell whether Whisper hears the wake name in a short clip (the second stage).

    Args:
        samples (np.ndarray): 16 kHz mono float32, the last `config.WAKE_CHECK_S` of audio.

    Returns:
        bool: True if a heard word is at least `config.WAKE_NAME_SIMILARITY` like `config.WAKE_NAME`.
    """
    text = stt.transcribe(samples)
    words = re.findall(r"[a-z]+", text.lower())
    best = max((difflib.SequenceMatcher(None, w, config.WAKE_NAME).ratio() for w in words), default=0.0)
    woke = best >= config.WAKE_NAME_SIMILARITY
    events.emit("voice.wake_check", text=text, similarity=round(best, 2), woke=woke)
    return woke


class Gate:
    """Turns the score of each frame into wakes: a sure score wakes, a "maybe" asks `confirm` once."""

    def __init__(
        self,
        threshold: float = config.WAKE_THRESHOLD,
        check_threshold: float | None = config.WAKE_CHECK_THRESHOLD,
        confirm: Callable[[np.ndarray], bool] | None = said_name,
    ):
        """Set the two thresholds.

        Args:
            threshold (float, optional): A sure wake. Defaults to `config.WAKE_THRESHOLD`.
            check_threshold (float | None, optional): A "maybe" that `confirm` checks. None = no second stage.
                Defaults to `config.WAKE_CHECK_THRESHOLD`.
            confirm (Callable[[np.ndarray], bool] | None, optional): Reads the recent audio and confirms a "maybe".
                None = no second stage. Defaults to `said_name` (Whisper).
        """
        self.threshold, self.check_threshold, self.confirm = threshold, check_threshold, confirm
        self.reset()

    def reset(self) -> None:
        """Forget earlier frames."""
        self.quiet_until = 0  # frame number until which a wake is ignored
        self.check_at: int | None = None  # frame number at which a pending "maybe" is checked
        self.no_check_until = 0  # one phrase is checked once

    def _wake(self, frame: int) -> bool:
        self.quiet_until = frame + int(config.WAKE_REFRACTORY_S * config.SAMPLE_RATE / FRAME)
        self.check_at = None
        return True

    def judge(self, frame: int, score: float, recent: Callable[[], np.ndarray]) -> bool:
        """Decide whether this frame ends a wake.

        Args:
            frame (int): The frame number (it grows by one per 80 ms).
            score (float): The model's score of this frame.
            recent (Callable[[], np.ndarray]): Returns the last `config.WAKE_CHECK_S` of audio, for `confirm`.

        Returns:
            bool: True once per wake.
        """
        if frame <= self.quiet_until:
            return False
        if score >= self.threshold:
            return self._wake(frame)
        maybe = self.check_threshold is not None and self.confirm is not None and score >= self.check_threshold
        if maybe and self.check_at is None and frame > self.no_check_until:
            self.check_at = frame + round(config.WAKE_CHECK_DELAY_S * config.SAMPLE_RATE / FRAME)
        if self.check_at is None or frame < self.check_at or self.confirm is None:
            return False
        self.check_at = None
        self.no_check_until = frame + round(config.WAKE_CHECK_S * config.SAMPLE_RATE / FRAME)
        return self.confirm(recent()) and self._wake(frame)


class Detector:
    """Streams 16 kHz audio through the three models and reports a score for each 80 ms frame."""

    def __init__(
        self,
        model: Path = config.WAKE_MODEL,
        threshold: float = config.WAKE_THRESHOLD,
        confirm: Callable[[np.ndarray], bool] | None = said_name,
    ):
        """Load the models.

        Args:
            model (Path, optional): The classifier. Defaults to `config.WAKE_MODEL`.
            threshold (float, optional): Score that counts as a wake. Defaults to `config.WAKE_THRESHOLD`.
            confirm (Callable[[np.ndarray], bool] | None, optional): The second stage for a "maybe" score. None =
                the model alone decides. Defaults to `said_name` (Whisper).

        Raises:
            FileNotFoundError: A model file is missing.
        """
        import onnxruntime as ort

        paths = (model.parent / "melspectrogram.onnx", model.parent / "embedding_model.onnx", model)
        for p in paths:
            if not p.exists():
                raise FileNotFoundError(f"{p.name} is missing: run python3 scripts/setup.py --only wake")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1  # tiny models: one thread, so the wake word never competes with Gemma
        self.mel, self.embed, self.clf = (
            ort.InferenceSession(str(p), options, providers=["CPUExecutionProvider"]) for p in paths
        )
        self.threshold = threshold
        self.gate = Gate(threshold, config.WAKE_CHECK_THRESHOLD, confirm)
        self.reset()

    def reset(self) -> None:
        """Forget all audio, as at start-up."""
        self.raw = np.zeros(0, dtype=np.int16)  # samples not yet in a full frame
        self.context = np.zeros(MEL_CONTEXT, dtype=np.int16)
        self.mels = np.ones((MEL_WINDOW, 32), dtype=np.float32)  # openWakeWord starts its mel buffer with ones
        rng = np.random.default_rng(0)  # ...and its feature buffer with the embeddings of 4 s of low noise
        noise = rng.integers(-1000, 1000, 16000 * 4).astype(np.int16)
        self.features = self._embed_clip(noise)
        self.frames = 0
        self.score = 0.0
        self.recent = np.zeros(0, dtype=np.int16)  # the last WAKE_CHECK_S of audio, for the second stage
        self.gate.reset()

    def _mel(self, samples: np.ndarray) -> np.ndarray:
        """Mel spectrogram of int16 samples, with openWakeWord's scaling (x / 10 + 2)."""
        out = np.asarray(self.mel.run(None, {"input": samples.astype(np.float32)[None, :]})[0])
        return np.squeeze(out) / 10 + 2

    def _embedding(self, window: np.ndarray) -> np.ndarray:
        """One 96-number embedding from 76 mel frames."""
        out = self.embed.run(None, {"input_1": window[None, :, :, None].astype(np.float32)})[0]
        return np.asarray(out).reshape(-1, 96)

    def _embed_clip(self, samples: np.ndarray) -> np.ndarray:
        """Embeddings of a whole clip: one per 8 mel frames (openWakeWord's `_get_embeddings`)."""
        mels = self._mel(samples)
        starts = range(0, len(mels) - MEL_WINDOW + 1, 8)
        return np.vstack([self._embedding(mels[i : i + MEL_WINDOW]) for i in starts])

    def _step(self, frame: np.ndarray) -> float:
        """Run one 80 ms frame through the three models and return its score."""
        self.mels = np.vstack((self.mels, self._mel(np.concatenate((self.context, frame)))))[-10 * 97 :]
        self.context = frame[-MEL_CONTEXT:]
        self.recent = np.concatenate((self.recent, frame))[-int(config.WAKE_CHECK_S * config.SAMPLE_RATE) :]
        self.features = np.vstack((self.features, self._embedding(self.mels[-MEL_WINDOW:])))[-120:]
        self.frames += 1
        x = self.features[-EMBED_FRAMES:][None, :, :].astype(np.float32)
        score = float(np.asarray(self.clf.run(None, {self.clf.get_inputs()[0].name: x})[0]).reshape(-1)[0])
        self.score = 0.0 if self.frames <= WARMUP_FRAMES else score  # the newest score, for `cli.py wake test`
        return self.score

    def push(self, samples: np.ndarray) -> list[float]:
        """Add audio and return the score of each 80 ms frame that it completes.

        Args:
            samples (np.ndarray): 16 kHz mono, float32 in [-1, 1] or int16.

        Returns:
            list[float]: One score per completed frame (none while a frame is still filling).
        """
        pcm = samples if samples.dtype == np.int16 else (np.clip(samples, -1, 1) * 32767).astype(np.int16)
        self.raw = np.concatenate((self.raw, pcm))
        scores = []
        while len(self.raw) >= FRAME:
            frame, self.raw = self.raw[:FRAME], self.raw[FRAME:]
            scores.append(self._step(frame))
        return scores

    def heard(self, samples: np.ndarray) -> bool:
        """Add audio and tell whether the wake phrase ended in it (outside the quiet time after a wake).

        Args:
            samples (np.ndarray): 16 kHz mono audio.

        Returns:
            bool: True once per wake.
        """
        pcm = samples if samples.dtype == np.int16 else (np.clip(samples, -1, 1) * 32767).astype(np.int16)
        self.raw = np.concatenate((self.raw, pcm))
        woke = False
        while len(self.raw) >= FRAME:
            frame, self.raw = self.raw[:FRAME], self.raw[FRAME:]
            score = self._step(frame)  # the gate may need the audio up to this frame, so one frame at a time
            woke = self.gate.judge(self.frames, score, lambda: self.recent.astype(np.float32) / 32767) or woke
        return woke


def wakes(detector: Detector, samples: np.ndarray) -> list[float]:
    """Find every wake in a recording, as the live detector would (the wake eval uses this).

    Args:
        detector (Detector): A detector; it is reset first.
        samples (np.ndarray): 16 kHz mono audio.

    Returns:
        list[float]: The time of each wake, in seconds from the start (the end of the frame that fired).
    """
    detector.reset()
    times = []
    for start in range(0, len(samples) - FRAME + 1, FRAME):
        if detector.heard(samples[start : start + FRAME]):
            times.append((start + FRAME) / 16000)
    return times
