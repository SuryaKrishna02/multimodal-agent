"""Synthetic audio cases: speech from the OS text-to-speech plug-in, mixed with generated sound.

Speech comes from `platforms.current().synthesize_speech`, so this module runs on any OS whose
plug-in implements it (else `Unsupported`). The sound around the speech (room tone, key clicks,
a chord) is numpy noise with a fixed seed, so a rebuild gives the same files.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from agent import config, platforms
from agent.senses import audio

SR = config.SAMPLE_RATE
EDGE_LEVEL = 0.02  # fraction of the peak that counts as sound when trimming the silence around speech

# Spoken sentences. Numbers are spelled out: a model may write digits, and that is a style choice, not an error.
SENTENCES = [
    "Please send the quarterly report to John Doe before lunch.",
    "Ravi said the demo will start at quarter to four.",
    "I need to buy oat milk, eggs and a loaf of bread.",
    "The plumber is coming on Monday morning to fix the kitchen sink.",
    "Remind me to renew my passport before the end of the month.",
    "Our meeting moved to Thursday afternoon in the small conference room.",
    "She booked a table for six at the Italian restaurant on Friday.",
    "Can you please check whether the backup finished without any errors?",
    "The launch plan covers marketing, hiring and the budget for next year.",
    "Turn left at the bakery and the library is on your right.",
]

# (sentences, action items). Pauses between sentences make the VAD cut the note into chunks.
NOTES = [
    (
        [
            "Okay so, I need to email John Doe the slides before lunch.",
            "Also, book a room for the Thursday meeting.",
        ],
        ["Email John Doe the slides", "Book a room for the Thursday meeting"],
    ),
    (
        [
            "The weather is lovely today.",
            "I had a really good coffee this morning.",
            "Nothing else is going on.",
        ],
        [],
    ),
    (
        [
            "Remind me to call the plumber on Monday.",
            "And I should pay the electricity bill by Friday.",
            "Oh, and pick up oat milk on the way home.",
        ],
        ["Call the plumber on Monday", "Pay the electricity bill by Friday", "Pick up oat milk"],
    ),
    (
        [
            "Ravi said the demo went well.",
            "I need to send him the notes tonight.",
        ],
        ["Send Ravi the notes"],
    ),
    (
        [
            "Don't forget to renew the passport.",
            "The report is due next Tuesday, so I have to finish the draft before then.",
        ],
        ["Renew the passport", "Finish the report draft"],
    ),
]


def trim(samples: np.ndarray) -> np.ndarray:
    """Cut the silence before and after the sound.

    Args:
        samples (np.ndarray): Mono audio.

    Returns:
        np.ndarray: The part from the first to the last sample above `EDGE_LEVEL` of the peak.
    """
    loud = np.flatnonzero(np.abs(samples) > EDGE_LEVEL * np.abs(samples).max())
    return samples[loud[0] : loud[-1] + 1]


def speak(text: str) -> np.ndarray:
    """Synthesize speech with the OS plug-in.

    Args:
        text (str): What to say.

    Returns:
        np.ndarray: 16 kHz mono float32 audio without the silence at both ends.

    Raises:
        Unsupported: If the OS plug-in has no speech hook.
    """
    with tempfile.TemporaryDirectory() as d:
        return trim(audio.load_wav(platforms.current().synthesize_speech(text, Path(d) / "speech")))


def silence(seconds: float) -> np.ndarray:
    """Return digital silence of the given length."""
    return np.zeros(int(seconds * SR), np.float32)


def room_tone(seconds: float, rng: np.random.Generator, level: float = 0.003) -> np.ndarray:
    """Return faint broadband noise, like a quiet room."""
    return rng.normal(0, level, int(seconds * SR)).astype(np.float32)


def key_clicks(seconds: float, rng: np.random.Generator) -> np.ndarray:
    """Return room tone with short, sharp clicks, like typing on a keyboard."""
    out = room_tone(seconds, rng)
    for start in rng.uniform(0, seconds - 0.05, int(seconds * 4)):
        i, n = int(start * SR), int(0.01 * SR)
        out[i : i + n] += rng.normal(0, 0.25, n) * np.exp(-np.arange(n) / (0.002 * SR))
    return out


def chord(seconds: float, rng: np.random.Generator) -> np.ndarray:
    """Return a soft A-major chord with a slow tremolo, like music in the background."""
    t = np.arange(int(seconds * SR)) / SR
    tones = sum(np.sin(2 * np.pi * f * t) for f in (220.0, 277.2, 329.6))
    return (0.08 * tones * (0.7 + 0.3 * np.sin(2 * np.pi * 0.5 * t)) + room_tone(seconds, rng)).astype(np.float32)


def add_noise(samples: np.ndarray, rng: np.random.Generator, snr_db: float) -> np.ndarray:
    """Mix white noise into a signal.

    Args:
        samples (np.ndarray): The signal.
        rng (np.random.Generator): Source of the noise.
        snr_db (float): Signal-to-noise ratio, measured over the whole signal.

    Returns:
        np.ndarray: The noisy signal.
    """
    noise_power = float(np.mean(samples**2)) / 10 ** (snr_db / 10)
    return (samples + rng.normal(0, np.sqrt(noise_power), len(samples))).astype(np.float32)


def write_wav(path: Path, samples: np.ndarray) -> None:
    """Write 16-bit mono WAV at the project sample rate, scaled down if it would clip.

    Args:
        path (Path): File to write. Its folder is created.
        samples (np.ndarray): Mono audio.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    peak = float(np.abs(samples).max())
    sf.write(path, samples * (0.9 / peak if peak > 0.9 else 1.0), SR, subtype="PCM_16")


def compose(parts: list[tuple[np.ndarray, bool]]) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """Join audio parts and note where the speech parts are.

    Args:
        parts (list[tuple[np.ndarray, bool]]): (audio, is_speech) pairs in order.

    Returns:
        tuple[np.ndarray, list[tuple[float, float]]]: The audio and the (start, end) seconds of each speech part.
    """
    labels, pos = [], 0
    for samples, is_speech in parts:
        if is_speech:
            labels.append((round(pos / SR, 2), round((pos + len(samples)) / SR, 2)))
        pos += len(samples)
    return np.concatenate([p for p, _ in parts]), labels


def build_asr(out: Path) -> None:
    """Write one clip per sentence, in clean, noisy and quiet conditions, with the transcript beside it.

    Args:
        out (Path): The case folder of the `asr` eval.
    """
    rng = np.random.default_rng(1)
    conditions = ["clean", "clean", "noisy", "quiet"]
    for i, text in enumerate(SENTENCES, 1):
        samples, condition = speak(text), conditions[i % len(conditions)]
        if condition == "noisy":
            samples = add_noise(samples, rng, snr_db=12)
        elif condition == "quiet":
            samples = add_noise(samples * 0.15, rng, snr_db=25)
        write_wav(out / f"clip{i:02d}-{condition}.wav", samples)
        (out / f"clip{i:02d}-{condition}.txt").write_text(text)


def _vad_clips(rng: np.random.Generator) -> dict[str, list[tuple[np.ndarray, bool]]]:
    """Describe the VAD clips as parts for `compose`."""
    a, b, c = speak(SENTENCES[0]), speak(SENTENCES[3]), speak(SENTENCES[7])
    return {
        "silence": [(room_tone(20, rng), False)],
        "keyboard": [(key_clicks(20, rng), False)],
        "music": [(chord(20, rng), False)],
        "pauses": [(room_tone(1.5, rng), False), (a, True), (room_tone(3.5, rng), False), (b, True)]
        + [(room_tone(2.5, rng), False), (c, True), (room_tone(1.5, rng), False)],
        "mixed": [(room_tone(1, rng), False), (a, True), (key_clicks(2.5, rng), False), (b, True)]
        + [(chord(2, rng), False), (c, True), (room_tone(1, rng), False)],
    }


def build_vad(out: Path) -> None:
    """Write clips with and without speech, and the speech regions of each.

    Args:
        out (Path): The case folder of the `vad` eval.
    """
    rng = np.random.default_rng(2)
    for name, parts in _vad_clips(rng).items():
        samples, labels = compose(parts)
        if name == "mixed":
            samples = add_noise(samples, rng, snr_db=15)
        write_wav(out / f"{name}.wav", samples)
        (out / f"{name}.txt").write_text("".join(f"{a:.2f}\t{b:.2f}\tspeech\n" for a, b in labels))


def note_audio(sentences: list[str], rng: np.random.Generator) -> np.ndarray:
    """Build a voice note: sentences separated by pauses that are long enough to split the note into chunks.

    Args:
        sentences (list[str]): What to say, one sentence per chunk.
        rng (np.random.Generator): Source of the room tone in the pauses.

    Returns:
        np.ndarray: The voice note.
    """
    parts: list[tuple[np.ndarray, bool]] = [(room_tone(0.5, rng), False)]
    for s in sentences:
        parts += [(speak(s), True), (room_tone(1.2, rng), False)]
    return compose(parts)[0]


def build_task1(out: Path) -> None:
    """Write voice notes and the action items in each.

    Args:
        out (Path): The case folder of the `task1` eval.
    """
    rng = np.random.default_rng(3)
    for i, (sentences, items) in enumerate(NOTES, 1):
        write_wav(out / f"note{i:02d}.wav", note_audio(sentences, rng))
        (out / f"note{i:02d}.json").write_text(json.dumps({"action_items": items}))
