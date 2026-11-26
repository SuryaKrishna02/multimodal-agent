"""Wake-word training clips from Kokoro, in place of Piper's `generate_samples`, which openWakeWord's train.py imports.

openWakeWord's automatic training makes its synthetic clips with Piper, and upstream supports that on Linux only.
train.py imports `generate_samples` from the folder in `piper_sample_generator_path`, so this file takes its place
and makes the clips with the agent's own voice (agent/voice/tts.py) instead. Each clip has a new speaker: two Kokoro
voices from the training set, mixed with a random weight, at a random speed. "Sanjaya" is spoken from its IPA in
`config.PRONUNCIATIONS`, the way the user says it. The voices in `config.WAKE_HELD_OUT_VOICES` never make a clip.

This file runs in the training venv (scripts/train_wake_word.py), which has torch; the agent venv never imports it.
"""

import sys
import uuid
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from agent import config  # noqa: E402
from agent.voice import tts  # noqa: E402

CLIP_RATE = 16000  # openWakeWord reads 16 kHz, 16-bit clips


def training_voices() -> list[str]:
    """List the Kokoro voices that may make a training clip.

    Returns:
        list[str]: Every voice in the voices file except the held-out ones.
    """
    return [v for v in tts.voices() if v not in config.WAKE_HELD_OUT_VOICES]


def near_miss_phrases() -> list[str]:
    """Build the near-miss phrases: every greeting with every name in config, except the wake phrase itself.

    openWakeWord makes these with DeepPhonemizer, whose pickled checkpoint torch 2.6 and later refuses to load by
    default. A hand-picked list is also easier to read: these are the sounds the model must learn to ignore.

    Returns:
        list[str]: For example "hey sanjay", "okay sanjaya", "sanjaya" (the name alone) and "hey" alone.
    """
    phrases = {f"{g} {n}".strip() for g in config.WAKE_NEAR_GREETINGS for n in config.WAKE_NEAR_NAMES}
    return sorted(phrases - {config.WAKE_PHRASE, ""})


def _speaker(rng: np.random.Generator, pool: list[str]) -> np.ndarray:
    """Mix two voices into a new speaker style."""
    k = tts.engine()
    a, b = rng.choice(pool, size=2, replace=False)
    w = rng.uniform(0.2, 0.8)
    return w * k.get_voice_style(str(a)) + (1 - w) * k.get_voice_style(str(b))


def generate_samples(text, max_samples: int, output_dir, file_names=None, length_scales=(1.0,), **_piper_only) -> None:
    """Write `max_samples` WAV clips of the phrases in `text`, each with a new mixed speaker.

    Args:
        text (str | list[str]): One phrase or a list; clips cycle through the list.
        max_samples (int): How many clips to write.
        output_dir (str | Path): Folder for the clips.
        file_names (list[str] | None, optional): Names to use, in order. Defaults to random names.
        length_scales (tuple[float, ...], optional): Piper's speaking-length factors; speed is 1 / factor.
        **_piper_only: Piper arguments (noise scales, batch size...) that Kokoro has no use for.
    """
    phrases = [text] if isinstance(text, str) else list(text)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    names = list(file_names or [])
    rng, pool, k = np.random.default_rng(), training_voices(), tts.engine()
    for i in range(max_samples):
        phrase = phrases[i % len(phrases)]
        speed = float(np.clip(rng.uniform(0.9, 1.1) / rng.choice(list(length_scales)), 0.6, 1.6))
        audio, rate = k.create(
            tts.phonemes(phrase), voice=_speaker(rng, pool), speed=speed, lang=config.TTS_LANG, is_phonemes=True
        )
        clip = resample_poly(audio, CLIP_RATE // 1000, rate // 1000)  # 24 kHz -> 16 kHz
        pcm = (np.clip(clip, -1, 1) * 32767).astype(np.int16)
        name = names[i] if i < len(names) else f"{uuid.uuid4().hex}.wav"
        sf.write(out / name, pcm, CLIP_RATE, subtype="PCM_16")
