"""Speech to text for always-on listening: Whisper or Gemma, chosen in config.STT_ENGINE.

Why not send everything to Gemma? Gemma 4 spends about 25 tokens per second of audio, so listening
all day keeps the language model busy. A dedicated speech model (Whisper, through the accelerator's
`transcribe` hook) turns speech into text cheaply, and Gemma then reads only text. The asr eval
compares the two engines on the same clips.
"""

import time

import numpy as np

from agent import accel, config, events

GEMMA_PROMPT = "Transcribe this audio exactly. Output only the transcript."


def gemma(samples: np.ndarray) -> str:
    """Transcribe with the Gemma 4 language model.

    Args:
        samples (np.ndarray): 16 kHz mono float32 audio, under 30 s.

    Returns:
        str: The transcript.
    """
    from agent import brain

    msgs = [{"role": "user", "content": [brain.part("audio", samples), brain.part("text", GEMMA_PROMPT)]}]
    return (brain.send(msgs)["choices"][0]["message"].get("content") or "").strip()


def transcribe(samples: np.ndarray, engine: str | None = None) -> str:
    """Turn speech into text with the chosen engine, and report how long it took.

    Args:
        samples (np.ndarray): 16 kHz mono float32 audio.
        engine (str, optional): "whisper" or "gemma". Defaults to `config.STT_ENGINE`.

    Returns:
        str: The transcript.

    Raises:
        ValueError: If the engine is unknown.
    """
    engine = engine or config.STT_ENGINE
    t0 = time.time()
    if engine == "whisper":
        text = accel.current().transcribe(samples, config.STT_LANGUAGE)
    elif engine == "gemma":
        text = gemma(samples)
    else:
        raise ValueError(f"unknown STT engine '{engine}': use 'whisper' or 'gemma'")
    seconds = len(samples) / config.SAMPLE_RATE
    events.emit("senses.stt", engine=engine, audio_s=round(seconds, 1), took_s=round(time.time() - t0, 2), text=text)
    return text
