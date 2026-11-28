"""Synthetic cases of the Session 3 voice evals: tts, wake and task6.

Labeled by construction: we know what each sentence says, whether a clip holds the wake phrase, and what reminder a
spoken request asks for. The wake clips use the Kokoro voices in `config.WAKE_EVAL_VOICES` and the OS voices (another
speech engine). They are still cleaner than a real voice: add your own with `python cli.py evals add wake`.
"""

import json
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from agent import config
from agent.platforms.base import Unsupported
from evals.synthetic.audio import SENTENCES, add_noise, room_tone, speak, write_wav
from evals.synthetic.text import write_jsonl

# Sentences like the agent's replies: names, numbers, times and file names, where text to speech goes wrong.
TTS_SENTENCES = [
    "I set a reminder to call Mike at 3:15.",
    "The test parse dates is failing with a key error on line 42.",
    "Your meeting with the design team starts in 10 minutes.",
    "I saved a note to buy milk and eggs tomorrow morning.",
    "The build finished, but 2 warnings need your attention.",
    "You decided to move the offsite to Lakeside Lodge on the 14th.",
    "Emily will send the deposit by Wednesday.",
    "It is 72 degrees Fahrenheit, which is about 22 degrees Celsius.",
    "There are 3 open reminders, and the next one is at 4:30.",
    "Jarvis heard you. Say the word and I will stop.",
]
WAKE_PHRASES = ["Hey Jarvis.", "Hey Jarvis, remind me to call Mike.", "Hey Jarvis, what time is it?"]
# The agent's own voice in the microphone while the user says the phrase (no headphones), as a share of the user's
# loudness (RMS).
ECHO_LEVELS = (0.25, 0.5)
NEAR_MISSES = [
    "Hey Travis, can you send me the report?",
    "Hey Harvey, how are you today?",
    "I went to the market on Sunday with Mavis.",
    "The jar is on the shelf by the service desk.",
    "Okay, let us start the meeting now.",
    "Jarvis was the butler in the old films.",
    "Say, Marvin, did you see the game?",
]
# (spoken request, expected reminder text phrase, minutes from now, or None when the request gives a clock time)
TASK6_REQUESTS = [
    ("Remind me to call Mike in two minutes.", "Mike", 2),
    ("Remind me in five minutes to check the oven.", "oven", 5),
    ("Set a reminder for ten minutes from now to stretch.", "stretch", 10),
    ("In three minutes, remind me to join the meeting.", "meeting", 3),
    ("Remind me to send the slides to the group in fifteen minutes.", "slides", 15),
    ("Please remind me to drink water in one minute.", "water", 1),
]


def kokoro(text: str, voice: str) -> np.ndarray:
    """Speak with Kokoro (agent/voice/tts.py) at 16 kHz.

    Args:
        text (str): What to say.
        voice (str): A Kokoro voice name.

    Returns:
        np.ndarray: 16 kHz mono float32 audio.
    """
    from agent.voice import tts

    k = tts.engine()
    audio, rate = k.create(tts.phonemes(text), voice=voice, lang=config.TTS_LANG, is_phonemes=True)
    return resample_poly(audio, config.SAMPLE_RATE, rate).astype(np.float32)


def build_tts(out: Path) -> None:
    """Write the sentences that the tts eval speaks and transcribes.

    Args:
        out (Path): The case folder of the `tts` eval.
    """
    write_jsonl(out / "sentences.jsonl", [{"id": f"s{i:02d}", "text": t} for i, t in enumerate(TTS_SENTENCES, 1)])


def _clip(out: Path, name: str, samples: np.ndarray, wakes: int, rng: np.random.Generator, note: str) -> None:
    """Write one wake clip: half a second of room tone on each side, and its label."""
    pad = room_tone(0.5, rng)
    write_wav(out / f"{name}.wav", np.concatenate([pad, samples, pad]))
    (out / f"{name}.json").write_text(json.dumps({"wakes": wakes, "note": note}))


def build_wake(out: Path) -> None:
    """Write the wake clips: the phrase in the eval voices (clean and noisy), and near misses that must not wake.

    Args:
        out (Path): The case folder of the `wake` eval.
    """
    rng = np.random.default_rng(3)
    for v, voice in enumerate(config.WAKE_EVAL_VOICES):
        for p, phrase in enumerate(WAKE_PHRASES):
            samples = kokoro(phrase, voice)
            noisy = p == 2  # one of each voice's clips gets noise at 15 dB
            _clip(out, f"pos-{voice}-{p}", add_noise(samples, rng, 15) if noisy else samples, 1, rng, phrase)
        _clip(out, f"neg-{voice}", kokoro(NEAR_MISSES[v % len(NEAR_MISSES)], voice), 0, rng, NEAR_MISSES[v])
    for i, sentence in enumerate(NEAR_MISSES):
        _clip(out, f"neg-kokoro-{i}", kokoro(sentence, config.WAKE_EVAL_VOICES[-1 - i % 3]), 0, rng, sentence)
    _talk(out, rng)
    _echo(out, rng)
    try:  # another speech engine: the OS voices say the name their own way, a hard test
        _clip(out, "pos-os-0", speak("Hey Jarvis."), 1, rng, "OS voice: Hey Jarvis.")
        _clip(out, "neg-os-0", speak("Hey Travis, the report is ready."), 0, rng, "OS voice: Hey Travis")
    except Unsupported:
        pass


def _rms(x: np.ndarray) -> float:
    """Loudness of a clip (root mean square), never 0."""
    return float(np.sqrt(np.mean(x**2))) + 1e-9


def _echo(out: Path, rng: np.random.Generator) -> None:
    """Write the phrase said over the agent's voice (barge-in without headphones), and the agent talking alone.

    The agent speaks with `config.TTS_VOICE`: in real use the echo is always that voice. The phrase is said by the
    eval voices.
    """
    agent = np.concatenate([kokoro(s, config.TTS_VOICE) for s in TTS_SENTENCES[:-1]])  # the last one holds the name
    for voice in config.WAKE_EVAL_VOICES:
        phrase = kokoro(WAKE_PHRASES[0], voice)
        for level in ECHO_LEVELS:
            start = int(rng.integers(0, len(agent) - len(phrase) - config.SAMPLE_RATE))
            bed = agent[start : start + len(phrase) + config.SAMPLE_RATE]  # the agent talks 0.5 s before and after
            mixed = bed * (level * _rms(phrase) / _rms(bed))
            mixed[config.SAMPLE_RATE // 2 : config.SAMPLE_RATE // 2 + len(phrase)] += phrase
            note = f"Hey Jarvis over the agent's voice at {level:.0%}"
            _clip(out, f"pos-echo-{voice}-{int(level * 100)}", np.clip(mixed, -1, 1), 1, rng, note)
    _clip(out, "neg-echo-agent", agent, 0, rng, f"the agent's voice ({config.TTS_VOICE}) alone: must not wake")


def _talk(out: Path, rng: np.random.Generator) -> None:
    """Write a few minutes of ordinary talk without the phrase, for false wakes: every eval voice reads sentences.

    Then the Task 3 conversation (the OS voices), when the OS can speak.
    """
    lines = [*SENTENCES, *TTS_SENTENCES[:-1]]  # the last TTS sentence holds the name
    parts = []
    for i, line in enumerate(lines * 2):
        parts += [kokoro(line, config.WAKE_EVAL_VOICES[i % len(config.WAKE_EVAL_VOICES)]), room_tone(0.8, rng)]
    _clip(out, "neg-talk", np.concatenate(parts), 0, rng, f"{len(lines) * 2} sentences of ordinary talk")
    try:
        from evals.synthetic.day import CONVERSATION, speak_as

        talk = [x for voice, text in CONVERSATION for x in (speak_as(text, voice), room_tone(1.0, rng))]
        _clip(out, "neg-conversation", np.concatenate(talk), 0, rng, "the Task 3 conversation, OS voices")
    except Unsupported:
        pass


def build_task6(out: Path) -> None:
    """Write the spoken reminder requests of Task 6 (the OS voice), with what the reminder must hold.

    Args:
        out (Path): The case folder of the `task6` eval.
    """
    for i, (request, mention, minutes) in enumerate(TASK6_REQUESTS, 1):
        write_wav(out / f"r{i:02d}.wav", speak(request))
        (out / f"r{i:02d}.json").write_text(json.dumps({"text": request, "mention": mention, "minutes": minutes}))
