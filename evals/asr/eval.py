"""Speech recognition eval: word error rate of Gemma 4 audio and of Whisper, on the same clips.

Cases: data/evals/{synthetic,user}/asr/
    clip01.wav   one utterance, under 30 s
    clip01.txt   exactly what you said (the reference transcript)

Metric: WER = (substitutions + deletions + insertions) / reference words,
after lowercasing, removing punctuation and removing fillers (um, uh, ...) on both sides.

Session 2 adds Whisper (the accelerator's `transcribe` hook) for always-on listening:
    wer / gemma_s / gemma_rtf               Gemma 4 through llama-server
    whisper_wer / whisper_s / whisper_rtf   Whisper large-v3-turbo
`_s` is the median seconds per clip. `_rtf` is the real-time factor: seconds of work per second of audio,
so 0.1 means 10 s of speech is transcribed in 1 s. Always-on listening needs an RTF well under 1.
Each engine first transcribes one second of silence, untimed: Whisper loads its weights on the first
call, and that one-time cost is not a property of each clip.
If the accelerator has no transcribe hook, the Whisper metrics are left out.
"""

import re
import statistics
import time
from pathlib import Path

import jiwer
import numpy as np

from agent import config
from agent.platforms.base import Unsupported
from agent.senses import audio, stt
from evals.common import NoCases

LEVEL, USES_LLM = "component", True
FILLERS = {"um", "uh", "erm", "er", "hmm", "mm"}
_warm: set[str] = set()  # engines that already ran once in this process


def normalize(text: str) -> str:
    """Normalize a transcript for word error rate.

    Args:
        text (str): Transcript text.

    Returns:
        str: Lowercase text without punctuation and fillers.
    """
    text = re.sub(r"[^\w\s']", " ", text.lower())
    return " ".join(w for w in text.split() if w not in FILLERS)


def load_cases(cases_dir: Path):
    """Load the cases: each WAV clip with the text of the .txt file beside it.

    Args:
        cases_dir (Path): The case folder of this eval.

    Returns:
        list[dict]: One item per case.

    Raises:
        NoCases: If there is no clip with a transcript.
    """
    items = [
        {"wav": w, "reference": w.with_suffix(".txt").read_text().strip()}
        for w in sorted(cases_dir.glob("*.wav"))
        if w.with_suffix(".txt").exists()
    ]
    if not items:
        raise NoCases(cases_dir)
    return items


def run_item(item):
    """Run one case: transcribe one clip with the model and count word errors.

    Args:
        item (dict): An item from `load_cases`.

    Returns:
        dict: A record with `reference`, `hypothesis`, `errors` and `ref_words`.
    """
    samples = audio.load_wav(item["wav"])
    rec = {"file": item["wav"].name, "audio": str(item["wav"]), "reference": item["reference"]}
    rec["audio_s"] = round(len(samples) / config.SAMPLE_RATE, 2)
    for engine, prefix in (("gemma", ""), ("whisper", "whisper_")):
        try:
            warm_up(engine)
            t0 = time.time()
            hyp = stt.transcribe(samples, engine)
        except Unsupported:
            continue  # this accelerator has no Whisper: report Gemma only
        rec[f"{engine}_s"] = round(time.time() - t0, 2)
        rec |= {f"{prefix}hypothesis": hyp, f"{prefix}errors": errors(item["reference"], hyp)}
    rec["ref_words"] = len(normalize(item["reference"]).split())
    return rec


def warm_up(engine: str) -> None:
    """Run an engine once on one second of silence, so loading its weights is not timed as a clip.

    Args:
        engine (str): "gemma" or "whisper".
    """
    if engine not in _warm:
        stt.transcribe(np.zeros(config.SAMPLE_RATE, np.float32), engine)
        _warm.add(engine)


def errors(reference: str, hypothesis: str) -> int:
    """Count word errors after normalizing both sides.

    Args:
        reference (str): What was said.
        hypothesis (str): What the engine heard.

    Returns:
        int: Substitutions + deletions + insertions.
    """
    out = jiwer.process_words(normalize(reference), normalize(hypothesis))
    return out.substitutions + out.deletions + out.insertions


def score(records):
    """Compute the metrics: word error rate over all clips.

    Args:
        records (list[dict]): The records from `run_item`.

    Returns:
        dict: Metric name to value.
    """
    words = max(sum(r["ref_words"] for r in records), 1)
    out = {"wer": round(sum(r["errors"] for r in records) / words, 3)}
    out |= _speed(records, "gemma")
    if all("whisper_errors" in r for r in records):
        out["whisper_wer"] = round(sum(r["whisper_errors"] for r in records) / words, 3)
        out |= _speed(records, "whisper")
    return out


def _speed(records: list[dict], engine: str) -> dict:
    """Median seconds per clip and the real-time factor of one engine."""
    took = [r[f"{engine}_s"] for r in records if f"{engine}_s" in r]
    audio_s = sum(r.get("audio_s", 0) for r in records if f"{engine}_s" in r)
    if not took:
        return {}
    out = {f"{engine}_s": round(statistics.median(took), 2)}
    if audio_s:
        out[f"{engine}_rtf"] = round(sum(took) / audio_s, 3)
    return out
