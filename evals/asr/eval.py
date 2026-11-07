"""Speech recognition eval: word error rate of Gemma 4 audio and of Whisper, on the same clips.

Cases: data/evals/{synthetic,user}/asr/
    clip01.wav   one utterance, under 30 s
    clip01.txt   exactly what you said (the reference transcript)

Metric: WER = (substitutions + deletions + insertions) / reference words,
after lowercasing, removing punctuation and removing fillers (um, uh, ...) on both sides.

Session 2 adds Whisper (the accelerator's `transcribe` hook) for always-on listening:
    wer / gemma_s           Gemma 4 through llama-server, and its seconds per clip
    whisper_wer / whisper_s Whisper large-v3-turbo, and its seconds per clip
If the accelerator has no transcribe hook, the Whisper metrics are left out.
"""

import re
import time
from pathlib import Path

import jiwer

from agent.platforms.base import Unsupported
from agent.senses import audio, stt
from evals.common import NoCases

LEVEL, USES_LLM = "component", True
FILLERS = {"um", "uh", "erm", "er", "hmm", "mm"}


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
    for engine, prefix in (("gemma", ""), ("whisper", "whisper_")):
        t0 = time.time()
        try:
            hyp = stt.transcribe(samples, engine)
        except Unsupported:
            continue  # this accelerator has no Whisper: report Gemma only
        rec |= {f"{prefix}hypothesis": hyp, f"{prefix}errors": errors(item["reference"], hyp)}
        rec[f"{engine}_s"] = round(time.time() - t0, 2)
    rec["ref_words"] = len(normalize(item["reference"]).split())
    return rec


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
    out["gemma_s"] = round(sum(r.get("gemma_s", 0) for r in records) / max(len(records), 1), 2)
    if all("whisper_errors" in r for r in records):
        out["whisper_wer"] = round(sum(r["whisper_errors"] for r in records) / words, 3)
        out["whisper_s"] = round(sum(r["whisper_s"] for r in records) / max(len(records), 1), 2)
    return out
