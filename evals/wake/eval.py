"""Wake-word eval: missed wakes and false wakes of "Hey Sanjaya".

Cases: data/evals/{synthetic,user}/wake/
    pos-01.wav   any recording, 16 kHz mono (`evals add` converts it)
    pos-01.json  {"wakes": 1, "note": "kitchen, far"}   how many times the phrase is said in it (0 for none)

Each recording streams through the live detector (agent/voice/wake.py: 80 ms frames, `config.WAKE_THRESHOLD`, the
quiet time after a wake), as `wakes()` does.

Metrics:
    miss_rate              share of expected wakes that did not happen (recordings with "wakes" >= 1), without the
                           echo recordings
    echo_miss_rate         the same for the phrase said over the agent's own voice (ids "pos-echo-..."): barge-in
                           without headphones. Reported, no threshold yet.
    false_wakes            wakes beyond the expected count, over all recordings
    false_wakes_per_hour   false wakes per hour of the recordings that hold no phrase. The synthetic ones last a
                           few minutes, so one false wake is already about 10 per hour: add a long recording of
                           normal talk with "wakes": 0 for a real number. (docs/VERIFICATION.md has the rate on
                           openWakeWord's 10.7 h validation set.)
    negative_minutes       how much audio without the phrase the rate is based on

The synthetic clips use voices that never made a training clip. Your own recordings are the real test: record new
ones (not the clips you trained with) and add each with `python cli.py evals add wake clip.wav --wakes 1`.
"""

import json
from pathlib import Path

from agent import config
from agent.senses import audio
from agent.voice import wake
from evals.common import NoCases

LEVEL, USES_LLM = "component", False
NEEDS_LLM_SERVER = False
_detector: list = []  # one detector for the whole run (loading the models once)


def load_cases(cases_dir: Path):
    """Load each recording with its label.

    Args:
        cases_dir (Path): The case folder of this eval.

    Returns:
        list[dict]: Items with `wav`, `wakes` and `note`.

    Raises:
        NoCases: If the folder has no labeled recording, or the wake-word model is not trained.
    """
    if not config.WAKE_MODEL.exists():
        raise NoCases(f"{config.WAKE_MODEL.name} is not trained yet (docs/WAKE_WORD.md)")
    items = []
    for label in sorted(cases_dir.glob("*.json")) if cases_dir.is_dir() else []:
        wav = label.with_suffix(".wav")
        if wav.exists():
            items.append({"id": label.stem, "wav": str(wav), **json.loads(label.read_text())})
    if not items:
        raise NoCases(cases_dir)
    return items


def run_item(item):
    """Stream one recording through the detector.

    Args:
        item (dict): An item from `load_cases`.

    Returns:
        dict: `expected`, `woke` (times in seconds) and `audio_s`.
    """
    if not _detector:
        _detector.append(wake.Detector())
    samples = audio.load_wav(item["wav"])
    return {
        "id": item["id"],
        "note": item.get("note", ""),
        "expected": item["wakes"],
        "woke": wake.wakes(_detector[0], samples),
        "audio_s": round(len(samples) / config.SAMPLE_RATE, 2),
    }


def _miss_rate(records: list[dict]) -> float | None:
    """Share of the expected wakes in these records that did not happen, or None if none was expected."""
    expected = sum(r["expected"] for r in records)
    found = sum(min(len(r["woke"]), r["expected"]) for r in records)
    return round(1 - found / expected, 3) if expected else None


def score(records):
    """Compute missed and false wakes.

    Args:
        records (list[dict]): The records from `run_item`.

    Returns:
        dict: Metric name to value.
    """
    false = sum(max(len(r["woke"]) - r["expected"], 0) for r in records)
    quiet_h = sum(r["audio_s"] for r in records if r["expected"] == 0) / 3600
    echo = [r for r in records if r["id"].startswith("pos-echo")]
    return {
        "miss_rate": _miss_rate([r for r in records if r not in echo]),
        "echo_miss_rate": _miss_rate(echo),
        "false_wakes": false,
        "false_wakes_per_hour": round(sum(len(r["woke"]) for r in records if r["expected"] == 0) / quiet_h, 2)
        if quiet_h
        else None,
        "negative_minutes": round(quiet_h * 60, 1),
    }
