"""VAD eval: does voice activity detection fire on noise, or miss real speech?

Gold set: data/gold/vad/
    clip01.wav          any length; include silence, noise, music, coughs, speech
    clip01.txt          Audacity label export: one line per speech region "start<TAB>end<TAB>speech"

Metrics (frame level, 32 ms frames, ignoring a 0.25 s collar around each labeled edge):
    false_trigger_rate  share of non-speech frames the VAD marked as speech
    missed_speech_rate  share of speech frames the VAD did not mark
    spurious_segments   detected segments that overlap no labeled speech at all
"""
from pathlib import Path

import numpy as np

from agent import config
from agent.senses import audio
from evals.common import NoGold

LEVEL, USES_LLM = "component", False
FRAME_S, COLLAR_S = 0.032, 0.25


def read_labels(path: Path) -> list[tuple[float, float]]:
    rows = [line.split("\t") for line in path.read_text().splitlines() if line.strip()]
    return [(float(r[0]), float(r[1])) for r in rows]


def load_gold(gold_dir: Path):
    items = [{"wav": w, "speech": read_labels(w.with_suffix(".txt"))}
             for w in sorted(gold_dir.glob("*.wav")) if w.with_suffix(".txt").exists()]
    if not items:
        raise NoGold(gold_dir)
    return items


def _mask(segments, n_frames):
    m = np.zeros(n_frames, bool)
    for a, b in segments:
        m[int(a / FRAME_S):int(b / FRAME_S) + 1] = True
    return m


def run_item(item):
    samples = audio.load_wav(item["wav"])
    detected = audio.VAD().segments(samples)
    n = int(len(samples) / config.SAMPLE_RATE / FRAME_S) + 1
    gold, pred = _mask(item["speech"], n), _mask(detected, n)
    edges = _mask([(t - COLLAR_S, t + COLLAR_S) for seg in item["speech"] for t in seg], n)
    scored = ~edges
    spurious = sum(1 for a, b in detected if not any(a < gb and b > ga for ga, gb in item["speech"]))
    return {"file": item["wav"].name, "gold": item["speech"], "detected": [(round(a, 2), round(b, 2)) for a, b in detected],
            "fp": int((pred & ~gold & scored).sum()), "neg": int((~gold & scored).sum()),
            "fn": int((~pred & gold & scored).sum()), "pos": int((gold & scored).sum()), "spurious": spurious}


def score(records):
    neg, pos = sum(r["neg"] for r in records), sum(r["pos"] for r in records)
    return {"false_trigger_rate": round(sum(r["fp"] for r in records) / max(neg, 1), 3),
            "missed_speech_rate": round(sum(r["fn"] for r in records) / max(pos, 1), 3),
            "spurious_segments": sum(r["spurious"] for r in records)}
