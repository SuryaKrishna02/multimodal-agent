"""Text-to-speech eval: how soon the first sound is ready, and whether the speech is understood.

Cases: data/evals/{synthetic,user}/tts/sentences.jsonl, one JSON object per line:
    {"id": "s01", "text": "I set a reminder to call Mike at 3:15."}
Write numbers as digits: Whisper writes them that way, so "3:15" in the reference matches "3:15" in the transcript.

Each sentence goes through the real voice path: `tts.sentences()` cuts it, Kokoro writes the first piece (timed:
that is the time to first audio before playback), then the whole text, and Whisper transcribes it back.

Metrics:
    first_audio_p50_s / first_audio_p95_s   seconds to write the first spoken piece (median, 95th percentile)
    wer                                     word error rate of Kokoro -> Whisper, normalized as in the asr eval
    rtf                                     seconds of synthesis per second of speech
The tab adds about 0.1 s of transport to the first audio; the Voice tab of the dashboard shows the full time.
"""

import json
import statistics
import time
from pathlib import Path

import jiwer
import numpy as np
from scipy.signal import resample_poly

from agent import config
from agent.senses import stt
from agent.voice import tts
from evals.asr.eval import normalize
from evals.common import NoCases

LEVEL, USES_LLM = "component", False
NEEDS_LLM_SERVER = False


def load_cases(cases_dir: Path):
    """Load the sentences.

    Args:
        cases_dir (Path): The case folder of this eval.

    Returns:
        list[dict]: One item per sentence.

    Raises:
        NoCases: If sentences.jsonl is missing.
    """
    f = cases_dir / "sentences.jsonl"
    if not f.exists():
        raise NoCases(cases_dir)
    tts.synthesize("Ready.")  # load the model once, untimed
    return [json.loads(line) for line in f.read_text().splitlines() if line.strip()]


def run_item(item):
    """Speak one sentence, time the first piece, and transcribe the whole of it.

    Args:
        item (dict): `id` and `text`.

    Returns:
        dict: `first_audio_s`, `synth_s`, `audio_s`, `heard` and `errors` / `words` for the word error rate.
    """
    first = tts.sentences(item["text"])[0]
    t0 = time.perf_counter()
    tts.synthesize(first)
    first_s = time.perf_counter() - t0
    t1 = time.perf_counter()
    audio = tts.synthesize(item["text"])
    synth_s = time.perf_counter() - t1
    heard = stt.transcribe(resample_poly(audio, config.SAMPLE_RATE, config.TTS_SAMPLE_RATE).astype(np.float32))
    ref, hyp = normalize(item["text"]), normalize(heard)
    out = jiwer.process_words(ref, hyp)
    return {
        "id": item["id"],
        "text": item["text"],
        "heard": heard,
        "first_audio_s": round(first_s, 3),
        "synth_s": round(synth_s, 3),
        "audio_s": round(len(audio) / config.TTS_SAMPLE_RATE, 2),
        "errors": out.substitutions + out.deletions + out.insertions,
        "words": len(ref.split()),
    }


def score(records):
    """Compute the time to first audio, the word error rate and the real-time factor.

    Args:
        records (list[dict]): The records from `run_item`.

    Returns:
        dict: Metric name to value.
    """
    firsts = sorted(r["first_audio_s"] for r in records)
    p95 = firsts[min(len(firsts) - 1, int(round(0.95 * (len(firsts) - 1))))]
    return {
        "first_audio_p50_s": round(statistics.median(firsts), 3),
        "first_audio_p95_s": round(p95, 3),
        "wer": round(sum(r["errors"] for r in records) / max(sum(r["words"] for r in records), 1), 3),
        "rtf": round(sum(r["synth_s"] for r in records) / max(sum(r["audio_s"] for r in records), 1e-9), 3),
    }
