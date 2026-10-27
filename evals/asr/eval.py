"""Speech recognition eval: word error rate of Gemma 4 audio on your own clips.

Gold set: data/gold/asr/
    clip01.wav   one utterance, under 30 s
    clip01.txt   exactly what you said (the reference transcript)

Metric: WER = (substitutions + deletions + insertions) / reference words,
after lowercasing, removing punctuation and removing fillers (um, uh, ...) on both sides.
"""
import re
from pathlib import Path

import jiwer

from agent import brain
from agent.senses import audio
from evals.common import NoGold

LEVEL, USES_LLM = "component", True
FILLERS = {"um", "uh", "erm", "er", "hmm", "mm"}
PROMPT = "Transcribe this audio exactly. Output only the transcript."


def normalize(text: str) -> str:
    text = re.sub(r"[^\w\s']", " ", text.lower())
    return " ".join(w for w in text.split() if w not in FILLERS)


def load_gold(gold_dir: Path):
    items = [{"wav": w, "reference": w.with_suffix(".txt").read_text().strip()}
             for w in sorted(gold_dir.glob("*.wav")) if w.with_suffix(".txt").exists()]
    if not items:
        raise NoGold(gold_dir)
    return items


def run_item(item):
    samples = audio.load_wav(item["wav"])
    msgs = [{"role": "user", "content": [brain.part("audio", samples), brain.part("text", PROMPT)]}]
    hyp = brain.send(msgs)["choices"][0]["message"].get("content") or ""
    ref_n, hyp_n = normalize(item["reference"]), normalize(hyp)
    out = jiwer.process_words(ref_n, hyp_n)
    return {"file": item["wav"].name, "audio": str(item["wav"]), "reference": item["reference"], "hypothesis": hyp,
            "errors": out.substitutions + out.deletions + out.insertions, "ref_words": len(ref_n.split())}


def score(records):
    return {"wer": round(sum(r["errors"] for r in records) / max(sum(r["ref_words"] for r in records), 1), 3)}
