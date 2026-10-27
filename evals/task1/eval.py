"""Task 1 end-to-end eval: voice note -> action items, against your gold list.

Gold set: data/gold/task1/
    note01.wav    a voice note as you would really say it (pauses, "um"s, asides are fine)
    note01.json   {"action_items": ["Email Priya the slides", "Book a room for Thursday"]}

Matching: each predicted item is paired with at most one gold item, greedily by word
overlap (F1 of content words). A pair counts if the overlap F1 >= MATCH_F1.

Metrics: precision (predicted items that match gold), recall (gold items that were found).
"""
import json
import re
from pathlib import Path

from agent import tasks
from agent.senses import audio
from evals.common import NoGold

LEVEL, USES_LLM = "end-to-end", True
MATCH_F1 = 0.5
STOP = {"the", "a", "an", "to", "for", "and", "of", "on", "by", "my", "me", "i", "at", "in", "with"}


def words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9']+", text.lower()) if w not in STOP}


def overlap_f1(a: str, b: str) -> float:
    x, y = words(a), words(b)
    common = len(x & y)
    return 0.0 if not common else 2 * common / (len(x) + len(y))


def match(predicted: list[str], gold: list[str]) -> list[tuple[str, str, float]]:
    pairs = sorted(((overlap_f1(p, g), p, g) for p in predicted for g in gold), reverse=True)
    used_p, used_g, out = set(), set(), []
    for f1, p, g in pairs:
        if f1 >= MATCH_F1 and p not in used_p and g not in used_g:
            used_p.add(p)
            used_g.add(g)
            out.append((p, g, round(f1, 2)))
    return out


def load_gold(gold_dir: Path):
    items = [{"wav": w, "gold": json.loads(w.with_suffix(".json").read_text())["action_items"]}
             for w in sorted(gold_dir.glob("*.wav")) if w.with_suffix(".json").exists()]
    if not items:
        raise NoGold(gold_dir)
    return items


def run_item(item):
    result = tasks.voice_note(audio.load_wav(item["wav"]), store=False)
    predicted = [a["task"] for a in result["action_items"]]
    matched = match(predicted, item["gold"])
    return {"file": item["wav"].name, "audio": str(item["wav"]), "transcript": result["transcript"],
            "predicted": predicted, "gold": item["gold"], "matched": matched, "chunks": result["chunks"]}


def score(records):
    m = sum(len(r["matched"]) for r in records)
    return {"precision": round(m / max(sum(len(r["predicted"]) for r in records), 1), 3),
            "recall": round(m / max(sum(len(r["gold"]) for r in records), 1), 3)}
