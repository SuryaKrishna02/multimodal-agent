"""Task 1 end-to-end eval: voice note -> action items, against the expected list.

Cases: data/evals/{synthetic,user}/task1/
    note01.wav    a voice note as you would really say it (pauses, "um"s, asides are fine)
    note01.json   {"action_items": ["Email John Doe the slides", "Book a room for Thursday"]}

Matching: each predicted item is paired with at most one expected item, greedily by word
overlap (F1 of content words). A pair counts if the overlap F1 >= MATCH_F1.

Metrics: precision (predicted items that match expected), recall (expected items that were found).
"""

import json
import re
from pathlib import Path

from agent import tasks
from agent.senses import audio
from evals.common import NoCases

LEVEL, USES_LLM = "end-to-end", True
MATCH_F1 = 0.5
STOP = {"the", "a", "an", "to", "for", "and", "of", "on", "by", "my", "me", "i", "at", "in", "with"}


def words(text: str) -> set[str]:
    """Split text into content words.

    Args:
        text (str): Any text.

    Returns:
        set[str]: Lowercase words without stop words.
    """
    return {w for w in re.findall(r"[a-z0-9']+", text.lower()) if w not in STOP}


def overlap_f1(a: str, b: str) -> float:
    """Measure how much two phrases share.

    Args:
        a (str): First phrase.
        b (str): Second phrase.

    Returns:
        float: F1 of the content words, 0.0 when they share none.
    """
    x, y = words(a), words(b)
    common = len(x & y)
    return 0.0 if not common else 2 * common / (len(x) + len(y))


def match(predicted: list[str], expected: list[str]) -> list[tuple[str, str, float]]:
    """Pair predicted items with expected items, best overlap first.

    Args:
        predicted (list[str]): Action items from the model.
        expected (list[str]): Action items that the case lists.

    Returns:
        list[tuple[str, str, float]]: (predicted, expected, F1) pairs. An item is used once and F1 must reach MATCH_F1.
    """
    pairs = sorted(((overlap_f1(p, g), p, g) for p in predicted for g in expected), reverse=True)
    used_p, used_g, out = set(), set(), []
    for f1, p, g in pairs:
        if f1 >= MATCH_F1 and p not in used_p and g not in used_g:
            used_p.add(p)
            used_g.add(g)
            out.append((p, g, round(f1, 2)))
    return out


def load_cases(cases_dir: Path):
    """Load the cases: each WAV note with the action items of the .json file beside it.

    Args:
        cases_dir (Path): The case folder of this eval.

    Returns:
        list[dict]: One item per case.

    Raises:
        NoCases: If there is no note with labels.
    """
    items = [
        {"wav": w, "expected": json.loads(w.with_suffix(".json").read_text())["action_items"]}
        for w in sorted(cases_dir.glob("*.wav"))
        if w.with_suffix(".json").exists()
    ]
    if not items:
        raise NoCases(cases_dir)
    return items


def run_item(item):
    """Run one case: run Task 1 on one note without storing memories.

    Args:
        item (dict): An item from `load_cases`.

    Returns:
        dict: A record with `predicted`, `expected`, `matched` and the transcript.
    """
    result = tasks.voice_note(audio.load_wav(item["wav"]), store=False)
    predicted = [a["task"] for a in result["action_items"]]
    matched = match(predicted, item["expected"])
    return {
        "file": item["wav"].name,
        "audio": str(item["wav"]),
        "transcript": result["transcript"],
        "predicted": predicted,
        "expected": item["expected"],
        "matched": matched,
        "chunks": result["chunks"],
    }


def score(records):
    """Compute the metrics: precision and recall of the action items.

    Args:
        records (list[dict]): The records from `run_item`.

    Returns:
        dict: Metric name to value.
    """
    m = sum(len(r["matched"]) for r in records)
    return {
        "precision": round(m / max(sum(len(r["predicted"]) for r in records), 1), 3),
        "recall": round(m / max(sum(len(r["expected"]) for r in records), 1), 3),
    }
