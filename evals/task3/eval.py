"""Task 3 end-to-end eval: hear a stretch of conversation, then answer questions about it hours later.

Cases: data/evals/{synthetic,user}/task3/
    conversation.wav    a recording (the synthetic one has three voices)
    conversation.json   {"start": "09:00", "ask_at": "14:00",
                         "questions": [{"question": "...", "must_mention": [["lakeside"]], "unanswerable": false}]}

For each recording, in a throwaway database: replay it through the ambient listener as if it started
at `start` today (the same Segmenter, STT engine and memory store as live listening), then consolidate.
Each question is answered by `tasks.recall()` as if asked at `ask_at`.

A question is correct when the answer has a phrase from every must_mention group (any case). An
unanswerable question is correct when the answer says it does not know and cites no memory.

Metrics: accuracy (all questions), abstain_rate (unanswerable questions answered "I do not know"),
cited_rate (answered questions that cite at least one memory).
"""

import json
import re
import time
from pathlib import Path

from agent import tasks
from agent.memory import consolidate
from agent.senses import ambient, audio
from evals import common
from evals.common import NoCases

LEVEL, USES_LLM = "end-to-end", True
NEEDS_EMBED_SERVER = True
DONT_KNOW = re.compile(
    r"\b(do not know|don't know|not sure|no information|not mentioned|no record|cannot find)\b", re.I
)
_heard: dict[str, list[str]] = {}  # recording -> transcripts, so each recording is replayed once per run


def load_cases(cases_dir: Path):
    """Load the questions of every recording.

    Args:
        cases_dir (Path): The case folder of this eval.

    Returns:
        list[dict]: One item per question, with its recording.

    Raises:
        NoCases: If no recording has a .json file beside it.
    """
    items = []
    for wav in sorted(cases_dir.glob("*.wav")):
        if not wav.with_suffix(".json").exists():
            continue
        case = json.loads(wav.with_suffix(".json").read_text())
        items += [{**q, "wav": str(wav), "start": case["start"], "ask_at": case["ask_at"]} for q in case["questions"]]
    if not items:
        raise NoCases(cases_dir)
    return items


def _today(hhmm: str) -> float:
    h, m = map(int, hhmm.split(":"))
    return time.mktime(time.localtime()[:3] + (h, m, 0, 0, 0, -1))


def hear_recording(wav: str, start: str) -> list[str]:
    """Replay a recording into the (empty) eval database and consolidate it.

    Args:
        wav (str): The recording.
        start (str): Local time it starts at, "HH:MM".

    Returns:
        list[str]: The transcripts that were heard.
    """
    samples = audio.load_wav(wav)
    heard = [ambient.hear(u, "replay:" + Path(wav).name)["text"] for u in ambient.utterances(samples, _today(start))]
    consolidate.consolidate(force=True)
    return heard


def correct(item: dict, answer: str, cited: list[int]) -> bool:
    """Check one answer.

    Args:
        item (dict): The question with `must_mention` and `unanswerable`.
        answer (str): The model's answer.
        cited (list[int]): The memory ids it cited.

    Returns:
        bool: True if correct.
    """
    if item["unanswerable"]:
        return bool(DONT_KNOW.search(answer)) and not cited
    low = answer.lower()
    return all(any(p.lower() in low for p in group) for group in item["must_mention"])


def run_item(item):
    """Answer one question, after replaying its recording once.

    Args:
        item (dict): A question from `load_cases`.

    Returns:
        dict: A record with the answer, the cited memories and whether it is correct.
    """
    with common.isolated("task3") as folder:
        if item["wav"] not in _heard:
            common.reset(folder)
            _heard.clear()
            _heard[item["wav"]] = hear_recording(item["wav"], item["start"])
        r = tasks.recall(item["question"], now=_today(item["ask_at"]))
    return {
        "question": item["question"],
        "audio": item["wav"],
        "answer": r["answer"],
        "memory_ids": r["memory_ids"],
        "window": r["window"],
        "unanswerable": item["unanswerable"],
        "correct": correct(item, r["answer"], r["memory_ids"]),
        "transcripts": _heard[item["wav"]],
    }


def score(records):
    """Compute accuracy, abstain rate and cited rate.

    Args:
        records (list[dict]): The records from `run_item`.

    Returns:
        dict: Metric name to value.
    """
    unans = [r for r in records if r["unanswerable"]]
    ans = [r for r in records if not r["unanswerable"]]
    out = {"accuracy": round(sum(r["correct"] for r in records) / max(len(records), 1), 3)}
    if unans:
        out["abstain_rate"] = round(sum(r["correct"] for r in unans) / len(unans), 3)
    if ans:
        out["cited_rate"] = round(sum(bool(r["memory_ids"]) for r in ans) / len(ans), 3)
    return out
