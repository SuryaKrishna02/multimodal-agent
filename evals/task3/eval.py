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

Where a wrong answer went wrong. An end-to-end score alone does not say which stage to fix, so each
wrong answer gets the first stage that lost the information (`failed_at`), checked by code:
    hearing    the transcripts do not contain the answer (an ASR or segmentation error)
    retrieval  the transcripts have it, but the memories the model read do not (search or consolidation)
    answer     the model read it and still answered wrong, or invented an answer to an unanswerable question
lost_hearing, lost_retrieval and lost_answer count the wrong answers per stage.
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
# Ways a small model says "I do not know". E2B wrote "The memories do not contain any information about ...".
# An abstention must also cite no memory, so a hedge next to an invented answer still fails.
DONT_KNOW = re.compile(
    r"\b(do not know|don't know|not sure|cannot find|can't find|no record|not mentioned|never mentioned"
    r"|(no|any) (information|mention|record)|(do|does|did)(n't| not) (contain|include|mention|say))\b",
    re.I,
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
    _heard.clear()  # a new run (or a rebuilt recording) replays from an empty database
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


def mentions(text: str, groups: list[list[str]]) -> bool:
    """Check that a text has at least one phrase of every group, in any case.

    Args:
        text (str): The text.
        groups (list[list[str]]): The must_mention groups.

    Returns:
        bool: True if every group has a phrase in the text.
    """
    low = text.lower()
    return all(any(p.lower() in low for p in group) for group in groups)


def failed_at(item: dict, ok: bool, heard: list[str], read: list[str]) -> str | None:
    """Name the first stage that lost the answer, or None for a correct answer.

    Args:
        item (dict): The question with `must_mention` and `unanswerable`.
        ok (bool): Whether the answer is correct.
        heard (list[str]): The transcripts of the recording.
        read (list[str]): The texts of the memories the model read.

    Returns:
        str | None: "hearing", "retrieval", "answer", or None.
    """
    if ok:
        return None
    if item["unanswerable"]:
        return "answer"
    if not mentions(" ".join(heard), item["must_mention"]):
        return "hearing"
    return "retrieval" if not mentions(" ".join(read), item["must_mention"]) else "answer"


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
    return mentions(answer, item["must_mention"])


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
    ok = correct(item, r["answer"], r["memory_ids"])
    heard = _heard[item["wav"]]
    return {
        "question": item["question"],
        "audio": item["wav"],
        "answer": r["answer"],
        "memory_ids": r["memory_ids"],
        "window": r["window"],
        "unanswerable": item["unanswerable"],
        "correct": ok,
        "failed_at": failed_at(item, ok, heard, [m["text"] for m in r["memories"]]),
        "memories_read": r["memories"],
        "transcripts": heard,
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
    for stage in ("hearing", "retrieval", "answer"):
        out[f"lost_{stage}"] = sum(r.get("failed_at") == stage for r in records)
    return out
