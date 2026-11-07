"""The tasks, built only from the components above.

Task 1  voice note -> chunks -> action items (JSON) -> memory
Task 2  hotkey -> screenshot (or a short screen video) -> explanation + fix (JSON) -> memory
Task 3  always-on listening -> transcripts -> summaries -> answer "what did I decide about X this morning?"
Task 4  today's memories -> daily brief with the evolving prompt -> your feedback -> evolve.reflect()
"""

import re
import time

from agent import brain, config, events, evolve
from agent.memory import episodic
from agent.senses import audio, screen, video

ACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "transcript": {"type": "string"},
        "action_items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"task": {"type": "string"}, "due": {"type": ["string", "null"]}},
                "required": ["task", "due"],
            },
        },
    },
    "required": ["transcript", "action_items"],
}

ACTION_PROMPT = """Above is a voice note (split into clips at pauses, in order).
Return the transcript and every action item: something the speaker must do.
Write each task as a short imperative phrase. Put any deadline in "due", else null.
Do not invent tasks that were not said."""

SCREEN_SCHEMA = {
    "type": "object",
    "properties": {
        "what_i_see": {"type": "string"},
        "problem": {"type": ["string", "null"]},
        "suggested_fix": {"type": ["string", "null"]},
    },
    "required": ["what_i_see", "problem", "suggested_fix"],
}

SCREEN_PROMPT = """This is a screenshot of my screen. Describe briefly what you see.
If there is an error or problem, name it exactly and suggest a concrete fix.
If there is no problem, set problem and suggested_fix to null."""

VIDEO_PROMPT = """This is a recording of my screen, with timestamps. Describe briefly what happened.
If an error or problem appeared, name it exactly, say when, and suggest a concrete fix.
If there is no problem, set problem and suggested_fix to null."""


def voice_note(samples, store: bool = True) -> dict:
    """Task 1: turn a voice recording into a transcript and action items.

    Args:
        samples (np.ndarray): 16 kHz mono float32 audio.
        store (bool, optional): Save the transcript and the action items in memory. Defaults to True.

    Returns:
        dict: `transcript`, `action_items`, `chunks` as (start_s, end_s) pairs, and the info of `ask_json`.
    """
    vad = audio.VAD()
    chunks = audio.chunk(samples, vad)
    if not chunks:
        events.emit("task1.empty", message="no speech detected")
        return {"transcript": "", "action_items": [], "chunks": []}
    parts = [brain.part("audio", c["samples"]) for c in chunks] + [brain.part("text", ACTION_PROMPT)]
    data, info = brain.ask_json(parts, ACTION_SCHEMA)
    data = data or {"transcript": "", "action_items": []}
    if store and data["transcript"]:
        note_id = episodic.store(data["transcript"], kind="voice_note", media=("audio", samples))
        for item in data["action_items"]:
            due = f" (due {item['due']})" if item.get("due") else ""
            episodic.store(item["task"] + due, kind="action_item", meta={"note_id": note_id, "due": item["due"]})
    result = {**data, "chunks": [(c["start_s"], c["end_s"]) for c in chunks], **info}
    events.emit("task1.done", **{k: v for k, v in result.items() if k != "tokens"})
    return result


def screen_help(img=None, store: bool = True, video_path=None) -> dict:
    """Explain the screen. Pass video_path to send a screen recording instead of one screenshot."""
    if video_path:
        path, parts = video_path, [brain.part("video", video_path), brain.part("text", VIDEO_PROMPT)]
    else:
        img = img or screen.capture()
        path, parts = screen.save(img), [brain.part("image", img), brain.part("text", SCREEN_PROMPT)]
    data, info = brain.ask_json(parts, SCREEN_SCHEMA)
    data = data or {"what_i_see": "", "problem": None, "suggested_fix": None}
    for k in ("problem", "suggested_fix"):  # small models often write "None" instead of null
        if str(data[k]).strip().lower() in ("", "none", "null", "n/a", "no problem"):
            data[k] = None
    if store and data["what_i_see"]:
        summary = data["what_i_see"]
        if data["problem"]:
            summary += f" Problem: {data['problem']}. Fix: {data['suggested_fix']}"
        episodic.store(summary, kind="screen_video" if video_path else "screen_help", media=str(path))
    media_key = "video" if video_path else "screenshot"
    result = {**data, media_key: str(path), **info}
    events.emit("task2.done", **{k: v for k, v in result.items() if k != "tokens"})
    return result


def watch_and_help(seconds: float = 10, store: bool = True) -> dict:
    """Task 2 over time: record the screen for a few seconds, then explain what happened."""
    return screen_help(video_path=video.record(seconds), store=store)


# --- Task 3: ambient listener with recall ----------------------------------------------

RECALL_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}, "memory_ids": {"type": "array", "items": {"type": "integer"}}},
    "required": ["answer", "memory_ids"],
}
RECALL_PROMPT = """Above are memories. Each starts with its [id], the time and its kind.
Answer the question from these memories only, in one or two sentences, and put the ids you used in memory_ids.
If they do not answer it, say that you do not know and leave memory_ids empty.
Question: {question}"""
# Phrase in a question -> (start hour, end hour, days back). None for the end hour means "now".
TIME_WORDS = {
    "this morning": (5, 12, 0),
    "this afternoon": (12, 18, 0),
    "this evening": (18, 24, 0),
    "tonight": (18, 24, 0),
    "earlier today": (0, None, 0),
    "today": (0, None, 0),
    "yesterday": (0, 24, 1),
}


def time_window(question: str, now: float | None = None) -> tuple[float | None, float | None, str]:
    """Turn a time phrase in a question into a search window.

    Args:
        question (str): The question, for example "what did I decide this morning?".
        now (float, optional): The current time. Defaults to `time.time()`.

    Returns:
        tuple[float | None, float | None, str]: (since, until, phrase), all None/"" when there is no phrase.
    """
    now = now or time.time()
    q = question.lower()
    for phrase, (start_h, end_h, back) in TIME_WORDS.items():
        if re.search(rf"\b{phrase}\b", q):
            midnight = time.mktime(time.localtime(now)[:3] + (0, 0, 0, 0, 0, -1)) - back * 86400
            until = now if end_h is None else min(now, midnight + end_h * 3600)
            return midnight + start_h * 3600, until, phrase
    return None, None, ""


def recall(question: str, now: float | None = None) -> dict:
    """Task 3: answer a question about earlier events from memory, and cite the memories used.

    Args:
        question (str): For example "what did I decide about the venue this morning?".
        now (float, optional): The current time, for the time phrase. Defaults to `time.time()`.

    Returns:
        dict: `answer`, `memory_ids`, `window` (the time phrase found) and `memories` (what the model read).
    """
    since, until, phrase = time_window(question, now)
    hits = episodic.search(question, k=config.RECALL_TOP_K, since=since, until=until)
    if not hits and phrase:  # the phrase may be wrong ("this morning" said at night): search all
        hits = episodic.search(question, k=config.RECALL_TOP_K)
    lines = [f"[{h['id']}] ({h['when']}, {h['level']} {h['kind']}) {h['text']}" for h in hits]
    text = "Memories:\n" + ("\n".join(lines) or "(none)") + "\n\n" + RECALL_PROMPT.format(question=question)
    data, info = brain.ask_json([brain.part("text", text)], RECALL_SCHEMA)
    data = data or {"answer": "", "memory_ids": []}
    known = {h["id"] for h in hits}
    data["memory_ids"] = [i for i in data["memory_ids"] if i in known]  # a cited id must be one it was shown
    result = {**data, "window": phrase, "memories": [{"id": h["id"], "text": h["text"]} for h in hits]}
    events.emit("task3.answer", question=question, **result)
    return result


# --- Task 4: self-improving daily brief ------------------------------------------------


def day_memories(day_start: float | None = None) -> list[str]:
    """Collect one day's summaries, facts, action items and screen episodes as lines for the brief.

    Raw transcripts that a summary covers are left out: the summary says the same in fewer words.

    Args:
        day_start (float, optional): Local midnight of the day. Defaults to today.

    Returns:
        list[str]: One line per memory, oldest first.
    """
    start = day_start or time.mktime(time.localtime()[:3] + (0, 0, 0, 0, 0, -1))
    rows = episodic.query_rows(
        "ts >= ? AND ts < ? AND superseded_by IS NULL AND (level != 'raw' OR covered_by IS NULL OR kind = 'action_item')",
        (start, start + 86400),
    )
    return [f"({time.strftime('%H:%M', time.localtime(r['ts']))}, {r['kind']}) {r['text']}" for r in rows]


def daily_brief(day_start: float | None = None) -> dict:
    """Task 4: write today's brief with the active prompt version.

    Args:
        day_start (float, optional): Local midnight of the day. Defaults to today.

    Returns:
        dict: The result of `evolve.write_brief`, with `memories`.
    """
    mems = day_memories(day_start)
    day = time.strftime("%Y-%m-%d", time.localtime(day_start or time.time()))
    return {**evolve.write_brief(mems, day=day), "memories": mems}
