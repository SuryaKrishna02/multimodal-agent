"""The two Session 1 tasks, built only from the components above.

Task 1  voice note -> chunks -> action items (JSON) -> memory
Task 2  hotkey -> screenshot -> explanation + fix (JSON) -> memory
"""
from agent import brain, events
from agent.memory import episodic
from agent.senses import audio, screen

ACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "transcript": {"type": "string"},
        "action_items": {"type": "array", "items": {
            "type": "object",
            "properties": {"task": {"type": "string"}, "due": {"type": ["string", "null"]}},
            "required": ["task", "due"]}},
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


def voice_note(samples, store: bool = True) -> dict:
    vad = audio.VAD()
    chunks = audio.chunk(samples, vad)
    if not chunks:
        events.emit("task1.empty", message="no speech detected")
        return {"transcript": "", "action_items": [], "chunks": []}
    parts = [brain.part("audio", c["samples"]) for c in chunks] + [brain.part("text", ACTION_PROMPT)]
    data, info = brain.ask_json(parts, ACTION_SCHEMA)
    data = data or {"transcript": "", "action_items": []}
    if store and data["transcript"]:
        note_id = episodic.store(data["transcript"], kind="voice_note")
        for item in data["action_items"]:
            due = f" (due {item['due']})" if item.get("due") else ""
            episodic.store(item["task"] + due, kind="action_item", meta={"note_id": note_id, "due": item["due"]})
    result = {**data, "chunks": [(c["start_s"], c["end_s"]) for c in chunks], **info}
    events.emit("task1.done", **{k: v for k, v in result.items() if k != "tokens"})
    return result


def screen_help(img=None, store: bool = True) -> dict:
    img = img or screen.capture()
    path = screen.save(img)
    data, info = brain.ask_json([brain.part("image", img), brain.part("text", SCREEN_PROMPT)], SCREEN_SCHEMA)
    data = data or {"what_i_see": "", "problem": None, "suggested_fix": None}
    for k in ("problem", "suggested_fix"):     # small models often write "None" instead of null
        if str(data[k]).strip().lower() in ("", "none", "null", "n/a", "no problem"):
            data[k] = None
    if store and data["what_i_see"]:
        summary = data["what_i_see"]
        if data["problem"]:
            summary += f" Problem: {data['problem']}. Fix: {data['suggested_fix']}"
        episodic.store(summary, kind="screen_help", media=str(path))
    result = {**data, "screenshot": str(path), **info}
    events.emit("task2.done", **{k: v for k, v in result.items() if k != "tokens"})
    return result
