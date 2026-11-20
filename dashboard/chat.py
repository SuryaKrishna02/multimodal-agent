"""The dashboard chat: type, paste a picture, share the screen or speak, and see what the agent used to reply.

    POST /api/chat               text, with an optional picture (a data URL) or a screenshot of the live screen
    POST /api/chat/voice/start   open the microphone for a spoken message
    POST /api/chat/voice/stop    close it, keep only the speech (VAD) and turn it into text (config.STT_ENGINE)
    POST /api/chat/brief         put a daily brief into the conversation, so you can rate it and ask about it
    GET  /api/chat/sessions      the saved conversations, newest first
    POST /api/chat/new           start a new conversation; the old one stays in the list
    POST /api/chat/open          continue a saved conversation
    POST /api/chat/delete        delete one conversation and its pictures

Each message has `role` and `content`, which the model reads back as history, and `ui`, which only the page
reads: pictures, the voice transcript, recalled memories, tool calls and timings. A session is one conversation.
It is saved in config.CHAT_DIR after every change, so it survives a restart of the dashboard. The model reads
only the messages of the open session. Long-term memory is separate and shared by all sessions: deleting a
session deletes its messages and its pictures, and keeps every memory.
"""

import base64
import binascii
import io
import json
import re
import secrets
import threading
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException
from PIL import Image, UnidentifiedImageError

from agent import brain, config
from agent.platforms.base import Unsupported
from agent.senses import audio, screen, stt

router = APIRouter(prefix="/api")
# Held while the model replies, so an approval during a reply cannot put its outcome before that turn in the history.
lock = threading.Lock()
state: dict = {"session": None, "mic": None}  # the open session is read from CHAT_DIR on first use
SESSION_ID = re.compile(r"^\d{8}-\d{6}-[0-9a-f]{4}$")  # an id is part of a file name: accept only our own format


# --- Sessions --------------------------------------------------------------------------------


def _new_session(messages: list[dict] | None = None) -> dict:
    now = time.time()
    sid = f"{time.strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(2)}"
    return {"id": sid, "created": now, "updated": now, "messages": messages or []}


def _read(path: Path) -> dict | None:
    """Read one session file. A missing file, or one cut short by a crash, is no session."""
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) and isinstance(data.get("messages"), list) else None


def _stored() -> list[dict]:
    """Return every saved session, newest first."""
    found = [s for p in config.CHAT_DIR.glob("*.json") if (s := _read(p))]
    return sorted(found, key=lambda s: s["updated"], reverse=True)


def _legacy() -> list[dict]:
    """Return the one conversation that older versions kept in CHAT_FILE."""
    try:
        data = json.loads(config.CHAT_FILE.read_text())
    except (OSError, ValueError):
        return []
    return data if isinstance(data, list) else []


def _save(s: dict) -> None:
    """Write one session to its file. An empty session leaves no file."""
    if not s["messages"]:
        return
    config.CHAT_DIR.mkdir(parents=True, exist_ok=True)
    path = config.CHAT_DIR / f"{s['id']}.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(s, default=str))
    tmp.replace(path)  # atomic: a crash leaves the old file or the new one, never half of one


def _delete(s: dict) -> None:
    """Delete one session: its file, and the pictures that its messages hold."""
    for m in s["messages"]:
        for p in m.get("ui", {}).get("images", []):
            if Path(p).resolve().is_relative_to(screen.CAPTURES.resolve()):
                Path(p).unlink(missing_ok=True)
    (config.CHAT_DIR / f"{s['id']}.json").unlink(missing_ok=True)


def session() -> dict:
    """Return the open session, reading the newest one from disk the first time.

    Returns:
        dict: `id`, `created`, `updated` and `messages`.
    """
    if state["session"] is None:
        saved = _stored()
        state["session"] = saved[0] if saved else _new_session(_legacy())
        _save(state["session"])
    return state["session"]


def title(s: dict) -> str:
    """Name a session after the start of its first message.

    Args:
        s (dict): A session.

    Returns:
        str: The title.
    """
    for m in s["messages"]:
        if m["role"] == "user":
            return m["content"][: config.CHAT_TITLE_CHARS]
    return "Daily brief" if s["messages"] else "New chat"


def _summary(s: dict) -> dict:
    return {
        "id": s["id"],
        "title": title(s),
        "updated": s["updated"],
        "messages": sum(1 for m in s["messages"] if not m.get("approval")),
        "current": s["id"] == session()["id"],
    }


@router.get("/chat/sessions")
def chat_sessions():
    """List the saved sessions, newest first.

    Returns:
        list[dict]: `id`, `title`, `updated`, `messages` (how many) and `current` for each session.
    """
    return [_summary(s) for s in _stored()]


@router.post("/chat/new")
def chat_new():
    """Start a new, empty session. The session that was open stays in the list.

    Returns:
        dict: The `id` of the new session.
    """
    with lock:
        state["session"] = _new_session()
    return {"id": state["session"]["id"]}


def _find(body: dict) -> dict:
    """Return the saved session that the request names, or raise 404."""
    sid = str(body.get("id", ""))
    found = _read(config.CHAT_DIR / f"{sid}.json") if SESSION_ID.match(sid) else None
    if found is None:
        raise HTTPException(404, "That chat is not there any more.")
    return found


@router.post("/chat/open")
def chat_open(body: dict):
    """Continue a saved session. The model then reads the messages of that session only.

    Args:
        body (dict): JSON with the `id` of the session.

    Returns:
        list[dict]: Its messages, oldest first.

    Raises:
        HTTPException: 404 if no session has that id.
    """
    with lock:
        state["session"] = _find(body)
    return state["session"]["messages"]


@router.post("/chat/delete")
def chat_delete(body: dict):
    """Delete one session and its pictures. Long-term memories stay. Deleting the open session starts a new one.

    Args:
        body (dict): JSON with the `id` of the session.

    Returns:
        list[dict]: The sessions that are left, as `chat_sessions` lists them.

    Raises:
        HTTPException: 404 if no session has that id.
    """
    with lock:
        gone = _find(body)
        _delete(gone)
        if gone["id"] == session()["id"]:
            state["session"] = _new_session()
    return chat_sessions()


# --- The conversation --------------------------------------------------------------------


def history() -> list[dict]:
    """Return the messages of the open session, oldest first.

    Returns:
        list[dict]: Messages with `role`, `content` and optional `ui` or `approval`.
    """
    return session()["messages"]


def append(*messages: dict) -> None:
    """Add messages to the open session and save it. Hold `lock` when the order of messages matters.

    Args:
        *messages (dict): Messages with `role` and `content`.
    """
    s = session()
    s["messages"].extend(messages)
    del s["messages"][: -config.CHAT_KEEP]
    s["updated"] = time.time()
    _save(s)


def _session_of_call(call_id: int) -> dict:
    """Return the session whose reply holds a tool call: the open one, or a saved one you left since."""
    for s in [session(), *_stored()]:
        if any(c.get("call_id") == call_id for m in s["messages"] for c in m.get("ui", {}).get("tool_calls", [])):
            return s
    return session()


def note_approval(call_id: int, outcome: dict) -> None:
    """Record how a call that waited for approval ended: a note for the model, and the new status on its chip.

    The note goes to the session that asked for the call, also when another session is open now.

    Args:
        call_id (int): The call.
        outcome (dict): The result of `tools.approve`.
    """
    with lock:
        s = _session_of_call(call_id)
        for m in s["messages"]:
            for c in m.get("ui", {}).get("tool_calls", []):
                if c.get("call_id") == call_id:
                    c["status"] = outcome["status"]
        s["messages"].append({**brain.approval_note(outcome), "approval": outcome})
        s["updated"] = time.time()
        _save(s)


@router.get("/chat")
def chat_history():
    """List the open session, so a reload, or a restart, shows the same one the model still has.

    Returns:
        list[dict]: Messages, oldest first.
    """
    return history()


@router.post("/chat/clear")
def chat_clear():
    """Delete the open session and its pictures, and start a new one. Long-term memories stay.

    Returns:
        list: An empty list.
    """
    with lock:
        _delete(session())
        state["session"] = _new_session()
    return []


# --- One turn ------------------------------------------------------------------------------


def _picture(data_url: str) -> Image.Image:
    """Decode a picture that the page sent as a data URL, and shrink it like a screenshot."""
    try:
        raw = base64.b64decode(data_url.split(",", 1)[-1], validate=True)
    except (binascii.Error, ValueError) as e:
        raise HTTPException(422, "The picture did not arrive in one piece. Try again.") from e
    if len(raw) > config.CHAT_IMAGE_MAX_BYTES:
        raise HTTPException(413, f"The picture is larger than {config.CHAT_IMAGE_MAX_BYTES // 1_000_000} MB.")
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except (UnidentifiedImageError, OSError) as e:
        raise HTTPException(422, "That file is not a picture the agent can read.") from e
    img.thumbnail((config.SCREEN_MAX_SIDE, config.SCREEN_MAX_SIDE))
    return img


def _pictures(body: dict) -> list[str]:
    """Save the picture and the screenshot of one message, and return their paths."""
    images = []
    if body.get("image"):
        images.append(("chat", _picture(body["image"])))
    if body.get("with_screen"):
        images.append(("screen", screen.capture()))
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return [str(screen.save(img, f"{kind}-{stamp}-{i}.png")) for i, (kind, img) in enumerate(images)]


@router.post("/chat")
def chat(body: dict):
    """Answer one chat message, with tools, memory and any picture. Earlier messages go along as history.

    Args:
        body (dict): JSON with `text`, and optional `image` (a data URL), `with_screen` (attach a screenshot)
            and `voice` (the transcript details, when the text was spoken).

    Returns:
        dict: The result of `brain.respond`, plus `seconds` and `images` (paths of the saved pictures).

    Raises:
        HTTPException: 422 if there is neither text nor a picture.
    """
    images = _pictures(body)
    default = "What is in this picture?" if body.get("image") else "What is on my screen?" if images else ""
    text = str(body.get("text", "")).strip() or default
    if not text:
        raise HTTPException(422, "Type a message, or attach a picture.")
    parts = [brain.part("image", p) for p in images] + [brain.part("text", text)]
    t0 = time.time()
    with lock:
        past = [{"role": m["role"], "content": m["content"]} for m in history()[-config.CHAT_HISTORY_MESSAGES :]]
        out = brain.respond(parts, history=past)
        seconds = round(time.time() - t0, 1)
        ui = {
            "tool_calls": out.get("tool_calls", []),
            "memories": out.get("memories", []),
            "seconds": seconds,
            "tokens": out.get("tokens", {}).get("total"),
        }
        append(
            {"role": "user", "content": text, "ui": {"images": images, "voice": body.get("voice")}},
            {"role": "assistant", "content": out["text"], "ui": ui},
        )
    return {**out, "seconds": seconds, "images": images}


# --- Voice -------------------------------------------------------------------------------


@router.post("/chat/voice/start")
def voice_start():
    """Open the microphone for one spoken message.

    Returns:
        dict: `recording` True.

    Raises:
        HTTPException: 409 if the microphone pause switch is on.
    """
    if audio.paused():
        raise HTTPException(409, "The microphone is paused. Turn on Mic at the top first.")
    if state["mic"] is None:
        state["mic"] = audio.Mic()
        state["mic"].start()
    return {"recording": True}


def _transcribe(clips: list) -> tuple[str, str]:
    """Transcribe speech clips with the configured engine, or with Gemma when this machine has no Whisper."""
    try:
        return " ".join(stt.transcribe(c) for c in clips).strip(), config.STT_ENGINE
    except Unsupported:
        return " ".join(stt.transcribe(c, "gemma") for c in clips).strip(), "gemma"


@router.post("/chat/voice/stop")
def voice_stop():
    """Close the microphone, keep the speech and turn it into text. The page then sends that text.

    Returns:
        dict: `text`, `engine`, `audio_s` (seconds recorded) and `took_s` (seconds to transcribe).

    Raises:
        HTTPException: 409 if nothing is recording, 422 if the recording has no speech.
    """
    mic, state["mic"] = state["mic"], None
    if mic is None:
        raise HTTPException(409, "The microphone is not recording.")
    samples = mic.stop()
    clips = [c["samples"] for c in audio.chunk(samples, audio.VAD())]  # Whisper invents words in silence
    if not clips:
        raise HTTPException(422, "No speech heard. Try again, a little closer to the microphone.")
    t0 = time.time()
    text, engine = _transcribe(clips)
    return {
        "text": text,
        "engine": engine,
        "audio_s": round(len(samples) / config.SAMPLE_RATE, 1),
        "took_s": round(time.time() - t0, 2),
    }


# --- The daily brief -------------------------------------------------------------------------


@router.post("/chat/brief")
def chat_brief(body: dict):
    """Put a stored daily brief into the conversation once, so you can rate it and ask about it.

    Args:
        body (dict): JSON with `brief_id`, `markdown` and `version`, from the `evolve.brief` event.

    Returns:
        dict: The chat message of the brief.
    """
    with lock:
        for m in history():
            if m.get("ui", {}).get("brief_id") == body["brief_id"]:  # several open pages post the same brief
                return m
        msg = {
            "role": "assistant",
            "content": "Here is your end-of-day brief:\n\n" + body["markdown"],
            "ui": {"kind": "brief", "brief_id": body["brief_id"], "version": body.get("version")},
        }
        append(msg)
    return msg
