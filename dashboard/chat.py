"""The dashboard chat: type, paste a picture, share the screen or speak, and see what the agent used to reply.

    POST /api/chat               text, with an optional picture (a data URL) or a screenshot of the live screen
    POST /api/chat/voice/start   open the microphone for a spoken message
    POST /api/chat/voice/stop    close it, keep only the speech (VAD) and turn it into text (config.STT_ENGINE)
    POST /api/chat/brief         put a daily brief into the conversation, so you can rate it and ask about it

Each message has `role` and `content`, which the model reads back as history, and `ui`, which only the page
reads: pictures, the voice transcript, recalled memories, tool calls and timings. The conversation is saved
in config.CHAT_FILE after every change, so it survives a restart of the dashboard. Long-term memory is
separate: Clear chat empties the conversation and deletes its pictures, and keeps every memory.
"""

import base64
import binascii
import io
import json
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
state: dict = {"history": None, "mic": None}  # history is read from CHAT_FILE on first use


# --- The conversation --------------------------------------------------------------------


def history() -> list[dict]:
    """Return the conversation, oldest first, reading it from disk the first time.

    Returns:
        list[dict]: Messages with `role`, `content` and optional `ui` or `approval`.
    """
    if state["history"] is None:
        try:
            data = json.loads(config.CHAT_FILE.read_text())
        except (OSError, ValueError):  # no file yet, or a file cut short by a crash: start empty
            data = []
        state["history"] = data if isinstance(data, list) else []
    return state["history"]


def append(*messages: dict) -> None:
    """Add messages to the conversation and save it. Hold `lock` when the order of messages matters.

    Args:
        *messages (dict): Messages with `role` and `content`.
    """
    h = history()
    h.extend(messages)
    del h[: -config.CHAT_KEEP]
    config.CHAT_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = config.CHAT_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(h, default=str))
    tmp.replace(config.CHAT_FILE)  # atomic: a crash leaves the old file or the new one, never half of one


def note_approval(call_id: int, outcome: dict) -> None:
    """Record how a call that waited for approval ended: a note for the model, and the new status on its chip.

    Args:
        call_id (int): The call.
        outcome (dict): The result of `tools.approve`.
    """
    with lock:
        for m in history():
            for c in m.get("ui", {}).get("tool_calls", []):
                if c.get("call_id") == call_id:
                    c["status"] = outcome["status"]
        append({**brain.approval_note(outcome), "approval": outcome})


@router.get("/chat")
def chat_history():
    """List the conversation, so a reload, or a restart, shows the same one the model still has.

    Returns:
        list[dict]: Messages, oldest first.
    """
    return history()


@router.post("/chat/clear")
def chat_clear():
    """Start a new conversation and delete the pictures it holds. Long-term memories stay.

    Returns:
        list: An empty list.
    """
    with lock:
        for m in history():
            for p in m.get("ui", {}).get("images", []):
                if Path(p).resolve().is_relative_to(screen.CAPTURES.resolve()):
                    Path(p).unlink(missing_ok=True)
        state["history"] = []
        append()
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
