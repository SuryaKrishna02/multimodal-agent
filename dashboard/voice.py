"""The voice endpoints: the tab plays the agent's speech, because Google Meet shares only the presented tab's audio.

    GET  /api/voice/stream          Server-Sent Events: each voice.* event, as soon as it is in the event log
    GET  /api/voice/audio/{utt}/{n} one WAV chunk of an utterance
    POST /api/voice/played          the tab started the first chunk (time to first audio)
    POST /api/voice/stopped         the tab went silent after a stop request (stop latency)
    POST /api/voice/stop            the Stop button
    POST /api/voice/say             speak a text (the Voice tab's test box)
    POST /api/voice/wake            listen for "Hey Sanjaya" (on or off)
    POST /api/voice/talk            start a spoken turn now (push to talk; works without a wake model)
    GET  /api/voice/state           is a tab listening, the voices, the latency budget, the wake word

The stream reads the event log every `config.VOICE_POLL_S`, so speech that the daemon writes reaches the tab
quickly. The page's 1 s poll of /api/events would be too slow for time to first audio. While the stream is open, it
writes the tab heartbeat that `speaker.tab_listening()` reads.
"""

import json
import threading
import time

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, StreamingResponse

from agent import config, events
from agent.voice import speaker, tts, turn
from dashboard import chat

router = APIRouter(prefix="/api/voice")
loop: dict = {"wake": None}  # the running turn.WakeLoop, if any
KEEPALIVE_S = 10.0  # an SSE comment this often, so a quiet stream is not closed by the browser


def _stream():
    """Yield each new voice.* event as one SSE message, and keep the tab heartbeat fresh."""
    newest = events.last("")
    after = newest["id"] if newest else 0  # only what happens from now on: old speech does not replay
    quiet_since = beat = 0.0
    while True:
        now = time.monotonic()
        if now - beat >= 1.0:
            speaker.mark_tab()
            beat = now
        batch = events.since(after, "voice.")
        for e in batch:
            after = e["id"]
            yield f"data: {json.dumps(e, default=str)}\n\n"
        if batch:
            quiet_since = now
        elif now - quiet_since >= KEEPALIVE_S:
            quiet_since = now
            yield ": keep-alive\n\n"
        time.sleep(config.VOICE_POLL_S)


def warm_up() -> None:
    """Load Kokoro and speak one word into nothing, so the first real reply does not wait for the model to load."""
    try:
        tts.synthesize("Ready.")
    except Exception as e:  # noqa: BLE001 - a missing model must not stop the dashboard; the Voice tab shows why
        events.emit("voice.error", error=f"Kokoro did not load: {e}")


@router.get("/stream")
def stream():
    """Open the event stream of the speech that the tab must play.

    Returns:
        StreamingResponse: text/event-stream.
    """
    return StreamingResponse(_stream(), media_type="text/event-stream", headers={"X-Accel-Buffering": "no"})


@router.get("/audio/{utt}/{n}")
def audio(utt: str, n: int):
    """Serve one WAV chunk.

    Args:
        utt (str): The utterance id. Only the format that `speaker.speak` makes is accepted.
        n (int): The chunk number.

    Returns:
        FileResponse: The WAV.

    Raises:
        HTTPException: 404 for an unknown or malformed id.
    """
    path = config.VOICE_DIR / utt / f"{n}.wav"
    if not speaker.UTTERANCE_ID.match(utt) or n < 0 or not path.is_file():
        raise HTTPException(404, "no such audio chunk")
    return FileResponse(path, media_type="audio/wav")


@router.post("/played")
def played(body: dict) -> dict:
    """Record that the tab started to play the first chunk of an utterance.

    Args:
        body (dict): `utt` and `say_ts`, the time of its voice.say event.

    Returns:
        dict: The voice.played event.
    """
    return events.emit(
        "voice.played", utt=str(body["utt"]), n=0, route="tab", since_say_s=time.time() - float(body["say_ts"])
    )


@router.post("/stopped")
def stopped(body: dict) -> dict:
    """Record that the tab went silent after a stop request.

    Args:
        body (dict): `utt` and `stop_ts`, the time of the voice.stop event.

    Returns:
        dict: The voice.stopped event.
    """
    return events.emit(
        "voice.stopped", utt=str(body["utt"]), route="tab", after_stop_s=time.time() - float(body["stop_ts"])
    )


@router.post("/stop")
def stop(body: dict | None = None) -> dict:
    """Stop all speech now.

    Args:
        body (dict | None, optional): `reason`, defaults to "button".

    Returns:
        dict: The voice.stop event.
    """
    return speaker.stop(str((body or {}).get("reason", "button")))


@router.post("/say")
def say(body: dict) -> dict:
    """Speak a text in the background, through the tab if one listens.

    Args:
        body (dict): `text`, and optionally `voice`.

    Returns:
        dict: `ok` and the route that it will take.

    Raises:
        HTTPException: 400 without text.
    """
    text = str(body.get("text", "")).strip()
    if not text:
        raise HTTPException(400, "nothing to say")
    voice = body.get("voice") or None
    threading.Thread(target=speaker.speak, args=(text, "test from the Voice tab", voice), daemon=True).start()
    return {"ok": True, "route": "tab" if speaker.tab_listening() else "local"}


def _ask(text: str, details: dict) -> str:
    """Answer a spoken request in the dashboard chat, so it shows there with its tools and approvals."""
    return chat.chat({"text": text, "voice": details, "spoken": True})["text"]


def _loop(use_wake: bool) -> turn.WakeLoop:
    """Return the running loop, or start one. A loop without the wake word answers only the Talk button."""
    current = loop["wake"]
    if current is not None and (current.detector is not None or not use_wake):
        return current
    if current is not None:
        current.stop()
    loop["wake"] = turn.WakeLoop(_ask, use_wake=use_wake)
    loop["wake"].start()
    return loop["wake"]


@router.post("/wake")
def wake(body: dict) -> dict:
    """Turn listening for the wake word on or off.

    Args:
        body (dict): `on` (bool).

    Returns:
        dict: The voice state.

    Raises:
        HTTPException: 409 when the wake-word model is not trained yet.
    """
    if body.get("on"):
        if not config.WAKE_MODEL.exists():
            raise HTTPException(409, f"{config.WAKE_MODEL.name} is not trained yet (docs/WAKE_WORD.md). Use Talk.")
        _loop(use_wake=True)
    elif loop["wake"] is not None:
        loop["wake"].stop()
        loop["wake"] = None
    return state()


@router.post("/talk")
def talk() -> dict:
    """Start a spoken turn now: the next thing you say is the request.

    Returns:
        dict: `ok`.

    Raises:
        HTTPException: 409 while the microphone pause switch is on.
    """
    from agent.senses import audio

    if audio.paused():
        raise HTTPException(409, "The microphone is paused. Turn on Mic at the top first.")
    _loop(use_wake=loop["wake"] is not None and loop["wake"].detector is not None).talk()
    return {"ok": True}


@router.get("/state")
def state() -> dict:
    """Describe the voice: is a tab listening, which voices exist, and the time budget.

    Returns:
        dict: `tab`, `voice`, `voices`, `budget`, `model`, `fade_ms`, `wake` (listening for the wake word),
            `wake_model` (the model exists) and `error` (None when Kokoro loads).
    """
    try:
        names = tts.voices()
    except FileNotFoundError as e:
        names, error = [], str(e)
    else:
        error = None
    return {
        "tab": speaker.tab_listening(),
        "voice": config.TTS_VOICE,
        "voices": names,
        "budget": config.LATENCY_BUDGET,
        "model": config.TTS_MODEL.name,
        "fade_ms": config.VOICE_FADE_MS,
        "wake": loop["wake"] is not None and loop["wake"].detector is not None,
        "wake_model": config.WAKE_MODEL.exists(),
        "error": error,
    }
