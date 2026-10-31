"""Local dashboard: one FastAPI app + one HTML page. Bound to 127.0.0.1 only.

    python cli.py dashboard      # then open http://127.0.0.1:8765 in Chrome

The server never listens on another address.
"""

import io
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from agent import brain, config, events, hotkey, serving, tasks, tools
from agent.memory import episodic
from agent.senses import audio, screen, video
from evals import common as evals_common

state = {"tier": config.TIER, "history": [], "mic": None}
llm, emb = serving.llm_server(), serving.embed_server()


@asynccontextmanager
async def lifespan(_app):
    """Start the model servers in the background and the global hotkey when the app starts.

    Args:
        _app (FastAPI): The application, not used.
    """

    def boot():
        try:
            llm.start()
            emb.start()
        except Exception as e:  # noqa: BLE001 - show the failure in the event log
            events.emit("serving.error", message=str(e))

    threading.Thread(target=boot, daemon=True).start()
    try:
        hotkey.listen(lambda: tasks.screen_help())
    except Exception as e:  # noqa: BLE001
        events.emit("hotkey.warning", message=str(e))
    yield


app = FastAPI(lifespan=lifespan)
STATIC = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.middleware("http")
async def no_cache(request: Request, call_next):
    """Always serve the latest page and scripts, so edits show up on a plain reload."""
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store"
    return response


@app.exception_handler(Exception)
async def show_error(_request: Request, error: Exception):
    """Send the reason for an unexpected failure to the page, so it can show more than "Internal Server Error".

    Args:
        _request (Request): The request that failed, not used.
        error (Exception): The failure.

    Returns:
        JSONResponse: Status 500 with the message in `detail`.
    """
    return JSONResponse({"detail": f"{type(error).__name__}: {error}"}, status_code=500)


@app.get("/")
def index():
    """Serve the dashboard page."""
    return FileResponse(STATIC / "index.html")


@app.get("/api/state")
def get_state():
    """Report the state of the servers, the microphone and pending tool approvals.

    Returns:
        dict: `tier`, `llm` and `embed` health, their ports, server memory in GB, `mic_paused`, `recording` and
            `pending` calls.
    """
    return {
        "tier": state["tier"],
        "llm": llm.healthy(),
        "embed": emb.healthy(),
        "llm_port": config.LLM_PORT,
        "embed_port": config.EMBED_PORT,
        "llm_rss_gb": round(llm.rss_gb(), 2),
        "embed_rss_gb": round(emb.rss_gb(), 2),
        "mic_paused": audio.Mic.paused,
        "recording": state["mic"] is not None,
        "pending": [{"call_id": i, "tool": n, "args": a} for i, (n, a) in tools.PENDING.items()],
    }


@app.get("/api/events")
def get_events(after: int = 0):
    """List events newer than an id.

    Args:
        after (int, optional): Return only events with a larger id. Defaults to 0.

    Returns:
        list[dict]: Events, oldest first.
    """
    return events.since(after)


# --- Serving -------------------------------------------------------------------


@app.post("/api/serving/switch")
def switch(body: dict):
    """Restart the language model server on another tier.

    Args:
        body (dict): JSON with `tier`, "e4b" or "e2b".

    Returns:
        dict: The new state.
    """
    global llm
    llm.stop()
    state["tier"] = body["tier"]
    llm = serving.llm_server(body["tier"])
    llm.start()
    return get_state()


@app.post("/api/serving/bench")
def bench():
    """Measure time to first token and generation speed with a fixed prompt.

    Returns:
        dict: The timings from `serving.stream_timing`.
    """
    r = serving.stream_timing([{"role": "user", "content": "Explain what a context window is in 80 words."}])
    events.emit("serving.bench", tier=state["tier"], rss_gb=round(llm.rss_gb(), 2), **r)
    return r


# --- Chat ----------------------------------------------------------------------


@app.get("/api/chat")
def chat_history():
    """List the chat so far, so a page reload shows the same conversation the model still has.

    Returns:
        list[dict]: Messages with `role` and `content`, oldest first.
    """
    return state["history"]


@app.post("/api/chat/clear")
def chat_clear():
    """Start a new chat. Long-term memories stay.

    Returns:
        list: An empty list.
    """
    state["history"] = []
    return []


@app.post("/api/chat")
def chat(body: dict):
    """Answer one chat message, with tools and memory. The last 6 history messages are sent.

    Args:
        body (dict): JSON with `text` and optional `with_screen`, which attaches a screenshot.

    Returns:
        dict: `text`, `tool_calls`, `tokens` and `tools_offered`.
    """
    user = [brain.part("text", body["text"])]
    if body.get("with_screen"):
        user.insert(0, brain.part("image", screen.capture()))
    out = brain.respond(user, history=state["history"][-6:])
    state["history"] += [{"role": "user", "content": body["text"]}, {"role": "assistant", "content": out["text"]}]
    return out


# --- Senses + tasks ------------------------------------------------------------


@app.post("/api/mic/pause")
def mic_pause(body: dict):
    """Turn the global microphone pause switch on or off.

    Args:
        body (dict): JSON with `paused` (bool).

    Returns:
        dict: The new state.
    """
    audio.Mic.paused = bool(body["paused"])
    events.emit("senses.mic", state="paused" if audio.Mic.paused else "unpaused")
    return get_state()


@app.post("/api/mic/start")
def mic_start():
    """Start recording from the microphone.

    Returns:
        dict: The new state.

    Raises:
        HTTPException: 409 if the microphone is paused.
    """
    if audio.Mic.paused:
        raise HTTPException(409, "Microphone is paused")
    state["mic"] = audio.Mic()
    state["mic"].start()
    return get_state()


@app.post("/api/mic/stop")
def mic_stop():
    """Stop recording and run Task 1 on the recording.

    Returns:
        dict: The result of `tasks.voice_note`.
    """
    mic, state["mic"] = state["mic"], None
    samples = mic.stop()
    return tasks.voice_note(samples)


@app.post("/api/task1/file")
async def task1_file(request: Request):
    """Run Task 1 on an uploaded audio file.

    Args:
        request (Request): The request whose body is a WAV, FLAC or AIFF file.

    Returns:
        dict: The result of `tasks.voice_note`.
    """
    data = await request.body()
    return await run_in_threadpool(lambda: tasks.voice_note(audio.load_wav(io.BytesIO(data))))


@app.post("/api/task2")
def task2():
    """Run Task 2 on a screenshot of the live screen.

    Returns:
        dict: The result of `tasks.screen_help`.
    """
    return tasks.screen_help()


@app.post("/api/task2/file")
async def task2_file(request: Request):
    """Run Task 2 on an uploaded image.

    Args:
        request (Request): The request whose body is an image file.

    Returns:
        dict: The result of `tasks.screen_help`.
    """
    img = Image.open(io.BytesIO(await request.body())).convert("RGB")
    img.thumbnail((config.SCREEN_MAX_SIDE, config.SCREEN_MAX_SIDE))
    return await run_in_threadpool(lambda: tasks.screen_help(img))


@app.post("/api/task2/watch")
def task2_watch(body: dict):
    """Run Task 2 on a fresh screen recording.

    Args:
        body (dict): JSON with optional `seconds`. Defaults to 10.

    Returns:
        dict: The result of `tasks.screen_help` for a video.
    """
    return tasks.watch_and_help(float(body.get("seconds", 10)))


@app.post("/api/task2/video_file")
async def task2_video_file(request: Request):
    """Run Task 2 on an uploaded screen recording.

    Args:
        request (Request): The request whose body is a video file.

    Returns:
        dict: The result of `tasks.screen_help` for a video.
    """
    video.CAPTURES.mkdir(parents=True, exist_ok=True)
    raw = video.CAPTURES / f"upload-{time.strftime('%Y%m%d-%H%M%S')}.bin"
    raw.write_bytes(await request.body())
    return await run_in_threadpool(lambda: tasks.screen_help(video_path=video.shrink(raw)))


@app.get("/media")
def media(path: str):
    """Serve a captured or recorded file.

    Args:
        path (str): File path under the runtime or eval case folder.

    Returns:
        FileResponse: The file.

    Raises:
        HTTPException: 403 if the path is outside those folders.
    """
    p = Path(path).resolve()
    if not (p.is_relative_to(config.RUNTIME.resolve()) or p.is_relative_to(config.EVAL_DATA.resolve())):
        raise HTTPException(403)
    return FileResponse(p)


# --- Memory, tools, evals ------------------------------------------------------


@app.get("/api/memory")
def memory_list():
    """List the newest memories.

    Returns:
        list[dict]: Up to 200 memories.
    """
    return episodic.all_memories()


@app.post("/api/memory/search")
def memory_search(body: dict):
    """Search memories.

    Args:
        body (dict): JSON with `q`, the query text.

    Returns:
        list[dict]: The 5 best matches.
    """
    return episodic.search(body["q"], k=5)


@app.post("/api/memory/clear")
def memory_clear():
    """Delete every memory.

    Returns:
        list: An empty list.
    """
    episodic.clear()
    return []


@app.get("/api/tools")
def tool_list():
    """List the registered tools.

    Returns:
        list[dict]: `name`, `tier`, `description` and `guarded` for each tool.
    """
    return [
        {"name": t.name, "tier": t.tier, "description": t.description, "guarded": t.guard is not None}
        for t in tools.REGISTRY.values()
    ]


@app.post("/api/tools/approve")
def approve(body: dict):
    """Approve or deny a tool call that waits for the user.

    Args:
        body (dict): JSON with `call_id` (int) and `yes` (bool).

    Returns:
        dict: The tool result, or `{"status": "denied"}`.
    """
    return tools.approve(int(body["call_id"]), bool(body["yes"]))


@app.get("/api/scorecard")
def scorecard():
    """Return the cached eval results.

    Returns:
        list[dict]: One row per component and metric.
    """
    return evals_common.scorecard_rows()


@app.post("/api/evals/live")
def eval_live(body: dict):
    """Run a few cases of one eval and show the inputs and outputs.

    Args:
        body (dict): JSON with `component` and optional `n` (items to run, default 1).

    Returns:
        list[dict]: One record per item.

    Raises:
        HTTPException: 404 if the component has no cases.
    """
    try:
        return evals_common.run_live(body["component"], int(body.get("n", 1)))
    except evals_common.NoCases as e:
        raise HTTPException(404, f"No eval cases: {e}. Add one with: python cli.py evals add -h") from e
