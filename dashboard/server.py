"""Local dashboard: one FastAPI app + one HTML page. Bound to 127.0.0.1 only.

    .venv/bin/python cli.py dashboard        # then open http://127.0.0.1:8765 in Chrome
"""
import io
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
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


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/state")
def get_state():
    return {"tier": state["tier"], "llm": llm.healthy(), "embed": emb.healthy(),
            "llm_rss_gb": round(llm.rss_gb(), 2), "embed_rss_gb": round(emb.rss_gb(), 2),
            "mic_paused": audio.Mic.paused, "recording": state["mic"] is not None,
            "pending": [{"call_id": i, "tool": n, "args": a} for i, (n, a) in tools.PENDING.items()]}


@app.get("/api/events")
def get_events(after: int = 0):
    return events.since(after)


# --- Serving -------------------------------------------------------------------

@app.post("/api/serving/switch")
def switch(body: dict):
    global llm
    llm.stop()
    state["tier"] = body["tier"]
    llm = serving.llm_server(body["tier"])
    llm.start()
    return get_state()


@app.post("/api/serving/bench")
def bench():
    r = serving.stream_timing([{"role": "user", "content": "Explain what a context window is in 80 words."}])
    events.emit("serving.bench", tier=state["tier"], rss_gb=round(llm.rss_gb(), 2), **r)
    return r


# --- Chat ----------------------------------------------------------------------

@app.post("/api/chat")
def chat(body: dict):
    user = [brain.part("text", body["text"])]
    if body.get("with_screen"):
        user.insert(0, brain.part("image", screen.capture()))
    out = brain.respond(user, history=state["history"][-6:])
    state["history"] += [{"role": "user", "content": body["text"]}, {"role": "assistant", "content": out["text"]}]
    return out


# --- Senses + tasks ------------------------------------------------------------

@app.post("/api/mic/pause")
def mic_pause(body: dict):
    audio.Mic.paused = bool(body["paused"])
    events.emit("senses.mic", state="paused" if audio.Mic.paused else "unpaused")
    return get_state()


@app.post("/api/mic/start")
def mic_start():
    if audio.Mic.paused:
        raise HTTPException(409, "Microphone is paused")
    state["mic"] = audio.Mic()
    state["mic"].start()
    return get_state()


@app.post("/api/mic/stop")
def mic_stop():
    mic, state["mic"] = state["mic"], None
    samples = mic.stop()
    return tasks.voice_note(samples)


@app.post("/api/task1/file")
async def task1_file(request: Request):
    data = await request.body()
    return await run_in_threadpool(lambda: tasks.voice_note(audio.load_wav(io.BytesIO(data))))


@app.post("/api/task2")
def task2():
    return tasks.screen_help()


@app.post("/api/task2/file")
async def task2_file(request: Request):
    img = Image.open(io.BytesIO(await request.body())).convert("RGB")
    img.thumbnail((config.SCREEN_MAX_SIDE, config.SCREEN_MAX_SIDE))
    return await run_in_threadpool(lambda: tasks.screen_help(img))


@app.post("/api/task2/watch")
def task2_watch(body: dict):
    return tasks.watch_and_help(float(body.get("seconds", 10)))


@app.post("/api/task2/video_file")
async def task2_video_file(request: Request):
    video.CAPTURES.mkdir(parents=True, exist_ok=True)
    raw = video.CAPTURES / f"upload-{time.strftime('%Y%m%d-%H%M%S')}.bin"
    raw.write_bytes(await request.body())
    return await run_in_threadpool(lambda: tasks.screen_help(video_path=video.shrink(raw)))


@app.get("/media")
def media(path: str):
    p = Path(path).resolve()
    if not (p.is_relative_to(config.RUNTIME.resolve()) or p.is_relative_to(config.GOLD.resolve())):
        raise HTTPException(403)
    return FileResponse(p)


# --- Memory, tools, evals ------------------------------------------------------

@app.get("/api/memory")
def memory_list():
    return episodic.all_memories()


@app.post("/api/memory/search")
def memory_search(body: dict):
    return episodic.search(body["q"], k=5)


@app.post("/api/memory/clear")
def memory_clear():
    episodic.clear()
    return []


@app.get("/api/tools")
def tool_list():
    return [{"name": t.name, "tier": t.tier, "description": t.description, "guarded": t.guard is not None}
            for t in tools.REGISTRY.values()]


@app.post("/api/tools/approve")
def approve(body: dict):
    return tools.approve(int(body["call_id"]), bool(body["yes"]))


@app.get("/api/scorecard")
def scorecard():
    return evals_common.scorecard_rows()


@app.post("/api/evals/live")
def eval_live(body: dict):
    try:
        return evals_common.run_live(body["component"], int(body.get("n", 1)))
    except evals_common.NoGold as e:
        raise HTTPException(404, f"No gold data recorded yet in {e} (see docs/RECORDING_GUIDE.md)")
