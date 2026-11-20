# multimodal-agent architecture

A personal agent that listens, sees, remembers and acts, entirely on the device. Inputs flow through senses and tasks
into the brain. The brain talks to a local `llama-server` running Gemma 4, with memory and a permissioned tool registry
on the side. The numbers in the diagram match the Session 1 segments in [Where each segment lives](#where-each-segment-lives).
Session 2 adds a daemon process around this core. See [Session 2: always on and learning](#session-2-always-on-and-learning).

## Diagram

Each box is one component, and [Components](#components) lists what is inside it. Dashed boxes and arrows are the
portability plug-ins.

```mermaid
flowchart TB
    inputs["Inputs<br/>microphone, screen, hotkey,<br/>cli.py, dashboard"]
    tasks["6 Tasks: agent/tasks.py<br/>voice_note(), screen_help(),<br/>watch_and_help()"]
    senses["2 Senses: agent/senses/<br/>audio.py, screen.py, video.py"]
    brain["3 Brain: agent/brain.py<br/>part(), fit_budget(), send(),<br/>ask_json(), respond()"]
    tools["5 Tools: agent/tools/<br/>select(), run(), check()"]
    memory["4 Memory: agent/memory/episodic.py<br/>SQLite memory.sqlite"]
    workspace["Workspace: data/workspace/<br/>sandbox for file and shell tools"]

    subgraph serving["1 Serving: agent/serving.py"]
        llm["llama-server :9931<br/>Gemma 4"]
        emb["llama-server :9932<br/>EmbeddingGemma-2"]
    end

    accel["agent/accel/<br/>GPU or runtime"]
    platforms["agent/platforms/<br/>operating system"]

    inputs -- "note, look, hotkey" --> tasks
    inputs -- "chat" --> brain
    tasks -- "capture" --> senses
    senses -- "samples, frames, clips" --> brain
    tasks -- "task prompts" --> brain
    tasks -- "task results" --> memory
    brain <-- "calls" --> tools
    brain -- "chat completions" --> llm
    brain -- "recall top 3" --> memory
    tools -- "rank tools" --> emb
    memory <-- "embed" --> emb
    tools -- "save_note, search_memory" --> memory
    tools -- "file and shell tools" --> workspace
    accel -. "binary, server flags" .-> serving
    platforms -. "screen recording, hotkey" .-> inputs

    classDef input fill:#f3f3f1,stroke:#8d8d86,color:#1d1d1b
    classDef sense fill:#e2f5ee,stroke:#12856a,color:#1d1d1b
    classDef task fill:#fdf2d9,stroke:#b57512,color:#1d1d1b
    classDef think fill:#e6effc,stroke:#2f6fd1,color:#1d1d1b
    classDef tool fill:#fde9e6,stroke:#c0392b,color:#1d1d1b
    classDef store fill:#e6f5e5,stroke:#2e8b3d,color:#1d1d1b
    classDef serve fill:#fdebdd,stroke:#cf6a1f,color:#1d1d1b
    classDef plugin fill:#eee9fb,stroke:#7048c9,stroke-dasharray:7 5,color:#1d1d1b

    class inputs,workspace input
    class senses sense
    class tasks task
    class brain think
    class tools tool
    class memory store
    class llm,emb serve
    class accel,platforms plugin
```

## Components

### Inputs

| Input | What it gives |
| ----- | ------------- |
| Microphone | A live or recorded clip |
| Screen | A screenshot or a screen recording |
| Hotkey | A global key combination, ctrl+alt+space. The OS plug-in provides it. |
| CLI, `cli.py` | `serve`, `stop`, `dashboard`, `hotkey`, `note`, `look`, `chat`, `memory`, `record-clip`, `record-screen`, `smoke`, `doctor`, `evals` |
| Dashboard | FastAPI on port 8765 with 9 tabs: chat, memory, tool approval, model switch, benchmark, daemon, evolution, tasks, evals |

### 1 Serving: `agent/serving.py`

Start, stop, health, chat and embed.

| Server | Details |
| ------ | ------- |
| `llama-server` on port 9931 | Gemma 4 E4B (default) or E2B, Q4_0. The mmproj handles audio, image and video. 16k context. Video runs at 1 fps, decoded with ffmpeg. The GPU plug-in provides the binary and flags. |
| `llama-server` on port 9932 | EmbeddingGemma-2, 768-d vectors |

The dashboard watches both servers. It restarts a server whose process stops, on the current tier, unless `cli.py stop`
stopped it.

### 2 Senses: `agent/senses/`

| File | What it does |
| ---- | ------------ |
| `audio.py` | Capture and Silero VAD, with chunks up to 28 s |
| `screen.py` | Downscale to at most 1280 px, and check for a flat image |
| `video.py` | Record a clip and shrink it to 640 px, as a 1 fps video part. The OS plug-in records the screen. |

### 3 Brain: `agent/brain.py`

| Function | What it does |
| -------- | ------------ |
| `part()` | Turns text, image, audio or video into OpenAI-style parts |
| `fit_budget()` | Adds the prompt, the top 3 memories and the tool schemas. Drops old history, then memories. |
| `send()` | Posts to `llama-server`, and logs the exact request to `last_request.json` and the `brain.request` event |
| `ask_json()` | Structured output: a JSON schema becomes a grammar. Validates, and retries once. |
| `respond()` | The tool loop, up to 4 steps. A bad call is asked again, up to 2 retries. Thinking is on after a tool result. |

Both `ask_json()` and `respond()` run the `part()`, `fit_budget()`, `send()` pipeline. Tasks call `ask_json()`. `cli.py
chat` and `/api/chat` call `respond()` directly. `respond()` also returns the memories it recalled, so the chat can
show what the reply was built from.

### 4 Memory: `agent/memory/episodic.py`

SQLite in `memory.sqlite`, with text and 768-d vectors. It stores and searches (top 3). The junk filter drops text that is
too short or nearly a duplicate.

### 5 Tools: `agent/tools/`

`tools.select()` embeds the request, ranks the tools by description and examples, and offers the top 5 only. `tools.run()`
calls `check()`: schema, tier, guard, then execute.

| Tier | Tools |
| ---- | ----- |
| Safe | `get_time`, `calculator`, `save_note`, `search_memory`, `list_files`, `read_file` |
| Confirm (approved in the dashboard) | `write_file`, `run_shell` |
| Blocked | `delete_file` |

The file and shell tools work in the sandbox `data/workspace/`, set by `AGENT_WORKSPACE` and ignored by Git. The path
guard keeps them inside the sandbox. The shell guard blocks risky commands.

### 6 Tasks: `agent/tasks.py`

| Task | What it does |
| ---- | ------------ |
| Task 1, `voice_note()` | Voice note to summary and action items. The result is stored in memory. |
| Task 2, `screen_help()` and `watch_and_help()` | Screen (image or video) to an explanation and a fix. The result is stored in memory. |

### Cross-cutting and quality

| Part | What it does |
| ---- | ------------ |
| `agent/events.py` | `events.emit(kind, ...)` from every component. A SQLite table (`data/runtime/events.sqlite`), so the daemon, the dashboard and the CLI share one log, plus `events.jsonl`. Each event names its `source` process. The dashboard polls `/api/events`. |
| `agent/sql.py` | Opens every SQLite file: commit, roll back, and always close. WAL mode lets one process read while another writes. |
| `evals/` (9 Scorecard) | 15 components: the 10 of Session 1, plus `daemon`, `multimodal`, `consolidation`, `task3` and `task4`. Synthetic cases and your cases in `data/evals/`, scored against `config.THRESHOLDS`. |
| `scripts/` | `setup.py` downloads and verifies pinned assets. `doctor.py` detects the OS and GPU and checks hooks. `smoke_test.py` runs end to end with real models. Run them through `cli.py doctor`, `smoke` and `evals`. |
| `tests/` and gates | `unittest` contract tests with no model and no GPU. `ruff check` and `ruff format`, `pyright`, `complexipy`. `test_plugin_contract.py` keeps OS and GPU code out of core modules. |

## Where each segment lives

| Segment | Files | Eval (folder in `evals/`) |
| ------- | ----- | ------------------------- |
| 0 Harness map, eval pattern | This page, `agent/events.py`, `evals/common.py`, `evals/cases.py`, `evals/synthetic/` | n/a |
| 1 Serving | `agent/config.py`, `agent/serving.py` | `serving` |
| 2 Senses | `agent/senses/audio.py`, `agent/senses/screen.py`, `agent/senses/video.py` | `vad`, `asr`, `screen` |
| 3 Brain | `agent/brain.py` | `structured`, `tool_calls` |
| 4 Memory v1 | `agent/memory/episodic.py` | `memory` |
| 5 Tools, CLI, hotkey | `agent/tools/`, `cli.py`, `agent/hotkey.py` | `tools` |
| 6 Live tasks | `agent/tasks.py` | `task1`, `task2` (end-to-end) |
| 7 Scorecard | `evals/run_all.py`, dashboard tab 9 | n/a |

## Portability layer

Everything that differs by OS or GPU sits behind two plug-in folders. The agent chooses the plug-in at start-up. Core code
never branches on the OS. See [PLATFORMS.md](PLATFORMS.md).

| Folder | Axis | Decides |
| ------ | ---- | ------- |
| `agent/platforms/<os>.py` | Operating system | Screen recording, speech synthesis, hotkey listener and permissions, install hints, `.exe` suffix, venv path |
| `agent/accel/<backend>.py` | GPU or runtime | Which llama.cpp build to download and its checksum, extra `llama-server` flags, how to measure server memory |
| `agent/pins.py` | Neither | Model weights and VAD checksums (same on every machine) |

Each plug-in folder holds a contract, a verified reference and a template:

| Folder | Contract | Verified reference | Template | Hooks |
| ------ | -------- | ------------------ | -------- | ----- |
| `agent/accel/` | `base.py`, `Accelerator` | `metal.py`, Apple Silicon | `_template.py`, for CUDA, Vulkan or CPU | `download_for`, `server_args`, `server_memory_gb`, `transcribe` (Session 2) |
| `agent/platforms/` | `base.py`, `Platform` and `Unsupported` | `macos.py` | `_template.py`, for Linux or Windows | `record_screen`, `synthesize_speech`, `start_hotkey`, and for Session 2 `keep_awake`, `notify`, `install_autostart`, `uninstall_autostart`, `spawn_background` |

Shared configuration: `config.py` holds ports, thresholds, paths and knobs, and `config.LLAMA_BUILD` pins the llama.cpp
build. `pins.py` holds the sha256 of the model and VAD weights. The environment variables are `AGENT_TIER`,
`AGENT_WORKSPACE`, `AGENT_PLATFORM`, `AGENT_ACCEL` and `AGENT_RUNTIME`. `AGENT_RUNTIME` moves `data/runtime/`; the daemon
eval and the soak test use it to run a throwaway daemon.

## Data flow for one request

1. A sense produces raw input: a 16 kHz float array, a PIL image or a video file.
2. `brain.part()` turns it into an OpenAI-style content part (base64 WAV, PNG or MP4).
3. For a chat turn, `brain.respond()` calls `tools.select()`. It embeds the request with the embedding server and ranks
   every tool by its description plus example requests. Only the top 5 tool schemas go into the request. If the embedding
   server is down, the first 5 registered tools are used.
4. `brain.fit_budget()` adds the system prompt, the top 3 memories and the tool schemas. It estimates tokens: audio at 25
   per second, an image at pixels / 2304 (up to 1,120), a video at frames x image, text at chars / 4. If the request does
   not fit, it drops the oldest history first, then the memories.
5. `brain.send()` writes the exact request to `data/runtime/last_request.json`, logs it to the event log, and posts it to
   `llama-server`.
6. A reply with `tool_calls` goes through `tools.run()`: schema check, tier, guard, execute. The result returns to the
   model as a `tool` message and the loop runs again, with thinking turned on from the second step (a Gemma 4 quirk: with
   thinking off, the reply after a tool result comes back empty).
   - A `confirm` call returns `pending` and ends the turn with a reply that the code writes, because the model claimed the
     call was done. While it waits, the system prompt names it. The model does not reply when the user decides. The
     dashboard adds the outcome (`brain.approval_note()`) to the chat history, so the next request tells the model what
     happened.
   - A `blocked` call never runs.
   - When the retries for invalid calls run out, the loop stops and says so.

## Session 2: always on and learning

Session 2 turns the agent from something you call into something that runs all the time and improves from feedback.
The daemon is its own process, so a kill -9 of the daemon leaves the dashboard up. The two processes share the SQLite
files in `data/runtime/`: the event log, the checkpoints, the memory and the prompt versions.

```mermaid
flowchart TB
    trig["Triggers<br/>config.SCHEDULE, many new transcripts,<br/>dashboard, cli.py"]
    subgraph daemon["daemon process: agent/daemon.py"]
        loop["Main loop<br/>due(), triggers(), claim_next()"]
        jobs["Jobs<br/>consolidate, brief, reflect,<br/>forget, replay, prune_events"]
        listen["Ambient listener<br/>agent/senses/ambient.py"]
        beat["Heartbeat thread"]
    end
    ckpt["Checkpoints: agent/checkpoint.py<br/>checkpoints.sqlite<br/>intent, execution, result"]
    stt["Speech to text: agent/senses/stt.py<br/>Whisper (accel hook) or Gemma"]
    cons["Consolidation: agent/memory/consolidate.py<br/>raw to summaries to facts, forget()"]
    mem["Memory: episodic.py<br/>levels, citations, image and audio vectors"]
    evo["Self-evolution: agent/evolve.py<br/>locked blocks, versions, reflect(),<br/>generality gate, rollback()"]
    dash["Dashboard<br/>Always on, Evolve, Tasks 3 and 4"]
    plat["agent/platforms/<br/>keep_awake, notify, autostart"]

    trig --> loop
    loop --> jobs
    jobs <--> ckpt
    listen --> stt --> mem
    jobs --> cons --> mem
    jobs --> evo
    evo -- "brief, and gate days, from day_memories()" --> mem
    beat --> ckpt
    dash -- "queue a job, read state" --> ckpt
    dash -- "feedback, rollback" --> evo
    plat -. "OS hooks" .-> daemon

    classDef store fill:#e6f5e5,stroke:#2e8b3d,color:#1d1d1b
    classDef think fill:#e6effc,stroke:#2f6fd1,color:#1d1d1b
    classDef sense fill:#e2f5ee,stroke:#12856a,color:#1d1d1b
    classDef plugin fill:#eee9fb,stroke:#7048c9,stroke-dasharray:7 5,color:#1d1d1b
    class ckpt,mem store
    class cons,evo think
    class listen,stt sense
    class plat plugin
```

### Components

| File | What it does |
| ---- | ------------ |
| `agent/daemon.py` | The main loop, once a second: queue the scheduled jobs that are due, check the triggers, run one job. A heartbeat thread writes "alive" every 5 s. On start, `checkpoint.recover()` re-queues the job that was running when the process died. Each finished job goes to the feed with its reason. |
| `agent/checkpoint.py` | A job queue and a step log in SQLite. `step()` records intent, then execution, then the result. On a rerun, a step with a result returns the stored output and does not run again. |
| `agent/senses/ambient.py` | `Segmenter` cuts a stream at pauses, while the audio arrives. `Listener` is the live microphone, and it closes the microphone when the pause switch is on. `SilenceCheck` emits `senses.ambient.warning` when the stream gives only silence for `AMBIENT_SILENT_S` after it opens (an OS without the Microphone permission gives zeros, not an error). `utterances()` feeds a recording through the same Segmenter for a replay. |
| `agent/senses/stt.py` | Speech to text for always-on listening: Whisper through the accelerator's `transcribe` hook, or Gemma 4. |
| `agent/memory/consolidate.py` | Groups raw rows into 30-minute windows. One request per window writes claims and facts, and each one cites raw ids. Code drops invalid citations. `forget()` deletes covered raw rows after 7 days. `judge()` checks one claim against only the rows it cites. |
| `agent/memory/episodic.py` | Adds a level (raw, summary, fact), `covered_by`, `superseded_by` and a second vector for the screenshot or the audio clip. `search()` takes a mode (text, media or both) and a time window. |
| `agent/evolve.py` | The brief prompt: locked blocks in code, plus a style block and a preference list in versioned rows. `reflect()` asks for a patch, and `validate_patch()` refuses anything outside the editable parts. Then the generality gate writes briefs with the parent and the new version for a few recent days (`config.EVOLVE_GATE_DAYS`): if `ungrounded()` finds names or numbers in the new briefs that the memories never say and the parent did not write, the patch is refused (`evolve.rejected`). `rollback()` makes any version active. |
| `agent/tasks.py` | Task 3, `recall()`: a time window from the question ("this morning"), a search, and an answer that cites memory ids. Task 4, `daily_brief()`: the day's summaries, facts, action items and screen episodes, written with the active prompt version. |
| `dashboard/chat.py` | The chat endpoints. A message can carry a picture, a screenshot or speech (the VAD keeps the speech, `stt.transcribe()` writes the text). Each message keeps `role` and `content` for the model and `ui` for the page (pictures, recalled memories, tool calls, timings). A session is one conversation with its own history for the model, and all sessions share the long-term memory. Each session is one file in `config.CHAT_DIR`, so it survives a restart (`config.CHAT_FILE` of older versions becomes the first session). A stored daily brief joins the conversation once. |
| `dashboard/always_on.py` | The Session 2 endpoints. While the daemon runs, the dashboard queues jobs for it. Without a daemon, it runs them itself through the same handlers. |
| `scripts/soak.py` | Runs a daemon for hours under load and records its memory and missed events. |

### Where each Session 2 segment lives

| Segment | Files | Eval (folder in `evals/`) |
| ------- | ----- | ------------------------- |
| 24/7 daemon | `agent/daemon.py`, `agent/checkpoint.py`, `scripts/soak.py`, keep-awake and autostart hooks | `daemon` |
| Memory consolidation | `agent/memory/consolidate.py`, `agent/memory/episodic.py` | `consolidation`, `multimodal` |
| Self-evolution | `agent/evolve.py`, dashboard tab 7 (prompt diff view) | `task4` |
| Live tasks | `agent/tasks.py` (`recall`, `daily_brief`), `agent/senses/ambient.py`, `agent/senses/stt.py` | `task3`, `task4`, `asr` (Gemma and Whisper on the same clips) |

### Data flow for one daemon job

1. A trigger queues a job: the schedule, a count of new transcripts, the dashboard or `cli.py`. The job row holds its
   kind, its inputs and the reason.
2. The main loop claims the oldest queued job and marks it `running`.
3. The handler runs each step through `checkpoint.step()`: intent, execution, then the result with its output.
4. If the process dies, the next start puts the job back in the queue. Steps with a result are skipped; the step that
   was in flight runs again. Steps must be safe to run twice. A summary, for example, is found again by its window key.
5. The job ends `done` or `failed`. The `daemon.action` event carries the reason and a summary to the dashboard feed. A
   finished brief also shows a desktop notification.
