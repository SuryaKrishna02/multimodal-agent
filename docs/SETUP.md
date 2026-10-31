# Setup

Run the whole agent on your own machine. Everything stays on it: the model servers listen on
`127.0.0.1` only, and the agent sends nothing to the internet after the one-time downloads.

## Requirements

| | |
|---|---|
| Machine | Verified: Apple Silicon Mac (M4, 24 GB, macOS 27.0.1). Other OSes and GPUs: see [PLATFORMS.md](PLATFORMS.md) |
| Memory | 16 GB works with the E2B tier. 24 GB or more is better for E4B |
| Disk | about 11 GB (both model tiers + embedding model) |
| Python | 3.10 or newer (tested with 3.13.3) |
| Optional | `ffmpeg` + `ffprobe` for screen-video input and the two synthetic video cases of the task2 eval (the doctor prints the install command for your OS) |

If your OS or GPU has no plug-in yet, `setup.py` stops with a message naming the file to add.
Adding one is a single-file PR: [CONTRIBUTING.md](../CONTRIBUTING.md).

## Install

```bash
git clone <this repo> && cd multimodal-agent
python3 scripts/setup.py
```

`setup.py` detects your OS and accelerator, then downloads and checksum-verifies:
- llama.cpp **b11960** (= v0.8.0), the official prebuilt binaries for your accelerator
  (on a Mac: Metal, from `agent/accel/metal.py`)
- Gemma 4 **E4B** and **E2B** (Q4_0, from Google's QAT checkpoints) + their Q8_0 image/audio projectors
- **EmbeddingGemma-2** (Q8_0) + its projector, for memory search over text, images and audio
- **Silero VAD** v6.2.3 (ONNX file only)
- a `.venv` with the pinned packages (`requirements/lock-<os>.txt`, else `requirements/base.txt`)

Skip what you do not need: `python3 scripts/setup.py --only e2b embed vad`.
Force a choice: `AGENT_ACCEL=<name>` or `AGENT_PLATFORM=<name>` (names: `cli.py doctor`).

Then check what the agent detected, and that text, image and audio all work. Use the venv's Python
(`.venv/bin/python`, or `.venv\Scripts\python.exe` on Windows):

```bash
.venv/bin/python cli.py doctor [--exercise]   # --exercise runs screen recording + speech for real
.venv/bin/python cli.py smoke                 # E4B
.venv/bin/python cli.py smoke --tier e2b      # E2B
```

## OS permissions

`cli.py doctor` lists the ones your OS needs. On macOS, open **System Settings > Privacy & Security**.
Grant these to the app that runs the agent (Terminal, iTerm, Visual Studio Code, or another terminal app). Then restart that app:

| Permission | Needed for | Symptom when missing |
|---|---|---|
| Microphone | recording voice notes | silent recordings |
| Screen Recording | Task 2, screen videos, screen evals | blank screenshots, and the dashboard shows a warning |
| Accessibility + Input Monitoring | the global hotkey (Ctrl+Option+Space) | the hotkey does nothing, and the event log shows a warning |

## Run

```bash
.venv/bin/python cli.py dashboard     # open http://127.0.0.1:8765 in Chrome
```

The dashboard starts both model servers (this takes about 10 to 20 s the first time). Tabs 1 to 7
follow the session agenda: Serving, Senses, Brain, Memory, Tools, Tasks, Scorecard.
Press the number keys to switch tabs. The page is light only. The chat sits on the right (press C to hide it and give
the tabs more room), and a bar at the bottom opens the Activity log (the event log). When a tool call waits for your
approval, Approve and Deny appear in the chat, above the message box. With the chat hidden or on a narrow window, they
appear in a banner at the top instead. A dot on a tab lights up while that part of the agent is working.
After you decide, the chat history holds the outcome, so you can ask the model "did that work?"

If a model server stops unexpectedly, the dashboard restarts it on the current tier. `data/runtime/llama-llm.previous.log`
or `llama-embed.previous.log` keeps the log of the server that stopped. The dashboard does not restart a server that
`python cli.py stop` stopped: the next dashboard start or `python cli.py serve` starts it again. Until then, the header
says "starting…".

Without the dashboard:

```bash
.venv/bin/python cli.py note                     # Task 1: speak, press Enter, get action items
.venv/bin/python cli.py look                     # Task 2: explain the current screen
.venv/bin/python cli.py look --watch 10          # Task 2 on a 10 s screen recording (needs ffmpeg)
.venv/bin/python cli.py chat "what do I need to do tomorrow?"
.venv/bin/python cli.py memory                   # list memories
.venv/bin/python cli.py stop                     # stop the model servers
```

Switch to the small model for any command: `AGENT_TIER=e2b .venv/bin/python cli.py ...`

## Evals

```bash
.venv/bin/python evals/run_all.py          # print the scorecard
.venv/bin/python evals/run_all.py --run    # run all evals on the synthetic cases and your own
.venv/bin/python evals/run_all.py status   # how many cases each eval has
```

There is nothing to record. The first run builds synthetic cases (speech, screenshots, tool calls, memories)
into `data/evals/synthetic/`. To add a case from your own use, see [EVALS.md](EVALS.md).

## Troubleshooting

- **`llama-server exited`**: read `data/runtime/llama-llm.log`. Most often a model file is missing:
  re-run `python3 scripts/setup.py`.
- **Port 9931 in use**: another llama-server is running. Run `.venv/bin/python cli.py stop`.
- **Slow or swapping**: close other large apps, or set `AGENT_TIER=e2b`.
- **`Apple could not verify llama-server`**: this happens only when a browser downloaded the
  binaries. `setup.py` downloads from Python, which does not set the quarantine flag.

## Session 3 preview

Session 3 adds MCP connectors. Many MCP servers start with `npx` (needs [Node.js](https://nodejs.org))
or `uvx` (needs [uv](https://docs.astral.sh/uv/)). Install these before Session 3 if you want to try
third-party servers.
