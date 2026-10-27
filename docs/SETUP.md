# Setup (after the session)

Run the whole agent on your own Mac. Everything stays on your machine: the model servers listen on
`127.0.0.1` only, and nothing is sent to the internet after the one-time downloads.

## Requirements

| | |
|---|---|
| Mac | Apple Silicon (M1 or newer). Tested on an M4 with 24 GB, macOS 27.0.1 |
| Memory | 16 GB works with the E2B tier. 24 GB or more is better for E4B |
| Disk | about 11 GB (both model tiers + embedding model) |
| Python | 3.10 or newer (tested with 3.13.3) |
| Tools | `curl`, `shasum`, `tar` (built into macOS) |

Not supported: Linux, Windows, Intel Macs, NVIDIA, Colab.

## Install

```bash
git clone <this repo> && cd multimodal-agent
scripts/setup.sh
```

`setup.sh` downloads and checksum-verifies:
- llama.cpp **b11669** (= v0.7.0), the official prebuilt macOS arm64 binaries with Metal
- Gemma 4 **E4B** and **E2B** (Q4_0, from Google's QAT checkpoints) + their Q8_0 image/audio projectors
- **EmbeddingGemma-2** (Q8_0) + its projector, for memory search over text, images and audio
- **Silero VAD** v6.2.3 (ONNX file only)
- a `.venv` with the pinned packages from `requirements.lock`

Then check that text, image and audio all work:

```bash
.venv/bin/python scripts/smoke_test.py             # E4B
.venv/bin/python scripts/smoke_test.py --tier e2b  # E2B
```

## macOS permissions

Grant these to the app you run the agent from (Terminal, iTerm, VS Code...) in
**System Settings > Privacy & Security**, then restart that app:

| Permission | Needed for | Symptom when missing |
|---|---|---|
| Microphone | recording voice notes | silent recordings |
| Screen Recording | Task 2, screen evals | blank screenshots; the dashboard shows a warning |
| Accessibility + Input Monitoring | the global hotkey (Ctrl+Option+Space) | the hotkey does nothing; warning in the event log |

## Run

```bash
.venv/bin/python cli.py dashboard     # open http://127.0.0.1:8765 in Chrome
```

The dashboard starts both model servers (this takes about 10 to 20 s the first time). Tabs 1 to 7
follow the session agenda: Serving, Senses, Brain, Memory, Tools, Tasks, Scorecard.
Press the number keys to switch tabs.

Without the dashboard:

```bash
.venv/bin/python cli.py note                     # Task 1: speak, press Enter, get action items
.venv/bin/python cli.py look                     # Task 2: explain the current screen
.venv/bin/python cli.py chat "what do I need to do tomorrow?"
.venv/bin/python cli.py memory                   # list memories
.venv/bin/python cli.py stop                     # stop the model servers
```

Switch to the small model for any command: `AGENT_TIER=e2b .venv/bin/python cli.py ...`

## Evals

```bash
.venv/bin/python evals/run_all.py          # print the scorecard
.venv/bin/python evals/run_all.py --run    # run all evals that have gold data
```

The gold sets are personal recordings and are not in the repo. To make your own, follow
[RECORDING_GUIDE.md](RECORDING_GUIDE.md).

## Troubleshooting

- **`llama-server exited`**: read `data/runtime/llama-llm.log`. Usually a missing model file:
  re-run `scripts/setup.sh`.
- **Port 9931 in use**: another llama-server is running. Run `.venv/bin/python cli.py stop`.
- **Slow or swapping**: close other large apps, or use `AGENT_TIER=e2b`.
- **"Apple could not verify llama-server"**: this happens only if the binaries were downloaded
  in a browser. `setup.sh` uses curl, which does not set the quarantine flag.

## Session 3 preview

Session 3 adds MCP connectors. Many MCP servers start with `npx` (needs [Node.js](https://nodejs.org))
or `uvx` (needs [uv](https://docs.astral.sh/uv/)). Install these before Session 3 if you want to try
third-party servers.
