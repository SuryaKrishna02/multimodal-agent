# multimodal-agent

A teaching harness for a personal agent that **listens, sees, remembers and acts**, entirely on-device
with open models. Built for a three-part coding series by the Multimodal AI Group at Cohere Labs Open Science.

- Inference: [llama.cpp](https://github.com/ggml-org/llama.cpp) `llama-server` (pinned build b11669) on Apple Metal
- Model: Gemma 4 **E4B** (default) or **E2B**: text, image, audio and screen video in, text out
- Memory search: EmbeddingGemma-2 (text, image and audio in one vector space)
- Everything else is plain Python you can read in one sitting

Each component has an eval: a small human-labeled gold set, a metric, a pass threshold and a script.
`python evals/run_all.py` prints the scorecard.

## Quick start (Apple Silicon Mac)

```bash
scripts/setup.sh                              # pinned binaries + models, checksum-verified (~11 GB)
.venv/bin/python scripts/smoke_test.py        # text, image, audio, tools: ALL PASS?
.venv/bin/python cli.py dashboard             # http://127.0.0.1:8765
```

Full instructions and macOS permissions: [docs/SETUP.md](docs/SETUP.md).

## Layout

```
agent/
  config.py          every knob: tier, paths, ports, thresholds
  serving.py         llama-server lifecycle + client                      (segment 1)
  senses/audio.py    mic, Silero VAD, chunks under 30 s                   (segment 2)
  senses/screen.py   screen capture + change-based sampling               (segment 2)
  senses/video.py    screen recording -> one video part (ffmpeg)          (segment 2)
  brain.py           request assembly, context budget, JSON, tool loop    (segment 3)
  memory/episodic.py SQLite log + embedding search + junk filter          (segment 4)
  tools/             registry, permission tiers, guards, built-in tools   (segment 5)
  hotkey.py          Ctrl+Option+Space -> Task 2                          (segment 5)
  tasks.py           Task 1 (voice note -> action items), Task 2 (screen help)
  events.py          event log shared by every component and the dashboard
dashboard/           FastAPI + one HTML page; tabs follow the agenda
cli.py               every action from the terminal
evals/<component>/   gold format (docstring), loader, scorer
evals/run_all.py     the scorecard
data/gold/           your recordings + labels (git-ignored by default)
docs/                SETUP, RECORDING_GUIDE, ARCHITECTURE (diagram), VERIFICATION
scripts/             setup.sh, smoke_test.py
```

## Status

Session 1 (the core loop) is built. Sessions 2 (always on, learning) and 3 (voice, skills, MCP)
come next. See [docs/VERIFICATION.md](docs/VERIFICATION.md) for what was tested and what was found.
