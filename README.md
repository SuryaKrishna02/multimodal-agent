# multimodal-agent

A teaching harness for a personal agent that **listens, sees, remembers and acts**, entirely on-device
with open models. Built for a three-part coding series of the Multimodal AI Group of the
[Cohere Labs Open Science Community](https://labscommunity.cohere.com/home).

- Inference: [llama.cpp](https://github.com/ggml-org/llama.cpp) `llama-server`, pinned and checksummed ([docs/SETUP.md](docs/SETUP.md)). Verified on Apple Metal. Other OSes and GPUs plug in as single files ([CONTRIBUTING.md](CONTRIBUTING.md))
- Model: Gemma 4 **E4B** (default) or **E2B**: text, image, audio and screen video (sampled frames, no soundtrack) in; text out (no speech output)
- Memory search: EmbeddingGemma-2 (text, image and audio in one vector space)
- Everything else is plain Python you can read in one sitting

Each component has an eval: test cases, a metric, a pass threshold and a script. The cases are synthetic (generated, so
nothing to record) plus any you add from your own use. `python evals/run_all.py` prints the scorecard. See
[docs/EVALS.md](docs/EVALS.md).

## Quick start

```bash
python3 scripts/setup.py                      # detects your OS + GPU; pinned binaries + models, checksum-verified (~11 GB)
.venv/bin/python cli.py doctor                # what was detected, what is missing
.venv/bin/python cli.py smoke                 # text, image, audio, tools: ALL PASS?
.venv/bin/python cli.py dashboard             # http://127.0.0.1:8765
```

On Windows the Python is `.venv\Scripts\python.exe`. For the machines that work today, and the machines that contributors can add, see
[docs/PLATFORMS.md](docs/PLATFORMS.md). Full instructions: [docs/SETUP.md](docs/SETUP.md).

## Layout

```
agent/
  config.py          every knob: tier, paths, ports, thresholds
  platforms/<os>.py  OS plug-ins: screen recording, speech, hotkey, install hints
  accel/<backend>.py GPU plug-ins: llama.cpp build + checksum, server flags, memory probe
  pins.py            model + VAD checksums (same on every machine)
  serving.py         llama-server lifecycle + client
  senses/audio.py    mic, Silero VAD, chunks under 30 s
  senses/screen.py   screen capture + change-based sampling
  senses/video.py    screen recording -> one video part (ffmpeg)
  brain.py           request assembly, context budget, JSON, tool loop
  memory/episodic.py SQLite log + embedding search + junk filter
  tools/             registry, permission tiers, guards, built-in tools
  hotkey.py          Ctrl+Alt/Option+Space -> Task 2
  tasks.py           Task 1 (voice note -> action items), Task 2 (screen help)
  events.py          event log shared by every component and the dashboard
dashboard/           FastAPI + one HTML page; one tab per component, plus tasks and the scorecard
cli.py               every action from the terminal
evals/<component>/   case format (docstring), loader, scorer
evals/synthetic/     generates the synthetic cases (speech, screenshots, tool calls, memories)
evals/cases.py       add your own cases, show what exists
evals/run_all.py     the scorecard
data/evals/          synthetic/ (generated) and user/ (yours), git-ignored
docs/                SETUP, PLATFORMS, EVALS, architecture.md (diagram), VERIFICATION, index.html + session-N-primer.html (slides)
CONTRIBUTING.md      add an OS or a GPU backend with one file
tests/               plug-in contract tests (stdlib only)
scripts/             setup.py, doctor.py, smoke_test.py
requirements/        base.txt (portable pins), lock-<os>.txt, dev.txt (ruff, pyright, complexipy)
```

## Status

Session 1 (the core loop) is complete. Sessions 2 (always on, learning) and 3 (voice, skills, MCP)
come next. [docs/VERIFICATION.md](docs/VERIFICATION.md) lists what we tested and what we found.

## Slides

The slides for each session are at <https://suryakrishna02.github.io/multimodal-agent/>. GitHub Pages publishes the
`docs/` folder of `main`, so everything in that folder is public. To add a session, put its slides in
`docs/session-N-primer.html` and link them from [docs/index.html](docs/index.html).
