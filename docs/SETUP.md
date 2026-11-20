# Setup

Run the whole agent on your own machine. Everything stays on it: the model servers listen on
`127.0.0.1` only, and the agent sends nothing to the internet after the one-time downloads.

## Requirements

| | |
|---|---|
| Machine | Verified: Apple Silicon Mac (M4, 24 GB, macOS 27.0.1). Other OSes and GPUs: see [PLATFORMS.md](PLATFORMS.md) |
| Memory | 16 GB works with the E2B tier. 24 GB or more is better for E4B |
| Disk | about 13 GB (both model tiers, the embedding model and Whisper large-v3-turbo) |
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
- **Whisper large-v3-turbo** in MLX format (1.6 GB), for always-on listening. It is the `stt_pins` of your
  accelerator plug-in, so only an accelerator with a `transcribe` hook downloads it (on a Mac: Metal)
- a `.venv` with the pinned packages (`requirements/lock-<os>.txt`, else `requirements/base.txt`)
- the packages of `requirements/nodeps-<os>.txt`, installed with `--no-deps`. On macOS this is `mlx-whisper`:
  it declares `torch`, but only its weight converter uses it. The lock file pins what `transcribe` needs instead.

Skip what you do not need: `python3 scripts/setup.py --only e2b embed vad whisper`.
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
| Notifications | daemon notifications, sent with `osascript`. Allow them for the app that System Settings > Notifications lists after the first one | the brief is ready, but no notification shows |

## Run

```bash
.venv/bin/python cli.py dashboard     # open http://127.0.0.1:8765 in Chrome
```

The dashboard starts both model servers (this takes about 10 to 20 s the first time). Tabs 1 to 9
follow the session agenda. Session 1 has Serving, Senses, Brain, Memory and Tools. Session 2 adds Always on and Evolve.
Tasks and Scorecard come last.

Press the number keys to switch tabs. The page is light only. The chat sits on the right. Press C to hide it and give
the tabs more room. A bar at the bottom opens the Activity log (the event log). A dot on a tab lights up while that part
of the agent is working.

Each tab opens in a simple view with its main cards only. Press Show details at the top (or the D key) to add the
other cards, the tables, the explanations and the metrics that have no target. The page remembers the choice.

The chat takes text, a picture, your screen or your voice:

- **Picture**: pick a file, paste one (Cmd+V / Ctrl+V), or drop one on the chat panel. It goes to Gemma 4 as an image.
- **Screen**: one screenshot goes with the next message, then the switch turns itself off.
- **Voice**: press Voice, speak, press it again. The dashboard records with the same microphone code as Task 1, keeps
  only the speech (Silero VAD) and turns it into text with `config.STT_ENGINE` (Whisper; Gemma when your accelerator has
  no Whisper). The text is sent as your message, and the bubble says which engine heard it. The Mic switch at the top
  still wins.

Under each reply, "Recalled N memories" opens the memories that search put into the prompt, with their match scores.
"Inspect in Brain" opens that turn in the Brain tab: the context budget, the exact request and the reply. A daily
brief (from the daemon at 18:00, or from the Tasks tab) also arrives in the chat, with buttons for your feedback.

The Brain tab shows only what you sent from the dashboard. Its Show menu adds the requests of the command line, the
evals and the daemon, which write to the same event log. The Evolve tab lists each prompt patch that the validator or
the generality gate refused, with the reason.

The chat has sessions. New chat starts an empty conversation, and the old one stays under History, where you can open
it again or delete it. The model reads only the messages of the open session (the last 6, `config.CHAT_HISTORY_MESSAGES`).
Long-term memory is shared by all sessions: each reply gets the 3 best matches (`config.MEMORY_TOP_K`), and each
`search_memory` call of the model returns 3 more for its own query. The number of stored memories has no limit. Each session is one file in `data/runtime/chats/`, so it survives a restart of the
dashboard. Deleting a session deletes its messages and its pictures. Long-term memories stay.

When a tool call waits for your approval, Approve and Deny appear in the chat, above the message box. With the chat
hidden or on a narrow window, they appear in a banner at the top instead. After you decide, the chat history holds the
outcome, so you can ask the model "did that work?"

If a model server stops unexpectedly, the dashboard restarts it on the current tier. While the daemon runs, the daemon
restarts it instead. `data/runtime/llama-llm.previous.log` or `llama-embed.previous.log` keeps the log of the server
that stopped.

The dashboard does not restart a server that
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

## Always on (Session 2)

The daemon is a separate process. It runs jobs from a queue, on a schedule (`config.SCHEDULE`) and on
triggers, and it records each step in `data/runtime/checkpoints.sqlite`. The dashboard shows it in the Always on tab.

```bash
.venv/bin/python cli.py daemon start             # in the background; the log is data/runtime/daemon.log
.venv/bin/python cli.py daemon status
.venv/bin/python cli.py listen on                # always-on listening: Whisper -> raw memories (the Mic switch pauses it)
.venv/bin/python cli.py replay data/evals/synthetic/task3/conversation.wav --at 09:00    # Task 3 demo input
.venv/bin/python cli.py ask "what did we decide about the offsite venue this morning?"
.venv/bin/python cli.py brief                    # Task 4: today's brief
.venv/bin/python cli.py feedback 1 --bad "too long, and put the deadlines first"
.venv/bin/python cli.py evolve reflect           # patch the prompt from the feedback
.venv/bin/python cli.py evolve versions          # then: evolve diff 1 2, evolve rollback 1
.venv/bin/python cli.py daemon kill              # a crash (SIGKILL): start it again and it resumes the job
.venv/bin/python cli.py daemon stop
```

For the live demo of Task 4, you can run some feedback rounds before the session:
`python cli.py evolve simulate --rounds 3`. The simulated user writes the feedback, and each row is labeled
`simulated`. Then you give one round of your own feedback live.

**Start at login.** `python cli.py daemon install` writes a LaunchAgent on macOS
(`~/Library/LaunchAgents/com.multimodal-agent.daemon.plist`). launchd then starts the daemon when you log in, and
again after a crash. A clean `daemon stop` exits with code 0, so launchd does not restart it.
`python cli.py daemon uninstall` removes the LaunchAgent.

A daemon that launchd starts does not get the Microphone permission of your terminal app, and macOS gives a process
without the permission silent audio instead of an error. So on macOS, `daemon install` builds a small app,
`.cache/autostart/multimodal-agent.app` (it needs `swiftc`, from the Xcode Command Line Tools:
`xcode-select --install`). launchd runs the daemon through this app. The first time it starts, macOS asks whether
"multimodal-agent" can use the microphone. Click **Allow**. After that, always-on listening works in the daemon that
launchd starts (tested on macOS 27 with real speech). If you clicked Don't Allow, turn on "multimodal-agent" in
System Settings > Privacy & Security > Microphone, then run `python cli.py daemon install` again.

The app keeps the permission while its source stays the same. If an update changes
`agent/platforms/macos_launcher.swift`, `daemon install` rebuilds the app and macOS asks again.
`daemon uninstall` leaves the app in `.cache/autostart/`, so a later `daemon install` keeps the permission.

The daemon gets the `PATH` of the shell that runs `daemon install` (launchd's own `PATH` has no Homebrew, so video
requests could not find `ffprobe`). Run `daemon install` again after you install a tool in a new place.

If the microphone still gives only silence, always-on listening emits `senses.ambient.warning` 5 s after it opens
the microphone (`AMBIENT_SILENT_S` and `AMBIENT_SILENT_RMS` in `agent/config.py`). The dashboard shows it, and
`python cli.py daemon status` shows it as `mic_warning`. Stop a daemon that `daemon start` started before you run
`daemon install`, because there is no lock against a second copy.

**Keep the Mac awake.** While the daemon runs, it holds `caffeinate -i`, so idle sleep does not stop it. The hold ends
when the daemon exits, even after a kill. A Mac laptop still sleeps when you close the lid, unless it is on power with
an external display. For hours of listening, keep the lid open and the charger connected.

## Evals

```bash
.venv/bin/python evals/run_all.py          # print the scorecard
.venv/bin/python evals/run_all.py --run    # run all evals on the synthetic cases and your own
.venv/bin/python evals/run_all.py status   # how many cases each eval has
```

There is nothing to record. The first run builds synthetic cases (speech, screenshots, tool calls, memories)
into `data/evals/synthetic/`. To add a case from your own use, see [EVALS.md](EVALS.md).

The soak test of the daemon runs for hours, so record it before the session. The `daemon` eval reads the newest result:

```bash
.venv/bin/python scripts/soak.py --hours 3
```

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
