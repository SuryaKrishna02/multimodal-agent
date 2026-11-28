# Setup

Run the whole agent on your own machine. Everything stays on it: the model servers listen on
`127.0.0.1` only, and the agent sends nothing to the internet after the one-time downloads.

## Requirements

| | |
|---|---|
| Machine | Verified: Apple Silicon Mac (M4, 24 GB, macOS 27.0.1). Other OSes and GPUs: see [PLATFORMS.md](PLATFORMS.md) |
| Memory | 16 GB works with the E2B tier. 24 GB or more is better for E4B |
| Disk | about 13 GB (both model tiers, the embedding model, Whisper large-v3-turbo, Kokoro and the wake-word models) |
| Python | 3.10 or newer (tested with 3.13.3) |
| Optional | `ffmpeg` + `ffprobe` for screen-video input and the two synthetic video cases of the task2 eval (the doctor prints the install command for your OS) |
| Optional | [Node.js](https://nodejs.org) for MCP servers started with `npx`, and [uv](https://docs.astral.sh/uv/) for those started with `uvx` (Session 3). The reminders server of this repo needs neither |

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
- **Kokoro-82M** (int8 ONNX, 92 MB) and its 54 voices (28 MB), the agent's voice (Session 3), group `tts`
- openWakeWord's two frozen feature models and its pretrained "hey jarvis" classifier (3.7 MB, release v0.5.1),
  group `wake`
- **Whisper large-v3-turbo** in MLX format (1.6 GB), for always-on listening. It is the `stt_pins` of your
  accelerator plug-in, so only an accelerator with a `transcribe` hook downloads it (on a Mac: Metal)
- a `.venv` with the pinned packages (`requirements/lock-<os>.txt`, else `requirements/base.txt`)
- the packages of `requirements/nodeps-<os>.txt`, installed with `--no-deps`. On macOS this is `mlx-whisper`:
  it declares `torch`, but only its weight converter uses it. The lock file pins what `transcribe` needs instead.

Skip what you do not need: `python3 scripts/setup.py --only e2b embed vad whisper tts wake`.
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

One command starts everything: the dashboard, the daemon (see [Always on](#always-on-session-2)) and both model
servers (this takes about 10 to 20 s the first time). Ctrl-C stops what the command started: first the daemon (it
finishes its current job; a second Ctrl-C kills it), then the model servers.

- If a daemon already runs (the login item, or `daemon start`), the dashboard uses it and leaves it running, with
  its model servers, after Ctrl-C.
- A model server that already ran (another dashboard, or `python cli.py serve`) stays up after Ctrl-C.
- `python cli.py dashboard --no-daemon` starts only the dashboard and the model servers, as in Session 1. Ctrl-C
  stops the model servers too, unless a daemon uses them.

The page has three views. Switch with the buttons at the top, or with the keys 1, 2 and 3:

1. **Agent** (the start page): the conversation, with a large message box, and a **Tasks** list beside it. Press a
   task to open its controls and its last result in place; "All tasks" goes back to the list. A task that records,
   watches or runs shows that on its button. Below the tasks, **Connect** has the MCP servers and Skills panes (see
   [Voice, skills and MCP](#voice-skills-and-mcp-session-3)). The Jarvis orb, a blue hologram of light streaks drawn
   on a canvas, turns slowly while idle, turns fast while the agent thinks, and pulses while it listens or speaks.
   Type `/` in the message box for commands, such as `/skill`.
2. **Under the hood**: one section per component, in a menu on the left: Serving, Senses, Brain, Memory, Tools, Always
   on, Evolve and Voice. A dot beside a section lights up while that part of the agent works, and a dot on the Under
   the hood button lights up when any part does. The Activity bar at the bottom opens the event log.
3. **Benchmarks**: the speed test of the language model and the scorecard of the evals.

The model menu at the top shows if both model servers are up; open it to see their memory and to switch between E4B
and E2B. The page is light only.

Under the hood and Benchmarks open in a simple view with the main cards only. Press Show details (or the D key) to add
the other cards, the tables, the explanations and the metrics that have no target. The page remembers the choice.

The message box takes text, a picture, your screen or your voice:

- **Picture**: pick a file, paste one (Cmd+V / Ctrl+V), or drop one on the conversation. It goes to Gemma 4 as an image.
- **Screen**: one screenshot goes with the next message, then the switch turns itself off.
- **Dictate**: press Dictate, speak, press it again. The dashboard records with the same microphone code as Task 1,
  keeps only the speech (Silero VAD) and turns it into text with `config.STT_ENGINE` (Whisper; Gemma when your
  accelerator has no Whisper). The text is sent as your message, and the bubble says which engine heard it. The Mic
  switch at the top still wins.
- **Talk** (key T): a spoken turn. The agent records until you pause, answers in the conversation and speaks the reply.
  "Wake word" above the conversation does the same when you say "Hey Jarvis".
- **Conversation** (on by default): after a reply has played, just keep talking. The header says "Your turn…" and the
  agent listens for `config.VOICE_FOLLOW_UP_S` (8 s) without the wake word. Silence ends the conversation, and the
  next request needs "Hey Jarvis" or Talk again. Turn it off for a call where the microphone hears other people.

Under each reply, "Recalled N memories" opens the memories that search put into the prompt, with their match scores.
"Inspect in Brain" opens that turn in Under the hood, Brain: the context budget, the exact request and the reply. A
daily brief (from the daemon at 18:00, or from the Daily brief task) also arrives in the conversation, with buttons for
your feedback.

Brain shows only what you sent from the dashboard. Its Show menu adds the requests of the command line, the evals and
the daemon, which write to the same event log. Evolve lists each prompt patch that the validator or the generality
gate refused, with the reason.

The chat has sessions. New chat starts an empty conversation, and the old one stays under History, where you can open
it again or delete it. The model reads only the messages of the open session (the last 6, `config.CHAT_HISTORY_MESSAGES`).
Long-term memory is shared by all sessions: each reply gets the 3 best matches (`config.MEMORY_TOP_K`), and each
`search_memory` call of the model returns 3 more for its own query. The number of stored memories has no limit. Each session is one file in `data/runtime/chats/`, so it survives a restart of the
dashboard. Deleting a session deletes its messages and its pictures. Long-term memories stay.

When a tool call waits for your approval, Approve and Deny appear in the conversation, above the message box. On the
other views, or on a narrow window, they appear in a banner at the top instead. After you decide, the chat history holds the
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
triggers, and it records each step in `data/runtime/checkpoints.sqlite`. The dashboard shows it under the hood, in Always on.
`python cli.py dashboard` starts it for you. To run it without the dashboard:

```bash
.venv/bin/python cli.py daemon start             # in the background; the log is data/runtime/daemon.log
.venv/bin/python cli.py daemon status
.venv/bin/python cli.py listen on                # always-on listening: Whisper -> raw memories (the Mic switch pauses it)
.venv/bin/python cli.py listen on --for 45m      # ...and stop after 45 minutes (also 2h, 1h30m)
.venv/bin/python cli.py listen on --window 09:00-17:00   # ...only between these local times each day
.venv/bin/python cli.py replay data/evals/synthetic/task3/conversation.wav --at 09:00    # Task 3 demo input
.venv/bin/python cli.py ask "what did we decide about the offsite venue this morning?"
.venv/bin/python cli.py brief                    # Task 4: today's brief
.venv/bin/python cli.py feedback 1 --bad "too long, and put the deadlines first"
.venv/bin/python cli.py evolve reflect           # patch the prompt from the feedback
.venv/bin/python cli.py evolve versions          # then: evolve diff 1 2, evolve rollback 1
.venv/bin/python cli.py daemon kill              # a crash (SIGKILL): start it again and it resumes the job
.venv/bin/python cli.py daemon stop
```

**Timed listening.** `--for` stores an end time; when it passes, the daemon turns the switch off. `--window` limits
listening to a daily window of local time (an end before the start crosses midnight); the switch stays on and the
daemon listens again when the window opens. Without `--window`, `LISTEN_WINDOW` in `agent/config.py` applies (None =
any time); `--window any` turns it off for this switch. Always on (under the hood) has the same two options next to the
listening switch. The Mic switch at the top still pauses the microphone over all of these.

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
`python cli.py daemon status` shows it as `mic_warning`.

**One daemon at a time.** A daemon that starts while another one still writes its heartbeat exits at once with code 0
and emits `daemon.refused`. So if `daemon start` started a daemon, the copy of the login item exits, and
`daemon install` says so: stop the first daemon, then run `daemon install` again.

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

## Voice, skills and MCP (Session 3)

**The voice plays in the dashboard tab.** Click **Voice off** at the top once after each page load (Chrome plays sound
only after a click). In Google Meet, present the dashboard as a **Chrome tab** and turn on **Also share tab audio**: Meet
shares only the audio of the presented tab. Without a dashboard tab, the agent speaks through this machine's speakers.

```bash
.venv/bin/python cli.py say "Hello, I am Jarvis."     # speak; --local for this machine's speakers
.venv/bin/python cli.py wake test                     # the wake word, live: say "Hey Jarvis"
```

- **Talk to it:** on the Agent page, press **Talk** (key T) or turn on **Wake word** and say "Hey Jarvis".
  The request and the reply appear in the conversation. **Stop speaking** (key S) silences it.
- **The wake word** is openWakeWord's pretrained "hey jarvis" model: setup downloads it, nothing is trained. A real voice
  often scores under its threshold, so a lower "maybe" score sends the last 2 s to Whisper, which must hear "Jarvis"
  (about 1 s more before it wakes). `cli.py wake test` prints each wake, each "maybe" score and what Whisper heard. Add
  recordings to the wake eval (docs/EVALS.md).
- **Conversation:** once you are talking, the switch beside "Wake word" lets you go on without the wake word (see
  the Agent page above).
- **MCP:** on the Agent page, below the tasks, open **MCP servers** and **+ Add a server**. Pick **Local command**
  (the command, one argument per row, environment variables, an optional working folder) or **Remote URL** (the URL
  and headers). An environment variable or a header takes its value from a variable of this machine's environment,
  which the form writes as `"${NAME}"`: the value itself is never saved. "See the JSON" shows the entry the form
  builds. Edit and Remove are on each server's card. **Paste JSON** still takes a standard `mcpServers` config (the
  JSON that Claude Desktop, Cursor or VS Code use), and it replaces the whole list. Then connect a server, tick the
  tools the model may use, and set each tool's tier. Every tool starts as `confirm`. Set `NAME` in the environment of
  the process that connects (the dashboard, or the daemon).
- **Task 6, a voice-set reminder:** the reminders server of this repo is in the list by default. Connect it, tick
  `add_reminder`, set it to `safe`, then say "Hey Jarvis, remind me to call Mike in two minutes". The daemon checks
  for due reminders every 15 s and speaks them; run `python cli.py daemon install` once after an update, so the
  launchd daemon runs the new code. The task "Spoken reminder" on the Agent page checks each of these steps live and
  links to the place that fixes a missing one.
- **Task 5, proactive alerts:** on the Agent page, open the task "Speak up when a test fails" and turn on
  **Watching**. When a test run fails on screen, the agent says so out loud, shows a notification and writes it to the
  feed. The dashboard runs it, because the terminal that
  started the dashboard holds the Screen Recording permission.
- **Native skills** are folders in `agent/skills/` with a `manifest.json`. They load at start; see
  `agent/skills/__init__.py` for the format and the rule that a skill which writes, runs code or uses the network
  cannot be `safe`. On the Agent page, below the tasks, **Skills** lists each folder with its tier and effects, says
  why a skill did not load, and puts an example into the message box when you press it. After you add or edit a
  folder, press **Reload skills**: no restart needed. "Write one by hand" there shows the folder layout and a manifest to
  copy.
- **Teach a new skill:** type `/skill` and a description in the message box (for example `/skill convert between
  dollars, euros and pounds at rates I give`), or use "Teach Jarvis a new skill" in the Skills pane. The local model
  writes a draft (about 30 s on E4B): name, description, examples and `skill.py`. The draft shows in the conversation
  and nothing is written yet. Read the code, edit it if you want, and pick the tier: a draft always starts at
  `confirm`, and the model never picks a tier. "What the code does" comes from reading the code's imports and calls,
  so a skill that writes files, runs programs or uses the network cannot be `safe`. This check is a hint, not a
  sandbox: read the code before you press **Create skill**. Create checks the draft again, writes
  `agent/skills/<name>/`, and loads it at once. If the code fails to load, the folder is removed again.
- **What comes next:** [ROADMAP.md](ROADMAP.md) maps these components to pi, OpenClaw and Pi Durable.
