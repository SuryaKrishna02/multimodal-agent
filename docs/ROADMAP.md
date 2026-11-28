# What comes next: our components next to pi, OpenClaw and Pi Durable

This harness was written from scratch so that each mechanism is visible. Three open projects solve the same problems
for real use. This page maps our components to theirs, so you know where to look after the series. It is content only:
we do not build on any of them.

Sources, read on 2026-11-25: Mario Zechner's post on pi ([mariozechner.at, 2025-11-30](https://www.mariozechner.at/posts/2025-11-30-pi-coding-agent/)),
Earendil's announcement of Pi Durable ([earendil.com, 2026-10-01](https://earendil.com/posts/pi-durable/)) and the
OpenClaw README ([github.com/openclaw/openclaw](https://github.com/openclaw/openclaw)). The pi post is older than
Pi 1.0 (2026-10-01), so pi today may differ from it. Pi Durable says of itself that it "is experimental, and the
API might still change". A cell that says "not stated" means the source did not say, not that the feature is missing.

## The three projects in one line each

| Project | What it is | Language, license |
| ------- | ---------- | ----------------- |
| **pi** | A minimal terminal coding agent. Four tools by default (read, write, edit, bash); system prompt and tool definitions under 1,000 tokens. Packages: `pi-ai` (one API for many model providers), `pi-agent-core` (the agent loop), `pi-tui`, `pi-coding-agent`. | TypeScript, MIT |
| **Pi Durable** | A framework for long-running agents that survive crashes, from the team behind pi (Earendil). It "does not replace the Pi coding agent"; it shares code such as `pi-ai`. Storage: memory, SQLite or JSONL. | TypeScript, experimental |
| **OpenClaw** | A personal assistant that runs on your own computer and talks to you in your chat apps (WhatsApp, Telegram, Slack, iMessage and others), with companion apps. A local Gateway is the control plane. Models and agent harnesses are plug-ins. | Node.js, MIT (OpenClaw Foundation) |

## Component by component

| Our component (session) | Ours | pi | Pi Durable | OpenClaw |
| ----------------------- | ---- | -- | ---------- | -------- |
| Model serving (1) | `llama-server` with Gemma 4 on this machine, `agent/serving.py` | `pi-ai`: many providers behind one API | `pi-ai` | Model providers are plug-ins, hosted or local |
| Senses (1) | Microphone with VAD, screenshots, screen video | Not its job (a coding agent) | Not stated | Companion apps and "nodes" add voice, camera and screen |
| Brain and tool loop (1) | `agent/brain.py`: context budget, JSON schema, at most 4 steps | `pi-agent-core` | Each run step is a task with a checkpoint | The agent harness is a plug-in |
| Tools and permissions (1) | Every call passes `tools.check()`: schema, tier, guard | "YOLO by default": no permission prompts | Hooks before and after each tool (`beforeTool`, `afterTool`) | Tools run on the host for the main session unless sandboxing is set up; unknown senders must pair first |
| Memory (1, 2) | SQLite with vectors, consolidation into summaries and facts | Context files (`AGENTS.md`), sessions you can resume and branch | Transcripts, typed documents and memos in the same atomic commits; background compaction | "State, memory, and credentials live on your hardware" |
| Always on and crash recovery (2) | Daemon with SQLite checkpoints: intent, execution, result per step | Not its job | The core idea: a new process opens the same storage and goes on from the last checkpoint; a tool reruns only if it is marked replay-safe | The Gateway runs all the time |
| Self-evolution (2) | Versioned prompt patches, locked blocks, a generality gate | Not stated | Extensions can be replaced while the agent runs | Not stated |
| Voice (3) | Kokoro TTS in the dashboard tab, the "Hey Jarvis" wake word | Not its job | Not stated | Voice through companion apps and nodes |
| Proactive output (3) | `agent/proactive.py`: feed, notification, speech, cool-down | Not its job | Not stated | Not stated in the part of the README we read |
| Native skills (3) | `agent/skills/<name>/manifest.json`: schema, tier, effects | Custom slash commands as markdown templates; the post describes no skill system | Extensions: "a named bundle of system prompt sections, tools, hooks, and tasks" | Skills and plug-ins, shared through the ClawHub registry |
| MCP (3) | `agent/mcp_client.py`: stdio and HTTP, labels, `${NAME}` secrets, per-tool tiers | "pi does not and will not support MCP" (context cost); it uses command-line tools with READMEs | Not stated | Not stated in the part of the README we read |
| Evals (1 to 3) | One eval per component, synthetic and your own cases, a scorecard | Not stated | A conformance suite for storage back ends | Not stated |

## What we would take from each

- **From pi: a small context.** Its whole system prompt and tool list stay under 1,000 tokens. Our tool selection
  (`tools.select()`, the `mcp_select` eval) attacks the same problem from the other side: many tools, but only the
  top 5 in each request.
- **From Pi Durable: finer recovery.** Our checkpoints are per job step. Pi Durable also stores the intent of each
  tool call before it runs and replays only tools marked safe. Our steps must be safe to run twice instead. Per-tool
  replay flags would be the next step for `agent/checkpoint.py`.
- **From OpenClaw: reach.** Our agent talks through one dashboard and one machine. OpenClaw meets the user in chat apps
  and on phones. A "channel" plug-in folder, like `agent/platforms/`, would be the way to add that here.

## Roadmap items that this series leaves out

These were out of scope from the start (see the brief): agent-written skills, on-device LoRA fine-tuning, phone
access and video input. The open problems after Session 3:

- Wake-word false wakes measured on hours of your own audio, not only on openWakeWord's validation set.
- A streamed reply: speak the first sentence while Gemma still writes the rest (`config.LATENCY_BUDGET`).
- Echo cancellation, so plain speech (not only the wake word) can interrupt the agent.
- Linux and Windows plug-ins, and CUDA (see [PLATFORMS.md](PLATFORMS.md)): contributions welcome.
