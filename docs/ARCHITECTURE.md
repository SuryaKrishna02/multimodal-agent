# Harness map

The diagram source below renders on GitHub, or paste it into https://mermaid.live to export a
slide image. Every box is one small file. The numbers are the Session 1 agenda segments.

```mermaid
flowchart LR
  subgraph IN["Inputs"]
    MIC["🎙 mic"]
    SCR["🖥 screen"]
    KEY["⌨ hotkey / CLI / dashboard"]
  end

  subgraph SENSES["2 · Senses  agent/senses/"]
    AUD["audio.py<br/>capture → Silero VAD → chunks &lt; 30 s"]
    SCN["screen.py<br/>capture → downscale → change sampler"]
  end

  subgraph BRAIN["3 · Brain  agent/brain.py"]
    ASM["assemble<br/>interleave text · image · audio"]
    BUD["context budget"]
    OUT["structured output<br/>JSON schema → grammar"]
    TC["tool loop<br/>validate · retry"]
  end

  subgraph SERVE["1 · Serving  agent/serving.py"]
    LLM["llama-server :9931<br/>Gemma 4 E4B / E2B"]
    EMB["llama-server :9932<br/>EmbeddingGemma-2"]
  end

  subgraph MEM["4 · Memory  agent/memory/episodic.py"]
    DB[("SQLite<br/>text + 768-d vectors")]
  end

  subgraph TOOLS["5 · Tools  agent/tools/"]
    REG["registry + tiers<br/>safe · confirm · blocked + guards"]
  end

  subgraph TASKS["6 · Tasks  agent/tasks.py"]
    T1["Task 1<br/>voice note → action items"]
    T2["Task 2<br/>screen → explain + fix"]
  end

  EVAL["7 · evals/  gold set · metric · threshold · script → scorecard"]

  MIC --> AUD --> ASM
  SCR --> SCN --> ASM
  KEY --> T1 & T2
  T1 --> AUD
  T2 --> SCN
  ASM --> BUD --> OUT & TC
  OUT --> LLM
  TC --> LLM
  TC <--> REG
  REG --> DB
  BRAIN <-- "top 3 memories" --> DB
  DB --> EMB
  T1 & T2 --> DB
  EVAL -. measures .-> SENSES & BRAIN & SERVE & MEM & TOOLS & TASKS
```

## Where each segment lives

| Segment | Files | Eval (folder in `evals/`) |
|---|---|---|
| 0 Harness map, eval pattern | this file, `evals/common.py` | n/a |
| 1 Serving | `agent/config.py`, `agent/serving.py` | `serving` |
| 2 Senses | `agent/senses/audio.py`, `agent/senses/screen.py` | `vad`, `asr`, `screen` |
| 3 Brain | `agent/brain.py` | `structured`, `tool_calls` |
| 4 Memory v1 | `agent/memory/episodic.py` | `memory` |
| 5 Tools, CLI, hotkey | `agent/tools/`, `cli.py`, `agent/hotkey.py` | `tools` |
| 6 Live tasks | `agent/tasks.py` | `task1`, `task2` (end-to-end) |
| 7 Scorecard | `evals/run_all.py`, dashboard tab 7 | n/a |

Cross-cutting: `agent/events.py` (every component logs here, and the dashboard shows it) and
`dashboard/` (FastAPI + one HTML page).

## Data flow for one request

1. A sense produces raw input: a 16 kHz float array or a PIL image.
2. `brain.part()` turns it into an OpenAI-style content part (base64 WAV or PNG).
3. `brain.fit_budget()` adds the system prompt and the top 3 memories, estimates tokens
   (audio 25/s, image ~280, text ~chars/4) and drops old history if it does not fit.
4. `brain.send()` writes the exact request to `data/runtime/last_request.json`, logs it to the
   event log (the dashboard's Brain tab), and posts it to llama-server.
5. A reply with `tool_calls` goes through `tools.run()`: schema check → tier → guard → execute.
   The result goes back to the model as a `tool` message, and the loop runs again.
