# Verification report (2026-10-26)

What was checked before building Session 1, on the target machine: Apple M4, 24 GB, macOS 27.0.1.

## Pinned versions

| Item | Version | Source |
|---|---|---|
| llama.cpp | **b11669** (= v0.7.0, commit `620d2225c`) | official `llama-b11669-bin-macos-arm64.tar.gz`, sha256 `dbed5549…13fd` |
| Gemma 4 E4B | `gemma-4-E4B-it-Q4_0.gguf` + `mmproj-gemma-4-E4B-it-Q8_0.gguf` | `ggml-org/gemma-4-E4B-it-GGUF` @ `a69f018` |
| Gemma 4 E2B | `gemma-4-E2B-it-Q4_0.gguf` + `mmproj-gemma-4-E2B-it-Q8_0.gguf` | `ggml-org/gemma-4-E2B-it-GGUF` @ `6938c58` |
| Embeddings | `embeddinggemma-2-Q8_0.gguf` + `mmproj-embeddinggemma-2-Q8_0.gguf` | `ggml-org/embeddinggemma-2-GGUF` @ `58faf90` |
| Silero VAD | `silero_vad.onnx` | `snakers4/silero-vad` tag v6.2.3 |
| Python | 3.13.3, packages in `requirements.lock` | |

All checksums are in `scripts/setup.sh`. The Q4_0 weights are converted from Google's QAT checkpoints
(`google/gemma-4-E4B-it-qat-q4_0-unquantized`), so Q4_0 is the intended quantization, not a shortcut.

## Highest risk: Gemma 4 audio through llama-server on Metal — WORKS

- Audio support (conformer encoder) was merged in llama.cpp PR #21421 (2026-04-12). The server
  bug that returned HTTP 500 for `input_audio` (issue #21868) was closed on 2026-04-13.
- On b11669, `/v1/chat/completions` accepts `{"type": "input_audio", "input_audio": {"data": <base64 wav>}}`
  on both E4B and E2B. `/props` reports `audio: true`. The server prints
  "audio input is in experimental stage", which is expected.
- `scripts/smoke_test.py` checks text, image, audio and tool calls in about 5 s. All pass on both tiers.

## Measured facts (and corrections to the brief)

| Brief said | Measured / found |
|---|---|
| Audio clips max 30 s | Confirmed. llama.cpp **splits longer clips every 30 s itself** (`tools/mtmd/mtmd-audio.cpp`, Gemma 4 preprocessor), even mid-word. A 35 s clip was fully transcribed, so there is no hard error. We still chunk at pauses, under 28 s. |
| ~25 tokens/s of audio | Confirmed: a 35.5 s clip used 906 prompt tokens, about 25/s plus the text. |
| Video not supported in llama.cpp | **Partly outdated**: b11669 reports `video: true` for Gemma 4 and the docs list video input. Out of scope here, but worth a sentence in the talk. |
| Function calling is weak | Simple cases work on both tiers. Found a real quirk: **with thinking off, the reply after a tool result is empty** (same as ollama/ollama#19044). Fix in `brain.respond()`: thinking on for that one step. |
| Measure peak memory | E4B server 4–7 GB resident (varies with paging of memory-mapped weights), E2B ~3.8 GB, embeddings ~1 GB. System total during evals with the dashboard and Chrome open: ~12–14.6 GB of 24 GB. |

Speed on the M4 (from `llama-bench` and the serving eval):

| | E4B | E2B |
|---|---|---|
| Prompt processing (2k tokens) | ~290 tok/s | ~490–600 tok/s |
| Generation | ~29 tok/s | ~53 tok/s |
| Time to first token, short prompt | 0.14 s | 0.09 s |
| Time to first token, ~2.4k-token prompt | **8.2 s** | 5.0 s |

The base M4 GPU is the limit for prompt processing: Metal is ~3x faster than CPU, and flash attention and
larger micro-batches did not help. So a 30 s clip plus a screenshot (~1.1k tokens) costs ~4 s before the
first word on E4B. Keep the system prompt stable so llama.cpp can reuse its cached prefix.

## Other things found while building (each is a good teaching moment)

- **Thinking is on by default** in Gemma 4. With a 300-token limit, structured output and tool calls came
  back empty because the thinking used up the budget. `config.THINKING = False` by default.
- **ASR slip**: E4B heard "oat milk" from the macOS `say` voice as "old milk". This is why the smoke test
  checks that paths work, and the evals measure accuracy.
- **Tool selection by embedding**: "Remember that the demo is Wednesday at 4pm" picked `get_time` and
  `write_file` over `save_note`. Embedding example requests with each tool fixed it.
- **A pending tool is not a done tool**: the model said "I have recorded that" while `write_file` waited
  for approval. The tool result now says "NOT done yet" explicitly.
- **Prompt cache hides latency**: repeated benchmark prompts reported 0.09 s to first token from the
  cache. The serving eval sends `cache_prompt: false`.
- **The screen sampler threshold** of 2% missed a traceback appearing (~1% of pixels). It is now 0.5%,
  and the eval shows the trade-off (`kept_ratio`).
- **Context budget** missed the tool declarations (~300 tokens). Fixed. The Brain tab shows the estimate
  next to llama-server's actual count.
- **Screen capture permission**: without Screen Recording permission, macOS returns a flat image instead
  of an error. `screen.capture()` detects this and warns.
- **Global hotkey** needs Accessibility + Input Monitoring for the terminal app. `hotkey.py` checks
  `AXIsProcessTrusted()` and warns in the event log.

## Embeddings (open decision 3): EmbeddingGemma-2 via a second llama-server

Text, image and audio share one 768-d space. On a quick check: the error screenshot retrieved the
matching text note (0.74), the voice note retrieved "Email Priya the slides" (0.73), and a spoken question
retrieved the right text memory (0.78). This makes "multimodal memory" in Session 2 cheap to build.

## Not verified yet

- Real recordings: every eval was run end to end on a **synthetic** set (macOS `say` voices, rendered
  screenshots) to check the code. Those results were deleted. Real numbers need your gold sets.
- Mic recording and the global hotkey need permissions that only you can grant. Please try
  `cli.py note` and the hotkey once permissions are set.
- Memory use with a Google Meet call running alongside.
