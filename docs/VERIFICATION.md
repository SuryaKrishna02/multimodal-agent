# Verification report (2026-10-26)

This report covers the checks before Session 1, on the target machine: Apple M4, 24 GB, macOS 27.0.1.

## Pinned versions

| Item | Version | Source |
|---|---|---|
| llama.cpp | **b11960** (= v0.8.0, commit `8642541b2`) | official `llama-b11960-bin-macos-arm64.tar.gz`, sha256 `58e5a5e1…1e3c` |
| Gemma 4 E4B | `gemma-4-E4B-it-Q4_0.gguf` + `mmproj-gemma-4-E4B-it-Q8_0.gguf` | `ggml-org/gemma-4-E4B-it-GGUF` @ `a69f018` |
| Gemma 4 E2B | `gemma-4-E2B-it-Q4_0.gguf` + `mmproj-gemma-4-E2B-it-Q8_0.gguf` | `ggml-org/gemma-4-E2B-it-GGUF` @ `6938c58` |
| Embeddings | `embeddinggemma-2-Q8_0.gguf` + `mmproj-embeddinggemma-2-Q8_0.gguf` | `ggml-org/embeddinggemma-2-GGUF` @ `58faf90` |
| Silero VAD | `silero_vad.onnx` | `snakers4/silero-vad` tag v6.2.3 |
| Python | 3.13.3, packages in `requirements/lock-macos.txt` | |
| ffmpeg + ffprobe | 9.0.2 (optional, for video) | Homebrew `brew install ffmpeg` |

All checksums are in `agent/pins.py` (models) and `agent/accel/metal.py` (llama.cpp). The Q4_0 weights come from Google's QAT checkpoints
(`google/gemma-4-E4B-it-qat-q4_0-unquantized`), so Q4_0 is the intended quantization, not a shortcut.

## Highest risk: Gemma 4 audio through llama-server on Metal — WORKS

- llama.cpp merged audio support (conformer encoder) in PR #21421 (2026-04-12). Issue #21868, the server
  bug that returned HTTP 500 for `input_audio`, closed on 2026-04-13.
- On b11960, `/v1/chat/completions` accepts `{"type": "input_audio", "input_audio": {"data": <base64 wav>}}`
  on both E4B and E2B. `/props` reports `audio: true`. The server prints
  "audio input is in experimental stage". This message is normal.
- `scripts/smoke_test.py` checks text, image, audio, tool calls and video in about 8 s on E4B and 5 s on E2B. All pass on both tiers.

## Measured facts (and corrections to the brief)

| Brief said | Measured / found |
|---|---|
| Audio clips max 30 s | Confirmed. llama.cpp **splits longer clips every 30 s itself** (`tools/mtmd/mtmd-audio.cpp`, Gemma 4 preprocessor), even mid-word. The model transcribed a 35 s clip in full, so there is no hard error. We still chunk at pauses, under 28 s. |
| ~25 tokens/s of audio | Confirmed: a 35.5 s clip used 906 prompt tokens, about 25/s plus the text. |
| Video not supported in llama.cpp | **Outdated**: b11960 accepts `input_video` for Gemma 4, so the harness now supports it (see "Video input" below). |
| Function calling is weak | Simple cases work on both tiers. An earlier check found a quirk: **with thinking off, the reply after a tool result is empty** (same as ollama/ollama#19044). `brain.respond()` turns thinking on for that one step. A one-tool check on b11960 returned text with thinking off too, but one case does not prove the quirk is gone, so the workaround stays. |
| Measure peak memory | Peak resident memory of the server: E4B 5.9 GB, E2B 3.7 GB, embeddings 1.1 GB. System total during the serving eval: 13.4 GB (E4B) and 12.1 GB (E2B) of 24 GB. |

Speed on the M4 (serving eval on b11960, `evals/results/<tier>/serving.json`):

| | E4B | E2B |
|---|---|---|
| Prompt processing | 328 tok/s | 627 tok/s |
| Generation | 29.3 tok/s | 53.7 tok/s |
| Time to first token, short prompt | 0.14 s | 0.08 s |
| Time to first token, ~2.4k-token prompt | **7.4 s** | 3.9 s |

Prompt processing is what costs time. A 30 s clip plus a screenshot (~1.1k tokens) takes ~3.4 s before the
first word on E4B. Keep the system prompt stable so llama.cpp can reuse its cached prefix.

## Checks on the current build (b11960 = llama.cpp v0.8.0, 2026-10-30)

The GitHub release `v0.8.0` has no binaries. Its only asset, `nightly-tag.txt`, names the build: `b11960`.
The tag `v0.8.0` and the release `b11960` point to the same commit, `8642541b255b`. The sha256 of the
downloaded archive matches the digest that GitHub publishes.

- Every `llama-server` flag in `agent/serving.py` exists in `--help`: `--mmproj`, `--host`, `--port`,
  `--no-webui`, `--metrics`, `--video-fps`, `--video-ffmpeg-dir`, `--embeddings`, `-c`, `-np`.
- `cli.py smoke` passes text, image, audio, tools and video on both tiers.
- `chat_template_kwargs.enable_thinking` works: with it on, the reply carries `reasoning_content`; with it off, it does not.
- The embedding server returns 768 dimensions. Similar sentences score 0.95 and an unrelated one 0.65.
- Evals: `serving` ran on both tiers (table above). The other evals ran on the **synthetic** cases
  ([EVALS.md](EVALS.md)) on 2026-10-30, E4B tier, M4 / macOS 27.0.1, with `python cli.py evals --run`.
  Every metric that has a threshold passed:

  | Eval | Cases | Result |
  |---|---|---|
  | tools | 28 | success 1.0, unsafe blocked 1.0, policy accuracy 1.0 (no model) |
  | tool_calls | 16 | E4B and E2B: tool accuracy 1.0, exact 1.0 |
  | memory | 35 | recall@3 0.958, junk stored 0.0, good rejected 0.0 (no chat model) |
  | asr | 10 | E4B and E2B: WER 0.044 (errors on the noisy and quiet clips; 4 of 10 transcripts differ between tiers) |
  | vad | 5 | false trigger 0.007, missed speech 0.0 |
  | screen | 4 | key-moment recall 1.0, kept ratio 0.065 |
  | structured | 10 | E4B and E2B: valid 1.0 (constrained and prompt-only) |
  | task1 | 5 | E4B and E2B: precision 1.0, recall 1.0 |
  | task2 | 10 | E4B: correct 1.0 (images and videos). E2B: correct 0.9 (images 0.875, videos 1.0) |

  The one memory miss: the query "what does Priya need from me?" returned the slides note.
  It did not return the action item "Email Priya the slides".
  The one E2B task2 miss was the failing-pytest screenshot (`test-failure.png`). E2B flagged the problem
  ("One test in `test_math.py` failed"). It did not name the failing test `test_add`, so the keyword check failed. These scores come from synthetic cases and are optimistic (see "Not verified yet").

## Video input (added after Session 1 was first committed)

One `{"type": "input_video", "input_video": {"data": <base64 mp4/mov>}}` part. Inside llama-server (`tools/mtmd/mtmd-helper.cpp`), this happens:

1. It runs **ffmpeg + ffprobe** (from PATH or `--video-ffmpeg-dir`).
2. It takes `--video-fps` frames per second (default 4, and we use 1).
3. It adds a text timestamp every 5 s.
4. It sends every frame to Gemma 4 as an ordinary image.

 There is no
Gemma-specific video encoder, and it **ignores the audio track**. Google's guidance for Gemma 4:
up to 60 s at 1 fps, with low per-frame budgets.

Measured on b11960 with a 12 s synthetic screen recording (traceback at 5 s, dialog at 9 s). The Retina row used a 3 s clip.
The table shows cost only, not answer quality:

| Tier | fps | Frame size | Prompt tokens | Time |
|---|---|---|---|---|
| E4B | 1 | 1280x720 | 6,574 | 43.6 s |
| E4B | 1 | 960x540 | 3,614 | 25.5 s |
| E4B | 1 | **640x360** | **1,758** | **17.9 s** |
| E4B | 4 | 640x360 | 6,740 | 52.8 s |
| E2B | 1 | 640x360 | 1,758 | 12.9 s |
| E2B | 1 | 2940x1912 (Retina, native), 3 s clip | 3,237 | 30.6 s |

Cost grows with the area of a frame and the number of frames.
So the harness records with macOS `screencapture -v`, shrinks to 640 px wide with ffmpeg, and runs
the server at 1 fps. Smaller frames lose some fine terminal text: test on your own recordings.

 (each is a good teaching moment)

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

Text, image and audio share one 768-d space. In a quick check, the error screenshot retrieved the
matching text note (0.74). The voice note retrieved "Email Priya the slides" (0.73). A spoken question
retrieved the right text memory (0.78). This makes "multimodal memory" in Session 2 cheap to build.

## Not verified yet

- Real use: the evals run on **synthetic** cases (macOS `say` voices, rendered screenshots, hand-written tool
  calls and memories). Text-to-speech is cleaner than a real voice, so ASR, VAD and Task 1 scores are optimistic.
  Add your own cases with `python cli.py evals add` for numbers that reflect your voice and your screen.
- Mic recording and the global hotkey need permissions that only you can grant. Please try
  `cli.py note` and the hotkey once you grant the permissions.
- Memory use with a Google Meet call running alongside.
