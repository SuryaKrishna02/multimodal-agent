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
| Measure peak memory | Peak resident memory of the server: E4B 5.9 GB, E2B 3.7 GB, embeddings 1.5 GB. System total during the serving eval: 13.4 GB (E4B) and 12.1 GB (E2B) of 24 GB. |

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
  `--no-webui`, `--video-fps`, `--video-ffmpeg-dir`, `--embeddings`, `-c`, `-b`, `-ub`, `-np`.
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
  | memory | 35 | recall@3 1.0, junk stored 0.0, good rejected 0.0 (no chat model) |
  | asr | 10 | E4B: WER 0.044. E2B: WER 0.053 (5 of 10 transcripts differ between tiers) |
  | vad | 5 | false trigger 0.008, missed speech 0.0 |
  | screen | 4 | key-moment recall 1.0, kept ratio 0.065 |
  | structured | 10 | E4B and E2B: valid 1.0 (constrained and prompt-only) |
  | task1 | 5 | E4B and E2B: precision 1.0, recall 1.0 |
  | task2 | 10 | E4B: correct 1.0 (images and videos). E2B: correct 0.9 (images 0.875, videos 1.0) |

  On 2026-10-30, the name in the cases changed to "John Doe", and these evals ran again. **Recall@3 went from 0.958 to 1.0 with no code change.** The old name made one query miss its action item, and the new name does not. With so few cases, one word can move a score. Read a small change here as noise, not as an improvement. E2B WER went from 0.044 to 0.053, and VAD false trigger from 0.007 to 0.008, in the same way. The other scores did not change. Task2, screen and serving did not run again, because their cases did not change.
  The one E2B task2 miss was the failing-pytest screenshot (`test-failure.png`). E2B flagged the problem
  ("One test in `test_math.py` failed"). It did not name the failing test `test_add`, so the keyword check failed. These scores come from synthetic cases and are optimistic (see "Not verified yet").

## Wait by input type (2026-10-30)

Measured on b11960, Apple M4, 24 GB, with `max_tokens=1` and `cache_prompt: false`, as the median of 3 runs for the Gemma 4 tiers and of 5 runs for text on EmbeddingGemma-2 (3 for image and audio). The inputs are synthetic: repeated text, quiet noise as audio, and a drawn screenshot, so read these as cost, not answer quality.

| Input | Prompt tokens | E4B | E2B | EmbeddingGemma-2 |
|---|---|---|---|---|
| Text, 1,000 characters | 223 | 0.73 s | 0.37 s | 0.05 s |
| Text, 10,000 characters | 2,141 | 6.59 s | 3.46 s | 0.50 s |
| Audio, 10 s | 264 | 1.01 s | 0.58 s | 0.16 s |
| Audio, 30 s | 764 | 2.81 s | 1.60 s | 0.48 s |
| Screenshot, 1280x720 | 418 | 2.36 s | 1.71 s | 1.00 s |
| Screen video, 12 s, 640x360 (1,304 tokens on this server) | see next section | see next section | 2.44 s |

An image costs more time per token than text or audio, so the token count alone does not predict the wait. Times for video on Gemma 4 are in the next section.

**Fixed: the embedding server used to stop at 512 tokens per input.** An embedding must fit in one batch, and the default batch is 512 tokens, so llama-server returned HTTP 500 for about 2,350 characters of text or 22 s of audio. A long voice note or typed question then failed in `episodic.store()` or `episodic.search()`. `serving.embed_server()` now sets `-b` and `-ub` to `config.EMBED_CTX` (8192 tokens), and sets `--video-fps` as the Gemma 4 server does. `tests/test_serving.py` checks the flags.

Checks on the fixed server:

- Text vectors did not change: cosine 1.000000 with the old server on a document and a query, so memories that are already stored stay valid.
- The longest plain text is now about 38,300 characters (about 6,300 words). A longer input still returns HTTP 500.
- A 4,659-character note was stored and found again, on a temporary database.
- `cli.py smoke` passes. The memory, tools, tool_calls and task1 evals gave the same scores as before.
- Resident memory was 1.5 GB after the inputs in the table, and 1.1 GB later. It was 0.43 GB on the old server. The cost is memory for the larger batch.
- A screenshot now costs 409 tokens and 1.0 s instead of 0.59 s, because the old batch capped images at 256 tokens. Memory stores only text today, so this does not affect it. The image scores in "Embeddings" below came from the old cap and were not measured again.

A server that is already running keeps its old flags, because `start()` reuses a server that runs the right model. Run `python cli.py stop` once to get the new flags.

## Image token cost (2026-10-30)

`brain.image_tokens()` estimates the cost of an image as `width * height // 2304` (about 1 token per 48x48 pixels), capped at 1,120. The estimate is for the context budget, so it only has to be close.

Measured on b11960, E4B, Apple M4. Each row is the `prompt_tokens` of one request with one random-noise image and a short text, minus the 13 tokens of the same request without the image (`max_tokens=1`, `cache_prompt: false`). One run per size.

| Image size | Actual tokens | Estimate | Estimate / actual |
|---|---|---|---|
| 320x180 | 86 | 25 | 0.29 |
| 640x360 | 106 | 100 | 0.94 |
| 960x540 | 222 | 225 | 1.01 |
| 1280x720 | 407 | 400 | 0.98 |
| 1280x800 | 461 | 444 | 0.96 |
| 1920x1080 | 922 | 900 | 0.98 |
| 2940x1912 (Retina, native) | 1,068 | 1,120 (cap) | 1.05 |

- From 640x360 up, the estimate is within 6% of the actual count. The slide value of 400 tokens for 1280x720 is 407 on this server.
- A small image costs more than the formula says: 320x180 uses 86 tokens, so there may be a minimum cost per image. We did not measure smaller sizes to find it. The harness shrinks screenshots to 1280 px wide at most, so it does not meet this case.
- The cap of 1,120 is not a hard limit of the server: a native Retina image used 1,068, a little less than the cap. The cap is a safe upper bound for the budget.
- Only one image per size was sent, and the content was noise. We did not test whether the content changes the count, and we did not repeat the runs. The image part of the Gemma 4 preprocessor in llama.cpp (`tools/mtmd`) is the place to check the exact rule.
- Video frames cost more than this formula says. The 12 s clip at 1280x720 in the next section used 6,574 tokens, about 548 per frame, against 400 for one image. The timestamps explain only a small part of it. The video budget in `estimate_tokens()` is therefore low for large frames.

## Video input (added after Session 1 was first committed)

One `{"type": "input_video", "input_video": {"data": <base64 mp4/mov>}}` part. Inside llama-server (`tools/mtmd/mtmd-helper.cpp`), this happens:

1. It runs **ffmpeg + ffprobe** (from PATH or `--video-ffmpeg-dir`).
2. It takes `--video-fps` frames per second (default 4, and we use 1).
3. It adds a text timestamp every 5 s.
4. It sends every frame to Gemma 4 as an ordinary image.

 There is no
Gemma-specific video encoder, and it **ignores the audio track**. Google's guidance for Gemma 4:
up to 60 s at 1 fps, with low per-frame budgets.

**The audio track, checked on 2026-10-31.** The pinned code does have a soundtrack path (`load_soundtrack()` in `tools/mtmd/mtmd-helper.cpp`, commit `8642541b255b`), but it returns early unless the video template is `MTMD_VIDEO_TEMPLATE_LFM3_OMNI`. Gemma 4 did not use it in this test, on b11960 and E4B, with a 3 s synthetic speech clip (`max_tokens=80`, `cache_prompt: false`):

| Request | Prompt tokens | Reply |
|---|---|---|
| Video, white frames, no audio stream (2 runs) | 358 | "NO SPEECH" |
| Same video with the clip as an AAC stream, confirmed with ffprobe (2 runs) | 358 | "NO SPEECH" |
| Same clip as an `input_audio` part (1 run) | 105 | The exact transcript |

Hugging Face says the small Gemma 4 models can take video with audio. That describes the model, not this llama.cpp path: to send both here, send the audio as its own `input_audio` part. We did not test that combination in one request.

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
  for approval. With "NOT done yet" in the tool result, it still said "I've created the file `plan.md`… It's waiting for
  your approval". The code now writes that reply and ends the turn. See the 2026-10-31 section.
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
matching text note (0.74). The voice note retrieved "Email John Doe the slides" (0.73; measured when that note used a different name, not measured again). A spoken question
retrieved the right text memory (0.78). This makes "multimodal memory" in Session 2 cheap to build.

## Approval outcome, pending wording and server restart (2026-10-31)

Build b11960, Gemma 4 E4B, on this Mac. The model does not reply when you approve or deny a call, so it could not
answer "did that work?". The dashboard now adds the outcome to the chat history.

The history has no tool calls, so a `tool` message has no call to answer. The pinned build accepted each shape below
with HTTP 200, including two `assistant` messages in a row. The question after the note was "did that work?".

| Note as | After approve | After deny |
| ------- | ------------- | ---------- |
| `assistant` (used) | "Yes, the file `plan.md` was created" | "No, it did not work. I was denied permission" |
| `user` | "Yes, `plan.md` was created" | "No, it did not work. The file `plan.md` was not created." |
| `system` | "Yes, `plan.md` was created" | "No, it did not work. I am waiting for your approval" (wrong) |
| `user`, then an `assistant` reply | "Yes, `plan.md` was created" | "No, the file was not created because you denied" |
| Added to the next `user` message | "Yes, it worked." | "No, it did not work." |

`assistant` is the choice because a command's output in the note must not read as something the user said. This is one
run of each shape, not an eval.

Through the real endpoints and the real page, with the model on port 9931:

- Approve, then "did that work?": "Yes, it worked. I created a file named `plan.md`". The file was on disk.
- Deny, then "did that work?": "No, it did not work. The action was denied by the user, so the file `other.md` was not
  created."
- The Approve button can be pressed while the model still replies. The outcome then lands after that reply
  (`chat_lock` in `dashboard/server.py`), because the model reads the newest message as the latest fact.
- After a page reload, the outcome chip shows from `/api/chat`, and the console shows no errors.

A call that still waits needs its own text. The reply on the turn that asked is now fixed:
"Waiting for your approval to run `write_file`. Not done yet." A fixed reply is not enough for the next question, so
each request also names the waiting calls. One run of each shape, with the real tool list, while `write_file` waited:

| Told as | "did that work?" | "is the file there?" | "did you create it?" | "what is 2 + 2?" |
| ------- | ---------------- | -------------------- | -------------------- | ---------------- |
| System prompt | calls `read_file` | calls `read_file` | "I have not created it yet" | "4" |
| System prompt with an instruction (used) | "still waiting for your approval" | "waiting for your approval to be created" | "still waiting for your approval" | "4" |
| `assistant` note | calls `read_file` | calls `run_shell` | "Yes, I have created the file" (wrong) | "4" |
| `assistant` note that says it will wait | calls `search_memory` | calls `read_file` | "I have not created the file yet" | "4" |

Through the real page, in this order: the fixed reply; "did that work?" gave "still waiting for your approval"; "what is
2 + 2?" gave 4; Approve; "did that work?" gave "Yes, the file `plan.md` was successfully created". This also saves the
second model step on that turn.

Restart, with throwaway servers on other ports and the dashboard's own startup code:

- `SIGKILL` of the server: `watch()` started a new process, and `/v1/embeddings` returned a 768-d vector. The log of the
  killed server stayed in `llama-embed.previous.log`.
- The same `SIGKILL` with the dashboard's own startup code: `/api/state` showed `embed` false, then true, with a new pid.
- `SIGKILL` of the E2B language model: it came back on E2B. After a switch to E4B through `/api/serving/switch`, one
  process held the port, and no second start happened. `SIGKILL` then: it came back on E4B, not on E2B.
- `stop()` from a new `LlamaServer`, as `cli.py stop` does: no process 7 s later. No `serving.error` event in the run.
- Not covered: a server that hangs while its process stays. That process may be loading or busy with a long input, so
  the watcher leaves it alone. The cause of the stop on 2026-10-31 at about 09:58 is still unknown.

## Session 2: always on and learning (2026-11-06)

Build b11960, Gemma 4 E4B, on the M4 Mac. Every number comes from the synthetic cases (OS voices and drawn screens).
They are a regression check, not a measure of real use.

### Pins added

| Item | Pin | How |
|---|---|---|
| Whisper large-v3-turbo, MLX format | `mlx-community/whisper-large-v3-turbo` @ `a4aaeec` (`config.json`, `weights.safetensors`, 1.61 GB) | downloaded and hashed; the sha256 values are in `agent/accel/metal.py` |
| `mlx-whisper` | 0.4.3 (2025-08-29, the newest on PyPI) | installed with `--no-deps` from `requirements/nodeps-macos.txt` |
| Runtime dependencies of `mlx-whisper` | `mlx` 0.32.4, `numba` 0.68.1, `scipy` 1.18.1, `tiktoken` 0.14.0, `huggingface_hub` 2.4.0 and their dependencies | resolved by pip on 2026-11-06, in `requirements/lock-macos.txt` |

`mlx-whisper` declares `torch` (a 138 MB wheel). In the 0.4.3 wheel, only `torch_whisper.py`, the weight converter,
imports it. `transcribe()` imports `mlx`, `numba`, `scipy` and `tiktoken`. So setup installs the package without its
declared dependencies, and the lock file pins the ones that `transcribe()` needs.

### Thinking leaked into a JSON reply, and how we stopped it

With `enable_thinking: false` and a JSON schema, E4B wrote a 3,137-character plan into `reasoning_content` on a
consolidation request. The plan used up all 1,024 output tokens, and the JSON content stopped after 356 characters.
Two runs gave the same result. Session 1 did not see this, because its JSON requests were shorter.

| Request field | finish_reason | completion tokens | reasoning | valid JSON |
|---|---|---|---|---|
| none | length | 1,024 | 3,137 chars | no (2 of 2 runs) |
| `reasoning_budget: 0` | length | 1,024 | 3,137 chars | no (2 of 2): the server ignores this field |
| `thinking_budget_tokens: 0` (used) | stop | 672 | none | yes (2 of 2) |

`brain.ask_json()` now sends `thinking_budget_tokens: 0` when `config.THINKING` is off. We did not use the server flag
`--reasoning-budget 0`: it turns off thinking for every request, and `brain.respond()` needs thinking after a tool
result. So the field goes on each JSON request only.

### Whisper and the streaming segmenter

The 57.4 s synthetic conversation has 11 spoken lines from 3 voices. The `Segmenter` cut it into exactly 11 utterances.
It got the audio in 1 s steps, as from the live microphone. Whisper took about 0.9 s per utterance. The first one took
3.9 s, because it also loads the model. Whisper made these errors:

- "standoff" for "standup", and "F site" for "offsite"
- "Kriya" for "Priya": Task 3 then answered "Kriya will send the deposit", and that question failed

The asr eval on the same 10 clips:

| Engine | WER | Seconds per clip |
|---|---|---|
| Gemma 4 E4B | 4.4% | 0.97 |
| Whisper large-v3-turbo | 2.6% | 1.20 (includes the model load on the first clip) |

Gemma dropped the last words of two clips ("a quarter to", "a lot of"). Both engines heard "oat milk" wrong.

### Consolidation

| Run | Faithfulness | Invalid citations | Compression | Recall before / after forgetting |
|---|---|---|---|---|
| First prompt | 0.97 | 19% | 0.51 | 1.00 / 1.00 |
| Merged claims, known facts as `[F12]` (used) | 1.00 | 0% | 1.31 (target 1.5: FAIL) | 1.00 / 0.93 |

With the first prompt, the model wrote one claim per raw row, plus a fact for each. The kept text was then twice as long
as the raw text. It also cited the ids of the known facts as sources, and the code dropped them. The prompt now asks
for at most one claim per three rows, and it labels known facts `[F<id>]`.

Compression now counts the summaries only, because facts are an index next to the summary. The synthetic logs have
about one fact per row, so they leave little to compress. The threshold stays at 1.5.

### Multimodal memory

Recall at 3 is 1.0 in every mode, because 8 images and 10 clips are easy. At 1, the media vectors did better:

| Modality | First result, text vectors | First result, media vectors | First result, both |
|---|---|---|---|
| Screenshots (8) | 0.875 | 1.00 | 1.00 |
| Speech clips (10) | 0.90 | 1.00 | 1.00 |

The text of a screenshot is Task 2's description, and the text of a clip is its Whisper transcript. A drawn terminal
is easier than a real screen.

### Crash recovery

- The `daemon` eval kills a daemon twice with SIGKILL, in a sleep job. It measured 0.11 s from the restart to the
  next step. No finished step ran again, and no job was lost. The step that ran during the kill ran again, as designed.
- With real models, a daemon replayed the conversation at real time on a throwaway runtime folder. We killed it with
  SIGKILL after 4 of 11 utterances had a result. The new daemon logged `checkpoint.recovered` and skipped utterances 0
  to 3 without transcribing them again. It finished the other 7, and the feed showed the job as done.
- `caffeinate -i -w <pid>` ended by itself when the daemon was killed.
- `daemon stop` (SIGTERM) let the daemon log `daemon.stopped` and exit.

### Soak and the dashboard endpoints

- `scripts/soak.py --hours 0.1 --every 120` ran for 8 minutes as a check of the script, not as the soak itself. It
  queued 3 replays, and all 3 finished. It found 0 missed slots and 0 late heartbeats. The daemon process peaked at
  520 MB of resident memory after it loaded Whisper. The run was too short for a memory growth figure. We did not check
  whether resident memory includes the Metal buffers that MLX allocates for Whisper.
- With real models and a throwaway runtime folder, a script called each Session 2 endpoint. It called replay,
  consolidate, ask, brief, feedback, reflect, diff, rollback, and a job kind that the page must refuse. Each returned
  what the code says. The Task 3 answer cited the summary row, not a raw transcript. Reflect wrote v2, rollback made v1
  active again, and the locked blocks stayed untouched.

### Self-evolution (Task 4)

First run. Held-out compliance went from 0.50 to 0.96, with no regressions and 0 locked edits. But the overfitting
example learned the opposite rule: "Do not start the headline with the word Dentist". The reflection prompt showed
feedback as `- (bad) Start the headline with the word Dentist.`, and the model read that as "this was bad, do not do
it". The prompt now says `The user rated a brief bad and said: "..."`.

Second run, with that fix:

| | Dev days | Held-out days |
|---|---|---|
| Seed prompt | 0.50 | 0.50 |
| After 3 rounds of simulated feedback | 0.96 | 1.00 |

No held-out check regressed, and no version changed a locked block.

The overfitting example then gave three rounds of
feedback that is true for one day only. Afterwards, that day followed the rule. All 4 held-out days, which have no
dentist, mentioned Dr. Lee, and held-out compliance dropped by 0.08. One held-out brief started with "Dr. Lee was
mentioned in the context of today's activities". That sentence breaks the locked rule "Never invent an event". A
learned preference cannot change the locked text, but it can still win over a locked instruction in the model.

### Found while testing

- **Two requests upgraded one database at once.** The dashboard opened a Session 1 memory database from two requests,
  and both added the new columns: "duplicate column name: level". `agent/sql.py` now creates tables under a lock, and
  the upgrade accepts a column that another process added a moment before.
- **`screencapture -v` waited forever** when `doctor --exercise` ran from the shell of the Claude app. It did not fail.
  That shell probably lacks Screen Recording permission. This is the Session 1 `record_screen` hook, and Session 2 did
  not change it.

## Session 2 review (2026-11-11)

Every eval ran again on E4B after the fixes below (`python evals/run_all.py --run --tier e4b`).

### Consolidation wrote longer claims than the raw rows

`compression_ratio` was 1.31, then 1.17 on the next run, below the 1.5 target. The raw rows of the synthetic day logs
are already short notes (about 55 characters), and E4B rewrote them as full sentences: "The plumber is scheduled to
fix the kitchen sink on Monday morning." The prompt said "keep each claim short", and that was not enough.

The prompt now asks for note style, "shorter than the memories it merges", and names the filler to drop. Two runs gave
`compression_ratio` 1.65, with `faithfulness` 1.0 and `recall_after` 0.933, the same as before the change, and Task 3
stayed at 8 of 8. The threshold did not change. A claim now reads "Plumber fixes kitchen sink Monday morning."

With the shorter style, E4B sometimes copied its citation into the text ("... Monday morning. Sources: [1]").
`consolidate._cited()` now removes a "Sources: [...]" suffix, because the ids belong in `sources`.

### Speech to text speed, after a warm-up

The asr eval now makes one untimed call per engine first and reports the real-time factor. On the 10 synthetic clips,
Whisper large-v3-turbo had an RTF of 0.30 and Gemma 4 E4B 0.29: on short clips Whisper is not faster. It is still the
default for always-on listening, because it keeps the language model free for the user (see `agent/senses/stt.py`).

### Session 2 on E2B, the first run

Every eval also ran on E2B (`--tier e2b`). Session 2 had only run on E4B before.

| Eval | E2B | E4B | Notes |
|---|---|---|---|
| `consolidation` | faithfulness 1.00, invalid citations **10.5% (FAIL)**, compression 1.57 | 1.00, 0%, 1.65 | E2B cited 1 id outside the window in each log. The code dropped both before storing, which is the reason the check exists. E2B also wrote far fewer facts (0.11 per raw row against 0.58). |
| `task3` | accuracy 1.00, abstain 1.00 | 1.00, 1.00 | See the scorer fix below. |
| `task4` | test gain +0.50, dev after 0.79, 1 patch refused, overfit leak 1.00 | +0.50, 0.96, 0, 1.00 | The one-day rule leaked into every held-out day on both tiers. |
| `asr` | Gemma WER 0.053, RTF 0.17; Whisper WER 0.026, RTF 0.33 | 0.044, 0.29; 0.026, 0.30 | E2B transcribes faster than Whisper on these short clips, at twice the word errors. |

**The Task 3 scorer missed an abstention.** E2B answered the question that nobody spoke about with "The memories do
not contain any information about what was decided regarding the catering", cited nothing, and the eval scored it
wrong: the "I do not know" pattern only knew a few phrasings. `evals/task3/eval.py` now accepts "does not contain /
mention / say" and "no / any information", and a test covers both sides (a hedge that cites a memory still fails).
With the fix, E2B scores 1.00 on `abstain_rate`. The model did not change; the measurement did.

### The soak and a sleeping Mac

`python scripts/soak.py --hours 1` ran on E4B and took 13.8 hours of wall clock. The samples stop at 14 minutes, one
comes at 46 minutes, and the loop ended the next morning: the Mac slept twice, although the daemon held `caffeinate`
(keep-awake stops idle sleep, not a closed lid or a sleep from the menu). The daemon reported 38 schedule slots as
missed, correctly, and the soak counted them as 38 missed events.

The soak now finds the sleep gaps (no sample for 3 minutes) and counts a slot inside one as `missed_while_asleep`.
The same run then reads: awake 0.23 h, missed events 0, missed while asleep 38, no late heartbeat, no job lost,
daemon memory 1.94 GB flat (Whisper loaded), growth 3.4 MB/h. With 0.23 awake hours that growth number rests on a few
samples, so the `daemon` record warns. A multi-hour soak with the lid open is still to do.

### Fixed

- **An eval reset left a database with no tables.** `sql.connect()` checked whether the file existed after
  `sqlite3.connect()`, which creates an empty file. After `evals.common.reset()` deleted a database, the next query in
  the same process failed with "no such table". Any second case folder (your own cases next to the synthetic ones) in
  `consolidation`, `multimodal`, `task3` or `task4` hit this.
- **Task 4 scored one folder's first round against another folder's last round**, and raised a KeyError when you
  added your own days. Each folder is now scored against its own seed.
- **A failed summary request marked the rows as covered.** With no valid JSON, the rows were marked covered by summary
  0, and `forget()` would delete them after 7 days without a summary. They now stay for the next run.
- **The soak script lost its result when the daemon had died**, because terminating a missing process raised. A sample
  where the daemon is not running now counts as a missed event.

## Not verified yet

- Real use: the evals run on **synthetic** cases (macOS `say` voices, rendered screenshots, hand-written tool
  calls and memories). Text-to-speech is cleaner than a real voice, so ASR, VAD and Task 1 scores are optimistic.
  Add your own cases with `python cli.py evals add` for numbers that reflect your voice and your screen.
- Mic recording and the global hotkey need permissions that only you can grant. Please try
  `cli.py note` and the hotkey once you grant the permissions.
- Memory use with a Google Meet call running alongside.
- Session 2, live microphone: the always-on `Listener` with a real microphone. Only replays of recordings were tested.
- Session 2, `daemon install` (launchd): the plist and the commands are written but were not run, because they change
  your login items. It is also unknown whether a process that launchd starts gets the Microphone permission.
- Session 2, notifications: `osascript` exited with code 0, but nobody looked at the screen to see the notification.
- Session 2, a soak of several hours while the machine stays awake: the longest awake stretch so far is 0.23 h
  (see "The soak and a sleeping Mac").
- The voice button of the dashboard chat with a real microphone: a unit test covers it with a mocked microphone. The
  picture and the daily brief in the chat were tried in the dashboard with E4B.
