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
- A misheard name (the person who sends the deposit): Task 3 then answered with the wrong name, and that question
  failed

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

### The generality gate (2026-11-13)

`evolve.reflect()` now checks a new version before it becomes active. It writes briefs with the parent and the new
version for `config.EVOLVE_GATE_DAYS` (3) recent days. Then `evolve.ungrounded()` finds names and numbers that the
day's memories never say. A patch whose briefs invent more than the parent's did (`EVOLVE_GATE_MAX_INVENTED` = 0) is
refused. In the eval the gate days are the dev days. `config.THRESHOLDS` did not change.

The first version of the check also took capitalized words in the headline for names. E4B writes title-case
headlines such as "End-of-Day Brief", so every brief was flagged for "Brief" and "Day", and `invented_rate` was 1.0
even with the seed prompt. Headlines are titles like headings, so the check now reads only their numbers. Run again on
the stored briefs of that run, the seed and the normal rounds had nothing flagged. After the overfit rounds, the
check flagged "Lee" on all 4 test days.

`python evals/run_all.py --run task4 --tier <tier>`. "Gate off" is the same code with `EVOLVE_GATE_DAYS = 0`, set by
a wrapper script for the run, not in `config.py`:

| E4B | Gate off | Gate on |
|---|---|---|
| `dev_before` → `dev_after` | 0.50 → 0.96 | 0.50 → 0.96 |
| `test_before` → `test_after` (`test_gain`) | 0.50 → 1.00 (+0.50) | 0.50 → 1.00 (+0.50) |
| `regression_rate`, `locked_edits`, `patches_rejected` | 0, 0, 0 | 0, 0, 0 |
| `gate_refused` (normal rounds) | 0 | 0 |
| `invented_rate_before`, `invented_rate` (test briefs) | 0.00, 0.00 | 0.00, 0.00 |
| `overfit_patches_refused` (of 3) | 0 | **3** |
| `overfit_test_leak` | **1.00** | **0.00** |
| `overfit_invented_rate` | 1.00 | 0.00 |
| `overfit_dev_gain` | 1 | 0 |
| `overfit_test_change` | -0.08 | 0.00 |
| Run time | 352 s | 664 s |

With the gate off, the overfit version learned "Always mention Dr. Lee in the first bullet", and each of the 4
held-out briefs named Dr. Lee ("Dr. Lee was mentioned in relation to the offsite agenda ..."). With the gate on, each
of the 3 overfit patches was refused with "Lee" as the invented word, so the seed prompt stayed active. The normal
rounds learned the same 3 preferences as with the gate off, and the gate refused none of them. `overfit_dev_gain`
drops to 0 because the one-day rule never becomes active, which is the intent. The gate costs 2 briefs per gate day
per reflection, so the eval took about twice as long.

| E2B | Gate off | Gate on |
|---|---|---|
| `dev_before` → `dev_after` | 0.50 → 0.79 | 0.50 → 0.79 |
| `test_before` → `test_after` (`test_gain`) | 0.50 → 1.00 (+0.50) | 0.50 → 1.00 (+0.50) |
| `regression_rate`, `locked_edits`, `patches_rejected` | 0, 0, 1 | 0, 0, 1 |
| `gate_refused` (normal rounds) | 0 | 0 |
| `invented_rate_before`, `invented_rate` (test briefs) | 0.00, 0.00 | 0.00, 0.00 |
| `overfit_patches_refused` (of 3) | 0 | **3** |
| `overfit_test_leak` | **1.00** | **0.00** |
| `overfit_invented_rate` | 0.75 | 0.00 |
| `overfit_dev_gain` | 1 | 0 |
| `overfit_test_change` | -0.08 | 0.00 |
| Run time | 226 s | 376 s |

On E2B the gate also refused all 3 overfit patches for "Lee". `patches_rejected` is 1 with the gate on and off. It
is a round 3 patch that the validator refused or that changed nothing (the record does not keep which), not the gate. With the gate off, `overfit_invented_rate` is 0.75,
not 1.00: the d5 brief leaked only through its headline "Dentist Summary", and the check does not read headline
words as names. That is the known gap (see `docs/EVALS.md`). The gate still caught the rule, because the same
version also named Dr. Lee on the other days.

What the gate does not prove: the eval has one overfitting example, and the gate days are 3 of the 4 dev days. A
one-day rule that invents nothing the check can see (lower-case words, or a word that some memory has) still passes.
The daemon gates on the days in the `briefs` table. No stored days means the gate checks nothing (`gate_days` 0 in
the result). That path is covered by a unit test, not by a live run.

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
| `task4` | test gain +0.50, dev after 0.79, 1 patch refused, overfit leak 1.00 | +0.50, 0.96, 0, 1.00 | The one-day rule leaked into every held-out day on both tiers. The generality gate (2026-11-13, above) takes the leak to 0.00 on both. |
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

## Before Session 2 (2026-11-13)

### The reply schema lists the allowed citation ids

On E2B, `consolidation.invalid_citation_rate` was 10.5% (limit 5%): the model cited one id from outside the window in
each synthetic log. The code dropped those ids, so nothing wrong was stored, but the check failed.

`consolidate.reply_schema()` now puts the row ids of the window into the JSON schema as an `enum`. Checked on the
pinned build (b11960) with E4B, same prompt, "Put every whole number from 40 to 45 into sources":

| Schema of `sources` items | Reply |
|---|---|
| `{"type": "integer"}` | 40, 41, 42, 43, 44, 45 |
| `{"type": "integer", "enum": [41, 43, 105]}` | 41, 43, 43, 43, 43, 43 |

The server enforces the list. It can repeat an id, so `_cited()` now counts the same id one time. `_cited()` still
drops an id outside the window, as a second check.

`python evals/run_all.py --run consolidation --tier <tier>`, 2 synthetic logs:

| | E4B before | E4B after | E2B before | E2B after |
|---|---|---|---|---|
| `invalid_citation_rate` (limit 0.05) | 0.00 | 0.00 | **0.105 (FAIL)** | **0.00** |
| `faithfulness` (limit 0.90) | 1.00 | 1.00 | 1.00 | 1.00 |
| `compression_ratio` (limit 1.5) | 1.75 | 1.60 | 1.57 | 1.56 |
| `facts_per_raw_row` | 0.58 | 0.37 | 0.11 | 0.11 |
| `recall_after` (limit 0.80) | 0.933 | 0.933 | 1.00 | 1.00 |

The metric is now 0 by construction, so it only shows whether the server still enforces the schema. On E4B the
compression went from 1.75 to 1.60 and the model wrote fewer facts. We did not find the reason, and one run of 2 logs
does not separate it from noise. `config.THRESHOLDS` did not change.

**A wrong run, and how we saw it.** The first "E2B" run happened while the dashboard was up. The dashboard's E4B
server held the port, so `--tier e2b` measured E4B and wrote the result under `e2b`. Both tiers then had the same
numbers to the last digit, which gave it away. The numbers above are from a run with the dashboard and the daemon
stopped, and `/props` showed `gemma-4-E2B-it-Q4_0.gguf` during the E2B run. Stop the dashboard before you run an eval
on another tier (GitHub issue #1 is the same weakness).

**Ids copied into the text.** In a real replay, E4B ended each claim with its ids: "... deposit due Wednesday.
(22, 23, 24, 25, 26)". `_cited()` now removes a list of numbers at the end of a claim when each number is one of its
sources. A test covers it, and a number that is not a source stays.

### Chat sessions and the simple view

- Unit tests cover the sessions: a new session starts with no history for the model, an old one opens again, a delete
  removes one session, an id that is not in our format is refused, and the conversation of an older version
  (`chat.json`) becomes the first session.
- In the dashboard with E4B: 4 new sessions were made through `/api/chat` (a note saved with `save_note` and found
  again, a picture of a failing test, a daily brief with feedback, and questions about the replayed conversation).
  History listed 5 sessions, one opened with its own messages, and the page had no console error.
- The reflection on the feedback "Too long. Six bullets at most, please." wrote v2 with the preference "Limit the
  brief to a maximum of six bullet points." The gate checked 1 stored day and found nothing invented. This is the
  first live run of the gate outside the eval. v1 was then made active again.
- The simple view and the Show details switch were looked at on the Serving, Brain, Memory, Evolve and Scorecard tabs.
  The voice button and a delete from the History list were not tried in the browser.

## Live microphone, launchd and a 1-hour soak (2026-11-14)

M4 24 GB, E4B, Whisper turbo. A person sat at the Mac for the microphone and the login.

### The 1-hour soak

`python scripts/soak.py --hours 1` (result `evals/results/soak/20261114-071851.json`), lid open, on power:

| Metric | Value |
|---|---|
| Jobs submitted / failed | 12 / 0 |
| Missed events (unfinished jobs, missed slots, late heartbeats, down samples) | 0 (0, 0, 0, 0) |
| Daemon memory | 1.90 GB, flat after the first 5 minutes; peak 1.95 GB |
| Growth fit | −25.2 MB/h (no leak seen; 1 hour is too short for a leak of a few MB/h) |
| Peak system memory | 15.6 GB |

The summary says `awake_hours` 0.98 and 1 `asleep_gaps`, but the Mac did not sleep. After the last sample the script
waited 5 minutes for the last job without sampling, and the sleep check (no sample for 3 minutes) read that wait as a
gap. So the `daemon` eval warns, because 0.98 is under `SOAK_MIN_HOURS` (1.0). `scripts/soak.py` now keeps sampling
during the wait. A 7-minute check with the 5-minute wait reads 0 gaps. The 1-hour result file was not changed.

### `daemon install` and launchd

- `python cli.py daemon install` wrote `~/Library/LaunchAgents/com.multimodal-agent.daemon.plist`, and launchd
  started the daemon at once (`RunAtLoad`).
- `python cli.py daemon kill`: launchd started a new daemon about 10 s later (`runs = 2`). This is the default
  launchd throttle.
- `python cli.py daemon stop`: exit code 0, and launchd did not start it again (`state = not running`).
- Log out and log in: the old daemon emitted `daemon.stopped` at 08:33:22. launchd started a new one at 08:33:41 in
  the new session (`daemon.started`, models loaded, nothing to recover).

The daemon has no lock against a second copy. Stop a daemon that `daemon start` started before you run
`daemon install`, or two daemons share `data/runtime/`.

### The live microphone

**A daemon that launchd starts hears only silence.** `listen on` opened the "MacBook Air Microphone" and emitted
`senses.ambient.state listening`, but in 2 minutes of speech the VAD found nothing. macOS showed no permission
dialog, and Python is not in System Settings > Privacy & Security > Microphone. macOS gives a process with no
Microphone permission silent audio instead of an error, so the listener cannot tell this apart from a quiet room.

**A daemon started from a terminal app with the permission works.** `daemon start` from Claude Code (which has the
permission), then `listen on`, then 90 s of real speech: 12 utterances of 2.7 to 11.1 s. Whisper took 0.93 to 1.08 s
for each one. Each one was stored as a `transcript` row with its audio. The text was close to what was said, with one
small Whisper error in 12 utterances.

## The Microphone permission for the launchd daemon (2026-11-14)

M4 MacBook Air, macOS 27.2.1, E4B, Whisper turbo. A person sat at the Mac, answered the macOS dialog and spoke.

**What Apple documents.** `NSMicrophoneUsageDescription` is the Info.plist key that an app that uses the microphone
must have (macOS 10.14 and later). `AVCaptureDevice.authorizationStatus(for:)` and
`requestAccess(for:completionHandler:)` read the permission and show the dialog once; the system remembers the answer
("Requesting authorization to capture and save media"). `AssociatedBundleIdentifiers` in a LaunchAgent plist names
the app that System Settings > Login Items shows (`man launchd.plist`). Apple does not document how macOS picks the
process whose permission counts for a child process. The result below is measured, not documented.

**The fix.** `install_autostart` builds `.cache/autostart/multimodal-agent.app`: an Info.plist with
`NSMicrophoneUsageDescription` and `LSUIElement`, an executable compiled with `swiftc` from
`agent/platforms/macos_launcher.swift`, and an ad-hoc signature (`codesign --sign -`). The LaunchAgent runs that
executable with the daemon command as its arguments. The launcher calls `requestAccess` without waiting for the
answer, starts Python as a child, forwards SIGTERM, SIGINT and SIGHUP to it, and exits with its exit code
(128 + the signal number after a kill). macOS knows an ad-hoc signed app by its code hash, so the app is rebuilt only
when its source or Info.plist changes; a rebuilt app asks again.

**Before: launchd runs Python directly.** A probe in a throwaway LaunchAgent recorded 4 s: RMS 0.0, 100% exact zeros.
The same probe through a test app built the same way: RMS 0.0020 (the quiet room; from Claude Code it measured 0.0019
to 0.0021).

**The warning on real silent audio.** `ambient.Listener` in a LaunchAgent that runs Python directly emitted
`senses.ambient.state listening` (MacBook Air Microphone), then `senses.ambient.warning` with the macOS hint
(`mic_silent_hint`), 5 s after the stream opened.

**After: `daemon install` with the launcher.**

- launchd started the launcher (pid 16000), and the launcher started Python (pid 16003, parent 16000). The log said
  `launcher: microphone permission not determined`. macOS showed the dialog for "multimodal-agent". The person clicked
  Allow, and the log said `launcher: microphone permission authorized in the dialog`.
- `listen on`, then real speech for about a minute: 4 utterances of 0.8 to 15.3 s. Whisper took 0.93 to 1.14 s for
  each one. 3 were stored as `transcript` rows (memories 50 to 52); the junk filter refused the 0.8 s "Mm-hmm".
  The transcripts are whole, sensible sentences; nobody checked them word by word against what was said. No
  `senses.ambient.warning`, and `daemon status` showed `mic_warning: null`.
- `daemon kill` (SIGKILL to Python): the launcher exited with code 137, and launchd started a new launcher and Python
  (`runs = 2`). The log said `microphone permission authorized`, with no second dialog.
- `daemon stop`: exit code 0, and launchd did not start it again (`state = not running`).
- `daemon install` again (same source, so no rebuild): `microphone permission authorized`, with no dialog.

**`PATH` in the LaunchAgent.** `python cli.py smoke` against the server of the launchd daemon failed the video check
(400). The server log said `ffprobe failed on buffer (is ffprobe in PATH?)`: launchd starts the daemon, and so its
`llama-server`, with `PATH=/usr/bin:/bin:/usr/sbin:/sbin`, and ffprobe is in `/opt/homebrew/bin`. This was so before
the launcher too. The plist now has `EnvironmentVariables.PATH` from the shell that runs `daemon install`. After a new
`daemon install`, the new server's `PATH` has `/opt/homebrew/bin`, and `smoke` passed all 5 checks (video 3.3 s,
server memory 5.7 GB).

Not tried: a log out and log in with the launcher (the launcher changes only what launchd runs, not when), and a
"Don't Allow" answer (the warning covers it, as the probe above shows).

## Timed listening, closed windows and one daemon (2026-11-24)

M4 24 GB, no model needed. The backlog items "Timed listening" and "Partial summaries of a conversation" are done,
and the daemon no longer runs twice on one runtime folder.

- **The daemon eval again.** `python cli.py evals --run daemon` now reads the 1-hour soak of 2026-11-14 (its cached
  record still read the 13.8-hour soak of 2026-11-11, when the Mac slept). recovery_s 0.11, redone_steps 0,
  lost_jobs 0, missed_events 0, rss_growth −25.2 MB/h. `soak_awake_hours` is 0.98, so the record still warns, as
  "The 1-hour soak" explains.
- **Timed listening.** In a throwaway runtime (`AGENT_RUNTIME`), with `daemon run --no-models`:
  `listen on --for 1h30m --window 22:00-06:00` printed `listening on (until 21:20; only 22:00-06:00, outside
  22:00-06:00 now)`. `listen on --for 2s`: 4 s later the daemon had turned the switch off (`daemon.listen`, reason
  "time up"). `--window 9am-5pm` and `--for soon` are refused with a message. `POST /api/daemon/listen` with
  `for_min` 60 and a window returned 200 with `note: "outside 09:00-17:00"`; a bad time returned 400. In the
  dashboard the duration select and the two time fields render beside the switch. With the request caught in the
  page (so the real daemon did not start listening), the switch sent `{"on": true, "for_min": 60, "window":
  ["09:00", "17:00"]}`, and with empty fields `{"for_min": null, "window": []}`. No console errors.
- **"Off" stays `False`.** The switch is now stored as `{"on": true, "until": ..., "window": ...}` when on and as plain
  `False` when off. A daemon of the older version reads `bool(value)`, so it still reads "off" as off. Restart such a
  daemon to get the end time and the window.
- **Closed windows.** A window is summarized only when no new row can join it: 30 minutes since its first row, 40
  rows, or a pause of `SUMMARY_GAP_S` (10 minutes). The trigger no longer passes `force`. The synthetic consolidation
  logs pause at most 5 minutes inside a window, so their windows do not change. A test stores 25 transcripts (over
  `TRANSCRIPTS_PER_SUMMARY`) while the talk goes on: the trigger queues nothing, 10 more rows arrive, and after the
  pause there is one summary for all 35 rows.
- **Notifications.** `platforms.current().notify("multimodal-agent", ...)` from a terminal at 19:53: the person at
  the Mac saw the notification on the screen. A notification from the daemon that launchd starts was not looked at.
- **The Session 2 primer.** The code moved, so 6 code panels of `docs/session-2-primer.html` pointed at old line
  numbers. Its build script is gone, so the line numbers in `data-src` and `data-blocks` were changed in the HTML,
  mapped from the old file to the new with a diff. The code in the panels did not change. The skill's `check.py`
  reads 0 errors.
- **A watcher thread died.** `data/runtime/daemon.log` of the launchd daemon started on 2026-11-18 had a traceback:
  `Thread-2 (watch)` ended with `SystemError: <built-in function proc_cmdline> returned a result with an exception
  set`, from `PermissionError ... sysctl(KERN_PROCARGS2)`. `LlamaServer._processes()` read the command line of every
  process, and psutil 7.2.2 (the newest on PyPI on this date) raised SystemError for one that macOS refused to
  describe, probably a process that was exiting. `process_iter` does not catch SystemError, so the watcher of the
  language model server stopped, and from then on a stopped server would not have been restarted. The log has no
  timestamps, so the time is unknown. Now `_processes()` reads the command line only of a process named
  `llama-server` and skips any process it cannot read, and `watch()` reports a failed check as `serving.error` and
  goes on after `SERVER_RETRY_S`. Two tests cover this; both fail on the old code.
- **One daemon.** In the throwaway runtime, a second `daemon run --no-models` printed `another daemon already runs
  (pid 44683), so this one exits` with exit code 0, so launchd does not start it again. The first heartbeat is
  written in one `BEGIN IMMEDIATE` transaction, so two daemons that start at the same moment cannot both win.

## Session 3: voice, MCP and the wake word (2026-11-24)

Branch `feature/session_3`, from `main` at `de0b96e`. Measured on the M4 (24 GB, macOS 27) with Python 3.13.3.

### Pins added

| What | Pin | How it was checked |
| ---- | --- | ------------------ |
| Kokoro TTS | `kokoro-onnx==0.6.1`, `phonemizer==3.6.0`, `espeakng-loader==0.2.4` | PyPI JSON, installed and run |
| Kokoro model | `kokoro-v1.0.int8.onnx` (92.4 MB), `voices-v1.0.bin` (28.2 MB), release `model-files-v1.0` of thewh1teagle/kokoro-onnx | Downloaded and hashed; sizes match the release (it lists no digest) |
| MCP SDK | `mcp==2.3.0` (official SDK, protocol revisions 2025-11-25 and 2026-07-28) | PyPI JSON, client and server run |
| Wake-word features | `melspectrogram.onnx`, `embedding_model.onnx`, openWakeWord release v0.5.1 | Downloaded and hashed |
| Wake-word classifier | `hey_jarvis_v0.1.onnx` (1.3 MB), the same openWakeWord release (added 2026-11-27, see below) | Downloaded and hashed |

### Kokoro runs through ONNX, not MLX

- The PyTorch package `kokoro` 0.9.4 needs Python < 3.13. The MLX path (`mlx-audio` 0.6.8) imports `misaki` for G2P,
  and `misaki` 0.9.4 also needs Python < 3.13. `kokoro-onnx` installs and runs on 3.13.
- kokoro-onnx uses the CPU unless `ONNX_PROVIDER` is set (or an accelerated onnxruntime is installed). We create the
  session ourselves with `config.TTS_PROVIDERS = ["CPUExecutionProvider"]`, so the GPU stays free for Gemma.
- Speed, warm, voice `af_heart`: int8 writes speech at a real-time factor of 0.11 (20 characters in 0.2 s), the
  full-precision file at 0.23. Through Whisper both had the same word error rate on 6 sentences x 3 voices; most
  "errors" were numbers written as digits ("315" for "three fifteen"), so the intelligibility eval must normalize
  numbers. int8 is the default.
- After a few seconds idle, the first sentence takes longer, and more so when it is long (52 characters: 0.35 to
  0.45 s). The first spoken piece is cut at a comma when it is over `TTS_FIRST_CHUNK_CHARS` (60).
- `config.PRONUNCIATIONS` gives Kokoro the IPA of a word that espeak-ng says wrong. It held "Sanjaya" (espeak:
  "SAN-jay-uh") while that was the wake word; it is empty now.

### Playback

- In the dashboard tab (Web Audio, chunks pushed over Server-Sent Events): first sound 0.65 to 0.88 s after the text
  arrived; Stop silenced it 61 ms after the stop event (including the 30 ms fade).
- On this machine: macOS `afplay` needs about 1.1 s per file on top of its length (0.3 s of sound: 1.4 s), which
  leaves a gap between sentences. One sounddevice (PortAudio) stream starts in about 50 ms, so it is the portable
  default of the `play_audio` hook. Stop on this machine: 140 ms, most of it the 130 ms output latency of the
  Bluetooth speaker in use (an Echo); `abort()` would block for another 130 ms, so stop() ends the stream at the next
  callback instead.
- Speech from another process (`cli.py say`) reaches the dashboard tab: the tab's heartbeat file picks the route.

### MCP

- Client: `mcp.Client(StdioServerParameters(...))` or `Client(streamable_http_client(url, http_client=httpx2.AsyncClient(headers=...)))`.
  The session is opened and closed in one asyncio task per server (anyio cancel scopes must exit in the task that
  entered them); calls reach it through a queue.
- A stdio server gets only a safe part of the client's environment (`get_default_environment()`: HOME, PATH...).
  An environment variable the agent sets does not reach it, so the reminders server takes its file as `--db`.
- `MCPServer` sends a tool's whole docstring as its description unless `description=` is given; the reminders tools
  give one line, for the small model. An exception inside a tool reaches the client as "Error executing tool X";
  `ToolError` keeps its message, so the model can correct the call.
- Task 6 with Gemma 4 E4B, typed (no voice yet), reminders tools offered by `tools.select()`: "Remind me to call
  <name> in 2 minutes" -> `add_reminder(in_minutes=2, text="Call <name>")`; "Set a reminder for 16:30 to send the slides"
  -> `add_reminder(at="16:30", ...)`; "What time is it?" -> `get_time`. Each turn took 6 to 10 s: the step after a
  tool result runs with thinking on (the Gemma 4 quirk of Session 1), so it is far over the 2.5 s reply budget.
- Due reminders: the daemon's watcher calls `due_reminders`, announces each one (feed, notification, speech) and
  then `complete_reminder`. Checked with a stubbed voice and notification; the same reminder is not announced twice.

### The first wake word, "Hey Sanjaya" (2026-11-24 to 2026-11-26, removed)

The first wake word was a custom openWakeWord classifier trained on this Mac (Kokoro clips, near misses, 2,000 h of
negative features, and 21 of the user's own clips). The trainer and its notes are in Git history at `902fc86`
(`scripts/train_wake_word.py`, `scripts/wakeword/`, `docs/WAKE_WORD.md`). On 2026-11-27 the user found it unreliable,
and it was replaced by openWakeWord's pretrained "hey jarvis" model (below). What still holds:

- Our 3-model streaming code (`agent/voice/wake.py`) gives the same scores as the openwakeword package: largest
  difference 1.2e-6 over 116 frames. The two feature models did not change.
- The Sanjaya models: false wakes per hour on openWakeWord's 10.7 h validation features were 1.12 (v1, threshold 0.5),
  0.28 (v1, 0.9) and 0.65 (`echo2`, the installed one, 0.9). Under the agent's echo the synthetic `echo_miss_rate`
  was 0.917 (v1) and 0.500 (`echo2`).

### Session 3 evals, first runs (2026-11-25, E4B, synthetic cases)

The thresholds were set from the approved plan before the first run (`config.THRESHOLDS`).

| Eval | Result | Threshold |
| ---- | ------ | --------- |
| `tts` | first audio p50 0.35 s, p95 0.45 s; WER 0.038 (10 sentences); real-time factor 0.12 | p95 <= 0.8 s PASS, WER <= 0.10 PASS |
| `voice_stop` | p95 153 ms; this machine 147 ms (Bluetooth speaker), dashboard tab 62 ms | <= 300 ms PASS |
| `wake` ("Hey Sanjaya", since replaced) | missed 1 of 19 (the OS voice); 1 false wake ("Sanjaya was the narrator...") in 4.6 min = 13 per hour | miss <= 0.10 PASS; false wakes per hour <= 0.5 **FAIL** |
| `proactive` | false alerts 0, missed 0, alert names the test 100% (after the change below) | PASS, PASS |
| `skills` | fires 100% when it should, 0% when it should not; selection offered the skill 100% | PASS, PASS |
| `mcp_select` | right tool 85% with 5, 20 and 40 tools all offered; 80% with the top 5 of 40; the right tool was in the top 5 in 95% | top-5 of 40 >= 0.80 PASS |
| `task5` | failure spoken at the right screen in all 3 failing sessions; false alerts 0; no repeats; 0.83 model calls per screen | PASS, PASS |
| `task6` | 6 of 6 reminders with the right text and time; 8.1 s per turn (Whisper + Gemma with one tool call) | >= 0.80 PASS |

- **The wake false-wake rate fails, and we keep the threshold.** One false wake in 4.6 minutes is 13 per hour; the
  word is the name alone, said by one held-out voice. On openWakeWord's 10.7 h validation audio the same model makes
  0.28 per hour. Both numbers are true; the short synthetic set cannot show a rate under 13 per hour
  without zero false wakes. An hour of the user's own normal talk (`evals add wake ... --wakes 0`) settles it.
- **`mcp_select` did not degrade with more tools.** The same three prompts fail at 5, 20 and 40 tools ("what changed
  in my repo", directions, an email draft): the model answered without a tool or chose a near neighbor. The top-5
  selection lost one more (the file tool was not among the 5 of 40). The brief expected accuracy to fall with more tools;
  with E4B and this catalog it did not. A run took 341 s, mostly the 40-tool requests (about 3,000 tokens of schemas).
- **Task 5: the first version let the model decide, and it raised false alerts.** The screen check asked for
  `alert: true/false`. Gemma said true for a script's `ModuleNotFoundError` traceback: proactive false alerts 0.143,
  task5 false alerts 0.20. Now the model reports `program` and `failed_tests` (names), and the code alerts only when
  the list is not empty: 0 false alerts in both evals, nothing missed. This is the agent's design, not the threshold,
  that changed. Two tests cover the rule (`tests/test_mcp.py`, `ScreenCheck`).
- **One label was wrong.** The jest screen's label named the file (`cart`) instead of the failing test ("adds two
  items"), which the model named correctly. The label is fixed in `evals/synthetic/alerts.py`, with a note.
- `cli.py smoke`: ALL PASS, including the new `voice` (Kokoro speech that Gemma transcribes) and `mcp` (the reminders
  server over stdio). `cli.py doctor`: all good.

### Barge-in by wake word, live in the dashboard (2026-11-25)

Measured with the "Hey Sanjaya" model; the code changes below still apply. The echo numbers of "Hey Jarvis" are in the
next section.

The user tried barge-in in the dashboard (Voice in the tab, no headphones): it worked sometimes. The event log showed
two replies of about a minute (28 and 23 sentences) where "Hey Sanjaya" fired only at the end, or not at all (the
Stop button ended it). The cause is the agent's own voice in the microphone. The user's 6 held-out clips, with
Kokoro speech mixed in at a level relative to the user's voice (the trainer's `usable_clips()`):

| Agent's voice in the mic | Wake scores of the 6 clips |
| ------------------------ | -------------------------- |
| none | 0.976 to 0.979: all wake |
| 25% of the user's level | 3 of 6 wake; the others fall to about 0.001 |
| 50% | 1 of 6 wakes |
| 100% or more | none wake |

- The score collapses instead of dropping a little, so a lower threshold while the agent speaks cannot help. The
  model never saw speech over speech in training.
- What the code changed: spoken turns ask for one to three plain sentences (`brain.SPOKEN`), so a reply rarely runs
  for a minute; only one dashboard tab plays the voice (two tabs played every reply twice, so twice the echo); the
  Stop button and key S always work.
- With headphones there is no echo, and barge-in by wake word behaves as in the table's first row.
- Options for speaking over the agent without headphones: retrain the wake model with Kokoro speech mixed in as
  background (tried: better under echo, more false wakes), or capture the microphone in the browser with `echoCancellation`, which removes the
  tab's own audio before the wake detector hears it (not built).

### Spoken replies read markdown aloud (2026-11-25)

Replies in the chat style ("**North Park**", bullets, emoji, tables) went to Kokoro as written, and espeak read the
symbols. `tts.speakable()` now turns a reply into plain sentences before it is cut into chunks (each list item ends
with a full stop, so it gets a pause), and spoken turns ask the model for plain speech. `voice.say` keeps the text as
written for the dashboard; `voice.chunk` holds what was said. Checked in a test dashboard (separate `AGENT_RUNTIME`):
"**Testing** the voice", an emoji and a two-item list were spoken as "Testing the voice.", "one." and "two."
- `task6` now sends `spoken=True`, as a dashboard voice turn does. Rerun with E4B: success 1.000 (text and time
  1.000), 7.5 s per turn (8.1 s before), so the plain-speech instruction did not hurt the tool call.

### The wake word is now "Hey Jarvis" (2026-11-27)

The user found "Hey Sanjaya" unreliable and chose "Jarvis". openWakeWord ships a pretrained `hey_jarvis` classifier in
release v0.5.1, beside the two feature models that we already pin, so nothing is trained on this machine.

- `hey_jarvis_v0.1.onnx`: input `x.1` of shape (1, 16, 96), output (1, 1), the same as our classifier, so
  `agent/voice/wake.py` runs it without a change. sha256 `94a13cfe...0d2cb`, in `agent/pins.py` (group `wake`).
- Its model card (`docs/models/hey_jarvis.md` upstream): about 200,000 synthetic clips from WaveGlow (LibriTTS) and
  VITS (VCTK) voices, with noise and room reverb; it warns that "jarvis" alone may also wake it. The release is from
  2023-09-02, before Kokoro, so the Kokoro voices of the eval are new to it.
- Kokoro, 7 voices, 0.5 s of room tone on each side: "Hey Jarvis." and "Hey Jarvis, what time is it?" score 0.993
  to 0.999. "Harvey said the jar is full" scores at most 0.30, "I saw Travis on Sunday" at most 0.12. "Hey Jervis"
  wakes it (0.89 to 0.998).
- False wakes on openWakeWord's 10.7 h validation features, live detector (2 s quiet time):

  | Threshold | 0.5 | 0.7 | 0.8 | 0.9 | 0.95 |
  | --------- | --- | --- | --- | --- | ---- |
  | False wakes per hour | 0.47 | 0.47 | 0.47 | 0.37 | 0.28 |

  `config.WAKE_THRESHOLD` is 0.5 (openWakeWord's default): a higher one buys little and loses clips said over the
  agent's echo (some score 0.96).
- Wake eval, synthetic cases rebuilt with the new phrase and near misses (48 clips, 5.1 min without the phrase):

  | Metric | Result | Threshold |
  | ------ | ------ | --------- |
  | `miss_rate` | 0.000 (19 of 19 woke, the OS voice included) | <= 0.10 PASS |
  | `echo_miss_rate` | 0.333 (4 of 12 missed, all at 50% echo; every 25% clip woke) | none |
  | `false_wakes` | 2, both "Jarvis was the butler in the old films." (the name alone, scores 0.89 and 0.996) | |
  | `false_wakes_per_hour` | 23.3 | <= 0.5 **FAIL** |

  No false wake on the ordinary talk, the Task 3 conversation or the agent's voice alone. The failure is the name
  without "hey", which the model card says can wake it. The eval threshold and the case stay as they are.
- The example names changed the same day (Ravi to Mike, Priya to Emily, Hyderabad to Chicago, the user's choice), so
  the synthetic cases were rebuilt and run again on E4B: `cli.py smoke` ALL PASS (Kokoro says "Remind me to call Mike
  at 4."); `asr` WER 0.044 (Whisper 0.018); `tts` WER 0.038, first audio p95 0.59 s; `task3` accuracy 1.0; `task6`
  success 1.0 at 8.97 s per turn. The wake numbers above are from this rebuild.
- Not measured yet: the user's own voice. Check it with `python cli.py wake test`, then add recordings with
  `python cli.py evals add wake`.

### The user's voice, and the Whisper check (2026-11-27)

The user said "Hey Jarvis" 5 times, about 2 s apart, recorded at once on three microphones (the MacBook Air
microphone, which is the default input; a Yeti Nano; a C922 webcam). The recordings stayed in a scratch folder and
were deleted.

- The classifier alone: **0 of 15 woke** at 0.5. Peak scores per phrase: MacBook 0.12, 0.47, 0.15, 0.39, under
  0.05; Yeti 0.15, 0.17, 0.06, 0.49, 0.11; C922 0.13, 0.43, 0.47, 0.42, 0.14. The level is not the cause: the
  synthetic positives still score 0.99 to 1.00 at -30 dB. The model learned from synthetic voices only.
- Whisper (warm, 2 s clips, 0.9 s each) heard "Hey Jarvis" 4 of 5 times and "Hey Cervis" once. Over the whole
  recording it heard "Hi Jarvis".
- Second stage added: a score from `WAKE_CHECK_THRESHOLD` (0.1) sends the last `WAKE_CHECK_S` (2 s) to Whisper,
  0.3 s after the first "maybe" frame, once per phrase. A word with a difflib ratio of `WAKE_NAME_SIMILARITY` (0.7)
  to "jarvis" or more wakes it. At 0.66 the eval's "Hey Travis" clips woke it 3 times ("Travis" and "Cervis" are
  both 0.67), so 0.7 it is.
- The same recordings through the two stages: each phrase alone (quiet time 0.8 s, so the 2 s gaps do not hide one):
  MacBook 4 of 5, Yeti 3 of 5, C922 3 of 5, **10 of 15** in all. With the live settings (2 s quiet time after a
  wake) the first phrase woke on every microphone.
- A wake through the second stage comes about 1.2 s after the phrase ends (0.3 s delay and Whisper), over the 0.3 s
  `wake` budget. A score of 0.5 or more still wakes at once.
- Wake eval, synthetic cases (48 clips, 5.1 min without the phrase), before / after:

  | Metric | Classifier only | With the Whisper check | Threshold |
  | ------ | --------------- | ---------------------- | --------- |
  | `miss_rate` | 0.000 | 0.000 | <= 0.10 PASS |
  | `echo_miss_rate` | 0.333 | **0.000** | none |
  | `false_wakes` | 2 | 2 (the same: "Jarvis was the butler in the old films.") | |
  | `false_wakes_per_hour` | 23.3 | 23.4 | <= 0.5 **FAIL** (as before) |

- `cli.py smoke`: ALL PASS. Unit tests cover the gate, the name check and the conversation without a model.
- Conversation mode (a follow-up without the wake word after the reply has played) is checked by unit tests only:
  the tab's `voice.ended` and this machine's end of playback, the follow-up trigger, and a silent follow-up that ends
  quietly. Not yet tried with the live microphone.

### The agent's voice is now `am_michael` (2026-11-27)

The user chose a male voice: `config.TTS_VOICE` changed from `af_heart` (US female) to `am_michael` (US male), one of
the 54 voices in `voices-v1.0.bin`. `python cli.py say --local` played it. The wake eval mixes the agent's own voice in
as echo, so its synthetic cases were rebuilt with the new voice.

| Metric | `af_heart` | `am_michael` | Threshold |
| ------ | ---------- | ------------ | --------- |
| `tts` first audio p95 | 0.59 s | 0.54 s | <= 0.8 PASS |
| `tts` WER | 0.038 | 0.038 | <= 0.10 PASS |
| `tts` real-time factor | 0.151 | 0.122 | |
| `wake` miss_rate | 0.000 | 0.000 | <= 0.10 PASS |
| `wake` echo_miss_rate | 0.000 | 0.167 (2 of 12, both at 50% echo: `am_echo` and `hf_beta`) | none |
| `wake` false_wakes_per_hour | 23.4 | 23.0 (the same 2 "Jarvis was the butler..." clips) | <= 0.5 FAIL (as before) |

The agent's voice alone still never wakes it, and Whisper hears the new voice as well as the old one (WER unchanged).

## Not verified yet

- Real use: the evals run on **synthetic** cases (macOS `say` voices, rendered screenshots, hand-written tool
  calls and memories). Text-to-speech is cleaner than a real voice, so ASR, VAD and Task 1 scores are optimistic.
  Add your own cases with `python cli.py evals add` for numbers that reflect your voice and your screen.
- Mic recording and the global hotkey need permissions that only you can grant. Please try
  `cli.py note` and the hotkey once you grant the permissions.
- Memory use with a Google Meet call running alongside.
- Session 2, the launcher app over a log out and log in, and the "Don't Allow" answer (see "The Microphone permission
  for the launchd daemon"). Listening with real speech in a daemon that launchd starts is verified.
- Session 2, a soak of several hours: the longest awake run is 1 h (see "Live microphone, launchd and a 1-hour soak").
  A memory leak of a few MB per hour shows only over 3 hours or more.
- The voice button of the dashboard chat with a real microphone: a unit test covers it with a mocked microphone. The
  picture and the daily brief in the chat were tried in the dashboard with E4B.
- Session 3: the voice in a Google Meet tab share (the user checks it on a second device).
- Session 3: the Talk button and the wake loop with a real microphone in the dashboard; the wake eval on new
  recordings of the user's voice and an hour of normal talk; the wake word under a running Meet call.
- Session 3: Task 6 by voice end to end (wake word, Whisper, the reminder spoken when due by the launchd daemon); the
  daemon must be reinstalled first (`cli.py daemon install`) to run the reminder watcher.
- Session 3: the E2B tier on the Session 3 evals (run with `--tier e2b`: it restarts the language model).
- Session 3: a remote MCP server over HTTP with a real API key; the HTTP path was written to the SDK's documented API
  but only stdio servers ran.
