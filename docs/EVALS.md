# Evals

Each component has an eval: test cases, a metric and a pass threshold in `agent/config.py`.
`python evals/run_all.py` prints the scorecard.

There is nothing to record. Cases come from two places, and an eval runs both:

| Source | Folder | Made by | Changed by |
|---|---|---|---|
| synthetic | `data/evals/synthetic/<component>/` | `evals/synthetic/` (code, committed) | rebuilt when that code changes |
| user | `data/evals/user/<component>/` | you, with `python cli.py evals add` | never touched by the tools |

Both folders are git-ignored. Every record in a result file has a `source` field, and when both sources ran,
the result file also has `by_source` metrics, so you can compare them.

```bash
python cli.py evals status        # cases per eval and the last run
python cli.py evals --run         # run every eval (builds the synthetic cases on the first run)
python cli.py evals --run asr     # run one
python cli.py evals build         # rebuild the synthetic cases
python cli.py evals add -h        # add your own case
```

## What the synthetic cases are

| Eval | Cases | How they are made |
|---|---|---|
| `tools` | 28 calls: safe, blocked, needs approval, bad arguments | written by hand in `evals/synthetic/text.py` |
| `tool_calls` | 16 prompts with the expected tool and arguments, 3 need no tool | written by hand |
| `memory` | 23 memories (6 are junk), 12 queries worded differently from the memory | written by hand |
| `asr` | 10 sentences, clean, noisy (12 dB SNR) and quiet | OS text-to-speech plus numpy noise |
| `vad` | silence, key clicks, music, speech with long pauses, speech with a clap | text-to-speech and generated sound; labels come from where the speech was placed |
| `task1` | 5 voice notes with 0 to 3 action items | text-to-speech with pauses |
| `task2` | 6 error screenshots, 2 clean screens, 2 short videos (with ffmpeg) | drawn with Pillow |
| `screen` | 4 frame sequences at 2 fps (error dialog, window switch, idle, log lines) | drawn with Pillow |
| `structured` | action-item texts, screen-help pictures, 2 spoken notes | reuses the above |
| `serving` | 3 fixed prompts, 3 repeats | built into the eval |

Speech needs the `synthesize_speech` hook of your OS plug-in (macOS has it). Without it, the audio evals
skip with a message, and text and screen evals still run.

**Read the scores with care.** We know the right answer by construction, so there is no labeling error. But
text-to-speech is cleaner than a real voice, and a drawn terminal is cleaner than a real screen. ASR, VAD and Task 1
scores are optimistic. A drop is a real regression; a pass does not prove it works for you.
Do not change `config.THRESHOLDS` to make a run pass.

## Add a case from your own use

Each command writes one case to `data/evals/user/` with an id `u001`, `u002`, ... and prints it.
The next `evals --run` includes it.

```bash
# The agent picked the wrong tool. Record what it should have done:
python cli.py evals add tool_calls "what did I say about the venue" --tool search_memory --args '{"query": "venue"}'
python cli.py evals add tool_calls "tell me a joke" --tool none

# A call the permission layer must stop, or must ask about:
python cli.py evals add tools --tool run_shell --args '{"command": "rm -rf ~"}' --expect blocked

# Memory: add memories, then a query that names their ids
python cli.py evals add memory "Parking is on level three of the blue garage"
python cli.py evals add memory "uh okay" --junk
python cli.py evals add query "where did I park" --relevant u001

# Speech: record a clip, then give the transcript
python cli.py record-clip clip.wav
python cli.py evals add asr clip.wav --text "Please send the report to Priya"
python cli.py evals add task1 note.wav --item "Send the report to Priya" --item "Book a room"
python cli.py evals add vad clip.wav --speech 1.2 4.8 --speech 7.0 9.5      # no --speech: no speech in the clip

# Screen: a screenshot or video, and what a correct answer must mention
python cli.py evals add task2 shot.png --problem --mention "torchvision" --mention "pip install|conda install"
python cli.py evals add task2 clean.png                                       # a screen with no problem

# Screen sampling: record, then mark when something important happens
python cli.py record-screen demo1 --seconds 60
python cli.py evals add screen demo1 --moment 12.5 "error dialog appears"

# JSON validity
python cli.py evals add structured --schema action_items --text "Call mum and pay rent by Friday."
```

In `--mention`, each flag is one group. The answer must contain at least one phrase (separate phrases with `|`)
from every group, in any case. Keep the phrases short and specific, so you can read why an answer passed or failed.

The files use the format of each eval (see the docstring of `evals/<component>/eval.py`), so you can also edit
them by hand. To label speech regions for `vad` by ear, [Audacity](https://www.audacityteam.org/) works: select a
region, press Cmd+B, then File > Export > Export Labels. Read the first column as `START` and the second as `END`.

Your cases hold your voice and your screen. They stay on your machine; do not commit them.
