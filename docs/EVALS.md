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
| `daemon` | 2 kills in the middle of a job, and the newest soak result | built into the eval; the soak comes from `scripts/soak.py` |
| `multimodal` | 8 screenshots and 10 speech clips, with a text query for each | drawn with Pillow and spoken by the OS voice |
| `consolidation` | 2 day logs (19 raw rows) with 15 queries | written by hand in `evals/synthetic/day.py` |
| `task3` | a 57 s morning conversation with 3 voices, and 8 questions (1 has no answer) | OS text-to-speech, 3 voices |
| `task4` | 8 days of memories (4 dev, 4 test), 6 preferences, 1 overfitting example | written by hand; the simulated user is code |

Speech needs the `synthesize_speech` hook of your OS plug-in (macOS has it). Without it, the audio evals
skip with a message, and text and screen evals still run.

**Read the scores with care.** We know the right answer by construction, so there is no labeling error. But
text-to-speech is cleaner than a real voice, and a drawn terminal is cleaner than a real screen. ASR, VAD and Task 1
scores are optimistic. A drop is a real regression; a pass does not prove it works for you.
Do not change `config.THRESHOLDS` to make a run pass.

## Session 2 evals

| Eval | Level | What it measures | How |
|---|---|---|---|
| `asr` (extended) | component | Word error rate of Gemma 4 and of Whisper on the same clips, and seconds per clip | each clip goes through both engines |
| `daemon` | component | Recovery time after a kill, finished steps that ran again (must be 0), lost jobs, memory growth and missed events over hours | a real daemon process on a throwaway runtime folder, killed with SIGKILL |
| `multimodal` | component | Recall of a screenshot or a clip by a text query, with text vectors, media vectors or both | top 3 and top 1 |
| `consolidation` | component | Faithfulness of each claim to the rows it cites, invalid citations, compression, recall before and after forgetting | a separate Gemma request judges each claim |
| `task3` | end-to-end | Questions about a conversation, hours later: correct answers, "I do not know" when nobody said it, answers that cite memory | the recording goes through the same listener as the microphone |
| `task4` | end-to-end | Preference compliance before and after feedback rounds, on held-out days; regressions; edits to locked blocks; one overfitting example | the simulated user gives feedback on dev days only |

**Held-out days in Task 4.** The simulated user gives feedback only on the 4 dev days. The 4 test days are only
scored, so `test_gain` is a gain on briefs that the reflection never saw. `regression_rate` counts test checks that
passed with the seed prompt and fail after the last round.

**The overfitting example.** The eval starts again from the seed prompt. It gives 3 rounds of feedback that is only
true for one dev day. The feedback is "Start the headline with the word Dentist" and "Always mention Dr. Lee in the
first bullet".

Then the eval checks two things. Does that day follow the rule (`overfit_dev_gain`)? Do the test days, which have no
dentist, now mention one (`overfit_test_leak`)? The reflection prompt asks for rules "that hold on every day". The
result shows whether that instruction is enough.

It was not. With the gate off, `overfit_test_leak` was 1.0 on E4B and E2B: every held-out brief mentioned the dentist,
and one said "Dr. Lee was mentioned in relation to the offsite agenda", although that day's memories never name Dr.
Lee. The locked identity block says "Never invent an event, a name, a time or a task", and the model still invented a
fact to obey a learned rule. An instruction is not a check.

**The generality gate is the check that code runs.** Before a new version becomes active, `evolve.reflect()` writes
briefs with the parent and the new version for a few recent days (`config.EVOLVE_GATE_DAYS`). `evolve.ungrounded()`
lists the names and numbers in a brief that none of that day's memories say:

- a name is a capitalized word in a bullet, not its first word, that no memory has in any case;
- a number or time in a bullet or the headline that no memory has (times compare in 24-hour form, so "2pm" grounds
  "14:00", and "six" grounds "6");
- the headline and headings are titles, so their capitalized words are not taken for names.

If the new version invents something that the parent did not (more than `config.EVOLVE_GATE_MAX_INVENTED`), the patch
is refused and `evolve.rejected` names the words. In the daemon the days come from the `briefs` table and
`tasks.day_memories()`. In the eval they are the dev days, the days that the simulated user saw, never a test day.
`overfit_patches_refused` counts refused overfit patches, `gate_refused` counts refusals in the normal rounds (a false
alarm costs a round), and `invented_rate` is the share of test briefs where the check finds something.

The check is a heuristic, and it misses things. A lower-case word ("the dentist") is not a name to it, and a headline
that starts with "Dentist" passes, because headline words are not read as names (E2B wrote "Dentist Summary" on a
day with no dentist, and the check found nothing in that brief). It catches the invented person, which is the harm,
and one such day is enough to refuse the patch. Read the refused words in the event log.

**The soak test runs beforehand.** `python scripts/soak.py --hours 3` runs its own daemon on a throwaway runtime folder
and queues a replay of the synthetic conversation every 5 minutes. The `daemon` eval reads the newest file in
`evals/results/soak/`. The soak metrics stay empty until you run it.

**The judges share the writer's blind spots.** Consolidation faithfulness and your own Task 4 preferences are judged by
Gemma 4, the same model family that wrote the text. A judge can miss what the writer missed. Read the failed claims.

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
python cli.py evals add asr clip.wav --text "Please send the report to John Doe"
python cli.py evals add task1 note.wav --item "Send the report to John Doe" --item "Book a room"
python cli.py evals add vad clip.wav --speech 1.2 4.8 --speech 7.0 9.5      # no --speech: no speech in the clip

# Screen: a screenshot or video, and what a correct answer must mention
python cli.py evals add task2 shot.png --problem --mention "torchvision" --mention "pip install|conda install"
python cli.py evals add task2 clean.png                                       # a screen with no problem

# Screen sampling: record, then mark when something important happens
python cli.py record-screen demo1 --seconds 60
python cli.py evals add screen demo1 --moment 12.5 "error dialog appears"

# JSON validity
python cli.py evals add structured --schema action_items --text "Call mum and pay rent by Friday."

# Session 2
python cli.py evals add multimodal shot.png --query "the screen with the failing build"
python cli.py evals add consolidation myday.json        # format: the docstring of evals/consolidation/eval.py
python cli.py evals add task3 morning.wav --start 09:00 --question "Which venue did we pick?" --mention "lakeside"
python cli.py evals add task3 --to u001 --question "What did we decide about catering?" --unanswerable
python cli.py evals add task4 --split test --memory "(09:00, fact) Demo is on Friday at 16:00" --memory "(11:00, action_item) Send slides (due tomorrow)"
python cli.py evals add task4 --prefer "Mention the weather first" --complaint "Start with the weather."
```

For `task4`, your folder needs both days and preferences. Your preferences have no code check, so Gemma judges them.

In `--mention`, each flag is one group. The answer must contain at least one phrase (separate phrases with `|`)
from every group, in any case. Keep the phrases short and specific, so you can read why an answer passed or failed.

The files use the format of each eval (see the docstring of `evals/<component>/eval.py`), so you can also edit
them by hand. To label speech regions for `vad` by ear, [Audacity](https://www.audacityteam.org/) works: select a
region, press Cmd+B, then File > Export > Export Labels. Read the first column as `START` and the second as `END`.

Your cases hold your voice and your screen. They stay on your machine; do not commit them.
