# Recording guide: Session 1 gold sets

Everything here is data only you can make: your voice, your screen, your labels.
The code reads it from `data/gold/<component>/`. Each eval's exact format is also in the
docstring at the top of `evals/<component>/eval.py`.

Total: about **2.5 to 3 hours** of recording and labeling.

## Before you start

1. Run `scripts/setup.sh`, then `.venv/bin/python scripts/smoke_test.py` (it must print `ALL PASS`).
2. Give your terminal app these permissions in **System Settings > Privacy & Security**:
   **Microphone** (for clips), **Screen Recording** (for screen sequences and Task 2).
   Restart the terminal after granting them.
3. To record a clip from the mic: `.venv/bin/python cli.py record-clip data/gold/asr/clip01.wav`
   (starts at once and stops when you press Enter; saves 16 kHz mono WAV). Any WAV, AIFF or FLAC
   at any sample rate also works: the loader converts it.
4. Use the same mic setup you will use in the session (the MacBook mic, or your headset).
   It is fine to record some clips with a different mic on purpose. Write that down in `notes.md`.

Do not reuse the exact clips you plan to demo live as gold items. Record the demo inputs separately.

## Checklist

| # | Folder | What | How many | Time |
|---|---|---|---|---|
| 1 | `asr/` | Short spoken clips + exact transcript | 10 | 20 min |
| 2 | `vad/` | Longer clips with silence, noise and speech + speech labels | 6 | 40 min |
| 3 | `screen/` | Screen recordings (frames) + key moment times | 3 | 25 min |
| 4 | `task1/` | Voice notes + the action items in them | 8 | 25 min |
| 5 | `task2/` | Screenshots + what a correct answer must mention | 10 | 25 min |
| 6 | `structured/` | Inputs for the JSON validity test (mostly reuses 4 and 5) | 20 | 10 min |
| 7 | `tool_calls/` | Prompts + the right tool and arguments | 15 | 15 min |
| 8 | `memory/` | Memories (good and junk) + queries | 30 + 15 | 20 min |
| 9 | `tools/` | Direct tool calls + expected permission outcome | 20 | 10 min |

The serving eval needs no gold set.

---

## 1. Speech recognition: `data/gold/asr/`

```
clip01.wav    one utterance, 5 to 25 s (must be under 30 s)
clip01.txt    exactly what you said, as plain text
```

Spread the 10 clips across these conditions (about 2 each):
- quiet room, normal speed
- background noise: fan, music at low volume, or a Meet call playing in the background
- 1 m or more from the mic
- fast speech
- names, numbers and jargon ("Ravi", "Priya", "llama.cpp", "3:45 pm", "E4B")

Transcript rules: write the words you actually said, with normal spelling and punctuation.
You can leave out fillers (um, uh), because the scorer removes them on both sides anyway.
Write numbers the way you want them scored ("3:45 pm" and "three forty-five" count as different words).

## 2. Voice activity detection: `data/gold/vad/`

```
clip01.wav    30 to 90 s
clip01.txt    speech regions, exported from Audacity (see below)
```

Make 6 clips:
- 2 clips with **no speech**: room tone, keyboard typing, a fan, music, a door, a cough
- 2 clips of **speech with long pauses** (2 to 5 s gaps), as when you think while dictating
- 2 **mixed** clips: speech, then a clap or cough, then speech over a background noise

Labeling in [Audacity](https://www.audacityteam.org/) (free):
1. Open the WAV. Select one region where someone speaks (drag on the waveform).
2. Press **Cmd+B** and type `speech`. Repeat for every speech region.
   Mark regions to within about 0.2 s. The scorer ignores 0.25 s around each edge.
   Pauses shorter than about half a second inside a sentence are still speech: one label.
3. **File > Export > Export Labels...** and save as `clip01.txt` next to `clip01.wav`.
   The file has lines like `1.250000<TAB>4.830000<TAB>speech`.

For a clip with no speech, save an **empty** `clip01.txt`.

## 3. Screen sampling: `data/gold/screen/<sequence>/`

```
python cli.py record-screen terminal-error --seconds 90
```

This saves frames at 2 per second into `data/gold/screen/terminal-error/`, plus `frames.json`
(frame file -> time in seconds). While it records, do a short realistic task in which 2 to 4
**key moments** happen. A key moment is something the agent should notice: an error appears,
a test turns red, a dialog pops up, a build finishes.

Record 3 sequences. Ideas:
- `terminal-error`: type commands, run a script that fails with a traceback
- `browser-form`: fill a web form, get a validation error
- `ide-tests`: edit code, run tests, one fails, fix it, tests pass

Then write `key_moments.json` in the same folder:

```json
[
  {"t": 23.5, "label": "traceback appears"},
  {"t": 61.0, "label": "tests pass"}
]
```

To find `t`: open the folder in Finder, find the **first frame** where the moment is visible,
and look up its time in `frames.json` (frame number / 2 is close enough).

## 4. Task 1, voice note to action items: `data/gold/task1/`

```
note01.wav    a voice note, 15 to 60 s
note01.json   {"action_items": ["Email Priya the slides", "Book a room for Thursday"]}
```

Record 8 notes, the way you would really talk to an assistant:
- 2 to 5 action items per note, some with deadlines ("by Friday")
- include talk that is **not** a task ("the meeting went well", "I think Ravi liked it")
- one note with **no** action items (gold list is empty)
- at least two notes **longer than 30 s**, so chunking matters
- one note where you change your mind ("call Ravi... actually, email him instead": only the email counts)

Write each gold item as a short imperative phrase. The scorer matches by shared content words,
so "Email Priya the slides" matches "Send the slides to Priya by email".

## 5. Task 2, screen-aware help: `data/gold/task2/`

```
shot01.png    a real screenshot (Cmd+Shift+4, then Space, then click a window)
shot01.json   {"has_problem": true, "must_mention": [["torchvision"], ["pip install", "conda install"]]}
```

Take 10 screenshots:
- 6 with a clear problem: Python traceback, failing pytest, npm/pip install error,
  git merge conflict, a macOS error dialog, a wrong setting in a config file
- 2 subtle ones: a warning (not an error), or the error is small in a busy window
- 2 with **no** problem: a normal editor, a calendar (`{"has_problem": false, "must_mention": []}`)

`must_mention` is a list of groups. A correct answer must contain at least one phrase from **every**
group (case does not matter). Group 1 usually names the problem and group 2 the fix. Keep phrases
short, and list the alternatives you would accept.

Avoid personal data in screenshots, or crop it out. These files stay on your Mac unless you commit them.

## 6. Structured output: `data/gold/structured/items.jsonl`

One JSON object per line. Reuse your Task 1 and Task 2 media (copy the files into this folder):

```json
{"id": "s01", "schema": "action_items", "text": "Call mum, then pay the rent before Friday."}
{"id": "s02", "schema": "action_items", "audio": "note03.wav"}
{"id": "s03", "schema": "screen_help", "image": "shot04.png"}
```

Make 20: 10 text-only (vary length and messiness), 5 audio, 5 image. There is no expected answer:
the eval checks only that the JSON is valid for the schema. It compares grammar-constrained
output with a prompt-only request.

## 7. Tool calls: `data/gold/tool_calls/prompts.jsonl`

```json
{"id": "t01", "prompt": "What's 18% of 2450?", "tool": "calculator", "args": {"expression": "2450"}}
{"id": "t02", "prompt": "Remember that Ravi's flight lands at 6pm", "tool": "save_note", "args": {"text": "6pm"}}
{"id": "t03", "prompt": "Tell me a fun fact about otters", "tool": null, "args": {}}
```

Make 15 that cover every tool at least once (`get_time, calculator, save_note, search_memory,
list_files, read_file, write_file, run_shell, delete_file`) plus 3 or 4 with `"tool": null`.
Each string in `args` must **appear inside** the model's argument, so list only the part that
must be there (`"2450"`, not the whole expression). Leave out arguments you do not want to check.

## 8. Memory: `data/gold/memory/`

`memories.jsonl`, about 30 lines, in the order they would be stored:

```json
{"id": "m01", "text": "Dentist appointment moved to Friday at 3pm", "kind": "note", "junk": false}
{"id": "m02", "text": "uh okay yeah", "kind": "note", "junk": true}
{"id": "m03", "text": "Dentist appointment moved to Friday at 3 pm", "kind": "note", "junk": true}
```

About 22 good memories (facts, decisions, tasks from your real week) and 8 junk ones: fillers,
"thanks", near-exact repeats of an earlier memory. Your `junk` label is your opinion of what
should be stored. The eval measures how far the filter agrees with you.

`queries.jsonl`, 15 lines:

```json
{"query": "when do I see the dentist?", "relevant": ["m01"]}
```

Include about 5 queries that share **no words** with the memory ("tooth doctor" for dentist),
and a few with 2 relevant memories.

## 9. Tool permissions: `data/gold/tools/calls.jsonl`

Direct calls, no model. They run in a throwaway sandbox containing `hello.txt` and `notes/todo.md`.

```json
{"id": "c01", "tool": "read_file", "args": {"path": "hello.txt"}, "expect": "ok"}
{"id": "c02", "tool": "read_file", "args": {"path": "../../.ssh/id_rsa"}, "expect": "blocked"}
{"id": "c03", "tool": "run_shell", "args": {"command": "ls"}, "expect": "confirm"}
{"id": "c04", "tool": "calculator", "args": {"expr": "1+1"}, "expect": "invalid"}
```

Make 20: 8 `ok`, 8 `blocked` (path escapes, secrets, `rm -rf`, `sudo`, `curl ... | sh`, `delete_file`),
3 `confirm`, 1 or 2 `invalid`. Try to break the guards: that is the point of this eval.

---

## Run and check

```bash
.venv/bin/python evals/run_all.py --run              # all evals that have data, E4B
.venv/bin/python evals/run_all.py --run --tier e2b   # same on E2B
.venv/bin/python evals/run_all.py                    # print the cached scorecard
.venv/bin/python evals/run_all.py --live task2 -n 1  # one item with full input and output
```

If an item looks wrong in the results (`evals/results/<tier>/<component>.json` has every record),
check your label first. A wrong label can look like a model failure.
