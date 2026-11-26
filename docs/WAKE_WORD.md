# The wake word: "Hey Sanjaya"

The agent wakes when you say "Hey Sanjaya" (Sanjaya, the narrator of the Mahabharata who was given sight to see the
battlefield and report it). No ready-made model knows this phrase, so we train one, once, on this machine. The
running agent then needs only three small ONNX files.

## How it works

Every 80 ms of microphone audio goes through three models (`agent/voice/wake.py`):

| Stage | Model | Trained by |
| ----- | ----- | ---------- |
| 1 | Mel spectrogram: audio to 32 frequency bands per 10 ms | openWakeWord, frozen |
| 2 | Speech embedding: the last 76 mel frames to 96 numbers | Google, frozen (shared by every openWakeWord model) |
| 3 | Classifier: the last 16 embeddings (1.28 s) to a score from 0 to 1 | **us**: `hey_sanjaya.onnx` |

A score of `config.WAKE_THRESHOLD` (0.9) or more is a wake. After a wake the detector ignores the phrase for
`config.WAKE_REFRACTORY_S`, so one "Hey Sanjaya" wakes once. The code repeats openWakeWord's streaming steps in numpy;
a check against the openwakeword package gave the same scores (largest difference 1.2e-6).

## What the training learns from

| Data | Where it comes from | Size |
| ---- | ------------------- | ---- |
| Positive clips | Kokoro (the agent's own voice) says "hey sanjaya" with two voices mixed per clip, at varied speeds. The name is spoken from IPA (`config.PRONUNCIATIONS`): espeak would say "SAN-jay-uh", you say "SUN-juh-yuh". Six voices are held out for the eval. | 10,000 + 2,000 for validation |
| Your clips | `python cli.py wake record`: each usable clip is copied `config.WAKE_USER_REPEAT` (50) times; augmentation makes each copy different | about 1,000 |
| Near misses | Kokoro says "hey sanjay", "okay sanjaya", "hello sonia", "sanjaya" alone... (`config.WAKE_NEAR_GREETINGS` x `WAKE_NEAR_NAMES`) | 10,000 + 2,000 |
| Rooms and noise | MIT room impulse responses (270), two parts of AudioSet (about 2.8 h of speech, music and noise) | 1.5 GB |
| Echo (only with `--echo`) | Kokoro reads sentences from this repo's docs, half in the agent's voice: the agent heard by the microphone | 500 x 10 s |
| General audio | openWakeWord's precomputed features of 2,000 h of ACAV100M audio: everything that is not the phrase | 17.28 GB |
| False-wake check | openWakeWord's 10.7 h validation features | 185 MB |

The 17 GB file is the reason false wakes are rare: the classifier sees 5.6 million windows of other sound. Only
training needs it; delete `.cache/wakeword-train/` afterwards if you need the space.

## Train it

Run the commands with the venv's Python (`.venv/bin/python`, on Windows `.venv\Scripts\python.exe`):

```bash
python scripts/train_wake_word.py setup
python scripts/train_wake_word.py data
python cli.py wake record --count 40
python scripts/train_wake_word.py train
python scripts/train_wake_word.py export
python cli.py wake test
```

1. `setup` makes a separate training venv in `.cache/wakeword-train/.venv` with torch and openWakeWord (pinned
   commit, `requirements/wakeword-train.txt`). The agent's venv never gets torch.
2. `data` downloads and checks the pinned data (about 19 GB) and decodes AudioSet to 16 kHz WAVs.
3. `wake record` records your own clips. Say the phrase once per clip, at different distances and volumes. A clip
   whose speech runs to its end is skipped (its end was cut off), and every 4th usable clip is kept for testing.
4. `train` makes the clips (about 80 min for 24,000 Kokoro clips), computes features (8 min) and trains (7 min) on
   an M4. `train --trial` is a quick run with 1,000 clips.
5. `export` writes `.cache/models/wakeword/hey_sanjaya.onnx` and prints its sha256.
6. `wake test` listens and prints each wake with its score. Then add new recordings to the eval:
   `python cli.py evals add wake clip.wav --wakes 1` (and long recordings of normal talk with `--wakes 0`).

## What it took on a Mac

openWakeWord's automatic training is written for Linux with a GPU and Piper voices. All changes are in our files
(`scripts/train_wake_word.py`, `scripts/wakeword/`), none in openWakeWord. The table of eight problems and their
fixes is in [VERIFICATION.md](VERIFICATION.md#wake-word-training-on-the-mac).

## Results of the first model (2026-11-24, M4)

| | Your 6 test clips | Held-out Kokoro voices | macOS voices | False wakes per hour (10.7 h) |
| - | - | - | - | - |
| Without your clips | 0 of 6 | 12 of 12 | 1 of 8 | 1.12 at 0.5 |
| With your clips x50 | 6 of 6 | 12 of 12 | 1 of 8 | 1.12 at 0.5, **0.28 at 0.9** |

Your own recordings decided it: a model trained only on synthetic voices did not wake on a real voice. With too few
copies of your clips (1% of the positives) it still did not; at 10% it did.

## Speaking over the agent (barge-in without headphones)

Without headphones the microphone also hears the agent, and its voice drowns out the phrase: with the agent at 25% of
your loudness, the installed model woke on 2 of your 6 test clips. `train --echo` adds Kokoro speech in the agent's
voice as background, so the model learns the phrase over it:

```bash
python scripts/train_wake_word.py train --name echo2 --clips-from full --echo --layer-size 64 --rounds 2 --only augment_clips train_model
python scripts/train_wake_word.py export --name echo2
```

`--clips-from full` reuses the clips of the full run, so it takes about 30 minutes. The result (2026-11-25): it wakes
over the agent's voice about twice as often, but it also wakes by mistake more than twice as often (0.65 instead of
0.28 per hour on the validation set). The installed model is `echo2` since 2026-11-26, chosen for barge-in; the
first model is kept as `hey_sanjaya.v1.onnx` and a plain `train` still makes it. The numbers are in
[VERIFICATION.md](VERIFICATION.md#retraining-the-wake-model-over-the-agents-echo-2026-11-25). Headphones, or the
Stop button (key S), always work. `export` replaces `hey_sanjaya.onnx`: keep a copy of the old one first.

## If it wakes too often, or not at all

- Too often: raise `config.WAKE_THRESHOLD`, or record long stretches of your normal day without the phrase and add them
  to the eval to see where it fires.
- Not at all: record more clips in the room and at the distance where you use it, and train again. Check the peak
  that `wake record` prints: under 0.05 is too quiet.
- The Talk button (key T in the Voice tab) starts a spoken turn without the wake word.
