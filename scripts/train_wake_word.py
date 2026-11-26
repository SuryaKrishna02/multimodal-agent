"""Train the "Hey Sanjaya" wake-word model on this machine, one time. See docs/WAKE_WORD.md.

    python scripts/train_wake_word.py setup           # training venv: torch + openWakeWord at a pinned commit
    python scripts/train_wake_word.py data            # download the pinned training data (about 19 GB) and check it
    python scripts/train_wake_word.py train --trial   # 1,000 clips and 5,000 steps, to time each stage
    python scripts/train_wake_word.py train           # the full run: upstream defaults, 10,000 clips, 50,000 steps
    python scripts/train_wake_word.py train --name echo2 --clips-from full --echo --layer-size 64 --rounds 2 \
        --only augment_clips train_model              # over the agent's echo, on the clips of the full run (25 min)
    python scripts/train_wake_word.py export          # copy the model to .cache/models/wakeword/, print its sha256

Run it with the agent venv's Python: the near-miss clips are made with Kokoro there.

What is trained: only openWakeWord's last stage, a small classifier that reads 16 speech embeddings (1.28 s of audio)
and says whether the phrase just ended. The two stages before it (mel spectrogram, Google's speech embedding) are
frozen. Positive clips: Kokoro speaks "hey sanjaya" with mixed voices (scripts/wakeword/generate_samples.py).
Negatives: near-miss phrases, and 2,000 hours of general audio as precomputed features. Clips get AudioSet sounds as
background; with `--echo`, also Kokoro speech in the agent's voice (its echo in the microphone, `_echo_clips`).

Everything goes to `config.WAKE_TRAIN_DIR` (ignored by Git). The training venv has torch and is separate from the
agent's venv, which never needs it. The torch steps run in the training venv.
"""

import argparse
import importlib
import importlib.util
import itertools
import json
import shutil
import subprocess
import sys
import time
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from agent import config, platforms  # noqa: E402

TRAIN = config.WAKE_TRAIN_DIR
DATA = TRAIN / "data"
OWW_COMMIT = "368c03716d1e92591906a84949bc477f3a834455"  # openWakeWord main, 2025-12-30: TensorFlow only for tflite
OWW_ZIP = f"https://github.com/dscripka/openWakeWord/archive/{OWW_COMMIT}.zip"
HF = "https://huggingface.co/datasets"
FEATURES_REV = "985bf1b47e7f19c07741af82bfe32d5a9dc56096"  # davidscripka/openwakeword_features
FEATURES = f"{HF}/davidscripka/openwakeword_features/resolve/{FEATURES_REV}"
RIR_REV = "b824a1ef2821f112fda0b9cb26e4278c62b425bb"  # davidscripka/MIT_environmental_impulse_responses
AUDIOSET_REV = "0c609e8302cf139307f639c57652032af0a88041"  # agkphysics/AudioSet (parquet, CC BY 4.0)
OWW_MODELS = "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1"

# (url, path under DATA, sha256). Hashed from the files downloaded on 2026-11-24; the Hugging Face files match the
# sha256 that the Hub lists for them (LFS oid). The GitHub release lists no digest.
DOWNLOADS = [
    (
        f"{FEATURES}/openwakeword_features_ACAV100M_2000_hrs_16bit.npy",
        "openwakeword_features_ACAV100M_2000_hrs_16bit.npy",
        "721a66d0682c65a1b5c1da0aa109409cede1d20e28b15235c344b000cbb7654f",
    ),
    (
        f"{FEATURES}/validation_set_features.npy",
        "validation_set_features.npy",
        "a56a8a0f8e0efb91900acc6de4c0cdf4c564842e8475a7d49b36c039e17a690f",
    ),
    (
        f"{OWW_MODELS}/melspectrogram.onnx",
        "models/melspectrogram.onnx",
        "ba2b0e0f8b7b875369a2c89cb13360ff53bac436f2895cced9f479fa65eb176f",
    ),
    (
        f"{OWW_MODELS}/embedding_model.onnx",
        "models/embedding_model.onnx",
        "70d164290c1d095d1d4ee149bc5e00543250a7316b59f31d056cff7bd3075c1f",
    ),
    (
        f"{HF}/agkphysics/AudioSet/resolve/{AUDIOSET_REV}/data/bal_train/08.parquet",
        "audioset/bal_train_08.parquet",
        "db17feaf9f5bd4bcabde4e7fe05161dcc0e8d1457f5b21bd2fed2675d99deb49",
    ),
    (
        f"{HF}/agkphysics/AudioSet/resolve/{AUDIOSET_REV}/data/bal_train/09.parquet",
        "audioset/bal_train_09.parquet",
        "1121a990b52fdc9805adf9e6d704cc053c7fed02b5ce6f5b83cf73d3e9eccad7",
    ),
]

TRIAL = {"n_samples": 1000, "n_samples_val": 500, "steps": 5000}
FULL = {"n_samples": 10000, "n_samples_val": 2000, "steps": 50000}  # the defaults of examples/custom_model.yml


def _setup_module():
    """Load scripts/setup.py, for its checksum-verified, resumable download."""
    spec = importlib.util.spec_from_file_location("agent_setup", ROOT / "scripts" / "setup.py")
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def train_python() -> Path:
    """Return the Python of the training venv.

    Returns:
        Path: The executable (it may not exist before `setup`).
    """
    return platforms.current().venv_python(TRAIN / ".venv")


def cmd_setup(_a) -> None:
    """Create the training venv and install the pinned packages, then openWakeWord without its declared extras."""
    env = TRAIN / ".venv"
    if not env.exists():
        venv.create(env, with_pip=True)
    py = str(train_python())
    reqs = ROOT / "requirements" / "wakeword-train.txt"
    subprocess.run([py, "-m", "pip", "install", "-q", "-r", str(reqs)], check=True)
    # Its 'full' extra pins torchaudio<1 and TensorFlow 2.8, which no current torch or Python can install.
    subprocess.run([py, "-m", "pip", "install", "-q", "--no-deps", f"openwakeword @ {OWW_ZIP}"], check=True)
    print(f"training venv ready: {py}")


def _rirs() -> None:
    """Download the 270 room impulse responses of the MIT survey (16 kHz WAVs, 8 MB), at a pinned revision."""
    from huggingface_hub import snapshot_download

    snapshot_download(
        "davidscripka/MIT_environmental_impulse_responses",
        repo_type="dataset",
        revision=RIR_REV,
        allow_patterns="16khz/*.wav",
        local_dir=DATA / "mit_rirs_download",
    )
    out = DATA / "mit_rirs"
    out.mkdir(parents=True, exist_ok=True)
    for wav in (DATA / "mit_rirs_download" / "16khz").glob("*.wav"):
        shutil.copy2(wav, out / wav.name)
    print(f"ok        {len(list(out.glob('*.wav')))} room impulse responses")


def cmd_data(_a) -> None:
    """Download the pinned training data and convert the AudioSet clips to 16 kHz WAVs."""
    setup = _setup_module()
    for url, rel, sha in DOWNLOADS:
        setup.fetch(url, DATA / rel, sha)
    subprocess.run([str(train_python()), __file__, "_prepare"], check=True)


def cmd_prepare(_a) -> None:
    """Fetch the room impulse responses and decode AudioSet into 16 kHz mono WAVs (runs in the training venv)."""
    import io

    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly

    pq = importlib.import_module("pyarrow.parquet")  # in the training venv only, so the agent's pyright cannot see it
    _rirs()
    out = DATA / "background"
    out.mkdir(parents=True, exist_ok=True)
    for part in sorted((DATA / "audioset").glob("*.parquet")):
        table = pq.read_table(part, columns=["audio"])
        for i, cell in enumerate(table.column("audio").to_pylist()):
            dest = out / f"{part.stem}_{i:04d}.wav"
            if dest.exists():
                continue
            samples, rate = sf.read(io.BytesIO(cell["bytes"]), dtype="float32", always_2d=True)
            mono = resample_poly(samples.mean(axis=1), 16000, rate)
            sf.write(dest, (np.clip(mono, -1, 1) * 32767).astype(np.int16), 16000, subtype="PCM_16")
    print(f"ok        {len(list(out.glob('*.wav')))} background clips")


def training_config(run: Path, sizes: dict, echo: bool = False) -> dict:
    """Build openWakeWord's training config (the keys of its examples/custom_model.yml).

    Args:
        run (Path): The run folder.
        sizes (dict): `n_samples`, `n_samples_val` and `steps`.
        echo (bool, optional): Also mix in the echo clips (Kokoro speech) as background. Defaults to False.

    Returns:
        dict: The config.
    """
    return {
        "model_name": config.WAKE_MODEL.stem,
        "target_phrase": [config.WAKE_PHRASE],
        "custom_negative_phrases": [],  # the near-miss clips exist before train.py runs (see _near_misses)
        **sizes,
        "tts_batch_size": 50,
        "augmentation_batch_size": 16,
        "piper_sample_generator_path": str(ROOT / "scripts" / "wakeword"),  # our Kokoro generate_samples
        "output_dir": str(run),
        "rir_paths": [str(DATA / "mit_rirs")],
        "background_paths": [str(DATA / "background")] + ([str(DATA / "echo")] if echo else []),
        "background_paths_duplication_rate": [1] + ([config.WAKE_ECHO_DUPLICATION] if echo else []),
        "false_positive_validation_data_path": str(DATA / "validation_set_features.npy"),
        "augmentation_rounds": 1,
        "feature_data_files": {"ACAV100M_sample": str(DATA / "openwakeword_features_ACAV100M_2000_hrs_16bit.npy")},
        "batch_n_per_class": {"ACAV100M_sample": 1024, "adversarial_negative": 50, "positive": 50},
        "model_type": "dnn",
        "layer_size": 32,
        "max_negative_weight": 1500,
        "target_false_positives_per_hour": 0.2,
    }


def _install_feature_models(py: str) -> None:
    """Copy the two frozen feature models to where openWakeWord looks for them in the training venv."""
    where = subprocess.run(
        [py, "-c", "import openwakeword, os; print(os.path.dirname(openwakeword.__file__))"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    models = Path(where) / "resources" / "models"
    models.mkdir(parents=True, exist_ok=True)
    for name in ("melspectrogram.onnx", "embedding_model.onnx"):
        shutil.copy2(DATA / "models" / name, models / name)


def _near_misses(run: Path, sizes: dict) -> None:
    """Write the near-miss clips before train.py runs, so it skips its own DeepPhonemizer step.

    train.py makes these only when its folder holds fewer than 95% of the clips it wants.
    """
    spec = importlib.util.spec_from_file_location(
        "generate_samples", ROOT / "scripts" / "wakeword" / "generate_samples.py"
    )
    shim = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(shim)  # type: ignore[union-attr]
    phrases = shim.near_miss_phrases()
    for folder, n in (("negative_train", sizes["n_samples"]), ("negative_test", sizes["n_samples_val"])):
        out = run / config.WAKE_MODEL.stem / folder
        have = len(list(out.glob("*.wav"))) if out.exists() else 0
        if have < n:
            print(f"near misses     {n - have} clips of {len(phrases)} phrases into {folder}", flush=True)
            shim.generate_samples(phrases, n - have, out, length_scales=(0.75, 1.0, 1.25))


def echo_texts() -> list[str]:
    """Collect sentences for the echo clips from this repo's documentation, as plain speech.

    The docs give varied, reply-like English. Sentences that hold the name or a near miss of it are left out, so the
    echo never says the phrase.

    Returns:
        list[str]: Sentences of 40 to 200 characters.
    """
    from agent.voice import tts

    names = ("sanj", "sonia", "sonya", "hey")
    text = " ".join(p.read_text() for p in sorted([ROOT / "README.md", *(ROOT / "docs").glob("*.md")]))
    sentences = tts.sentences(tts.speakable(text), max_chars=200, first_chars=200)
    return [x for x in sentences if 40 <= len(x) <= 200 and not any(n in x.lower() for n in names)]


def _echo_clip(sentences, voice: str):
    """Speak sentences in one voice until the clip is `config.WAKE_ECHO_SECONDS` long; return 16 kHz int16 audio."""
    import numpy as np
    from scipy.signal import resample_poly

    from agent.voice import tts

    parts, seconds = [], 0.0
    while seconds < config.WAKE_ECHO_SECONDS:
        audio = tts.synthesize(next(sentences), voice)
        parts.append(audio)
        seconds += len(audio) / config.TTS_SAMPLE_RATE
    mono = resample_poly(np.concatenate(parts), config.SAMPLE_RATE, config.TTS_SAMPLE_RATE)
    mono = mono[: int(config.WAKE_ECHO_SECONDS * config.SAMPLE_RATE)]
    return (np.clip(mono, -1, 1) * 32767).astype(np.int16)


def _echo_clips() -> None:
    """Write `config.WAKE_ECHO_CLIPS` background clips of Kokoro speech, half of them in the agent's own voice."""
    import soundfile as sf

    out = DATA / "echo"
    out.mkdir(parents=True, exist_ok=True)
    missing = [i for i in range(config.WAKE_ECHO_CLIPS) if not (out / f"echo_{i:04d}.wav").exists()]
    if not missing:
        return
    print(f"echo clips      {len(missing)} clips of Kokoro speech into {out}", flush=True)
    sentences = itertools.cycle(echo_texts())
    others = itertools.cycle(config.WAKE_ECHO_OTHER_VOICES)
    for i in missing:
        voice = config.TTS_VOICE if i % 2 == 0 else next(others)
        sf.write(out / f"echo_{i:04d}.wav", _echo_clip(sentences, voice), config.SAMPLE_RATE, subtype="PCM_16")


def _reuse_clips(run: Path, source: str) -> None:
    """Link the generated clip folders of an earlier run into this one, so generate_clips has nothing to do."""
    for folder in ("positive_train", "positive_test", "negative_train", "negative_test"):
        src, dest = TRAIN / source / config.WAKE_MODEL.stem / folder, run / config.WAKE_MODEL.stem / folder
        if not src.is_dir():
            sys.exit(f"{src} not found: train the run '{source}' first")
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            dest.symlink_to(src.resolve(), target_is_directory=True)


def usable_clips() -> tuple[list[Path], list[Path]]:
    """Split your recordings into training and test clips, without the ones that cannot teach the phrase.

    A clip without speech is dropped, and so is one whose speech runs to its last 50 ms: its end was cut off, and a
    cut "hey sanj..." would teach the model to fire too early. Every `config.WAKE_USER_TEST_EVERY`th usable clip is
    kept for testing.

    Returns:
        tuple[list[Path], list[Path]]: (training clips, test clips).
    """
    from agent.senses import audio

    vad, usable = audio.VAD(), []
    for clip in sorted(config.WAKE_USER_CLIPS.glob("*.wav")):
        samples = audio.load_wav(clip)
        speech = vad.segments(samples)
        if speech and speech[-1][1] < len(samples) / config.SAMPLE_RATE - 0.05:
            usable.append(clip)
    test = usable[config.WAKE_USER_TEST_EVERY - 1 :: config.WAKE_USER_TEST_EVERY]
    return [c for c in usable if c not in test], test


def _user_clips(run: Path) -> int:
    """Copy your usable training recordings into the positive clips, trimmed, each `WAKE_USER_REPEAT` times.

    The silence after the phrase is cut to 0.1 s, like the Kokoro clips: openWakeWord learns from the end of a clip.

    Returns:
        int: How many recordings were used.
    """
    import soundfile as sf

    from agent.senses import audio

    train, test = usable_clips()
    out = run / config.WAKE_MODEL.stem / "positive_train"
    out.mkdir(parents=True, exist_ok=True)
    vad = audio.VAD()
    for clip in train:
        samples = audio.load_wav(clip)
        start, end = vad.segments(samples)[0][0], vad.segments(samples)[-1][1]
        cut = samples[max(0, int((start - 0.2) * 16000)) : int((end + 0.1) * 16000)]
        for i in range(config.WAKE_USER_REPEAT):
            sf.write(out / f"user_{clip.stem}_{i}.wav", (cut * 32767).astype("int16"), 16000, subtype="PCM_16")
    (run / "user_test_clips.txt").write_text("\n".join(str(c) for c in test))
    if train:
        print(
            f"your clips      {len(train)} x {config.WAKE_USER_REPEAT} into positive_train, {len(test)} kept for testing"
        )
    return len(train)


def cmd_train(a) -> None:
    """Run openWakeWord's three training stages and time each one."""
    py = str(train_python())
    run, sizes = run_dir(a), dict(TRIAL if a.trial else FULL)
    sizes["steps"] = a.steps or sizes["steps"]
    run.mkdir(parents=True, exist_ok=True)
    cfg = run / "config.json"  # JSON is valid YAML, so train.py reads it with yaml.load
    settings = training_config(run, sizes, echo=a.echo)
    settings["max_negative_weight"] = a.max_negative_weight or settings["max_negative_weight"]
    settings["layer_size"] = a.layer_size or settings["layer_size"]
    settings["augmentation_rounds"] = a.rounds or settings["augmentation_rounds"]
    cfg.write_text(json.dumps(settings, indent=1))
    _install_feature_models(py)
    start = time.monotonic()
    if a.clips_from:
        _reuse_clips(run, a.clips_from)
    else:
        _user_clips(run)
        _near_misses(run, sizes)
    timings = {"near_misses": round(time.monotonic() - start)}
    if a.echo:
        start = time.monotonic()
        _echo_clips()
        timings["echo_clips"] = round(time.monotonic() - start)
    train_py = str(ROOT / "scripts" / "wakeword" / "run_train.py")  # train.py, reading WAVs with soundfile
    with open(run / "train.log", "a") as log:
        for stage in [f"--{s}" for s in a.stages or ("generate_clips", "augment_clips", "train_model")]:
            start = time.monotonic()
            print(f"{stage[2:]:15s} started {time.strftime('%H:%M:%S')}", flush=True)
            subprocess.run([py, train_py, "--training_config", str(cfg), stage], check=True, stdout=log, stderr=log)
            timings[stage[2:]] = round(time.monotonic() - start)
            print(f"{stage[2:]:15s} took {timings[stage[2:]] / 60:.1f} min", flush=True)
    (run / "timings.json").write_text(json.dumps(timings, indent=1))


def run_dir(a) -> Path:
    """Return the run folder: `--name`, else "trial" or "full".

    Args:
        a (argparse.Namespace): The parsed command line.

    Returns:
        Path: The folder under `config.WAKE_TRAIN_DIR`.
    """
    return TRAIN / (a.name or ("trial" if a.trial else "full"))


def cmd_export(a) -> None:
    """Write the trained model to `config.WAKE_MODEL` as one file and print its sha256 for agent/pins.py.

    torch's ONNX exporter keeps the weights in a second file (`.onnx.data`); one file is simpler to pin and ship.
    """
    run = run_dir(a)
    model = run / f"{config.WAKE_MODEL.stem}.onnx"
    if not model.exists():
        sys.exit(f"{model} not found: run `train` first")
    config.WAKE_MODEL.parent.mkdir(parents=True, exist_ok=True)
    merge = "import onnx, sys; onnx.save(onnx.load(sys.argv[1]), sys.argv[2], save_as_external_data=False)"
    subprocess.run([str(train_python()), "-c", merge, str(model), str(config.WAKE_MODEL)], check=True)
    print(f"{config.WAKE_MODEL}  sha256 {_setup_module().sha256(config.WAKE_MODEL)}")


def main() -> None:
    """Parse the command line and run one step."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("setup").set_defaults(fn=cmd_setup)
    sub.add_parser("data").set_defaults(fn=cmd_data)
    sub.add_parser("_prepare").set_defaults(fn=cmd_prepare)
    for name, fn in (("train", cmd_train), ("export", cmd_export)):
        p = sub.add_parser(name)
        p.add_argument("--trial", action="store_true", help="the short run: 1,000 clips, 5,000 steps")
        p.add_argument("--name", help="the run folder under .cache/wakeword-train (default: trial or full)")
        p.set_defaults(fn=fn)
        if name == "train":
            p.add_argument("--steps", type=int, help="training steps (default: 5,000 trial, 50,000 full)")
            p.add_argument("--max-negative-weight", type=float, help="final weight of negatives (default 1500)")
            p.add_argument("--echo", action="store_true", help="also train over Kokoro speech (the agent's echo)")
            p.add_argument("--layer-size", type=int, help="width of the classifier's hidden layers (default 32)")
            p.add_argument("--rounds", type=int, help="augmented copies of each clip, each with new noise (default 1)")
            p.add_argument(
                "--clips-from", help="link the generated clips of this earlier run instead of making new ones"
            )
            p.add_argument(
                "--only",
                dest="stages",
                nargs="+",
                choices=["generate_clips", "augment_clips", "train_model"],
                help="run only these stages of openwakeword/train.py",
            )
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
