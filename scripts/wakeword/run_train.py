"""Run openWakeWord's train.py with one change: WAV files are read with soundfile.

torchaudio 2.9 and later reads audio only through TorchCodec, which needs FFmpeg's libraries, and has no `info()`
any more. openWakeWord and torch-audiomentations call both. Every clip here is a
plain 16-bit WAV (our Kokoro clips, the MIT room responses and the AudioSet clips that train_wake_word.py decodes),
so soundfile reads them without FFmpeg.

A second change: train.py computes features in a pool of half the CPUs, and loads training batches in DataLoader
worker processes. Both pickle a lambda, which works only with Linux's `fork`; macOS starts workers with `spawn`.
Here train.py sees 2 CPUs (one feature process, no pool), and every DataLoader loads in the main process.

A third: train.py always tries a TensorFlow Lite conversion at the end (see the note at the bottom).

Runs in the training venv:

    python scripts/wakeword/run_train.py --training_config config.json --train_model
"""

import importlib
import logging
import os
import runpy
import sys
from types import SimpleNamespace
from typing import Any

import soundfile as sf

# In the training venv only, so the agent's pyright cannot see them.
torch: Any = importlib.import_module("torch")
torchaudio: Any = importlib.import_module("torchaudio")
train: Any = importlib.import_module("openwakeword.train")


def load(uri, frame_offset: int = 0, num_frames: int = -1, **_kwargs):
    """Read a WAV the way torchaudio.load returns it: a (channels, samples) float tensor and the sample rate."""
    stop = None if num_frames < 0 else frame_offset + num_frames
    data, rate = sf.read(str(uri), start=frame_offset, stop=stop, dtype="float32", always_2d=True)
    return torch.from_numpy(data.T.copy()), rate


def info(uri, **_kwargs) -> SimpleNamespace:
    """Read a WAV header the way torchaudio.info returned it (the fields that the callers use)."""
    meta = sf.info(str(uri))
    return SimpleNamespace(num_frames=meta.frames, sample_rate=meta.samplerate, num_channels=meta.channels)


torchaudio.load, torchaudio.info = load, info  # for every caller in train.py, data.py and torch-audiomentations
os.cpu_count = lambda: 2  # train.py uses cpu_count() // 2 feature processes: one, no pool
_DataLoader = torch.utils.data.DataLoader


def main_process_loader(*args, **kwargs):
    """A DataLoader without worker processes (prefetch_factor is allowed only with workers)."""
    kwargs.pop("prefetch_factor", None)
    return _DataLoader(*args, **{**kwargs, "num_workers": 0})


torch.utils.data.DataLoader = main_process_loader
logging.basicConfig(level=logging.INFO)  # train.py reports recall and false positives per hour at INFO
script = str(train.__file__)
sys.argv = [script, *sys.argv[1:]]
try:
    runpy.run_path(script, run_name="__main__")
except ModuleNotFoundError as e:
    # train.py's --convert_to_tflite defaults to the string "False", which is true, so after it saves the ONNX
    # model it always tries the TensorFlow Lite conversion. We use ONNX only, so a missing onnx_tf is the end.
    if e.name != "onnx_tf":
        raise
    print("ONNX model saved; TensorFlow Lite conversion skipped (onnx_tf is not installed).")
