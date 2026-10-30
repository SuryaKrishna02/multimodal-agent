"""Synthetic cases for the structured-output eval: inputs only. The eval checks the JSON, not the content."""

import contextlib
from pathlib import Path

import numpy as np

from agent.platforms.base import Unsupported
from evals.synthetic import audio, screens
from evals.synthetic.text import write_jsonl

IMAGES = ("module-missing", "syntax-error", "all-clean")
AUDIO_NOTES = (0, 2)  # indexes into audio.NOTES


def build_structured(out: Path) -> None:
    """Write action-item inputs as text and as speech, and screen-help inputs as pictures.

    Speech is skipped when the OS plug-in cannot synthesize it, so the text and picture cases still run.

    Args:
        out (Path): The case folder of the `structured` eval.
    """
    out.mkdir(parents=True, exist_ok=True)
    rows = [
        {"id": f"s{i:02d}", "schema": "action_items", "text": " ".join(s)} for i, (s, _) in enumerate(audio.NOTES, 1)
    ]
    for name in IMAGES:
        screens.render(screens.SHOTS[name][0]).save(out / f"{name}.png")
        rows.append({"id": f"s-{name}", "schema": "screen_help", "image": f"{name}.png"})
    with contextlib.suppress(Unsupported):
        for i in AUDIO_NOTES:
            name = f"note{i + 1:02d}.wav"
            audio.write_wav(out / name, audio.note_audio(audio.NOTES[i][0], np.random.default_rng(i)))
            rows.append({"id": f"s-{name[:-4]}", "schema": "action_items", "audio": name})
    write_jsonl(out / "items.jsonl", rows)
