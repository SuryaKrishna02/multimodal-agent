"""Multimodal memory eval: find a screenshot or a voice clip by text query, with text, media or both vectors.

Cases: data/evals/{synthetic,user}/multimodal/
    module-missing.png, clip01.wav, ...      the things to remember
    queries.jsonl   {"query": "when port 8000 was taken", "item": "port-in-use.png", "modality": "image"}

Each item is stored the way the agent stores it: a screenshot with the text that Task 2 writes about it,
a clip with its Whisper transcript (or Gemma, if the accelerator has no Whisper), plus the image or audio
vector of EmbeddingGemma-2. Then each query searches three ways:
    text   only the text vectors (Session 1 memory)
    media  only the image/audio vectors (no model had to describe anything)
    both   the better score of the two for each row (the default search)

Metrics: <modality>_recall_<mode>, the share of queries whose item is in the top 3, for example image_recall_both,
and <modality>_top1_<mode>, the share whose item comes first. With 8 to 10 items per modality, the top 3 is easy
(chance alone is 30 to 37%), so the top-1 numbers tell the modes apart better.
"""

import json
from pathlib import Path

from PIL import Image

from agent import tasks
from agent.memory import episodic
from agent.platforms.base import Unsupported
from agent.senses import audio, stt
from evals import common
from evals.common import NoCases

LEVEL, USES_LLM = "component", True
NEEDS_EMBED_SERVER = True
MODES = ("text", "media", "both")
_stored: dict[str, dict[int, str]] = {}  # case folder -> {row id: item file name}


def load_cases(cases_dir: Path):
    """Load the queries. The items are stored when the first query of a folder runs.

    Args:
        cases_dir (Path): The case folder of this eval.

    Returns:
        list[dict]: One item per query, with its folder.

    Raises:
        NoCases: If `queries.jsonl` is missing.
    """
    q = cases_dir / "queries.jsonl"
    if not q.exists():
        raise NoCases(cases_dir)
    rows = [json.loads(x) for x in q.read_text().splitlines() if x.strip()]
    return [{**r, "folder": str(cases_dir)} for r in rows]


def describe(path: Path, modality: str) -> str:
    """Write the text the agent would keep about an item: Task 2's description, or a transcript.

    Args:
        path (Path): The PNG or WAV file.
        modality (str): "image" or "audio".

    Returns:
        str: The text.
    """
    if modality == "image":
        r = tasks.screen_help(Image.open(path).convert("RGB"), store=False)
        return " ".join(str(r[k]) for k in ("what_i_see", "problem", "suggested_fix") if r[k])
    try:
        return stt.transcribe(audio.load_wav(path), "whisper")
    except Unsupported:
        return stt.transcribe(audio.load_wav(path), "gemma")


def _prepare(folder: Path, rows: list[dict]) -> dict[int, str]:
    """Store every item of a folder once, in the (empty) eval database."""
    ids = {}
    for name, modality in sorted({(r["item"], r["modality"]) for r in rows}):
        path = folder / name
        if modality == "image":
            media, kind = ("image", Image.open(path).convert("RGB")), "screen_help"
        else:
            media, kind = ("audio", audio.load_wav(path)), "voice_note"
        if row := episodic.store(describe(path, modality), kind=kind, media=media):
            ids[row] = name
    return ids


def run_item(item):
    """Run one query in all three modes, after storing the items of its folder.

    Args:
        item (dict): A query from `load_cases`.

    Returns:
        dict: A record with the top 3 items of each mode and whether the item was found.
    """
    folder = Path(item["folder"])
    with common.isolated("multimodal") as db_folder:
        if item["folder"] not in _stored:  # first query of this folder: start from an empty database
            common.reset(db_folder)
            _stored.clear()
            _stored[item["folder"]] = _prepare(folder, load_cases(folder))
        ids = _stored[item["folder"]]
        rec = {"query": item["query"], "item": item["item"], "modality": item["modality"]}
        for mode in MODES:
            top = [ids.get(h["id"]) for h in episodic.search(item["query"], k=3, mode=mode)]
            rec[f"top3_{mode}"], rec[f"found_{mode}"] = top, item["item"] in top
            rec[f"first_{mode}"] = bool(top) and top[0] == item["item"]
    return rec


def score(records):
    """Compute recall at 3 for each modality and search mode.

    Args:
        records (list[dict]): The records from `run_item`.

    Returns:
        dict: Metric name to value.
    """
    out = {}
    for modality in sorted({r["modality"] for r in records}):
        rs = [r for r in records if r["modality"] == modality]
        for mode in MODES:
            out[f"{modality}_recall_{mode}"] = round(sum(r[f"found_{mode}"] for r in rs) / len(rs), 3)
            out[f"{modality}_top1_{mode}"] = round(sum(r[f"first_{mode}"] for r in rs) / len(rs), 3)
    return out
