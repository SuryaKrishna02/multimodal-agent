"""The `evals add` commands of the Session 3 evals. `evals/cases.py` registers them.

python cli.py evals add tts "Your train leaves at 6:40 from platform 2."
python cli.py evals add wake kitchen.wav --wakes 1 --note "kitchen, far"   (new recordings, not training clips)
python cli.py evals add wake meeting.wav --wakes 0 --note "30 min of a meeting"
python cli.py evals add proactive shot.png --alert --mention test_parse
python cli.py evals add skills "how many cups is a liter" --skill unit_convert
python cli.py evals add mcp_select "what is on my calendar friday" --tool calendar__list_events
python cli.py evals add task5 a.png b.png c.png --fail-at 2 --mention test_login
python cli.py evals add task6 request.wav --text "Remind me to call mum in five minutes" --mention mum --minutes 5
"""

import argparse
import json
from pathlib import Path

from PIL import Image

from evals.cases import append_jsonl, folder, new_stem, read_jsonl, save_audio, save_media


def add_tts(a: argparse.Namespace) -> str:
    """Add a sentence for the text-to-speech eval.

    Args:
        a (argparse.Namespace): `text`.

    Returns:
        str: What was added.
    """
    return f"{append_jsonl('tts', 'sentences.jsonl', {'text': a.text})} added to tts"


def add_wake(a: argparse.Namespace) -> str:
    """Add a recording and how many times the wake phrase is said in it.

    Args:
        a (argparse.Namespace): `audio`, `wakes` and `note`.

    Returns:
        str: What was added.
    """
    wav = save_audio("wake", a.audio)
    wav.with_suffix(".json").write_text(json.dumps({"wakes": a.wakes, "note": a.note or ""}))
    return f"{wav.stem} added to wake ({a.wakes} wakes expected)"


def add_proactive(a: argparse.Namespace) -> str:
    """Add a screenshot and whether the agent should speak up about it.

    Args:
        a (argparse.Namespace): `image`, `alert` and `mention`.

    Returns:
        str: What was added.
    """
    path = save_media("proactive", a.image)
    path.with_suffix(".json").write_text(json.dumps({"alert": a.alert, "mention": a.mention}))
    return f"{path.stem} added to proactive ({'alert' if a.alert else 'no alert'})"


def add_skills(a: argparse.Namespace) -> str:
    """Add a prompt and the skill that must fire (or none).

    Args:
        a (argparse.Namespace): `prompt` and `skill`.

    Returns:
        str: What was added.

    Raises:
        SystemExit: The skill is not one of the loaded skills.
    """
    from agent import tools

    skills = sorted(n for n, t in tools.REGISTRY.items() if getattr(t, "skill", None))
    skill = None if a.skill == "none" else a.skill
    if skill and skill not in skills:
        raise SystemExit(f"unknown skill '{a.skill}': use one of {', '.join(skills)} or none")
    return f"{append_jsonl('skills', 'prompts.jsonl', {'prompt': a.prompt, 'skill': skill})} added to skills"


def add_mcp_select(a: argparse.Namespace) -> str:
    """Add a prompt and the catalog tool it needs. Your folder gets a copy of the synthetic catalog first.

    Args:
        a (argparse.Namespace): `prompt` and `tool`.

    Returns:
        str: What was added.

    Raises:
        SystemExit: The tool is not in the catalog.
    """
    from evals.synthetic.bolt_ons import CATALOG
    from evals.synthetic.text import write_jsonl

    catalog = folder("mcp_select") / "catalog.jsonl"
    if not catalog.exists():
        write_jsonl(catalog, [{"name": n, "description": d, "input_schema": s} for n, d, s in CATALOG])
    names = {row["name"] for row in read_jsonl(catalog)}
    if a.tool not in names:
        raise SystemExit(f"'{a.tool}' is not in {catalog} (edit that file to add tools)")
    return f"{append_jsonl('mcp_select', 'prompts.jsonl', {'prompt': a.prompt, 'tool': a.tool})} added to mcp_select"


def add_task5(a: argparse.Namespace) -> str:
    """Add a screen session: screenshots in order and the first one that shows a failed test run.

    Args:
        a (argparse.Namespace): `images`, `fail_at` and `mention`.

    Returns:
        str: What was added.
    """
    session = folder("task5") / new_stem("task5")
    session.mkdir()
    for i, src in enumerate(a.images):
        Image.open(src).convert("RGB").save(session / f"{i:02d}.png")
    (session / "sequence.json").write_text(json.dumps({"fail_at": a.fail_at, "mention": a.mention}))
    return f"{session.name} added to task5 ({len(a.images)} frames)"


def add_task6(a: argparse.Namespace) -> str:
    """Add a spoken reminder request (without the wake word) and the reminder it must make.

    Args:
        a (argparse.Namespace): `audio`, `text`, `mention` and `minutes`.

    Returns:
        str: What was added.
    """
    wav = save_audio("task6", a.audio)
    label = {"text": a.text, "mention": a.mention, "minutes": a.minutes}
    wav.with_suffix(".json").write_text(json.dumps(label))
    return f"{wav.stem} added to task6"


def add_parsers(sub) -> None:
    """Register the `add <component>` commands of the Session 3 evals.

    Args:
        sub: The subparsers of `evals add`.
    """
    p = sub.add_parser("tts", help="a sentence to speak and transcribe back (numbers as digits)")
    p.add_argument("text")
    p.set_defaults(handler=add_tts)
    p = sub.add_parser("wake", help="a recording and how many times 'Hey Sanjaya' is said in it")
    p.add_argument("audio", type=Path)
    p.add_argument("--wakes", type=int, required=True, help="0 for a recording without the phrase")
    p.add_argument("--note", help="where and how it was recorded")
    p.set_defaults(handler=add_wake)
    p = sub.add_parser("proactive", help="a screenshot and whether the agent should speak up")
    p.add_argument("image", type=Path)
    p.add_argument("--alert", action="store_true", help="it shows a failed test run")
    p.add_argument("--mention", help="a word a right alert names, e.g. the failing test")
    p.set_defaults(handler=add_proactive)
    p = sub.add_parser("skills", help="a prompt and the skill that must fire (or none)")
    p.add_argument("prompt")
    p.add_argument("--skill", required=True)
    p.set_defaults(handler=add_skills)
    p = sub.add_parser("mcp_select", help="a prompt and the catalog tool it needs")
    p.add_argument("prompt")
    p.add_argument("--tool", required=True)
    p.set_defaults(handler=add_mcp_select)
    p = sub.add_parser("task5", help="screenshots in order, and the first one with a failed test run")
    p.add_argument("images", type=Path, nargs="+")
    p.add_argument("--fail-at", type=int, help="index of the first failing screenshot (leave out: none fails)")
    p.add_argument("--mention", help="a word the spoken alert must hold")
    p.set_defaults(handler=add_task5)
    p = sub.add_parser("task6", help="a spoken reminder request and the reminder it must make")
    p.add_argument("audio", type=Path)
    p.add_argument("--text", required=True, help="what you said")
    p.add_argument("--mention", required=True, help="a word the reminder must hold")
    p.add_argument("--minutes", type=float, required=True, help="when it is due, in minutes from the request")
    p.set_defaults(handler=add_task6)
