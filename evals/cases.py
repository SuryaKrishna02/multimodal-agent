"""Add your own eval cases, see what exists, and rebuild the synthetic ones.

    python cli.py evals status                              how many cases each eval has, and its last run
    python cli.py evals build [component ...]               rebuild the synthetic cases
    python cli.py evals add tool_calls "remind me to call mum" --tool save_note --args '{"text": "mum"}'
    python cli.py evals add asr clip.wav --text "exactly what was said"
    python cli.py evals add task2 shot.png --problem --mention "torchvision" --mention "pip install|conda install"

Your cases go to `config.EVAL_USER/<component>/` with ids u001, u002, ... and are run together with the
synthetic ones. The files use the format of each eval (see the docstring of `evals/<component>/eval.py`),
so you can also edit them by hand. Nothing here is ever overwritten by `build`.
"""

import argparse
import json
import re
import shutil
from pathlib import Path

import soundfile as sf
from PIL import Image

from agent import config
from agent.senses import audio
from evals import common

EXPECTATIONS = ("ok", "blocked", "confirm", "invalid")
SCHEMAS = ("action_items", "screen_help")


def folder(component: str) -> Path:
    """Return the folder of your cases for a component, creating it.

    Args:
        component (str): One of `common.COMPONENTS`.

    Returns:
        Path: The folder.
    """
    path = config.EVAL_USER / component
    path.mkdir(parents=True, exist_ok=True)
    return path


def next_id(taken: list[str]) -> str:
    """Pick the next free case id.

    Args:
        taken (list[str]): Ids or file stems that exist already. Only `u<number>` entries count.

    Returns:
        str: For example "u001" when nothing is taken, or "u004" after "u003".
    """
    numbers = [int(m.group(1)) for t in taken if (m := re.fullmatch(r"u(\d+)", t))]
    return f"u{max(numbers, default=0) + 1:03d}"


def read_jsonl(path: Path) -> list[dict]:
    """Read a JSON-lines file, or an empty list when it does not exist."""
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def append_jsonl(component: str, name: str, row: dict) -> str:
    """Add a case to a JSON-lines file, with the next free id.

    Args:
        component (str): One of `common.COMPONENTS`.
        name (str): File name inside the component folder, for example "prompts.jsonl".
        row (dict): The case, without an `id`.

    Returns:
        str: The id given to the case.
    """
    path = folder(component) / name
    case_id = next_id([r.get("id", "") for r in read_jsonl(path)])
    with path.open("a") as f:
        f.write(json.dumps({"id": case_id, **row}) + "\n")
    return case_id


def new_stem(component: str) -> str:
    """Pick a free file stem (u001, u002, ...) for a case that is stored as files."""
    return next_id([p.stem for p in folder(component).iterdir()])


def save_audio(component: str, src: Path) -> Path:
    """Convert an audio file to 16 kHz mono WAV inside the folder of a component.

    Args:
        component (str): One of `common.COMPONENTS`.
        src (Path): A WAV, AIFF or FLAC file.

    Returns:
        Path: The new WAV file.
    """
    dst = folder(component) / f"{new_stem(component)}.wav"
    sf.write(dst, audio.load_wav(src), config.SAMPLE_RATE, subtype="PCM_16")
    return dst


def save_media(component: str, src: Path) -> Path:
    """Copy a screenshot or screen recording into the folder of a component.

    Args:
        component (str): One of `common.COMPONENTS`.
        src (Path): An image (saved as PNG) or an MP4/MOV video (copied).

    Returns:
        Path: The new file.
    """
    stem = new_stem(component)
    if src.suffix.lower() in (".mp4", ".mov"):
        return Path(shutil.copy2(src, folder(component) / f"{stem}{src.suffix.lower()}"))
    dst = folder(component) / f"{stem}.png"
    Image.open(src).convert("RGB").save(dst)
    return dst


def tool_names() -> set[str]:
    """Return the names of the registered tools."""
    from agent import tools

    return set(tools.REGISTRY)


def add_tool_call(a: argparse.Namespace) -> str:
    """Add a tool selection case.

    Args:
        a (argparse.Namespace): `prompt`, `tool` (a tool name or "none") and `args` (dict).

    Returns:
        str: What was added.

    Raises:
        SystemExit: If the tool does not exist.
    """
    tool = None if a.tool == "none" else a.tool
    if tool and tool not in tool_names():
        raise SystemExit(f"unknown tool '{tool}'. Tools: {', '.join(sorted(tool_names()))}, or 'none'")
    case_id = append_jsonl("tool_calls", "prompts.jsonl", {"prompt": a.prompt, "tool": tool, "args": a.args})
    return f"{case_id} added to tool_calls"


def add_tool_policy(a: argparse.Namespace) -> str:
    """Add a tool permission case: a call and what the permission layer must do with it.

    Args:
        a (argparse.Namespace): `tool`, `args` (dict) and `expect` (one of `EXPECTATIONS`).

    Returns:
        str: What was added.
    """
    case_id = append_jsonl("tools", "calls.jsonl", {"tool": a.tool, "args": a.args, "expect": a.expect})
    return f"{case_id} added to tools"


def add_memory(a: argparse.Namespace) -> str:
    """Add a memory to store. Use its id in `add query --relevant`.

    Args:
        a (argparse.Namespace): `text` and `junk` (True if the junk filter should reject it).

    Returns:
        str: What was added.
    """
    case_id = append_jsonl("memory", "memories.jsonl", {"text": a.text, "kind": "note", "junk": a.junk})
    return f"{case_id} added to memory ({'junk' if a.junk else 'good'})"


def add_query(a: argparse.Namespace) -> str:
    """Add a memory query and the memories that must come back.

    Args:
        a (argparse.Namespace): `query` and `relevant` (ids of memories that you added).

    Returns:
        str: What was added.

    Raises:
        SystemExit: If an id is not one of your memories.
    """
    known = {m["id"] for m in read_jsonl(folder("memory") / "memories.jsonl")}
    if unknown := [r for r in a.relevant if r not in known]:
        raise SystemExit(f"unknown memory id(s) {unknown}. Add the memory first: evals add memory TEXT")
    case_id = append_jsonl("memory", "queries.jsonl", {"query": a.query, "relevant": a.relevant})
    return f"{case_id} added to memory queries"


def add_structured(a: argparse.Namespace) -> str:
    """Add an input for the structured output eval.

    Args:
        a (argparse.Namespace): `schema`, and any of `text`, `image` and `audio`.

    Returns:
        str: What was added.
    """
    row: dict = {"schema": a.schema}
    if a.text:
        row["text"] = a.text
    if a.image:
        row["image"] = save_media("structured", a.image).name
    if a.audio:
        row["audio"] = save_audio("structured", a.audio).name
    case_id = append_jsonl("structured", "items.jsonl", row)
    return f"{case_id} added to structured"


def add_asr(a: argparse.Namespace) -> str:
    """Add a speech clip and what was said.

    Args:
        a (argparse.Namespace): `audio` (file) and `text` (the transcript).

    Returns:
        str: What was added.
    """
    wav = save_audio("asr", a.audio)
    wav.with_suffix(".txt").write_text(a.text)
    return f"{wav.stem} added to asr"


def add_task1(a: argparse.Namespace) -> str:
    """Add a voice note and the action items in it. Give no `--item` for a note without any.

    Args:
        a (argparse.Namespace): `audio` (file) and `item` (list of action items).

    Returns:
        str: What was added.
    """
    wav = save_audio("task1", a.audio)
    wav.with_suffix(".json").write_text(json.dumps({"action_items": a.item}))
    return f"{wav.stem} added to task1 ({len(a.item)} action items)"


def add_task2(a: argparse.Namespace) -> str:
    """Add a screenshot or recording and what a correct answer must mention.

    Args:
        a (argparse.Namespace): `media` (file), `problem` (bool) and `mention` (list of "a|b" groups).

    Returns:
        str: What was added.
    """
    media = save_media("task2", a.media)
    groups = [[p.strip() for p in g.split("|") if p.strip()] for g in a.mention]
    media.with_suffix(".json").write_text(json.dumps({"has_problem": a.problem, "must_mention": groups}))
    return f"{media.stem} added to task2"


def add_vad(a: argparse.Namespace) -> str:
    """Add an audio clip and its speech regions. Give no `--speech` for a clip without speech.

    Args:
        a (argparse.Namespace): `audio` (file) and `speech` (list of [start, end] seconds).

    Returns:
        str: What was added.
    """
    wav = save_audio("vad", a.audio)
    wav.with_suffix(".txt").write_text("".join(f"{s:.2f}\t{e:.2f}\tspeech\n" for s, e in a.speech))
    return f"{wav.stem} added to vad ({len(a.speech)} speech regions)"


def add_screen(a: argparse.Namespace) -> str:
    """Label the key moments of a sequence that `python cli.py record-screen NAME` recorded.

    Args:
        a (argparse.Namespace): `name` and `moment` (list of [seconds, label]).

    Returns:
        str: What was added.

    Raises:
        SystemExit: If the sequence has not been recorded.
    """
    seq = config.EVAL_USER / "screen" / a.name
    if not (seq / "frames.json").exists():
        raise SystemExit(f"no recording at {seq}. Record it first: python cli.py record-screen {a.name}")
    moments = [{"t": float(t), "label": label} for t, label in a.moment]
    (seq / "key_moments.json").write_text(json.dumps(moments))
    return f"{a.name} labeled with {len(moments)} key moments"


def _json_arg(text: str) -> dict:
    """Parse a JSON object given on the command line."""
    value = json.loads(text)
    if not isinstance(value, dict):
        raise argparse.ArgumentTypeError('expected a JSON object such as \'{"path": "a.txt"}\'')
    return value


def _add_parsers(sub) -> None:
    """Register the `add <component>` commands."""
    p = sub.add_parser("tool_calls", help="a prompt and the tool the model should call")
    p.add_argument("prompt")
    p.add_argument("--tool", required=True, help="tool name, or 'none' if the model should just answer")
    p.add_argument(
        "--args", type=_json_arg, default={}, help='JSON of arguments that must appear, e.g. \'{"query": "Priya"}\''
    )
    p.set_defaults(handler=add_tool_call)
    p = sub.add_parser("tools", help="a tool call and what the permission layer must do")
    p.add_argument("--tool", required=True)
    p.add_argument("--args", type=_json_arg, default={})
    p.add_argument("--expect", required=True, choices=EXPECTATIONS)
    p.set_defaults(handler=add_tool_policy)
    p = sub.add_parser("memory", help="a memory to store")
    p.add_argument("text")
    p.add_argument("--junk", action="store_true", help="the junk filter should reject it")
    p.set_defaults(handler=add_memory)
    p = sub.add_parser("query", help="a memory search and the memories (ids) that must come back")
    p.add_argument("query")
    p.add_argument("--relevant", nargs="+", required=True, metavar="ID")
    p.set_defaults(handler=add_query)
    p = sub.add_parser("structured", help="an input for the JSON validity test")
    p.add_argument("--schema", required=True, choices=SCHEMAS)
    p.add_argument("--text")
    p.add_argument("--image", type=Path)
    p.add_argument("--audio", type=Path)
    p.set_defaults(handler=add_structured)
    p = sub.add_parser("asr", help="a speech clip and its transcript")
    p.add_argument("audio", type=Path)
    p.add_argument("--text", required=True, help="exactly what was said")
    p.set_defaults(handler=add_asr)
    p = sub.add_parser("task1", help="a voice note and its action items")
    p.add_argument("audio", type=Path)
    p.add_argument("--item", action="append", default=[], help="an action item; repeat for more, none for no items")
    p.set_defaults(handler=add_task1)
    p = sub.add_parser("task2", help="a screenshot or recording and what the answer must mention")
    p.add_argument("media", type=Path)
    p.add_argument("--problem", action="store_true", help="the screen shows an error or problem")
    p.add_argument("--mention", action="append", default=[], help="phrases of which one must appear; separate with |")
    p.set_defaults(handler=add_task2)
    p = sub.add_parser("vad", help="an audio clip and where the speech is")
    p.add_argument("audio", type=Path)
    p.add_argument("--speech", nargs=2, type=float, action="append", default=[], metavar=("START", "END"))
    p.set_defaults(handler=add_vad)
    p = sub.add_parser("screen", help="label the key moments of a recorded sequence")
    p.add_argument("name")
    p.add_argument("--moment", nargs=2, action="append", required=True, metavar=("SECONDS", "LABEL"))
    p.set_defaults(handler=add_screen)


def print_status() -> None:
    """Print, for each eval, how many synthetic and user cases it has and when it last ran."""
    print(f"{'component':11s} {'synthetic':>9s} {'user':>5s}   last run")
    for c in common.COMPONENTS:
        mod = common.module(c)
        counts = common.case_counts(c)
        cached = common.cached_results(c, mod)
        last = ", ".join(f"{t} {r['when']} (n={r['n']})" for t, r in cached.items()) or "never"
        if getattr(mod, "BUILTIN", False):
            print(f"{c:11s} {'built in':>9s} {'-':>5s}   {last}")
            continue
        synthetic = str(counts["synthetic"]) if counts["synthetic"] else "not built"
        print(f"{c:11s} {synthetic:>9s} {counts['user']:>5d}   {last}")


def build_synthetic(components: list[str]) -> None:
    """Rebuild the synthetic cases, even if they are current.

    Args:
        components (list[str]): Components to rebuild, or an empty list for all.
    """
    from evals import synthetic

    if unknown := [c for c in components if c not in synthetic.BUILDERS]:
        raise SystemExit(f"nothing to build for {unknown}. Choose from: {', '.join(synthetic.BUILDERS)}")
    for c in components or list(synthetic.BUILDERS):
        print(f"building {c}: {synthetic.build(c)}")


def main(argv: list[str]) -> None:
    """Run `status`, `build` or `add` from the command line.

    Args:
        argv (list[str]): Arguments after the command name.
    """
    ap = argparse.ArgumentParser(
        prog="evals", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    p = sub.add_parser("build")
    p.add_argument("components", nargs="*", metavar="component")
    adder = sub.add_parser("add", help="add one of your own cases").add_subparsers(dest="component", required=True)
    _add_parsers(adder)
    a = ap.parse_args(argv)
    if a.cmd == "status":
        print_status()
    elif a.cmd == "build":
        build_synthetic(a.components)
    else:
        print(a.handler(a))
