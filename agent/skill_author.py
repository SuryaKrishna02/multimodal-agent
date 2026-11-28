"""Draft a native skill from a description, check it, and write it only after the user approves.

1. `draft(description)`: the local model writes a draft (name, description, examples, parameters, the code of
   `run`). Nothing is written to disk. The draft always starts at tier `confirm`: the model never picks a tier.
2. The dashboard shows the draft with its code. The user reads it, can edit it, picks the tier, and presses Create.
3. `create(draft)` checks everything again (`build`) and only then writes agent/skills/<name>/ and loads it. If the
   code fails to load, the folder is removed again, so a broken draft leaves nothing behind.

`effects_in()` reads the code (its imports and calls) and adds every effect it finds to the manifest. So a skill
that opens a URL or writes a file cannot be `safe` (skills.check, ground rule 10), whatever the model declared. It
is a hint, not a sandbox: a person reads the code before pressing Create.
"""

import ast
import json
import re
import shutil

from agent import brain, events, skills, tools

NAME = re.compile(r"^[a-z][a-z0-9_]{1,40}$")
EFFECTS = ["reads_memory", "reads_files", "writes", "runs_code", "network"]
TYPES = {"str": "string", "int": "integer", "float": "number", "bool": "boolean", "list": "array", "dict": "object"}

DRAFT_SCHEMA = {
    "type": "object",
    "required": ["name", "description", "examples", "params", "effects", "code"],
    "additionalProperties": False,
    "properties": {
        "name": {"type": "string", "pattern": "^[a-z][a-z0-9_]{1,40}$"},
        "description": {"type": "string", "minLength": 10, "maxLength": 300},
        "examples": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 4},
        "params": {
            "type": "array",
            "maxItems": 6,
            "items": {
                "type": "object",
                "required": ["name", "type", "description"],
                "additionalProperties": False,
                "properties": {
                    "name": {"type": "string", "pattern": "^[a-z_][a-z0-9_]*$"},
                    "type": {"enum": ["string", "integer", "number", "boolean"]},
                    "description": {"type": "string"},
                },
            },
        },
        "effects": {"type": "array", "items": {"enum": EFFECTS}},
        "code": {"type": "string", "minLength": 20},
    },
}

SYSTEM = """You write native skills for an agent that runs on this computer. A skill is one Python file with a
function run(...) that the agent calls as a tool.

Rules:
- Use only the Python standard library. Keep the file under 60 lines.
- Give run one keyword argument per input, with a type hint (str, int, float or bool). List the same inputs in params.
- Return a short string, a dict or a list. Raise ValueError with a clear message for bad input.
- Do not read or write files, start programs or use the network unless the description asks for it. List each such
  effect in effects: reads_files, writes, runs_code, network.
- name: lowercase snake_case, 2 to 40 characters. description: one sentence that says what it does and when to use it.
- examples: 2 to 4 requests a user might say that should call this skill."""

# Modules and calls that show an effect. Matched on the first part of a module name and on the called name.
RISKY_MODULES = {
    "network": {"socket", "ssl", "urllib", "http", "requests", "httpx", "aiohttp", "ftplib", "smtplib", "websocket"},
    "runs_code": {"subprocess", "multiprocessing", "ctypes", "importlib", "pty"},
    "writes": {"shutil", "tempfile", "sqlite3"},
}
RISKY_CALLS = {
    "runs_code": {"eval", "exec", "compile", "__import__", "system", "popen", "spawnl", "execv", "execvp"},
    "writes": {
        "write_text",
        "write_bytes",
        "unlink",
        "remove",
        "rmdir",
        "rename",
        "replace",
        "mkdir",
        "makedirs",
        "rmtree",
    },
    "reads_files": {"read_text", "read_bytes", "listdir", "scandir", "glob", "iterdir", "walk"},
}


class SkillProblem(ValueError):
    """A draft that cannot become a skill; the message lists every problem, one per line."""


def _import_effects(node: ast.AST) -> set[str]:
    """Return the effects shown by one import statement."""
    if isinstance(node, ast.Import):
        names = [a.name for a in node.names]
    elif isinstance(node, ast.ImportFrom):
        names = [node.module or ""]
    else:
        return set()
    roots = {n.split(".")[0] for n in names}
    found = {effect for effect, modules in RISKY_MODULES.items() if roots & modules}
    if any(n.startswith("agent.memory") for n in names) or (
        isinstance(node, ast.ImportFrom) and node.module == "agent"
    ):
        found.add("reads_memory")
    return found


def _open_effect(node: ast.Call) -> str:
    """Return the effect of an open() call: writes for a mode with w, a, x or +, or when the mode is not a literal."""
    mode = node.args[1] if len(node.args) > 1 else next((k.value for k in node.keywords if k.arg == "mode"), None)
    if mode is None:
        return "reads_files"
    if isinstance(mode, ast.Constant) and isinstance(mode.value, str):
        return "writes" if set(mode.value) & set("wax+") else "reads_files"
    return "writes"


def _call_effects(node: ast.AST) -> set[str]:
    """Return the effects shown by one call."""
    if not isinstance(node, ast.Call):
        return set()
    func = node.func
    name = func.attr if isinstance(func, ast.Attribute) else func.id if isinstance(func, ast.Name) else ""
    found = {effect for effect, calls in RISKY_CALLS.items() if name in calls}
    if name == "open":
        found.add(_open_effect(node))
    return found


def effects_in(code: str) -> set[str]:
    """Find the effects that the code of a skill shows, from its imports and calls.

    Args:
        code (str): The source of skill.py.

    Returns:
        set[str]: Effects from EFFECTS. Empty if the code does not parse.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set()
    return {e for node in ast.walk(tree) for e in _import_effects(node) | _call_effects(node)}


def _annotation_type(node: ast.expr | None) -> str | None:
    """Turn a type hint such as `int` into a JSON schema type, or None when there is no simple hint."""
    return TYPES.get(node.id) if isinstance(node, ast.Name) else None


def run_params(code: str) -> list[dict] | None:
    """Read the parameters of the top-level `run` function.

    Args:
        code (str): The source of skill.py.

    Returns:
        list[dict] | None: `name`, `type` (or None) and `required` for each keyword argument; None without `run`.
    """
    tree = ast.parse(code)
    run = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run"), None)
    if run is None:
        return None
    args = run.args.posonlyargs + run.args.args
    first_default = len(args) - len(run.args.defaults)
    params = [
        {"name": a.arg, "type": _annotation_type(a.annotation), "required": i < first_default}
        for i, a in enumerate(args)
    ]
    for a, default in zip(run.args.kwonlyargs, run.args.kw_defaults, strict=True):
        params.append({"name": a.arg, "type": _annotation_type(a.annotation), "required": default is None})
    return params


def input_schema(params: list[dict], notes: list[dict]) -> dict:
    """Build the manifest's input schema from `run`'s signature, with the draft's types and descriptions as notes.

    Args:
        params (list[dict]): From run_params(): the code is the source of truth for the names.
        notes (list[dict]): The draft's params (`name`, `type`, `description`).

    Returns:
        dict: A JSON schema of type object.
    """
    by_name = {n.get("name"): n for n in notes}
    props = {}
    for p in params:
        note = by_name.get(p["name"], {})
        props[p["name"]] = {"type": p["type"] or note.get("type") or "string"}
        if note.get("description"):
            props[p["name"]]["description"] = note["description"]
    schema: dict = {"type": "object", "properties": props}
    required = [p["name"] for p in params if p["required"]]
    if required:
        schema["required"] = required
    return schema


def strip_fences(code: str) -> str:
    """Remove a Markdown code fence that the model may put around the code."""
    code = code.strip()
    match = re.match(r"^```[a-zA-Z]*\n(.*?)\n?```$", code, re.S)
    return (match.group(1) if match else code).strip() + "\n"


def build(draft: dict) -> tuple[dict, list[str]]:
    """Turn a draft (from the model, maybe edited by the user) into a manifest, and list its problems.

    Args:
        draft (dict): `name`, `description`, `examples`, `tier`, `effects`, `params` and `code`.

    Returns:
        tuple[dict, list[str]]: The manifest, and the problems (empty when it may be written).
    """
    code = strip_fences(str(draft.get("code", "")))
    try:
        params = run_params(code)
    except SyntaxError as e:
        return {}, [f"skill.py, line {e.lineno}: {e.msg}"]
    name = str(draft.get("name", "")).strip()
    declared = {e for e in draft.get("effects", []) if e in EFFECTS}
    manifest = {
        "name": name,
        "description": str(draft.get("description", "")).strip(),
        "version": "1.0",
        "tier": draft.get("tier", "confirm"),
        "effects": [e for e in EFFECTS if e in declared | effects_in(code)],
        "input_schema": input_schema(params or [], list(draft.get("params", []))),
        "examples": [str(x).strip() for x in draft.get("examples", []) if str(x).strip()],
        "entry": "skill.py:run",
    }
    problems = [] if params is not None else ["skill.py must define a top-level function run(...)"]
    folder = skills.SKILLS_DIR / name
    if NAME.match(name) and (folder.exists() or name in tools.REGISTRY):
        problems.append(f"a skill or tool called '{name}' already exists: pick another name")
    return manifest, problems + skills.check(manifest, folder)


def draft(description: str) -> dict:
    """Ask the local model for a draft skill. Nothing is written.

    Args:
        description (str): What the skill should do, in the user's words.

    Returns:
        dict: The draft with `tier` "confirm", the scanned `effects`, `params` and `problems`.

    Raises:
        SkillProblem: The model did not return a valid draft.
    """
    data, _info = brain.ask_json(
        [brain.part("text", f"Write a skill that does this:\n{description}")], DRAFT_SCHEMA, system=SYSTEM
    )
    if data is None:
        raise SkillProblem("The model did not write a valid draft. Try again, or describe the skill more plainly.")
    data = {**data, "code": strip_fences(data["code"]), "tier": "confirm"}
    manifest, problems = build(data)
    data["effects"] = manifest.get("effects", data["effects"])
    events.emit("skills.drafted", name=data["name"], effects=data["effects"], problems=problems)
    return {**data, "problems": problems}


def create(draft: dict) -> dict:
    """Check an approved draft again, write its folder, and load the skill.

    Args:
        draft (dict): The draft as the user approved it (they may have edited the code, name or tier).

    Returns:
        dict: The manifest that was written.

    Raises:
        SkillProblem: The draft has a problem, or the code failed to load (then nothing stays on disk).
    """
    manifest, problems = build(draft)
    if problems:
        raise SkillProblem("\n".join(problems))
    folder = skills.SKILLS_DIR / manifest["name"]
    folder.mkdir()
    (folder / "skill.py").write_text(strip_fences(str(draft["code"])))
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if skills.load_one(folder) is None:
        reasons = skills.PROBLEMS.pop(folder.name, ["it did not load"])
        shutil.rmtree(folder)
        raise SkillProblem("\n".join(reasons))
    events.emit("skills.created", name=manifest["name"], tier=manifest["tier"], effects=manifest["effects"])
    return manifest
