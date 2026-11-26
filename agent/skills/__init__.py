"""Native skills: Python bolt-ons that run inside the agent, each declared by a manifest that we read first.

    agent/skills/<name>/manifest.json    what it is, when to use it, its input schema and its permission tier
    agent/skills/<name>/skill.py          def run(**args) -> str | dict | list

`load()` checks every manifest against MANIFEST_SCHEMA, checks that the tier fits the declared effects, imports the
entry function and registers it in the same registry as the built-in tools. So `tools.check()` (schema, tier,
guard) and `tools.select()` treat a skill like any other tool. A skill with a problem is skipped and reported as a
`skills.error` event: one broken skill never stops the agent.

Skill or MCP server? A skill is code in this process: fast, and it can read memory, but a bug in it is a bug in the
agent, so its manifest is reviewed like code. An MCP server is a separate process behind a protocol: any language,
any machine, and the agent sees only its tool list.

The effects rule (ground rule 10, as data): a skill that declares `writes`, `runs_code` or `network` cannot be `safe`.
"""

import importlib.util
import json
from pathlib import Path

import jsonschema

from agent import events

SKILLS_DIR = Path(__file__).parent
RISKY_EFFECTS = {"writes", "runs_code", "network"}  # these need tier confirm or blocked
MANIFEST_SCHEMA = {
    "type": "object",
    "required": ["name", "description", "version", "tier", "effects", "input_schema", "examples", "entry"],
    "additionalProperties": False,
    "properties": {
        "name": {"type": "string", "pattern": "^[a-z][a-z0-9_]{1,40}$"},
        "description": {"type": "string", "minLength": 10},
        "version": {"type": "string"},
        "tier": {"enum": ["safe", "confirm", "blocked"]},
        "effects": {"type": "array", "items": {"enum": ["reads_memory", "reads_files", *sorted(RISKY_EFFECTS)]}},
        "input_schema": {"type": "object", "required": ["type"], "properties": {"type": {"const": "object"}}},
        "examples": {"type": "array", "items": {"type": "string"}, "minItems": 2},
        "entry": {"type": "string", "pattern": r"^[a-z_]+\.py:[a-z_]+$"},
    },
}


def check(manifest: dict, folder: Path) -> list[str]:
    """List the problems of one manifest. An empty list means it may load.

    Args:
        manifest (dict): The parsed manifest.json.
        folder (Path): The skill folder; its name must equal the manifest's name.

    Returns:
        list[str]: Problems, in words.
    """
    from agent import tools  # here, not at the top: agent/tools/builtin.py calls load() while tools is importing

    problems = [
        f"manifest: {e.message}" for e in jsonschema.Draft202012Validator(MANIFEST_SCHEMA).iter_errors(manifest)
    ]
    if problems:
        return problems
    if manifest["name"] != folder.name:
        problems.append(f"name '{manifest['name']}' must equal the folder name '{folder.name}'")
    risky = RISKY_EFFECTS & set(manifest["effects"])
    if risky and manifest["tier"] == tools.SAFE:
        problems.append(f"declares {', '.join(sorted(risky))} but asks for tier safe: use confirm (ground rule 10)")
    if manifest["name"] in tools.REGISTRY and getattr(tools.REGISTRY[manifest["name"]], "skill", None) is None:
        problems.append(f"a built-in tool is already called '{manifest['name']}'")
    return problems


def _entry(folder: Path, entry: str):
    """Import the entry function named in the manifest, from a file inside the skill folder only."""
    file, func = entry.split(":")
    spec = importlib.util.spec_from_file_location(f"agent.skills.{folder.name}.{Path(file).stem}", folder / file)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, func)


def load_one(folder: Path) -> dict | None:
    """Check one skill folder and register its skill.

    Args:
        folder (Path): agent/skills/<name>.

    Returns:
        dict | None: The manifest, or None if the skill was refused (a skills.error event says why).
    """
    try:
        manifest = json.loads((folder / "manifest.json").read_text())
        problems = check(manifest, folder)
        fn = None if problems else _entry(folder, manifest["entry"])
    except Exception as e:  # any error in skill.py (a SyntaxError, a NameError at import) refuses only this skill
        manifest, problems, fn = {}, [f"{type(e).__name__}: {e}"], None
    if problems or fn is None:
        events.emit("skills.error", skill=folder.name, problems=problems)
        return None
    from agent import tools

    t = tools.Tool(
        manifest["name"], manifest["description"], manifest["input_schema"], manifest["tier"], fn,
        examples=list(manifest["examples"]),
    )  # fmt: skip
    t.skill = manifest  # type: ignore[attr-defined]  # marks it as a skill for the dashboard and check()
    tools.REGISTRY[manifest["name"]] = t
    return manifest


def load(folder: Path = SKILLS_DIR) -> list[dict]:
    """Load every skill under a folder.

    Args:
        folder (Path, optional): The folder of skill folders. Defaults to agent/skills/.

    Returns:
        list[dict]: The manifests of the skills that loaded.
    """
    # No event for a skill that loads: this runs when agent.tools is imported, in every process (tests included).
    return [m for d in sorted(folder.iterdir()) if (d / "manifest.json").is_file() and (m := load_one(d))]
