"""The native-skill endpoints: what each skill folder declares, whether it loaded, and a reload.

    GET  /api/skills          every skill folder: its manifest, whether it loaded, and why not
    POST /api/skills/reload   load every skill folder again, after you add or edit one
    POST /api/skills/draft    the local model drafts a skill from a description ({"description": "..."}); nothing is written
    POST /api/skills/create   write and load a draft the user approved (agent/skill_author.py checks it again)

A loaded skill's tier lives in its manifest.json, which is reviewed like code, so the page shows the tier but cannot
change it. For a new skill, the user picks the tier on the draft before it is written; the model never does.
"""

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool

from agent import events, skill_author, skills, tools

router = APIRouter(prefix="/api/skills")


def _manifest(folder: Path) -> tuple[dict, list[str]]:
    """Read one manifest and list its problems, without importing the skill's code."""
    try:
        manifest = json.loads((folder / "manifest.json").read_text())
    except (OSError, ValueError) as e:
        return {}, [f"manifest.json: {e}"]
    if not isinstance(manifest, dict):
        return {}, ["manifest.json: the top level must be an object"]
    return manifest, skills.check(manifest, folder)


def folder_view(folder: Path) -> dict:
    """Describe one skill folder for the page.

    Args:
        folder (Path): agent/skills/<name>.

    Returns:
        dict: The manifest fields, `loaded` (bool), `state` (loaded, refused or new) and `problems`.
    """
    manifest, problems = _manifest(folder)
    problems = problems or skills.PROBLEMS.get(folder.name, [])  # an import error shows only at load time
    loaded = getattr(tools.REGISTRY.get(manifest.get("name", folder.name)), "skill", None) is not None
    state = "refused" if problems else "loaded" if loaded else "new"
    return {
        "folder": folder.name,
        "name": manifest.get("name", folder.name),
        "description": manifest.get("description", ""),
        "version": manifest.get("version", ""),
        "tier": manifest.get("tier", ""),
        "effects": manifest.get("effects", []),
        "examples": manifest.get("examples", []),
        "loaded": loaded and not problems,
        "state": state,
        "problems": problems,
    }


@router.get("")
def overview() -> dict:
    """List every skill folder.

    Returns:
        dict: `skills` (one view per folder), `folder` (where skills live) and `risky_effects`.
    """
    folders = sorted(d for d in skills.SKILLS_DIR.iterdir() if (d / "manifest.json").is_file())
    return {
        "skills": [folder_view(d) for d in folders],
        "folder": "agent/skills",
        "risky_effects": sorted(skills.RISKY_EFFECTS),
    }


@router.post("/reload")
def reload() -> dict:
    """Load every skill folder again, so a new or edited skill takes effect without a restart.

    Returns:
        dict: The same view as GET /api/skills.
    """
    loaded = [m["name"] for m in skills.load()]
    view = overview()
    refused = [s["folder"] for s in view["skills"] if s["state"] == "refused"]
    events.emit("skills.reloaded", loaded=loaded, refused=refused)
    return view


@router.post("/draft")
async def draft(body: dict) -> dict:
    """Ask the local model to draft a skill from a description. Nothing is written.

    Args:
        body (dict): `description`, what the skill should do.

    Returns:
        dict: The draft: name, description, examples, params, effects, tier (always confirm), code and problems.
    """
    text = str(body.get("description", "")).strip()
    if len(text) < 8:
        raise HTTPException(400, "Describe the skill in a sentence: what it does, and what it needs as input.")
    try:
        return await run_in_threadpool(skill_author.draft, text)
    except skill_author.SkillProblem as e:
        raise HTTPException(422, str(e)) from e


@router.post("/create")
def create(body: dict) -> dict:
    """Write and load a skill the user approved.

    Args:
        body (dict): The draft as approved: name, description, examples, params, effects, tier and code.

    Returns:
        dict: The new skill's view, as in GET /api/skills.
    """
    try:
        manifest = skill_author.create(body)
    except skill_author.SkillProblem as e:
        raise HTTPException(400, str(e)) from e
    return folder_view(skills.SKILLS_DIR / manifest["name"])
