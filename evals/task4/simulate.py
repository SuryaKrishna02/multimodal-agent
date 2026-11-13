"""The simulated user of Task 4: preferences that code can check, and the feedback rounds built on them.

Its feedback is synthetic and labeled so (source "simulated"). Feedback is only ever given on dev days.
Test days are only scored, so a gain on them is a gain on briefs the reflection step never saw.

    check(pref, brief, day)   does one brief follow one preference? (code, or the Gemma judge for your own)
    evaluate(version, days)   write each day's brief with a version and check every preference
    complaints(results)       the feedback the simulated user gives on the briefs that break a preference
"""

import json
import re
from pathlib import Path

from agent import brain, config, evolve

GREETING = re.compile(r"^\s*(good (morning|afternoon|evening|day)|hi|hello|hey|greetings|dear)\b", re.I)
AM_PM = re.compile(r"\b\d{1,2}(:\d{2})?\s?(am|pm|a\.m\.|p\.m\.)", re.I)


def bullets(data: dict) -> list[str]:
    """Return every bullet of a brief, in order."""
    return [b for s in data.get("sections", []) for b in s.get("bullets", [])]


def _text(data: dict) -> str:
    return " ".join(
        [data.get("headline", ""), *(s.get("heading", "") for s in data.get("sections", [])), *bullets(data)]
    )


def _heading(data: dict, i: int) -> str:
    sections = data.get("sections", [])
    return sections[i].get("heading", "").lower() if sections else ""


CHECKS = {
    "no_greeting": lambda d: bool(d.get("headline")) and not GREETING.search(d["headline"]),
    "max_bullets": lambda d: 0 < len(bullets(d)) <= 6,
    "deadlines_first": lambda d: "deadline" in _heading(d, 0),
    "times_24h": lambda d: bool(bullets(d)) and not AM_PM.search(_text(d)),
    "end_tomorrow": lambda d: "tomorrow" in _heading(d, -1),
    "short_bullets": lambda d: bool(bullets(d)) and all(len(b.split()) < 15 for b in bullets(d)),
}
JUDGE = {
    "type": "object",
    "properties": {"follows": {"type": "boolean"}, "reason": {"type": "string"}},
    "required": ["follows", "reason"],
}


def judge(pref_text: str, data: dict) -> bool:
    """Ask Gemma whether a brief follows a preference that has no code check (your own preferences).

    Args:
        pref_text (str): The preference.
        data (dict): The brief.

    Returns:
        bool: True if the judge says it follows.
    """
    prompt = f"Brief:\n{evolve.to_markdown(data)}\n\nPreference: {pref_text}\nDoes the brief follow the preference?"
    out, _ = brain.ask_json([brain.part("text", prompt)], JUDGE)
    return bool(out and out["follows"])


def check(pref: dict, data: dict) -> bool:
    """Check one brief against one preference.

    Args:
        pref (dict): `id` and `text`. A known `id` has a code check, else the Gemma judge decides.
        data (dict): The brief (`headline`, `sections`).

    Returns:
        bool: True if the brief follows the preference.
    """
    fn = CHECKS.get(pref["id"])
    return fn(data) if fn else judge(pref["text"], data)


def leaks(data: dict, words: list[str]) -> bool:
    """Return True if a brief mentions any of the words (the overfitting example's day-only details)."""
    low = _text(data).lower()
    return any(w in low for w in words)


def follows_rule(data: dict, spec: dict) -> bool:
    """Check the one-day rule of the overfitting example: the headline start and the first bullet.

    Args:
        data (dict): The brief.
        spec (dict): The overfitting example, with `headline_starts` and `first_bullet_has`.

    Returns:
        bool: True if the brief follows both parts of the rule.
    """
    first = (bullets(data) or [""])[0].lower()
    return data.get("headline", "").lower().startswith(spec["headline_starts"]) and spec["first_bullet_has"] in first


def evaluate(version: dict, days: list[dict], prefs: list[dict]) -> dict:
    """Write each day's brief with one version and check every preference.

    Args:
        version (dict): A prompt version from `evolve.get_version`.
        days (list[dict]): Days with `id`, `split` and `memories`.
        prefs (list[dict]): Preferences with `id` and `text`.

    Returns:
        dict: Day id -> {"split", "brief", "checks": {pref id: bool}}.
    """
    out = {}
    for d in days:
        data = evolve.write_brief(d["memories"], version, day=d["id"], store=False)["data"]
        out[d["id"]] = {"split": d["split"], "brief": data, "checks": {p["id"]: check(p, data) for p in prefs}}
    return out


def complaints(results: dict, prefs: list[dict], split: str = "dev") -> list[dict]:
    """Build the simulated user's feedback: one complaint per broken preference, on days of one split only.

    Args:
        results (dict): The result of `evaluate`.
        prefs (list[dict]): Preferences with `id` and `complaint`.
        split (str, optional): Only days of this split get feedback. Defaults to "dev".

    Returns:
        list[dict]: Feedback rows with `rating`, `text` and `day`.
    """
    by_id = {p["id"]: p for p in prefs}
    out = []
    for day, r in results.items():
        if r["split"] != split:
            continue
        broken = [by_id[p]["complaint"] for p, ok in r["checks"].items() if not ok]
        out += [{"rating": -1, "text": t, "day": day} for t in broken] or [
            {"rating": 1, "text": "Good brief.", "day": day}
        ]
    return out


def compliance(results: dict, split: str) -> float:
    """Share of (day, preference) checks that pass, on one split.

    Args:
        results (dict): The result of `evaluate`.
        split (str): "dev" or "test".

    Returns:
        float: Between 0 and 1.
    """
    checks = [ok for r in results.values() if r["split"] == split for ok in r["checks"].values()]
    return round(sum(checks) / max(len(checks), 1), 3)


def load(folder: Path) -> tuple[list[dict], list[dict], dict | None]:
    """Read the days, preferences and overfitting example of a case folder.

    Args:
        folder (Path): A `task4` case folder.

    Returns:
        tuple[list[dict], list[dict], dict | None]: Days, preferences and the overfitting example (or None).
    """
    read = lambda f: [json.loads(x) for x in f.read_text().splitlines() if x.strip()] if f.exists() else []  # noqa: E731
    overfit = folder / "overfit.json"
    return (
        read(folder / "days.jsonl"),
        read(folder / "preferences.jsonl"),
        (json.loads(overfit.read_text()) if overfit.exists() else None),
    )


def rounds_into_real_db(n: int) -> list[dict]:
    """Run `n` simulated feedback rounds on the synthetic dev days, into your real evolution history.

    Use it before a live demo: the dashboard then shows the versions and their diffs, and you give one
    round of your own feedback live. Every feedback row is stored with source "simulated".

    Args:
        n (int): Rounds.

    Returns:
        list[dict]: For each round, the new version (or why there is none) and the dev compliance before it.
    """
    from evals import synthetic

    synthetic.ensure("task4")
    days, prefs, _ = load(config.EVAL_SYNTHETIC / "task4")
    dev = [d for d in days if d["split"] == "dev"]
    out = []
    for _ in range(n):
        version = evolve.get_version()
        results = evaluate(version, dev, prefs)
        feedback = complaints(results, prefs)
        for f in feedback:
            evolve.add_feedback(None, f["rating"], f["text"], source="simulated")
        r = evolve.reflect(source="simulated", days=[d["memories"] for d in dev])
        out.append({"from": version["id"], "dev_compliance": compliance(results, "dev"), **r})
    return out
