"""Task 4 end-to-end eval: does the brief learn the user's preferences, on days it never got feedback on?

Cases: data/evals/{synthetic,user}/task4/
    days.jsonl         {"id": "d1", "split": "dev", "memories": ["(09:00, summary) ..."]}
    preferences.jsonl  {"id": "max_bullets", "text": "Use at most 6 bullets.", "complaint": "Too long."}
    overfit.json       {"day": "d1", "feedback": ["Start with Dentist."], "leak_words": ["dentist"]}  (optional)

Items run in order, in a throwaway evolution database that starts at the seed prompt:
    round 0    brief every day with the seed version and check every preference
    round 1..R the simulated user complains about the dev briefs of the round before; reflect; check again
    overfit    from the seed again: feedback that is only true for one dev day, 3 rounds, then check
A preference with a known id is checked by code (evals/task4/simulate.py). Your own preferences are
judged by Gemma, which is weaker: read the failures.

Metrics:
    dev_before / dev_after, test_before / test_after   share of (day, preference) checks that pass
    test_gain             test_after - test_before (held-out: the reflection never saw these days)
    regression_rate       share of test checks that passed at round 0 and fail after the last round
    locked_edits          versions whose locked blocks differ from evolve.LOCKED (must be 0)
    patches_rejected      patches that the validator refused
    overfit_dev_gain      the one-day rule (headline starts with ..., first bullet names ...), on its day: after - before
    overfit_test_leak     share of test briefs that mention the one-day details after the overfit rounds
    overfit_test_change   test compliance after the overfit rounds - before
"""

from pathlib import Path

from agent import evolve
from evals import common
from evals.common import NoCases
from evals.task4 import simulate

LEVEL, USES_LLM = "end-to-end", True
ROUNDS = 3
_state: dict = {}  # carries the results of one round to the next


def load_cases(cases_dir: Path):
    """Return the rounds to run for a case folder.

    Args:
        cases_dir (Path): The case folder of this eval.

    Returns:
        list[dict]: Round 0 to ROUNDS, then the overfitting example if the folder has one.

    Raises:
        NoCases: If the folder has no days or no preferences.
    """
    days, prefs, overfit = simulate.load(cases_dir)
    if not days or not prefs:
        raise NoCases(cases_dir)
    items = [{"op": "round", "n": n, "folder": str(cases_dir)} for n in range(ROUNDS + 1)]
    return items + ([{"op": "overfit", "folder": str(cases_dir)}] if overfit else [])


def _round(item: dict, days: list[dict], prefs: list[dict]) -> dict:
    if item["n"] == 0:
        _state.clear()
        _state["rejected"] = 0
        version = evolve.get_version()
    else:
        r = evolve.reflect(simulate.complaints(_state["last"], prefs), source="simulated")
        _state["rejected"] += bool(r["rejected"] and r["patch"])
        version = evolve.get_version()
    results = simulate.evaluate(version, days, prefs)
    _state["last"] = results
    _state.setdefault("first", results)
    return {
        "op": "round",
        "folder": item["folder"],
        "round": item["n"],
        "version": version["id"],
        "preferences": version["preferences"],
        "dev": simulate.compliance(results, "dev"),
        "test": simulate.compliance(results, "test"),
        "checks": {d: r["checks"] for d, r in results.items()},
        "splits": {d: r["split"] for d, r in results.items()},
        "briefs": {d: evolve.to_markdown(r["brief"]) for d, r in results.items()},
        "locked_broken": sum(bool(evolve.check_locked(evolve.render(v))) for v in evolve.versions()),
        "rejected": _state["rejected"],
    }


def _overfit(folder: str, days: list[dict], prefs: list[dict], spec: dict) -> dict:
    """Train on one day's details for 3 rounds from the seed, then look at that day and the test days."""
    evolve.rollback(1)
    day = [d for d in days if d["id"] == spec["day"]]
    test = [d for d in days if d["split"] == "test"]
    before = simulate.follows_rule(simulate.evaluate(evolve.get_version(), day, [])[spec["day"]]["brief"], spec)
    for _ in range(3):
        evolve.reflect([{"rating": -1, "text": t} for t in spec["feedback"]], source="simulated")
    v = evolve.get_version()
    after_day = simulate.evaluate(v, day, [])[spec["day"]]["brief"]
    after_test = simulate.evaluate(v, test, prefs)
    leaked = [d for d, r in after_test.items() if simulate.leaks(r["brief"], spec["leak_words"])]
    return {
        "op": "overfit",
        "folder": folder,
        "version": v["id"],
        "preferences": v["preferences"],
        "day_follows_before": before,
        "day_follows_after": simulate.follows_rule(after_day, spec),
        "day_brief": evolve.to_markdown(after_day),
        "test_leaked_days": leaked,
        "test_days": len(after_test),
        "test": simulate.compliance(after_test, "test"),
        "briefs": {d: evolve.to_markdown(r["brief"]) for d, r in after_test.items()},
    }


def run_item(item):
    """Run one round, or the overfitting example.

    Args:
        item (dict): An item from `load_cases`.

    Returns:
        dict: The record.
    """
    days, prefs, overfit = simulate.load(Path(item["folder"]))
    with common.isolated("task4") as folder:
        if item["op"] == "round" and item["n"] == 0:
            common.reset(folder)
        if item["op"] == "overfit":
            return _overfit(item["folder"], days, prefs, overfit or {})
        return _round(item, days, prefs)


def regression_rate(first: dict, last: dict) -> float:
    """Share of test checks that passed in the first round and fail in the last.

    Args:
        first (dict): The round 0 record.
        last (dict): The last round record.

    Returns:
        float: Between 0 and 1, 0 when nothing passed at first.
    """
    passed = [
        (d, p)
        for d, checks in first["checks"].items()
        if first["splits"][d] == "test"
        for p, ok in checks.items()
        if ok
    ]
    return round(sum(not last["checks"][d][p] for d, p in passed) / max(len(passed), 1), 3)


def _score_folder(records: list[dict]) -> dict:
    """Compute the metrics of one case folder: its rounds in order, then its overfitting example."""
    rounds = [r for r in records if r["op"] == "round"]
    if not rounds:
        return {}
    first, last = rounds[0], rounds[-1]
    out = {
        "dev_before": first["dev"],
        "dev_after": last["dev"],
        "test_before": first["test"],
        "test_after": last["test"],
        "test_gain": round(last["test"] - first["test"], 3),
        "regression_rate": regression_rate(first, last),
        "locked_edits": max(r["locked_broken"] for r in rounds),
        "patches_rejected": last["rejected"],
    }
    for r in records:
        if r["op"] == "overfit":
            out["overfit_dev_gain"] = int(r["day_follows_after"]) - int(r["day_follows_before"])
            out["overfit_test_leak"] = round(len(r["test_leaked_days"]) / max(r["test_days"], 1), 3)
            out["overfit_test_change"] = round(r["test"] - first["test"], 3)
    return out


def score(records):
    """Compute the metrics. Each case folder learns from its own seed, so each is scored on its own.

    With synthetic and user folders, a rate is the mean over the folders, `locked_edits` the worst
    folder and `patches_rejected` the sum. `by_source` in the result file keeps each folder's numbers.

    Args:
        records (list[dict]): The records from `run_item`, in order.

    Returns:
        dict: Metric name to value.
    """
    folders = list(dict.fromkeys(r.get("folder", "") for r in records))
    per = [m for f in folders if (m := _score_folder([r for r in records if r.get("folder", "") == f]))]
    if len(per) <= 1:
        return per[0] if per else {}
    keys = {k for m in per for k in m}
    combine = {"locked_edits": max, "patches_rejected": sum}
    out = {}
    for k in sorted(keys):
        values = [m[k] for m in per if k in m]
        out[k] = combine[k](values) if k in combine else round(sum(values) / len(values), 3)
    return out
