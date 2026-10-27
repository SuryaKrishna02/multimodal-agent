"""The eval pattern, shared by every component.

Each evals/<component>/eval.py defines:
    LEVEL       "component" or "end-to-end"
    USES_LLM    True if results depend on the model tier (E4B vs E2B)
    load_gold(gold_dir) -> list of items          (raises NoGold if nothing is recorded yet)
    run_item(item)      -> record dict            (one input -> one output, shown in live mode)
    score(records)      -> {metric: value}

run() caches results in evals/results/<tier>/<component>.json, so the scorecard
can be shown instantly and a full run can happen before the session.
"""
import importlib
import json
import time
from pathlib import Path

from agent import config

COMPONENTS = ["serving", "vad", "asr", "screen", "structured", "tool_calls", "memory", "tools", "task1", "task2"]


class NoGold(Exception):
    """The gold set for this component has not been recorded yet."""


def module(component: str):
    return importlib.import_module(f"evals.{component}.eval")


def _tier_dir(mod) -> str:
    return config.TIER if mod.USES_LLM else "any"


def _start_servers(mod) -> None:
    from agent import serving
    if getattr(mod, "NEEDS_LLM_SERVER", mod.USES_LLM):
        serving.llm_server().start()
    if getattr(mod, "NEEDS_EMBED_SERVER", False):
        serving.embed_server().start()


def run(component: str, gold_dir: Path | None = None, limit: int | None = None, verbose: bool = False) -> dict:
    mod = module(component)
    items = mod.load_gold((gold_dir or config.GOLD) / component)[:limit]
    _start_servers(mod)
    records, t0 = [], time.time()
    for i, item in enumerate(items, 1):
        t_item = time.time()
        rec = {**mod.run_item(item), "seconds": round(time.time() - t_item, 2)}
        records.append(rec)
        if verbose:
            print(f"  [{i}/{len(items)}] {json.dumps(rec, default=str)[:160]}")
    result = {"component": component, "level": mod.LEVEL, "tier": _tier_dir(mod), "n": len(records),
              "metrics": mod.score(records), "seconds": round(time.time() - t0, 1),
              "when": time.strftime("%Y-%m-%d %H:%M"), "gold_dir": str(gold_dir or config.GOLD),
              "records": records}
    out = config.RESULTS / result["tier"] / f"{component}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, default=str))
    return result


def run_live(component: str, n: int = 1, gold_dir: Path | None = None) -> list[dict]:
    """Run a few items with visible inputs and outputs. Not cached."""
    mod = module(component)
    items = mod.load_gold((gold_dir or config.GOLD) / component)[:n]
    _start_servers(mod)
    return [mod.run_item(item) for item in items]


def passes(metric_key: str, value) -> bool | None:
    if metric_key not in config.THRESHOLDS or value is None:
        return None
    op, threshold = config.THRESHOLDS[metric_key]
    return value <= threshold if op == "<=" else value >= threshold


def scorecard_rows() -> list[dict]:
    """One row per (component, metric), with a column per tier found in the cache."""
    rows = []
    for component in COMPONENTS:
        mod = module(component)
        tiers = list(config.TIERS) if mod.USES_LLM else ["any"]
        cached = {}
        for t in tiers:
            f = config.RESULTS / t / f"{component}.json"
            if f.exists():
                cached[t] = json.loads(f.read_text())
        if not cached:
            rows.append({"component": component, "level": mod.LEVEL, "metric": "-", "status": "not run"})
            continue
        metrics = sorted({m for r in cached.values() for m in r["metrics"]})
        for m in metrics:
            key = f"{component}.{m}"
            row = {"component": component, "level": mod.LEVEL, "metric": m,
                   "threshold": " ".join(map(str, config.THRESHOLDS[key])) if key in config.THRESHOLDS else ""}
            for t, r in cached.items():
                v = r["metrics"].get(m)
                row[t] = {"value": v, "pass": passes(key, v), "n": r["n"], "when": r["when"]}
            rows.append(row)
    return rows


def print_scorecard() -> None:
    rows = scorecard_rows()
    tiers = ["any", *config.TIERS]
    print(f"\n{'level':11s} {'component':11s} {'metric':24s} {'threshold':10s} " +
          " ".join(f"{t:>16s}" for t in tiers))
    print("-" * (60 + 17 * len(tiers)))
    for level in ("component", "end-to-end"):
        for r in [r for r in rows if r["level"] == level]:
            cells = []
            for t in tiers:
                c = r.get(t)
                if not c:
                    cells.append(f"{'':>16s}")
                    continue
                v = c["value"]
                mark = {True: "PASS", False: "FAIL", None: ""}[c["pass"]]
                vs = f"{v:.3f}" if isinstance(v, float) else str(v)
                cells.append(f"{vs + ' ' + mark:>16s}")
            print(f"{level:11s} {r['component']:11s} {r['metric']:24s} {r.get('threshold', ''):10s} " +
                  (" ".join(cells) if r.get("status") != "not run" else "   (not run: see docs/RECORDING_GUIDE.md)"))
    print()
