"""Consolidation eval: are summaries faithful, how much smaller are they, and can you still find things after forgetting?

Cases: data/evals/{synthetic,user}/consolidation/
    workday.json   {"rows": [{"at": "09:02", "kind": "transcript", "text": "..."}],
                    "queries": [{"query": "which venue did we pick?", "relevant": [0]}]}
    (`relevant` holds row numbers of `rows`, starting at 0)

One case = one day log, run in a throwaway database:
    1. store the rows at their times (today), then run each query: recall before
    2. consolidate(force=True): summaries and facts, each claim citing raw ids
    3. judge every claim against only the rows it cites (a separate Gemma request)
    4. forget() as if RAW_RETENTION_DAYS had passed: the covered raw rows are deleted
    5. run each query again: a summary or fact that cites a relevant row counts as a hit: recall after

Metrics:
    faithfulness          share of claims and facts that the judge finds supported by their cited rows
    invalid_citation_rate share of cited ids that were not in the window (dropped by code before storing)
    compression_ratio     characters of raw rows / characters of the summaries that replace them
    facts_per_raw_row     facts kept per raw row (facts are an index next to the summary, not a replacement)
    recall_before / recall_after   share of queries with a relevant row (or its summary) in the top 3
The judge is the same model family as the writer, so it can share its blind spots: read the failed claims.
"""

import json
import time
from pathlib import Path

from agent import config
from agent.memory import consolidate, episodic
from evals import common
from evals.common import NoCases

LEVEL, USES_LLM = "component", True
NEEDS_EMBED_SERVER = True


def load_cases(cases_dir: Path):
    """Load the day logs.

    Args:
        cases_dir (Path): The case folder of this eval.

    Returns:
        list[dict]: One item per log file.

    Raises:
        NoCases: If there is no log.
    """
    items = [{"name": f.stem, **json.loads(f.read_text())} for f in sorted(cases_dir.glob("*.json"))]
    if not items:
        raise NoCases(cases_dir)
    return items


def _at(hhmm: str) -> float:
    h, m = map(int, hhmm.split(":"))
    return time.mktime(time.localtime()[:3] + (h, m, 0, 0, 0, -1))


def _recall(queries: list[dict], ids: list[int | None]) -> list[float]:
    """Run each query and score it. A hit is a relevant raw row, or a summary/fact that cites one."""
    out = []
    for q in queries:
        want = {ids[i] for i in q["relevant"] if ids[i]}
        hits = episodic.search(q["query"], k=3)
        cited = [{h["id"], *h["meta"].get("sources", [])} for h in hits]
        out.append(float(any(want & c for c in cited)))
    return out


def _judge_all(results: list[dict]) -> list[dict]:
    """Judge every claim and every fact against the rows it cites."""
    checks = [{"claim": c["text"], "sources": c["sources"]} for r in results for c in r["claims"]]
    for r in results:
        for fid in r.get("fact_ids", []):
            fact = episodic.all_memories(ids=[fid])[0]
            checks.append({"claim": fact["text"], "sources": fact["meta"]["sources"]})
    return [{**c, **consolidate.judge(c["claim"], c["sources"])} for c in checks]


def run_item(item):
    """Run one day log through consolidation, judging and forgetting.

    Args:
        item (dict): A log from `load_cases`.

    Returns:
        dict: A record with the claims, the judge's verdicts and the recall before and after.
    """
    with common.isolated("consolidation") as folder:
        common.reset(folder)
        ids = [episodic.store(r["text"], kind=r["kind"], ts=_at(r["at"])) for r in item["rows"]]
        before = _recall(item["queries"], ids)
        results = consolidate.consolidate(force=True)
        judged = _judge_all(results)
        stats = consolidate.stats()
        consolidate.forget(now=time.time() + (config.RAW_RETENTION_DAYS + 1) * 86400)
        after = _recall(item["queries"], ids)
    return {
        "log": item["name"],
        "raw_rows": len(item["rows"]),
        "summaries": stats["summary"]["rows"],
        "facts": stats["fact"]["rows"],
        "raw_chars": stats["raw"]["chars"],
        "summary_chars": stats["summary"]["chars"],
        "citations": sum(len(c["sources"]) for c in judged),
        "invalid_citations": sum(r.get("invalid_citations", 0) for r in results),
        "judged": judged,
        "recall_before": before,
        "recall_after": after,
    }


def score(records):
    """Compute the metrics over all logs.

    Args:
        records (list[dict]): The records from `run_item`.

    Returns:
        dict: Metric name to value.
    """
    judged = [j for r in records for j in r["judged"]]
    before = [x for r in records for x in r["recall_before"]]
    after = [x for r in records for x in r["recall_after"]]
    cited = sum(r["citations"] + r["invalid_citations"] for r in records)
    return {
        "faithfulness": round(sum(j["supported"] for j in judged) / max(len(judged), 1), 3),
        "invalid_citation_rate": round(sum(r["invalid_citations"] for r in records) / max(cited, 1), 3),
        "compression_ratio": round(
            sum(r["raw_chars"] for r in records) / max(sum(r["summary_chars"] for r in records), 1), 2
        ),
        "facts_per_raw_row": round(sum(r["facts"] for r in records) / max(sum(r["raw_rows"] for r in records), 1), 2),
        "recall_before": round(sum(before) / max(len(before), 1), 3),
        "recall_after": round(sum(after) / max(len(after), 1), 3),
    }
