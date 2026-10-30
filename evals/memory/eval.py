"""Memory eval: are the right memories found, and is junk kept out?

Cases: data/evals/{synthetic,user}/memory/
    memories.jsonl   {"id": "m01", "text": "Dentist moved to Friday 3pm", "kind": "note", "junk": false}
                     {"id": "m02", "text": "uh okay", "kind": "note", "junk": true}
    queries.jsonl    {"query": "when is the dentist?", "relevant": ["m01"]}

Every memory is offered to store() in order (junk filter on), into a separate eval database.
Then every query runs search(k=3).

Metrics:
    recall_at_3         mean share of relevant memories found in the top 3
    junk_stored_rate    share of junk items that got stored anyway
    good_rejected_rate  share of good items the filter wrongly threw away
"""

import json
from contextlib import contextmanager
from pathlib import Path

from agent import config
from agent.memory import episodic
from evals.common import NoCases

LEVEL, USES_LLM = "component", False
NEEDS_EMBED_SERVER = True
_db_ids: dict[int, str] = {}  # database row id -> case id


@contextmanager
def eval_db():
    """Point memory at a separate database for the duration, so your real memory is untouched."""
    real, config.MEMORY_DB = config.MEMORY_DB, config.RESULTS / "memory-eval.sqlite"
    try:
        yield
    finally:
        config.MEMORY_DB = real


def load_cases(cases_dir: Path):
    """Load the cases: memories to store, then queries to search. It also empties the eval database.

    Args:
        cases_dir (Path): The case folder of this eval.

    Returns:
        list[dict]: One item per case.

    Raises:
        NoCases: If `memories.jsonl` or `queries.jsonl` is missing.
    """
    mem_f, q_f = cases_dir / "memories.jsonl", cases_dir / "queries.jsonl"
    if not (mem_f.exists() and q_f.exists()):
        raise NoCases(cases_dir)
    read = lambda f: [json.loads(x) for x in f.read_text().splitlines() if x.strip()]  # noqa: E731
    with eval_db():
        episodic.clear()  # each run starts from an empty eval database
    _db_ids.clear()
    return [{"op": "store", **m} for m in read(mem_f)] + [{"op": "query", **q} for q in read(q_f)]


def run_item(item):
    """Run one case: store one memory or run one query, in the eval database.

    Args:
        item (dict): An item from `load_cases`.

    Returns:
        dict: A record with `op` and the outcome.
    """
    with eval_db():
        return _run(item)


def _run(item):
    if item["op"] == "store":
        row = episodic.store(item["text"], kind=item.get("kind", "note"))
        if row:
            _db_ids[row] = item["id"]
        return {"op": "store", "id": item["id"], "junk": item["junk"], "stored": row is not None}
    hits = [_db_ids.get(h["id"]) for h in episodic.search(item["query"], k=3)]
    found = [r for r in item["relevant"] if r in hits]
    return {
        "op": "query",
        "query": item["query"],
        "relevant": item["relevant"],
        "top3": hits,
        "recall": len(found) / max(len(item["relevant"]), 1),
    }


def score(records):
    """Compute the metrics: recall at 3, junk stored rate and good rejected rate.

    Args:
        records (list[dict]): The records from `run_item`.

    Returns:
        dict: Metric name to value.
    """
    stores = [r for r in records if r["op"] == "store"]
    queries = [r for r in records if r["op"] == "query"]
    junk, good = [r for r in stores if r["junk"]], [r for r in stores if not r["junk"]]
    return {
        "recall_at_3": round(sum(r["recall"] for r in queries) / max(len(queries), 1), 3),
        "junk_stored_rate": round(sum(r["stored"] for r in junk) / max(len(junk), 1), 3),
        "good_rejected_rate": round(sum(not r["stored"] for r in good) / max(len(good), 1), 3),
    }
