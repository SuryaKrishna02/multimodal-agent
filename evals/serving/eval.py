"""Serving eval: time to first token, generation speed, peak memory. No case files needed.

Run it once per tier to compare:  python evals/run_all.py --run serving --tier e2b

Memory: peak_rss_gb is the server's resident memory. Weights are memory-mapped, so macOS can
page some out and RSS moves by a GB or more between runs. system_used_gb is the whole Mac
(model + embeddings + browser + Meet...), which is what decides whether you start swapping.
"""

import statistics
import threading
import time

import psutil

from agent import serving

LEVEL, USES_LLM = "component", True
BUILTIN = True  # the prompts are fixed in code, so there are no case files
LONG_CONTEXT = (
    "Meeting notes: the team reviewed the launch plan, the budget, hiring, and the demo schedule. " * 120
)  # ~2,000 tokens, like a prompt with memories and a transcript
PROMPTS = [  # fixed prompts so runs are comparable; short, medium and long input
    "Say hello.",
    "Explain what a context window is in about 80 words.",
    LONG_CONTEXT + "\nSummarize these notes in two sentences.",
]
REPEATS = 3


def load_cases(_cases_dir):
    """Build the fixed prompt list. This eval needs no recorded data.

    Args:
        _cases_dir: Not used.

    Returns:
        list[dict]: Each prompt repeated REPEATS times.
    """
    return [{"prompt": p, "repeat": r} for p in PROMPTS for r in range(REPEATS)]


def run_item(item):
    """Run one case: time one streamed reply while sampling memory.

    Args:
        item (dict): An item from `load_cases`.

    Returns:
        dict: A record with the timings and the peak memory.
    """
    server = serving.llm_server()
    peak = {"rss": server.rss_gb(), "system_used": psutil.virtual_memory().used / 1e9}
    done = threading.Event()

    def sample():  # watch memory while the request runs
        while not done.is_set():
            peak["rss"] = max(peak["rss"], server.rss_gb())
            peak["system_used"] = max(peak["system_used"], psutil.virtual_memory().used / 1e9)
            time.sleep(0.1)

    t = threading.Thread(target=sample)
    t.start()
    r = serving.stream_timing([{"role": "user", "content": item["prompt"]}], max_tokens=128)
    done.set()
    t.join()
    return {
        "prompt": item["prompt"][:40],
        **r,
        "peak_rss_gb": round(peak["rss"], 2),
        "system_used_gb": round(peak["system_used"], 2),
    }


def score(records):
    """Compute the metrics: median time to first token, generation speed and peak memory.

    Args:
        records (list[dict]): The records from `run_item`.

    Returns:
        dict: Metric name to value.
    """

    def med(key, prompt_index=None):
        rows = [r for r in records if prompt_index is None or r["prompt"] == PROMPTS[prompt_index][:40]]
        return round(statistics.median(r[key] for r in rows if r[key] is not None), 3)

    return {
        "ttft_s": med("ttft_s", 0),  # short prompt: the floor
        "ttft_2k_s": med("ttft_s", 2),  # ~2k-token prompt: what a real request costs
        "prompt_tok_s": med("prompt_tok_s", 2),
        "gen_tok_s": med("gen_tok_s"),
        "peak_rss_gb": max(r["peak_rss_gb"] for r in records),
        "system_used_gb": max(r["system_used_gb"] for r in records),
    }
