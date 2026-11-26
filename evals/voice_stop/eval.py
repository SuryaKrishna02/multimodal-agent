"""Stop eval: how long speech keeps playing after someone asks for silence.

Built in code (no case files). Two kinds of record:
    local   play 5 s of near-silence through the OS plug-in (`play_audio`), stop it at a random moment, and time
            until the stream reports done. Quiet on purpose: the eval runs on your speakers.
    tab     the stops that a dashboard tab reported (`voice.stopped` events with route "tab", the newest 20 in the
            event log): the time from the stop event to silence in the tab, fade included.
Run the Voice tab's Stop button (or say the wake word over the agent) a few times to get tab records.

Metric:
    stop_p95_ms   95th percentile of the stop latency, both kinds together (budget: config.LATENCY_BUDGET["stop"])
    local_p50_ms / tab_p50_ms   the median of each kind
"""

import statistics
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from agent import config, events, platforms
from evals.common import NoCases

LEVEL, USES_LLM, BUILTIN = "component", False, True
NEEDS_LLM_SERVER = False
TRIALS = 8


def load_cases(cases_dir: Path):
    """Make the local trials and read the tab's stops from the event log.

    Args:
        cases_dir (Path): Not used: the cases are built in code.

    Returns:
        list[dict]: Items with `kind` "local" (with a stop time) or "tab" (with the measured latency).
    """
    rng = np.random.default_rng(5)
    items = [{"kind": "local", "id": f"local{i}", "stop_after_s": float(rng.uniform(0.3, 1.5))} for i in range(TRIALS)]
    tab = [e for e in events.since(0, "voice.stopped") if e.get("route") == "tab"][-20:]
    items += [{"kind": "tab", "id": f"tab{e['id']}", "after_stop_s": e["after_stop_s"]} for e in tab]
    if not items:
        raise NoCases("no stops to measure")
    return items


def _local(stop_after_s: float) -> float:
    """Play near-silence, stop it after `stop_after_s`, and return the seconds until it is done."""
    with tempfile.TemporaryDirectory() as d:
        wav = Path(d) / "quiet.wav"
        rate = config.TTS_SAMPLE_RATE
        sf.write(wav, np.random.default_rng(0).normal(0, 1e-4, 5 * rate).astype(np.float32), rate)
        playing = platforms.current().play_audio(wav)
        time.sleep(stop_after_s)
        t0 = time.perf_counter()
        playing.stop()
        while not playing.done() and time.perf_counter() - t0 < 2:
            time.sleep(0.002)
        return time.perf_counter() - t0


def run_item(item):
    """Measure one stop, or pass on one that the tab measured.

    Args:
        item (dict): An item from `load_cases`.

    Returns:
        dict: `id`, `kind` and `stop_ms`.
    """
    seconds = _local(item["stop_after_s"]) if item["kind"] == "local" else item["after_stop_s"]
    return {"id": item["id"], "kind": item["kind"], "stop_ms": round(seconds * 1000, 1)}


def score(records):
    """Compute the stop latency percentiles.

    Args:
        records (list[dict]): The records from `run_item`.

    Returns:
        dict: Metric name to value.
    """
    ms = sorted(r["stop_ms"] for r in records)
    out = {"stop_p95_ms": ms[min(len(ms) - 1, int(round(0.95 * (len(ms) - 1))))]}
    for kind in ("local", "tab"):
        if kind_ms := [r["stop_ms"] for r in records if r["kind"] == kind]:
            out[f"{kind}_p50_ms"] = round(statistics.median(kind_ms), 1)
    return out
