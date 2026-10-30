"""Print the scorecard. Optionally (re)run evals first.

    python evals/run_all.py                         # print cached scorecard (instant)
    python evals/run_all.py --run                   # run every eval (synthetic + your cases), then print
    python evals/run_all.py --run asr task1         # run only these
    python evals/run_all.py --run --tier e2b        # same, on the small tier
    python evals/run_all.py --live asr -n 2         # run 2 items with full inputs/outputs (not cached)
    python evals/run_all.py status                  # how many cases each eval has
    python evals/run_all.py build [component]       # rebuild the synthetic cases
    python evals/run_all.py add <component> ...     # add your own case (see: add -h)

Run it from the repository root with the Python of the venv.
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    """Print the scorecard, after running or showing evals if the options ask for it."""
    if sys.argv[1:2] and sys.argv[1] in ("add", "build", "status"):
        from evals import cases

        return cases.main(sys.argv[1:])
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", nargs="*", metavar="COMPONENT", help="run these evals (none listed = all)")
    ap.add_argument("--live", metavar="COMPONENT")
    ap.add_argument("-n", type=int, default=1, help="items for --live")
    ap.add_argument("--tier", help="e4b or e2b (default: AGENT_TIER or e4b)")
    ap.add_argument(
        "--cases-dir", type=Path, help="run only the cases in this folder, which holds one folder per component"
    )
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    if args.tier:
        os.environ["AGENT_TIER"] = args.tier

    from agent import config
    from evals import common

    if args.tier:
        config.TIER = args.tier
    if args.live:
        for rec in common.run_live(args.live, args.n, args.cases_dir):
            print(json.dumps(rec, indent=1, default=str))
        return
    if args.run is not None:
        for c in args.run or common.COMPONENTS:
            try:
                r = common.run(c, args.cases_dir, args.limit, verbose=True)
                print(f"{c:11s} n={r['n']:<3d} {r['seconds']:6.1f}s  {r['sources']}  {r['metrics']}")
            except common.NoCases as e:
                print(f"{c:11s} skipped: {e}")
    common.print_scorecard()


if __name__ == "__main__":
    main()
