"""Tests for the scorecard helpers in evals.common.

Skipped on a bare Python: the scorecard imports every eval module, and those need the agent packages.
"""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from agent import config
from evals import common


def write_result(root: Path, tier: str, component: str, metrics: dict) -> None:
    """Write a cached eval result file like `common.run` does.

    Args:
        root (Path): Results folder.
        tier (str): Tier folder name, for example "any" or "e4b".
        component (str): Eval component name.
        metrics (dict): Metric values to store.
    """
    out = root / tier / f"{component}.json"
    out.parent.mkdir(parents=True)
    out.write_text(json.dumps({"n": 3, "when": "2026-10-28 10:00", "metrics": metrics}))


class Scorecard(unittest.TestCase):
    """scorecard_rows and print_scorecard over a temporary results folder."""

    @classmethod
    def setUpClass(cls):
        """Skip the class when an eval module cannot be imported."""
        try:
            for component in common.COMPONENTS:
                common.module(component)
        except ImportError as e:
            raise unittest.SkipTest(f"agent dependencies are not installed: {e}") from e

    def setUp(self):
        """Point config.RESULTS at an empty temporary folder."""
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        patcher = mock.patch.object(config, "RESULTS", self.root)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_components_without_results_are_marked_not_run(self):
        """Every component has one 'not run' row when nothing is cached."""
        rows = common.scorecard_rows()
        self.assertEqual(len(rows), len(common.COMPONENTS))
        self.assertTrue(all(r["status"] == "not run" and r["metric"] == "-" for r in rows))

    def test_cached_metrics_become_rows_with_pass_flags(self):
        """A cached result gives one row per metric, with the threshold and the pass flag."""
        write_result(self.root, "any", "vad", {"false_trigger_rate": 0.05, "missed_speech_rate": 0.5})
        rows = {r["metric"]: r for r in common.scorecard_rows() if r["component"] == "vad"}
        self.assertEqual(set(rows), {"false_trigger_rate", "missed_speech_rate"})
        self.assertEqual(rows["false_trigger_rate"]["threshold"], "<= 0.1")
        self.assertIs(rows["false_trigger_rate"]["any"]["pass"], True)
        self.assertIs(rows["missed_speech_rate"]["any"]["pass"], False)
        self.assertEqual(rows["missed_speech_rate"]["any"]["n"], 3)

    def test_metric_without_threshold_has_no_pass_flag(self):
        """A metric that has no entry in THRESHOLDS shows an empty threshold and pass None."""
        write_result(self.root, "any", "vad", {"extra": 1.0})
        row = next(r for r in common.scorecard_rows() if r["metric"] == "extra")
        self.assertEqual(row["threshold"], "")
        self.assertIsNone(row["any"]["pass"])

    def test_cell_formatting(self):
        """Cells are 16 characters wide: blank, float with PASS, or plain value."""
        self.assertEqual(common._scorecard_cell(None), " " * 16)
        self.assertEqual(common._scorecard_cell({"value": 0.5, "pass": True}), f"{'0.500 PASS':>16s}")
        self.assertEqual(common._scorecard_cell({"value": 7, "pass": False}), f"{'7 FAIL':>16s}")
        self.assertEqual(common._scorecard_cell({"value": 7, "pass": None}), f"{'7 ':>16s}")

    def test_print_scorecard_lists_every_component(self):
        """The printed table has a header and a line for each component."""
        write_result(self.root, "any", "vad", {"false_trigger_rate": 0.05})
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            common.print_scorecard()
        text = buf.getvalue()
        self.assertIn("threshold", text)
        self.assertIn("0.050 PASS", text)
        self.assertEqual(text.count("not run: python cli.py evals --run"), len(common.COMPONENTS) - 1)


if __name__ == "__main__":
    unittest.main()
