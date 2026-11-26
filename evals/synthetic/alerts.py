"""Synthetic screens for the Session 3 alert evals: proactive (one screen) and task5 (a sequence of screens).

Labeled by construction: we draw each screen, so we know whether it shows a failed test run (the condition in
`config.WATCH_CONDITION`). The negatives are the hard part: errors that are not tests, code that only mentions tests,
and passing runs. Rendered text is cleaner than a real screen: add your own screenshots with `evals add proactive`.
"""

import json
from pathlib import Path

from evals.synthetic.screens import SHOTS, render

# name -> (lines, shows a failed test run, a word that the alert must name, or None)
ALERT_SHOTS = {
    "pytest-fail": (SHOTS["test-failure"][0], True, "test_add"),
    "unittest-fail": (
        [
            "$ python -m unittest",
            "F.....",
            "FAIL: test_parse_dates (tests.test_io.ParseTest)",
            "AssertionError: KeyError not raised",
            "Ran 6 tests in 0.04s",
            "FAILED (failures=1)",
        ],
        True,
        "test_parse_dates",
    ),
    "jest-fail": (
        [
            "$ npm test",
            " FAIL  src/cart.test.js",
            "  x adds two items (5 ms)",
            "Tests:       1 failed, 7 passed, 8 total",
        ],
        True,
        "adds two items",  # the failing test (cart.test.js is its file); fixed after the first run named the test
    ),
    "go-fail": (
        ["$ go test ./...", "--- FAIL: TestSlugify (0.00s)", '    slug_test.go:14: got "a-b", want "a_b"', "FAIL"],
        True,
        "TestSlugify",
    ),
    "pytest-pass": (SHOTS["tests-pass"][0], False, None),
    "clean-tree": (SHOTS["all-clean"][0], False, None),
    "import-error": (SHOTS["module-missing"][0], False, None),
    "port-in-use": (SHOTS["port-in-use"][0], False, None),
    "merge-conflict": (SHOTS["merge-conflict"][0], False, None),
    "test-code": (
        ["# tests/test_math.py", "def test_add():", "    assert add(2, 2) == 4", "", "def test_sub():"],
        False,
        None,
    ),
    "unittest-pass": (["$ python -m unittest", "......", "Ran 6 tests in 0.04s", "", "OK"], False, None),
}

# A screen session for Task 5: (name, frames as names of ALERT_SHOTS, index of the first failing frame or None)
SEQUENCES = [
    ("edit-run-fail", ["test-code", "test-code", "pytest-pass", "pytest-fail", "pytest-fail"], 3),
    ("fail-then-fix", ["clean-tree", "unittest-fail", "unittest-fail", "unittest-pass"], 1),
    ("go-fail", ["clean-tree", "go-fail"], 1),
    ("errors-not-tests", ["import-error", "port-in-use", "merge-conflict"], None),
    ("all-green", ["test-code", "pytest-pass", "unittest-pass", "clean-tree"], None),
]


def build_proactive(out: Path) -> None:
    """Write each alert screen with its label.

    Args:
        out (Path): The case folder of the `proactive` eval.
    """
    out.mkdir(parents=True, exist_ok=True)
    for name, (lines, alert, mention) in ALERT_SHOTS.items():
        render(lines).save(out / f"{name}.png")
        (out / f"{name}.json").write_text(json.dumps({"alert": alert, "mention": mention}))


def build_task5(out: Path) -> None:
    """Write each screen session: one folder of numbered frames and the first frame that must alert.

    Args:
        out (Path): The case folder of the `task5` eval.
    """
    for name, frames, fail_at in SEQUENCES:
        folder = out / name
        folder.mkdir(parents=True, exist_ok=True)
        for i, shot in enumerate(frames):
            render(ALERT_SHOTS[shot][0]).save(folder / f"{i:02d}.png")
        mention = ALERT_SHOTS[frames[fail_at]][2] if fail_at is not None else None
        (folder / "sequence.json").write_text(json.dumps({"fail_at": fail_at, "mention": mention}))
