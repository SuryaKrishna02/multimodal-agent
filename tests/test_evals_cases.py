"""Tests for the synthetic eval cases and for adding your own.

They use only the text-based evals (no model, no speech), a temporary folder and a temporary results folder.
Skipped on a bare Python: the eval modules need the agent packages.
"""

import argparse
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from agent import config
from evals import cases, common, synthetic
from evals.synthetic import text


def args(**kwargs) -> argparse.Namespace:
    """Build the namespace that `cases.main` would pass to a handler."""
    return argparse.Namespace(**kwargs)


class SyntheticTables(unittest.TestCase):
    """The hand-written case tables agree with the code they test."""

    def test_tool_names_exist(self):
        """Every tool in the tool cases is registered, except the one that tests an unknown name."""
        from agent import tools

        names = {t for t, _, _ in text.TOOL_POLICY} | {t for _, t, _ in text.TOOL_CALLS if t}
        self.assertEqual(names - set(tools.REGISTRY), {"no_such_tool"})

    def test_policy_expectations_are_known(self):
        """Every permission case expects one of the outcomes that the tools eval knows."""
        self.assertLessEqual({e for _, _, e in text.TOOL_POLICY}, set(cases.EXPECTATIONS))

    def test_memory_queries_point_at_good_memories(self):
        """Ids are unique, and every query is answered by memories that are not junk."""
        ids = [i for i, _, _, _ in text.MEMORIES]
        self.assertEqual(len(ids), len(set(ids)))
        good = {i for i, _, _, junk in text.MEMORIES if not junk}
        for _query, relevant in text.QUERIES:
            self.assertTrue(relevant and set(relevant) <= good)

    def test_junk_is_short_or_a_copy(self):
        """Junk follows the filter design: under the minimum word count, or a copy of a stored memory."""
        stored = {t for _, t, _, junk in text.MEMORIES if not junk}
        for _, memory, _, junk in text.MEMORIES:
            if junk:
                self.assertTrue(len(memory.split()) < config.MIN_MEMORY_WORDS or memory in stored, memory)


class Workspace(unittest.TestCase):
    """Base class: point the case folders and the results folder at temporary folders."""

    @classmethod
    def setUpClass(cls):
        """Skip the class when an eval module cannot be imported."""
        try:
            for component in common.COMPONENTS:
                common.module(component)
        except ImportError as e:
            raise unittest.SkipTest(f"agent dependencies are not installed: {e}") from e

    def setUp(self):
        """Create the temporary folders and patch config to use them."""
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        for name in ("EVAL_SYNTHETIC", "EVAL_USER", "RESULTS"):
            patcher = mock.patch.object(config, name, self.root / name.lower())
            patcher.start()
            self.addCleanup(patcher.stop)
        silence = mock.patch("agent.events.emit")  # tool calls would write to the real event log
        silence.start()
        self.addCleanup(silence.stop)


class SyntheticBuild(Workspace):
    """Building the synthetic cases."""

    def test_text_builders_write_what_the_loaders_read(self):
        """Each text-based builder gives a folder that the loader of its eval accepts."""
        for component in ("tools", "tool_calls", "memory"):
            folder = synthetic.build(component)
            self.assertTrue(common.module(component).load_cases(folder), component)

    def test_ensure_builds_once_until_the_code_changes(self):
        """A second ensure() does nothing, and a different code version rebuilds."""
        self.assertEqual(synthetic.ensure("tools"), ["tools"])
        self.assertEqual(synthetic.ensure("tools"), [])
        with mock.patch.object(synthetic, "code_version", return_value="other"):
            self.assertEqual(synthetic.ensure("tools"), ["tools"])

    def test_failed_build_keeps_the_old_cases(self):
        """If a builder fails, the folder from the last good build stays."""
        folder = synthetic.build("tools")
        failing = {"tools": mock.Mock(side_effect=RuntimeError("boom"))}
        with mock.patch.dict(synthetic.BUILDERS, failing), self.assertRaises(RuntimeError):
            synthetic.build("tools")
        self.assertTrue((folder / "calls.jsonl").exists())
        self.assertFalse((folder.parent / ".tools.tmp").exists())


class AddCases(Workspace):
    """Adding your own cases."""

    def test_ids_increase(self):
        """next_id skips ids that are taken and ignores other names."""
        self.assertEqual(cases.next_id([]), "u001")
        self.assertEqual(cases.next_id(["u001", "u007", "m03", "clip"]), "u008")

    def test_tool_call_case_is_loaded_with_the_synthetic_ones(self):
        """An added case is written in the eval's format, and run() counts both sources."""
        added = cases.add_tool_call(args(prompt="what is 2 plus 2", tool="calculator", args={"expression": "2"}))
        self.assertEqual(added, "u001 added to tool_calls")
        self.assertEqual(cases.add_tool_call(args(prompt="hi", tool="none", args={})), "u002 added to tool_calls")
        mine = common.module("tool_calls").load_cases(config.EVAL_USER / "tool_calls")
        self.assertEqual([m["id"] for m in mine], ["u001", "u002"])
        self.assertIsNone(mine[1]["tool"])

    def test_unknown_tool_is_refused(self):
        """A tool name that does not exist stops the command."""
        with self.assertRaises(SystemExit):
            cases.add_tool_call(args(prompt="x", tool="no_such_tool", args={}))

    def test_query_needs_a_known_memory(self):
        """A query can only point at memories that were added before."""
        with self.assertRaises(SystemExit):
            cases.add_query(args(query="where", relevant=["u001"]))
        cases.add_memory(args(text="Parking is on level three", junk=False))
        cases.add_query(args(query="where did I park", relevant=["u001"]))
        items = common.module("memory").load_cases(config.EVAL_USER / "memory")
        self.assertEqual([i["op"] for i in items], ["store", "query"])

    def test_run_uses_both_sources(self):
        """The tools eval runs the synthetic and the user cases, and reports each source."""
        cases.add_tool_policy(args(tool="get_time", args={}, expect="ok"))
        result = common.run("tools")
        self.assertEqual(set(result["sources"]), {"synthetic", "user"})
        self.assertEqual(result["sources"]["user"], 1)
        self.assertEqual({r["source"] for r in result["records"]}, {"synthetic", "user"})
        self.assertEqual(set(result["by_source"]), {"synthetic", "user"})
        self.assertEqual(common.case_counts("tools")["user"], 1)

    def test_audio_case_is_converted_to_wav_with_its_transcript(self):
        """An audio file becomes 16 kHz mono WAV with the transcript beside it."""
        import numpy as np
        import soundfile as sf

        src = self.root / "in.flac"
        sf.write(src, np.zeros(8000, np.float32), 8000)
        cases.add_asr(args(audio=src, text="hello there"))
        [item] = common.module("asr").load_cases(config.EVAL_USER / "asr")
        self.assertEqual(item["reference"], "hello there")
        self.assertEqual(sf.info(item["wav"]).samplerate, config.SAMPLE_RATE)

    def test_task2_groups_split_on_bar(self):
        """Each --mention becomes one group, and | separates the phrases in it."""
        from PIL import Image

        shot = self.root / "shot.jpg"
        Image.new("RGB", (20, 20)).save(shot)
        cases.add_task2(args(media=shot, problem=True, mention=["torch", "pip install | conda install"]))
        [item] = common.module("task2").load_cases(config.EVAL_USER / "task2")
        self.assertEqual(item["must_mention"], [["torch"], ["pip install", "conda install"]])
        self.assertTrue(item["has_problem"])


if __name__ == "__main__":
    unittest.main()
