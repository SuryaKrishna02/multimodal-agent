"""Tests for skill authoring: the effects scan, the checks of a draft, and writing a skill only when it is sound.

No model runs: draft() gets a canned answer. The module is skipped when the agent's packages are not installed.
"""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

try:
    from agent import brain, config, skill_author, skills, tools
except ImportError as e:  # bare interpreter without requirements/base.txt
    raise unittest.SkipTest(f"agent dependencies are not installed: {e}") from e

CODE = '''def run(text: str, upper: bool = False) -> str:
    """Count words."""
    n = len(text.split())
    return f"{n} WORDS" if upper else f"{n} words"
'''
DRAFT = {
    "name": "word_count_test",
    "description": "Count the words in a piece of text.",
    "examples": ["how many words is this", "count the words"],
    "params": [{"name": "text", "type": "string", "description": "the text"}],
    "effects": [],
    "tier": "safe",
    "code": CODE,
}


class EffectsScan(unittest.TestCase):
    """The scan finds what the code does, whatever the model declared."""

    def test_each_effect(self):
        """Network, writes, running code and reading files are found; plain code shows none."""
        cases = {
            "import urllib.request\ndef run(): pass": {"network"},
            "from http import client\ndef run(): pass": {"network"},
            "def run(p):\n    open(p, 'w').write('x')": {"writes"},
            "def run(p, m):\n    open(p, m)": {"writes"},
            "def run(p):\n    return open(p).read()": {"reads_files"},
            "import subprocess\ndef run(): subprocess.run(['ls'])": {"runs_code"},
            "def run(s):\n    return eval(s)": {"runs_code"},
            "from agent.memory import episodic\ndef run(): pass": {"reads_memory"},
            CODE: set(),
        }
        for code, expected in cases.items():
            with self.subTest(code=code.splitlines()[0]):
                self.assertEqual(skill_author.effects_in(code), expected)


class Build(unittest.TestCase):
    """A draft becomes a manifest whose input schema comes from run()'s signature."""

    def test_schema_from_signature(self):
        """Names, types and required come from the code; descriptions from the draft."""
        manifest, problems = skill_author.build(DRAFT)
        self.assertEqual(problems, [])
        self.assertEqual(manifest["input_schema"]["required"], ["text"])
        self.assertEqual(manifest["input_schema"]["properties"]["upper"], {"type": "boolean"})
        self.assertEqual(manifest["input_schema"]["properties"]["text"]["description"], "the text")

    def test_problems(self):
        """No run(), a syntax error, an undeclared risky effect at tier safe, and a taken name are refused."""
        cases = {
            "no run": dict(DRAFT, code="def main():\n    return 1\n"),
            "syntax": dict(DRAFT, code="def run(:\n"),
            "network but safe": dict(DRAFT, code="import socket\n" + CODE),
            "built-in name": dict(DRAFT, name="get_time"),
        }
        for label, draft in cases.items():
            with self.subTest(label):
                self.assertTrue(skill_author.build(draft)[1])

    def test_fences_are_removed(self):
        """A Markdown fence around the code does not end up in skill.py."""
        self.assertEqual(skill_author.strip_fences("```python\nx = 1\n```"), "x = 1\n")


class Create(unittest.TestCase):
    """create() writes a folder only for a sound draft, and loads it."""

    def setUp(self) -> None:
        """Write skills into a temporary folder, and keep events out of the real runtime."""
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        (root / "skills").mkdir()
        for patcher in (
            mock.patch.object(config, "RUNTIME", root),
            mock.patch.object(skills, "SKILLS_DIR", root / "skills"),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.addCleanup(lambda: tools.REGISTRY.pop(DRAFT["name"], None))
        self.skills_dir = root / "skills"

    def test_create_writes_and_loads(self):
        """The new skill is a tool at once, and a second skill of the same name is refused."""
        skill_author.create(DRAFT)
        self.assertTrue((self.skills_dir / DRAFT["name"] / "manifest.json").is_file())
        self.assertEqual(tools.run(DRAFT["name"], {"text": "one two three"})["result"], "3 words")
        with self.assertRaises(skill_author.SkillProblem):
            skill_author.create(DRAFT)

    def test_code_that_fails_to_load_leaves_nothing(self):
        """An import error at load time removes the folder and reports the reason."""
        with self.assertRaises(skill_author.SkillProblem) as caught:
            skill_author.create(dict(DRAFT, code="import no_such_module_x\n" + CODE))
        self.assertIn("no_such_module_x", str(caught.exception))
        self.assertFalse((self.skills_dir / DRAFT["name"]).exists())

    def test_draft_starts_at_confirm(self):
        """The model's draft never sets the tier, and the scan adds the effects it finds."""
        answer = dict(DRAFT, code="import urllib.request\n" + CODE)
        del answer["tier"]
        with mock.patch.object(brain, "ask_json", return_value=(answer, {})):
            out = skill_author.draft("count words in text from a web page")
        self.assertEqual((out["tier"], out["effects"], out["problems"]), ("confirm", ["network"], []))


if __name__ == "__main__":
    unittest.main()
