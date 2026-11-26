"""Tests for the native skill interface: manifests, the effects rule, registration, and the two example skills.

The module is skipped when the third-party packages of the agent are not installed.
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

try:
    from agent import config, skills, tools
except ImportError as e:  # bare interpreter without requirements/base.txt
    raise unittest.SkipTest(f"agent dependencies are not installed: {e}") from e

GOOD = {
    "name": "echo_back",
    "description": "Say the given text back to the user.",
    "version": "1.0",
    "tier": "safe",
    "effects": [],
    "input_schema": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
    "examples": ["repeat after me", "say this back"],
    "entry": "skill.py:run",
}


class Manifests(unittest.TestCase):
    """A skill loads only when its manifest is complete and its tier fits its effects."""

    def setUp(self) -> None:
        """Make a temporary skills folder and keep the event log out of the real runtime."""
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        patcher = mock.patch.object(config, "RUNTIME", self.root)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(lambda: tools.REGISTRY.pop("echo_back", None))

    def skill(self, manifest: dict, name: str = "echo_back") -> Path:
        """Write one skill folder."""
        folder = self.root / "skills" / name
        folder.mkdir(parents=True)
        (folder / "manifest.json").write_text(json.dumps(manifest))
        (folder / "skill.py").write_text("def run(text):\n    return text\n")
        return folder

    def test_good_skill_registers_as_a_tool(self):
        """A good skill joins the registry and runs through tools.run with its tier."""
        self.assertIsNotNone(skills.load_one(self.skill(GOOD)))
        self.assertEqual(tools.run("echo_back", {"text": "hi"}), {"status": "ok", "result": "hi"})
        self.assertEqual(tools.check("echo_back", {})[0], "invalid")  # the manifest's schema is enforced

    def test_problems_are_refused(self):
        """Missing fields, a risky effect at tier safe, a wrong folder name or a built-in's name are refused."""
        cases = {
            "missing tier": ({k: v for k, v in GOOD.items() if k != "tier"}, "echo_back"),
            "writes but safe": (dict(GOOD, effects=["writes"]), "echo_back"),
            "folder name": (GOOD, "other_name"),
            "built-in name": (dict(GOOD, name="get_time"), "get_time"),
            "entry outside": (dict(GOOD, entry="../x.py:run"), "echo_back"),
        }
        for label, (manifest, folder) in cases.items():
            with self.subTest(label):
                path = self.skill(manifest, folder)
                self.assertTrue(skills.check(manifest, path) or skills.load_one(path) is None)
                (path / "manifest.json").unlink()
                (path / "skill.py").unlink()
                path.rmdir()

    def test_broken_code_is_refused_not_raised(self):
        """A skill.py that fails to import is reported, and the other skills still load."""
        for label, code in {"syntax": "def run(:\n", "import": "import no_such_module_x\n", "name": "x = y\n"}.items():
            with self.subTest(label):
                folder = self.skill(GOOD)
                (folder / "skill.py").write_text(code)
                self.assertIsNone(skills.load_one(folder))
                self.assertEqual(skills.load(folder.parent), [])
                shutil.rmtree(folder)  # import errors leave a __pycache__ behind

    def test_risky_skill_may_ask_for_confirm(self):
        """The same effect is fine at tier confirm."""
        manifest = dict(GOOD, effects=["writes"], tier="confirm")
        self.assertEqual(skills.check(manifest, self.skill(manifest)), [])


class ExampleSkills(unittest.TestCase):
    """The two skills that ship with the repo."""

    def setUp(self) -> None:
        """Keep the tool-call events out of the real runtime folder."""
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        patcher = mock.patch.object(config, "RUNTIME", Path(tmp.name))
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_both_load(self):
        """Both manifests pass and both skills are in the registry."""
        names = [m["name"] for m in skills.load()]
        self.assertEqual(names, ["standup", "unit_convert"])

    def test_unit_convert(self):
        """Length, temperature and a refused pair."""
        self.assertEqual(
            tools.run("unit_convert", {"value": 5, "from_unit": "miles", "to_unit": "km"})["result"], "5 mi = 8.047 km"
        )
        self.assertEqual(
            tools.run("unit_convert", {"value": 100, "from_unit": "C", "to_unit": "F"})["result"], "100 c = 212 f"
        )
        self.assertEqual(tools.run("unit_convert", {"value": 1, "from_unit": "kg", "to_unit": "km"})["status"], "error")


if __name__ == "__main__":
    unittest.main()
