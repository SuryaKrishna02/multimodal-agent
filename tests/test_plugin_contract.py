"""Contract tests for OS and accelerator plug-ins. Standard library only: runs on any OS.

    python -m unittest discover tests -v

A plug-in PR must keep these green. They check shape (names, imports, pins), not hardware:
`python cli.py doctor --exercise` is how you show it works on your machine.
"""

import re
import unittest

from agent import accel, platforms
from agent.accel.base import Accelerator
from agent.platforms.base import HOOKS, Platform, Unsupported


class PlatformContract(unittest.TestCase):
    """Contract tests that every platform plug-in must pass."""

    def test_every_plugin_imports_on_this_os(self):
        """Every platform file imports on this OS."""
        for name in platforms.available():
            with self.subTest(name):
                platforms.load(name)  # fails if a top-level import is OS-specific

    def test_name_matches_file_and_hooks_exist(self):
        """The name equals the file name and every hook is callable."""
        for name in platforms.available():
            with self.subTest(name):
                p = platforms.load(name)
                self.assertEqual(p.name, name)
                self.assertIsInstance(p, Platform)
                for hook in HOOKS:
                    self.assertTrue(callable(getattr(p, hook)), hook)

    def test_unimplemented_hooks_raise_unsupported_not_crash(self):
        """The base class raises Unsupported for hooks that have no portable default."""
        with tempfile_path() as tmp:
            with self.assertRaises(Unsupported):
                Platform().record_screen(tmp, 1)
            with self.assertRaises(Unsupported):
                Platform().synthesize_speech("hi", tmp)

    def test_current_always_returns_something(self):
        """`current()` falls back to the generic platform."""
        self.assertIsInstance(platforms.current(), Platform)

    def test_override_by_name(self):
        """Every plug-in can be loaded by name."""
        for name in platforms.available():
            with self.subTest(name):
                self.assertEqual(platforms.load(name).name, name)

    def test_unknown_name_is_a_clear_error(self):
        """An unknown name raises an error that lists the available plug-ins."""
        with self.assertRaisesRegex(ValueError, "Available"):
            platforms.load("no-such-os")


class AcceleratorContract(unittest.TestCase):
    """Contract tests that every accelerator plug-in must pass."""

    def test_every_plugin_imports_on_this_os(self):
        """Every accelerator file imports on this OS."""
        for name in accel.available():
            with self.subTest(name):
                accel.load(name)

    def test_name_matches_file(self):
        """The name equals the file name and `server_args()` returns a list."""
        for name in accel.available():
            with self.subTest(name):
                a = accel.load(name)
                self.assertEqual(a.name, name)
                self.assertIsInstance(a, Accelerator)
                self.assertIsInstance(a.server_args(), list)

    def test_downloads_are_keyed_by_a_real_platform_and_pinned(self):
        """Each download has a real platform key, an https URL, a sha256 and a matching archive type."""
        for name in accel.available():
            a = accel.load(name)
            for plat, dl in a.downloads.items():
                with self.subTest(f"{name}/{plat}"):
                    self.assertIn(plat, platforms.available(), "key must be a platform plug-in name")
                    self.assertRegex(dl.sha256, r"^[0-9a-f]{64}$")
                    self.assertTrue(dl.url.startswith("https://"))
                    self.assertIn(dl.archive, ("tar.gz", "zip"))
                    self.assertTrue(dl.url.endswith("." + dl.archive))

    def test_missing_download_is_a_clear_error(self):
        """A missing download raises Unsupported."""
        with self.assertRaisesRegex(Unsupported, "no llama.cpp download"):
            Accelerator().download_for("macos")


class NoOsLeaksInCore(unittest.TestCase):
    """Core files must go through the plug-ins, not run OS commands themselves."""

    BANNED = [
        r"\bscreencapture\b",
        r"\bafconvert\b",
        r"\bHIServices\b",
        r"sys\.platform",
        r"\bbrew install\b",
        r"/System/Library",
        r"\.venv/bin",
    ]
    ALLOWED = (
        "agent/platforms/",
        "agent/accel/",
        "tests/",
        "docs/",
        "README.md",
        "CONTRIBUTING.md",
        "requirements/",
    )

    def test_core_python_has_no_os_specific_commands(self):
        """No core Python file contains an OS-specific command or path."""
        import pathlib

        root = pathlib.Path(__file__).resolve().parent.parent
        for path in root.rglob("*.py"):
            rel = path.relative_to(root).as_posix()
            if rel.startswith((".venv", ".cache")) or rel.startswith(self.ALLOWED):
                continue
            text = path.read_text()
            for pattern in self.BANNED:
                with self.subTest(f"{rel}: {pattern}"):
                    self.assertIsNone(
                        re.search(pattern, text), f"{rel} is OS-specific: move it into agent/platforms/ or agent/accel/"
                    )


class tempfile_path:  # noqa: N801 - tiny context manager
    """Context manager that gives a path inside a temporary folder and deletes the folder on exit."""

    def __enter__(self):
        """Create the temporary folder and return a path inside it."""
        import pathlib
        import tempfile

        self._d = tempfile.TemporaryDirectory()
        return pathlib.Path(self._d.name) / "x"

    def __exit__(self, *exc):
        """Delete the temporary folder."""
        self._d.cleanup()


if __name__ == "__main__":
    unittest.main()
