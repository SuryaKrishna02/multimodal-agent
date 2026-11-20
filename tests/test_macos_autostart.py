"""Tests for the LaunchAgent that the macOS plug-in writes. No launchctl, no swiftc: both are mocked.

The plug-in imports on every OS, so these tests run everywhere.
"""

import os
import plistlib
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from agent.platforms import macos


class InstallAutostart(unittest.TestCase):
    """`install_autostart` runs the daemon through the launcher app, with the installer's PATH."""

    def test_plist_runs_the_launcher_with_the_path_of_the_installer(self) -> None:
        """The plist carries PATH: launchd's own has no Homebrew, so llama-server could not find ffprobe for video."""
        with tempfile.TemporaryDirectory() as tmp:
            agents, launcher = Path(tmp) / "LaunchAgents", Path(tmp) / "app" / "launcher"
            with (
                mock.patch.object(macos, "_agents_dir", lambda: agents),
                mock.patch.object(macos, "build_launcher", lambda _label: launcher),
                mock.patch.object(macos, "_bootout"),
                mock.patch.object(macos.subprocess, "run") as run,
                mock.patch.dict(os.environ, {"PATH": "/opt/homebrew/bin:/usr/bin"}),
            ):
                plist = macos.MacOS().install_autostart("x.test", ["python", "cli.py"], Path(tmp), Path(tmp) / "log")
            spec = plistlib.loads(plist.read_bytes())
        self.assertEqual(spec["ProgramArguments"], [str(launcher), "python", "cli.py"])
        self.assertEqual(spec["EnvironmentVariables"]["PATH"], "/opt/homebrew/bin:/usr/bin")
        self.assertEqual(spec["AssociatedBundleIdentifiers"], "x.test")
        self.assertEqual(spec["KeepAlive"], {"SuccessfulExit": False})
        self.assertEqual(run.call_args.args[0][:2], ["launchctl", "bootstrap"])

    def test_the_app_asks_for_the_microphone(self) -> None:
        """The Info.plist of the launcher app has the usage text that macOS shows in the permission dialog."""
        info = macos._app_info("x.test")
        self.assertTrue(info["NSMicrophoneUsageDescription"])
        self.assertEqual(info["CFBundleIdentifier"], "x.test")
        self.assertTrue(macos.LAUNCHER_SOURCE.exists())


if __name__ == "__main__":
    unittest.main()
