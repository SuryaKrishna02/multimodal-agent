"""Tests for the silent-microphone warning of the always-on listener. No microphone: a fake stream feeds audio.

An OS that blocks the microphone can deliver zeros instead of an error (macOS does this for a process without the
Microphone permission). The listener must say so in the event log instead of failing silently.
The module is skipped when the third-party packages of the agent are not installed.
"""

import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

try:
    import numpy as np

    from agent import config, daemon, events
    from agent.senses import ambient
except ImportError as e:  # bare interpreter without requirements/base.txt
    raise unittest.SkipTest(f"agent dependencies are not installed: {e}") from e

SR = config.SAMPLE_RATE


class SilenceCheckTest(unittest.TestCase):
    """`SilenceCheck` decides once: silent for the whole window, or sound seen."""

    def test_zeros_for_the_whole_window_warn_once(self) -> None:
        """Five 1-second blocks of zeros complete a 5-second window; later blocks do not warn again."""
        check = ambient.SilenceCheck(seconds=5.0, rms=1e-5)
        results = [check.feed(np.zeros(SR, np.float32)) for _ in range(7)]
        self.assertEqual(results, [False] * 4 + [True] + [False] * 2)

    def test_a_quiet_room_is_not_silent(self) -> None:
        """Room noise at the measured level (RMS ~2e-3) passes the check, so a quiet room never warns."""
        check = ambient.SilenceCheck(seconds=5.0, rms=1e-5)
        noise = np.random.default_rng(0).normal(0, 2e-3, SR).astype(np.float32)
        self.assertFalse(check.feed(noise))
        self.assertTrue(check.done)
        self.assertFalse(any(check.feed(np.zeros(SR, np.float32)) for _ in range(10)))

    def test_empty_blocks_do_not_count(self) -> None:
        """A step with no audio yet adds no silent seconds."""
        check = ambient.SilenceCheck(seconds=1.0, rms=1e-5)
        self.assertFalse(check.feed(np.zeros(0, np.float32)))
        self.assertEqual(check.silent_s, 0.0)


class FakeStream:
    """Stand-in for `sounddevice.InputStream`: a thread calls the callback with 0.25 s blocks of `level` noise."""

    level = 0.0

    def __init__(self, callback, **_kw) -> None:
        """Keep the callback.

        Args:
            callback: The listener's audio callback.
            **_kw: The stream settings, not used.
        """
        self.callback, self.closed = callback, threading.Event()

    def start(self) -> None:
        """Deliver audio until `close()`."""
        threading.Thread(target=self._feed, daemon=True).start()

    def _feed(self) -> None:
        rng = np.random.default_rng(1)
        while not self.closed.wait(0.01):
            block = rng.normal(0, self.level, (SR // 4, 1)).astype(np.float32) if self.level else np.zeros((SR // 4, 1))
            self.callback(block.astype(np.float32), SR // 4, None, None)

    def close(self) -> None:
        """Stop delivering audio."""
        self.closed.set()


class ListenerWarningTest(unittest.TestCase):
    """The live listener emits `senses.ambient.warning` for a silent stream, and the heartbeat carries it."""

    def setUp(self) -> None:
        """Send events to a temporary folder, speed up the loop and fake the microphone."""
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        patches = [
            mock.patch.object(config, "RUNTIME", Path(tmp.name)),
            mock.patch.object(config, "AMBIENT_STEP_S", 0.05),
            mock.patch.object(config, "AMBIENT_SILENT_S", 1.0),
            mock.patch.object(ambient.sd, "InputStream", FakeStream),
            mock.patch.object(ambient.sd, "query_devices", lambda **_kw: {"name": "Fake Microphone"}),
            mock.patch.object(ambient.audio, "paused", lambda: False),
            mock.patch.object(ambient.audio, "VAD", lambda: mock.Mock(segments=lambda _buf: [])),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def listen(self, level: float, seconds: float) -> ambient.Listener:
        """Run a listener on a fake stream of `level` noise for `seconds`, then stop it.

        Args:
            level (float): Standard deviation of the fake audio; 0.0 = exact zeros.
            seconds (float): How long to listen.

        Returns:
            ambient.Listener: The stopped listener.
        """
        FakeStream.level = level
        listener = ambient.Listener(on_utterance=lambda _u: None)
        listener.start()
        threading.Event().wait(seconds)
        listener.stop()
        return listener

    def test_zeros_emit_a_warning_that_names_the_permission(self) -> None:
        """Only zeros for longer than AMBIENT_SILENT_S: one warning event, with the platform's permission hint."""
        listener = self.listen(0.0, 2.0)
        warnings = events.since(0, "senses.ambient.warning")
        self.assertEqual(len(warnings), 1)
        self.assertIn("Microphone", warnings[0]["message"])
        self.assertEqual(warnings[0]["device"], "Fake Microphone")
        self.assertEqual(listener.warning, warnings[0]["message"])

    def test_sound_emits_no_warning(self) -> None:
        """A stream with room noise never warns."""
        listener = self.listen(2e-3, 2.0)
        self.assertEqual(events.since(0, "senses.ambient.warning"), [])
        self.assertIsNone(listener.warning)

    def test_the_heartbeat_reports_the_warning(self) -> None:
        """`daemon status` reads `mic_warning` from the heartbeat that the daemon writes."""
        listener = ambient.Listener(on_utterance=lambda _u: None)
        listener.warning = "silent"
        d = daemon.Daemon(models=False)
        d.listener = listener
        with mock.patch.object(daemon.checkpoint, "put") as put:
            threading.Timer(0.05, d.stop_flag.set).start()  # one beat, then return
            d.beat()
        self.assertEqual(put.call_args_list[0].args[1]["mic_warning"], "silent")


if __name__ == "__main__":
    unittest.main()
