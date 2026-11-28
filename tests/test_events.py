"""Contract tests for the event log. The dashboard pages through it by `id`, so ids must only grow."""

import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from agent import config, events

EPISODIC = Path(__file__).parent.parent / "agent" / "memory" / "episodic.py"


class EventIdTest(unittest.TestCase):
    """Check that a payload cannot break paging through the event log."""

    def test_payload_cannot_replace_the_event_id(self):
        """A payload key named `id` does not replace the id of the event."""
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(config, "RUNTIME", Path(tmp)):
            first = events.emit("test.first")
            second = events.emit("test.second", id=1)
            self.assertEqual(second["id"], first["id"] + 1)
            self.assertEqual([e["kind"] for e in events.since(first["id"])], ["test.second"])

    def test_memory_stored_does_not_pass_an_id_field(self):
        """`memory.stored` names its row id `memory_id`, so the value stays in the event."""
        calls = re.findall(r'events\.emit\(\s*"memory\.stored",[^)]*\)', EPISODIC.read_text())  # the call can wrap
        self.assertEqual(len(calls), 1)
        self.assertIn("memory_id=", calls[0])
        self.assertNotRegex(calls[0], r"\bid=")


if __name__ == "__main__":
    unittest.main()


class HotkeyTrustTest(unittest.TestCase):
    """Check that a trusted start can clear the hotkey warning of an earlier start in the replayed log."""

    def listening(self, warning: str | None) -> dict:
        """Start the hotkey with a fake OS plug-in and return the `hotkey.listening` payload.

        Args:
            warning (str | None): What the fake `hotkey_warning()` returns.

        Returns:
            dict: The keyword arguments of the `hotkey.listening` event.
        """
        from agent import hotkey

        plat = mock.MagicMock()
        plat.hotkey_warning.return_value = warning
        with (
            mock.patch.object(hotkey.platforms, "current", return_value=plat),
            mock.patch.object(hotkey.events, "emit") as emit,
        ):
            hotkey.listen(lambda: None)
        return dict(next(c.kwargs for c in emit.call_args_list if c.args == ("hotkey.listening",)))

    def test_a_trusted_start_says_so(self):
        """No warning from the plug-in: the event carries `trusted=True`."""
        self.assertTrue(self.listening(None)["trusted"])

    def test_an_untrusted_start_says_so(self):
        """A warning from the plug-in: the event carries `trusted=False`."""
        self.assertFalse(self.listening("not trusted")["trusted"])
