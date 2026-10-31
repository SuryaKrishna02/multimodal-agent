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
        calls = re.findall(r'events\.emit\("memory\.stored",[^\n]*', EPISODIC.read_text())
        self.assertEqual(len(calls), 1)
        self.assertIn("memory_id=", calls[0])
        self.assertNotRegex(calls[0], r"\bid=")


if __name__ == "__main__":
    unittest.main()
