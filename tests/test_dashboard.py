"""Tests for the chat history in dashboard.server, with no model server started.

The whole module is skipped when the third-party packages of the agent are not installed, because the
contract tests must run on a bare Python.
"""

import threading
import unittest
from pathlib import Path
from unittest import mock

try:
    from agent import brain, serving, tools

    with (
        mock.patch.object(serving.accel, "llama_server", return_value=Path("llama-server")),
        mock.patch.object(serving.accel, "current", return_value=mock.Mock(server_args=list)),
    ):
        from dashboard import server
except ImportError as e:  # bare interpreter without requirements/base.txt
    raise unittest.SkipTest(f"agent dependencies are not installed: {e}") from e


class ApprovalInHistory(unittest.TestCase):
    """An approval outcome reaches the model through the chat history."""

    def setUp(self) -> None:
        """Start with an empty chat and a pending call of a mocked tool."""
        patcher = mock.patch.dict(server.state, {"history": []})
        patcher.start()
        self.addCleanup(patcher.stop)
        tool = mock.patch.object(tools.REGISTRY["write_file"], "fn", return_value="wrote it")
        tool.start()
        self.addCleanup(tool.stop)
        self.call_id = tools.run("write_file", {"path": "a.txt", "content": "hi"})["call_id"]

    def chat(self, text: str = "did that work?", respond=None) -> mock.MagicMock:
        """Send one chat message with the model mocked.

        Args:
            text (str, optional): The message. Defaults to "did that work?".
            respond (Callable, optional): Replaces the model reply. Defaults to a fixed reply.

        Returns:
            mock.MagicMock: The mock of `brain.respond`.
        """
        with mock.patch.object(brain, "respond", side_effect=respond, return_value={"text": "ok"}) as mocked:
            server.chat({"text": text})
        return mocked

    def test_next_request_carries_the_outcome_as_plain_messages(self) -> None:
        """The model gets the note with `role` and `content` only, and the page also gets the details."""
        out = server.approve({"call_id": self.call_id, "yes": True})
        self.assertEqual(out["status"], "ok")
        sent = self.chat().call_args.kwargs["history"]
        self.assertEqual(sent, [brain.approval_note(out)])
        self.assertEqual(server.chat_history()[0]["approval"]["tool"], "write_file")

    def test_denial_is_recorded_too(self) -> None:
        """A denied call adds a note that says nothing ran."""
        server.approve({"call_id": self.call_id, "yes": False})
        self.assertIn("did not run", server.chat_history()[0]["content"])

    def test_approval_during_a_reply_lands_after_that_turn(self) -> None:
        """The reply that says "waiting for approval" must come before the outcome, or the model reads it as newer."""
        approving = threading.Thread(target=server.approve, args=({"call_id": self.call_id, "yes": True},))

        def respond(*_args, **_kwargs) -> dict:
            approving.start()
            approving.join(0.2)  # still blocked on the lock that this reply holds
            self.assertTrue(approving.is_alive())
            return {"text": "It is waiting for your approval."}

        self.chat("write a.txt", respond=respond)
        approving.join()
        roles = [(m["role"], "approval" in m) for m in server.chat_history()]
        self.assertEqual(roles, [("user", False), ("assistant", False), ("assistant", True)])


if __name__ == "__main__":
    unittest.main()
