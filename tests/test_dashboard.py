"""Tests for the dashboard chat (dashboard.chat and the approval endpoint), with no model server started.

The whole module is skipped when the third-party packages of the agent are not installed, because the
contract tests must run on a bare Python.
"""

import base64
import io
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

try:
    from PIL import Image

    from agent import brain, config, serving, tools

    with (
        mock.patch.object(serving.accel, "llama_server", return_value=Path("llama-server")),
        mock.patch.object(serving.accel, "current", return_value=mock.Mock(server_args=list)),
    ):
        from dashboard import chat, server
except ImportError as e:  # bare interpreter without requirements/base.txt
    raise unittest.SkipTest(f"agent dependencies are not installed: {e}") from e


class TempChat(unittest.TestCase):
    """Keep the conversation and its pictures in a temporary folder."""

    def setUp(self) -> None:
        """Start with an empty chat saved under a temporary folder."""
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        for patcher in (
            mock.patch.object(config, "RUNTIME", self.root),  # the event log too: tests never write to yours
            mock.patch.object(config, "CHAT_FILE", self.root / "chat.json"),
            mock.patch.object(chat.screen, "CAPTURES", self.root / "captures"),
            mock.patch.dict(chat.state, {"history": None, "mic": None}),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)


class ApprovalInHistory(TempChat):
    """An approval outcome reaches the model through the chat history."""

    def setUp(self) -> None:
        """Start with an empty chat and a pending call of a mocked tool."""
        super().setUp()
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
            chat.chat({"text": text})
        return mocked

    def test_next_request_carries_the_outcome_as_plain_messages(self) -> None:
        """The model gets the note with `role` and `content` only, and the page also gets the details."""
        out = server.approve({"call_id": self.call_id, "yes": True})
        self.assertEqual(out["status"], "ok")
        sent = self.chat().call_args.kwargs["history"]
        self.assertEqual(sent, [brain.approval_note(out)])
        self.assertEqual(chat.chat_history()[0]["approval"]["tool"], "write_file")

    def test_denial_is_recorded_too(self) -> None:
        """A denied call adds a note that says nothing ran."""
        server.approve({"call_id": self.call_id, "yes": False})
        self.assertIn("did not run", chat.chat_history()[0]["content"])

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
        roles = [(m["role"], "approval" in m) for m in chat.chat_history()]
        self.assertEqual(roles, [("user", False), ("assistant", False), ("assistant", True)])


class Conversation(TempChat):
    """The conversation survives a restart, carries pictures and shows each brief once."""

    def reply(self, text: str = "ok") -> mock._patch:
        """Mock the model with a fixed reply that recalled one memory."""
        memory = {"id": 7, "when": "Tue 09:02", "kind": "transcript", "level": "raw", "text": "x", "score": 0.5}
        return mock.patch.object(brain, "respond", return_value={"text": text, "memories": [memory]})

    def test_a_restart_reads_the_same_conversation_back(self) -> None:
        """The file holds every message; a new process (history None) loads it, with the recalled memories."""
        with self.reply():
            chat.chat({"text": "hello"})
        chat.state["history"] = None  # what a restarted dashboard sees
        saved = chat.chat_history()
        self.assertEqual([m["role"] for m in saved], ["user", "assistant"])
        self.assertEqual(saved[1]["ui"]["memories"][0]["id"], 7)
        self.assertEqual(json.loads(config.CHAT_FILE.read_text()), saved)

    def test_a_picture_goes_to_the_model_and_clear_deletes_it(self) -> None:
        """A pasted picture is saved, sent as an image part, and removed with the conversation."""
        buf = io.BytesIO()
        Image.new("RGB", (40, 30), "red").save(buf, "PNG")
        data_url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        with self.reply() as respond:
            out = chat.chat({"text": "", "image": data_url})
        parts = respond.call_args.args[0]
        self.assertEqual([p["type"] for p in parts], ["image_url", "text"])
        self.assertEqual(parts[1]["text"], "What is in this picture?")
        self.assertTrue(Path(out["images"][0]).exists())
        chat.chat_clear()
        self.assertFalse(Path(out["images"][0]).exists())
        self.assertEqual(chat.chat_history(), [])

    def test_a_file_that_is_not_a_picture_is_refused(self) -> None:
        """Bad bytes give a 422 with a reason, not a crash in the image library."""
        with self.assertRaises(chat.HTTPException) as caught:
            chat.chat({"text": "look", "image": "data:image/png;base64," + base64.b64encode(b"nope").decode()})
        self.assertEqual(caught.exception.status_code, 422)

    def test_voice_keeps_only_speech_and_falls_back_to_gemma(self) -> None:
        """Silence is refused before any speech model runs; without Whisper, Gemma writes the text."""
        mic = mock.Mock(stop=mock.Mock(return_value=[0.0] * 16000))
        clip = {"samples": [0.1] * 8000}
        with (
            mock.patch.object(chat.audio, "paused", return_value=False),
            mock.patch.object(chat.audio, "Mic", return_value=mic),
            mock.patch.object(chat.audio, "VAD"),
            mock.patch.object(chat.audio, "chunk", side_effect=[[], [clip]]),
            mock.patch.object(chat.stt, "transcribe", side_effect=[chat.Unsupported("no whisper"), "hello there"]),
        ):
            chat.voice_start()
            with self.assertRaises(chat.HTTPException) as caught:
                chat.voice_stop()
            self.assertEqual(caught.exception.status_code, 422)
            chat.voice_start()
            heard = chat.voice_stop()
        self.assertEqual((heard["text"], heard["engine"]), ("hello there", "gemma"))

    def test_the_same_brief_is_added_once(self) -> None:
        """Two open pages post the same brief event; the chat shows it once."""
        brief = {"brief_id": 3, "markdown": "**Quiet day**", "version": 2}
        chat.chat_brief(brief)
        chat.chat_brief(brief)
        self.assertEqual(len(chat.chat_history()), 1)
        self.assertEqual(chat.chat_history()[0]["ui"]["kind"], "brief")


if __name__ == "__main__":
    unittest.main()
