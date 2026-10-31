"""Tests for the tool loop in agent.brain, with the model server mocked out.

The whole module is skipped when the third-party packages of the agent are not installed, because the
contract tests must run on a bare Python.
"""

import json
import unittest
from unittest import mock

try:
    from agent import brain, tools
except ImportError as e:  # bare interpreter without requirements/base.txt
    raise unittest.SkipTest(f"agent dependencies are not installed: {e}") from e


def reply(content: str = "", calls: list[dict] | None = None) -> dict:
    """Build the part of a chat completion that `respond` reads.

    Args:
        content (str, optional): Text of the assistant message. Defaults to "".
        calls (list[dict], optional): Tool calls in the message. Defaults to None.

    Returns:
        dict: A chat completion with one choice.
    """
    msg: dict = {"content": content}
    if calls:
        msg["tool_calls"] = calls
    return {"choices": [{"message": msg}]}


def call(name: str, arguments: str) -> dict:
    """Build one tool call as llama-server returns it.

    Args:
        name (str): Tool name.
        arguments (str): Arguments as a JSON string.

    Returns:
        dict: A tool call.
    """
    return {"id": f"id-{name}", "function": {"name": name, "arguments": arguments}}


class RespondLoop(unittest.TestCase):
    """The loop in brain.respond: tool calls, retries and the step limit."""

    def run_respond(self, replies: list[dict], **kwargs) -> tuple[dict, mock.MagicMock]:
        """Run `respond` with a scripted model and no memory.

        Args:
            replies (list[dict]): The completions that the mocked `send` returns in order.
            **kwargs: Extra arguments for `respond`.

        Returns:
            tuple[dict, mock.MagicMock]: The result and the mocked `send`.
        """
        with (
            mock.patch.object(brain, "send", side_effect=replies) as send,
            mock.patch.object(tools, "select", return_value=["get_time"]),
        ):
            out = brain.respond([brain.part("text", "what time is it")], use_memory=False, **kwargs)
        return out, send

    def test_plain_answer_has_no_tool_calls(self):
        """A reply without tool calls is returned as text."""
        out, send = self.run_respond([reply("Hello")])
        self.assertEqual(out["text"], "Hello")
        self.assertEqual(out["tool_calls"], [])
        self.assertEqual(out["tools_offered"], ["get_time"])
        self.assertEqual(send.call_count, 1)

    def test_tool_call_then_answer(self):
        """A valid tool call runs, its result goes back to the model, and thinking is on for step 2."""
        out, send = self.run_respond([reply(calls=[call("get_time", "{}")]), reply("It is noon")])
        self.assertEqual(out["text"], "It is noon")
        self.assertEqual([t["tool"] for t in out["tool_calls"]], ["get_time"])
        self.assertEqual(out["tool_calls"][0]["status"], "ok")
        second = send.call_args_list[1]
        self.assertEqual(second.kwargs["chat_template_kwargs"], {"enable_thinking": True})
        self.assertNotIn("chat_template_kwargs", send.call_args_list[0].kwargs)
        roles = [m["role"] for m in second.args[0]]
        self.assertEqual(roles[-2:], ["assistant", "tool"])

    def test_invalid_json_arguments_are_reported_as_invalid(self):
        """Arguments that are not JSON become an invalid result, and the model may retry."""
        out, _ = self.run_respond([reply(calls=[call("get_time", "{oops")]), reply("sorry")])
        self.assertEqual(out["tool_calls"][0]["status"], "invalid")
        self.assertEqual(out["tool_calls"][0]["args"], {})
        self.assertEqual(out["text"], "sorry")

    def test_gives_up_when_retries_are_used_up(self):
        """After TOOL_RETRIES + 1 invalid calls the loop stops with a fixed message."""
        bad = reply(calls=[call("get_time", "{oops")])
        out, send = self.run_respond([bad] * 10, max_steps=10)
        self.assertEqual(out["text"], "I could not form a valid tool call.")
        self.assertEqual(len(out["tool_calls"]), brain.config.TOOL_RETRIES + 1)
        self.assertEqual(send.call_count, brain.config.TOOL_RETRIES + 1)

    def test_stops_after_max_steps(self):
        """A model that keeps calling tools is stopped at max_steps."""
        again = reply(calls=[call("get_time", "{}")])
        out, send = self.run_respond([again] * 5, max_steps=2)
        self.assertEqual(out["text"], "Stopped after too many tool steps.")
        self.assertEqual(send.call_count, 2)

    def test_later_calls_in_a_batch_do_not_run_after_giving_up(self):
        """When the retries run out inside one batch, the remaining calls are not executed."""
        batch = reply(calls=[call("get_time", "{oops")] * (brain.config.TOOL_RETRIES + 1) + [call("get_time", "{}")])
        out, _ = self.run_respond([batch])
        self.assertEqual(out["text"], "I could not form a valid tool call.")
        self.assertEqual(len(out["tool_calls"]), brain.config.TOOL_RETRIES + 1)
        self.assertTrue(all(t["status"] == "invalid" for t in out["tool_calls"]))


class PendingReply(unittest.TestCase):
    """A call that waits for approval ends the turn with fixed text, so the reply cannot claim it is done."""

    WRITE = call("write_file", json.dumps({"path": "a.txt", "content": "hi"}))

    def run_batch(self, *calls: dict) -> tuple[dict, mock.MagicMock]:
        """Run `respond` on one model turn that asks for some calls, and drop the pending calls after.

        Args:
            *calls (dict): The tool calls.

        Returns:
            tuple[dict, mock.MagicMock]: The result and the mocked `send`.
        """
        self.addCleanup(tools.PENDING.clear)
        with (
            mock.patch.object(brain, "send", side_effect=[reply("I have created the file", list(calls))]) as send,
            mock.patch.object(tools, "select", return_value=["write_file"]),
        ):
            return brain.respond([brain.part("text", "write a.txt")], use_memory=False), send

    def test_reply_comes_from_the_code_and_the_model_is_not_asked_again(self):
        """The text of the model is dropped, and no second request goes out."""
        out, send = self.run_batch(self.WRITE)
        self.assertEqual(out["text"], "Waiting for your approval to run `write_file`. Not done yet.")
        self.assertEqual(send.call_count, 1)
        self.assertEqual(out["tool_calls"][0]["status"], "pending")

    def test_safe_call_in_the_same_batch_still_runs(self):
        """Only the waiting call is named. The trace keeps the result of the safe call."""
        out, _ = self.run_batch(call("get_time", "{}"), self.WRITE)
        self.assertEqual([t["status"] for t in out["tool_calls"]], ["ok", "pending"])
        self.assertEqual(out["text"], "Waiting for your approval to run `write_file`. Not done yet.")

    def test_next_request_names_the_calls_that_still_wait(self):
        """Asked "did that work?" before a decision, the model must know the call has not run."""
        self.run_batch(self.WRITE)
        with (
            mock.patch.object(brain, "send", return_value=reply("Not yet")) as send,
            mock.patch.object(tools, "select", return_value=["write_file"]),
        ):
            brain.respond([brain.part("text", "did that work?")], use_memory=False)
        system = send.call_args.args[0][0]["content"]
        self.assertIn("not run yet: write_file(", system)
        self.assertIn("a.txt", system)


class ApprovalNote(unittest.TestCase):
    """brain.approval_note tells the model how a call that waited for approval ended."""

    def outcome(self, yes: bool, **patches) -> dict:
        """Resolve a real pending call of a mocked tool.

        Args:
            yes (bool): True to approve the call.
            **patches: Fields of the registered tool to replace, such as `fn`.

        Returns:
            dict: The result of `tools.approve`.
        """
        with mock.patch.multiple(tools.REGISTRY["write_file"], **patches):
            pending = tools.run("write_file", {"path": "a.txt", "content": "hi"})
            return tools.approve(pending["call_id"], yes)

    def test_denied_call_did_not_run(self):
        """A denied call does not run, and the note says so."""
        fn = mock.Mock()
        out = self.outcome(False, fn=fn)
        fn.assert_not_called()
        note = brain.approval_note(out)
        self.assertEqual(note["role"], "assistant")
        self.assertIn("denied write_file", note["content"])
        self.assertIn("did not run", note["content"])

    def test_approved_call_reports_its_result(self):
        """An approved call runs once, and the note holds its result and arguments."""
        fn = mock.Mock(return_value="wrote 2 characters")
        note = brain.approval_note(self.outcome(True, fn=fn))
        fn.assert_called_once_with(path="a.txt", content="hi")
        self.assertIn("approved write_file", note["content"])
        self.assertIn("a.txt", note["content"])
        self.assertIn("wrote 2 characters", note["content"])

    def test_failed_call_reports_the_error(self):
        """An approved call that fails must not read as done."""
        note = brain.approval_note(self.outcome(True, fn=mock.Mock(side_effect=OSError("disk full"))))
        self.assertIn("did not finish (error)", note["content"])
        self.assertIn("disk full", note["content"])


class ElideMedia(unittest.TestCase):
    """brain._elide_media shortens base64 blobs for the debug view."""

    def test_blobs_are_shortened_and_the_original_is_untouched(self):
        """Image, audio and video data are cut to 48 characters plus a length note."""
        blob = "A" * 1000
        body = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": blob}},
                        {"type": "input_audio", "input_audio": {"data": blob, "format": "wav"}},
                        {"type": "input_video", "input_video": {"data": blob}},
                        {"type": "text", "text": "hi"},
                    ],
                },
                {"role": "assistant", "content": "plain"},
            ]
        }
        out = brain._elide_media(body)
        parts = out["messages"][0]["content"]
        self.assertEqual(parts[0]["image_url"]["url"], "A" * 48 + "...<1,000 chars>")
        self.assertEqual(parts[1]["input_audio"]["data"], "A" * 48 + "...<1,000 chars>")
        self.assertEqual(parts[2]["input_video"]["data"], "A" * 48 + "...<1,000 chars>")
        self.assertEqual(parts[3]["text"], "hi")
        self.assertEqual(body["messages"][0]["content"][0]["image_url"]["url"], blob)
        self.assertEqual(json.dumps(out["messages"][1]), json.dumps(body["messages"][1]))


if __name__ == "__main__":
    unittest.main()
