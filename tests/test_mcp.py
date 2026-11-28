"""Tests for the MCP client's config handling, the reminders server's times and proactive alerts.

No server process starts here: these check the parts that decide what is allowed (labels, secrets, tiers, the
registry) and the alert cool-down. The module is skipped when the third-party packages are not installed.
"""

import asyncio
import concurrent.futures
import importlib.util
import json
import tempfile
import time
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

try:
    from agent import config, events, mcp_client, proactive, tools

    _spec = importlib.util.spec_from_file_location("reminders_server", config.ROOT / "mcp_servers/reminders/server.py")
    reminders = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
    _spec.loader.exec_module(reminders)  # type: ignore[union-attr]  # the server module needs the mcp package
except ImportError as e:  # bare interpreter without requirements/base.txt
    raise unittest.SkipTest(f"agent dependencies are not installed: {e}") from e

GITHUB = {"command": "npx", "args": ["-y", "server-github"], "env": {"GITHUB_PERSONAL_ACCESS_TOKEN": "${GH_TOKEN}"}}


def pasted(**servers) -> str:
    """Build a pasted config."""
    return json.dumps({"mcpServers": servers})


class Parse(unittest.TestCase):
    """A pasted config is labeled and checked before anything connects."""

    def test_labels(self):
        """A command is on-device, a URL is remote, unless the URL is this machine."""
        servers = mcp_client.parse(
            pasted(gh=GITHUB, far={"url": "https://example.com/mcp"}, near={"url": "http://127.0.0.1:9000/mcp"})
        )
        self.assertEqual(
            {n: s.label for n, s in servers.items()},
            {"gh": mcp_client.ON_DEVICE, "far": mcp_client.REMOTE, "near": mcp_client.LOCAL_HTTP},
        )

    def test_bare_object_of_servers_works(self):
        """Some apps paste the servers without the outer mcpServers key."""
        self.assertIn("gh", mcp_client.parse(json.dumps({"gh": GITHUB})))

    def test_literal_secret_is_refused(self):
        """A key typed into env or headers is refused, with the fix in the message."""
        bad = dict(GITHUB, env={"GITHUB_PERSONAL_ACCESS_TOKEN": "ghp_123"})
        with self.assertRaises(mcp_client.ConfigError) as caught:
            mcp_client.parse(pasted(gh=bad, far={"url": "https://x.org", "headers": {"Authorization": "Bearer k"}}))
        message = str(caught.exception)
        self.assertIn("gh: env.GITHUB_PERSONAL_ACCESS_TOKEN holds a literal value", message)
        self.assertIn("far: headers.Authorization", message)  # every problem is listed, not only the first

    def test_bad_entries(self):
        """Neither or both of command and url, a bad name or a non-HTTP url are refused."""
        for servers in (
            {"a": {}},
            {"a": {"command": "x", "url": "http://x"}},
            {"a b": GITHUB},
            {"a": {"url": "file:///x"}},
        ):
            with self.subTest(servers=servers), self.assertRaises(mcp_client.ConfigError):
                mcp_client.parse(json.dumps({"mcpServers": servers}))

    def test_not_json(self):
        """A paste that is not JSON says so."""
        with self.assertRaises(mcp_client.ConfigError):
            mcp_client.parse("{mcpServers:")


class Secrets(unittest.TestCase):
    """`${NAME}` is read from the environment when the client connects."""

    def test_resolve(self):
        """A set variable is used, a missing one is named."""
        with mock.patch.dict("os.environ", {"GH_TOKEN": "secret"}, clear=True):
            self.assertEqual(mcp_client.resolve({"T": "${GH_TOKEN}"}), {"T": "secret"})
            with self.assertRaises(mcp_client.ConfigError) as caught:
                mcp_client.resolve({"K": "${MISSING_KEY}"})
        self.assertIn("MISSING_KEY", str(caught.exception))


class Saved(unittest.TestCase):
    """The saved state keeps references, enabled tools and tiers; the registry follows it."""

    def setUp(self) -> None:
        """Keep the state and the event log in a temporary folder."""
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        for patcher in (
            mock.patch.object(config, "RUNTIME", root),
            mock.patch.object(config, "MCP_CONFIG", root / "mcp.json"),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.addCleanup(mcp_client._unregister, "gh")

    def test_default_is_the_reminders_server(self):
        """With nothing saved, the config holds this repo's reminders server, with no tool enabled."""
        state = mcp_client.load()
        self.assertEqual(list(state["mcpServers"]), [config.REMINDERS_SERVER])
        self.assertEqual(state["enabled"], {})

    def test_secret_never_reaches_the_file(self):
        """The saved file holds the reference, not the value."""
        with mock.patch.dict("os.environ", {"GH_TOKEN": "s3cr3t"}):
            mcp_client.set_config(pasted(gh=GITHUB))
        text = config.MCP_CONFIG.read_text()
        self.assertIn("${GH_TOKEN}", text)
        self.assertNotIn("s3cr3t", text)

    def test_enabled_tools_register_with_their_tier(self):
        """Only enabled tools join the registry, as <server>__<tool>, at the default tier unless the user set one."""
        mcp_client.set_config(pasted(gh=GITHUB))
        state = mcp_client.load()
        schema = {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}
        state["tools"]["gh"] = [
            {"name": n, "description": n, "input_schema": schema, "annotations": {}} for n in ("search", "delete_repo")
        ]
        mcp_client.save(state)
        with mock.patch.object(mcp_client.hub, "queues", {"gh": object()}):
            mcp_client.enable("gh", ["search"])
            self.assertEqual(tools.REGISTRY["gh__search"].tier, config.MCP_DEFAULT_TIER)
            self.assertNotIn("gh__delete_repo", tools.REGISTRY)
            mcp_client.set_tier("gh", "search", "safe")
            self.assertEqual(tools.REGISTRY["gh__search"].tier, "safe")
            self.assertEqual(tools.check("gh__search", {})[0], "invalid")  # the server's schema is enforced
        with self.assertRaises(ValueError):
            mcp_client.set_tier("gh", "search", "always")

    def test_tool_of_a_disconnected_server_is_not_registered(self):
        """A server without a session offers nothing to the model."""
        mcp_client.set_config(pasted(gh=GITHUB))
        mcp_client.enable("gh", ["search"])
        self.assertNotIn("gh__search", tools.REGISTRY)


class FakeClient:
    """Stands in for `mcp.Client`: opens at once and offers no tools."""

    def __init__(self, target) -> None:
        """Ignore the transport."""

    async def __aenter__(self):
        """Open the fake session."""
        return self

    async def __aexit__(self, *exc) -> None:
        """Close the fake session."""

    async def list_tools(self):
        """Offer no tools."""
        return SimpleNamespace(tools=[])


class Reconnect(unittest.TestCase):
    """A session that closes after a reconnect must not take the new session's queue with it."""

    def test_old_session_leaves_the_new_queue(self):
        """The old session ends after the new one registered: the new queue stays."""
        hub = mcp_client.Hub()
        server = mcp_client.Server("gh", mcp_client.ON_DEVICE, command="x")

        async def scenario() -> asyncio.Queue:
            ready: concurrent.futures.Future = concurrent.futures.Future()
            task = asyncio.create_task(hub._session(server, ready))
            await asyncio.wrap_future(ready)  # the old session listed its tools and waits for calls
            old, newer = hub.queues["gh"], asyncio.Queue()
            hub.queues["gh"] = newer  # a reconnect opened its session first
            old.put_nowait(None)  # ...then the old session is told to close
            await task
            return newer

        with (
            mock.patch("mcp.Client", FakeClient),
            mock.patch.object(mcp_client, "_transport", lambda _s: nullcontext("transport")),
        ):
            newer = asyncio.run(scenario())
        self.assertIs(hub.queues.get("gh"), newer)


class ReminderTimes(unittest.TestCase):
    """The reminders server turns in_minutes or HH:MM into a time."""

    def test_in_minutes(self):
        """Minutes count from now."""
        self.assertEqual(reminders.due_time(2, None, now=1000.0), 1120.0)

    def test_at_rolls_to_tomorrow(self):
        """A time that has passed today means tomorrow."""
        now = time.mktime((2026, 11, 28, 15, 0, 0, 0, 0, -1))
        self.assertEqual(reminders.due_time(None, "16:30", now) - now, 90 * 60)
        self.assertEqual(reminders.due_time(None, "14:00", now) - now, 23 * 3600)

    def test_bad_times(self):
        """Both, neither, a negative delay or a bad clock time go back to the model as a ToolError."""
        for args in ((None, None), (5, "10:00"), (-1, None), (None, "7pm")):
            with self.subTest(args=args), self.assertRaises(reminders.ToolError):
                reminders.due_time(*args)


class ScreenCheck(unittest.TestCase):
    """Task 5: the model reports the failed tests it sees, and the code decides whether to alert."""

    def check(self, reply: dict) -> dict:
        """Run check_screen on a blank image with the model's reply mocked."""
        from PIL import Image

        from agent import tasks

        with (
            mock.patch.object(tasks.brain, "ask_json", return_value=(reply, {})),
            mock.patch.object(tasks.screen, "save", return_value=Path("shot.png")),
            mock.patch.object(tasks.events, "emit"),
        ):
            return tasks.check_screen(Image.new("RGB", (8, 8)))

    def test_no_failed_test_no_alert(self):
        """A script's traceback with no failed test does not alert, even if the model wrote something to say."""
        out = self.check({"program": "script", "failed_tests": [], "say": "The script failed."})
        self.assertEqual((out["alert"], out["say"]), (False, ""))

    def test_failed_test_alerts_with_its_name(self):
        """A named failed test alerts, and `what` holds the name."""
        out = self.check({"program": "pytest", "failed_tests": ["test_add", " "], "say": "test_add failed."})
        self.assertEqual((out["alert"], out["what"]), (True, "test_add"))


class Alerts(unittest.TestCase):
    """An alert goes to the feed, the notification and the voice, once per cool-down."""

    def setUp(self) -> None:
        """Keep the event log in a temporary folder and start with no sent alerts."""
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        for patcher in (
            mock.patch.object(config, "RUNTIME", Path(tmp.name)),
            mock.patch.dict(proactive._last, clear=True),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_once_per_cool_down(self):
        """The same key twice sends once; the voice failing does not stop the feed or the notification."""
        plat = mock.Mock()
        with (
            mock.patch.object(proactive.platforms, "current", return_value=plat),
            mock.patch("agent.voice.speaker.speak", side_effect=FileNotFoundError("no model")),
        ):
            first = proactive.alert("k", "Title", "Message", "because")
            second = proactive.alert("k", "Title", "Message", "because")
        self.assertIsNotNone(first)
        self.assertIsNone(second)
        plat.notify.assert_called_once_with("Title", "Message")
        kinds = [e["kind"] for e in events.since(0, "proactive.")]
        self.assertEqual(kinds, ["proactive.alert", "proactive.warning", "proactive.skipped"])

    def test_due_reminder_is_announced_then_completed(self):
        """Each due reminder becomes an alert, and is marked done after it."""
        due = [{"id": 7, "text": "call Mike", "due": "2026-11-28 15:02:00"}]
        calls = []

        def call(server, tool, args):
            calls.append(tool)
            return due if tool == "due_reminders" else {}

        with (
            mock.patch.object(mcp_client.hub, "queues", {config.REMINDERS_SERVER: object()}),
            mock.patch.object(mcp_client.hub, "call", side_effect=call),
            mock.patch.object(proactive, "alert") as alert,
        ):
            self.assertEqual(proactive.check_reminders(), 1)
        self.assertEqual(calls, ["due_reminders", "complete_reminder"])
        self.assertEqual(alert.call_args.kwargs["spoken"], "Reminder: call Mike.")


if __name__ == "__main__":
    unittest.main()
