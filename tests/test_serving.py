"""Tests for the server commands in agent.serving, with no server started.

The whole module is skipped when the third-party packages of the agent are not installed, so the
contract tests must run on a bare Python.
"""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

try:
    from agent import config, serving
except ImportError as e:  # bare interpreter without requirements/base.txt
    raise unittest.SkipTest(f"agent dependencies are not installed: {e}") from e


def embed_server() -> serving.LlamaServer:
    """Build the embedding server on any machine, without starting it.

    Returns:
        serving.LlamaServer: The server.
    """
    with (
        mock.patch.object(serving.accel, "llama_server", return_value=Path("llama-server")),
        mock.patch.object(serving.accel, "current", return_value=mock.Mock(server_args=list)),
    ):
        return serving.embed_server()


def embed_args() -> list[str]:
    """Build the command line of the embedding server on any machine.

    Returns:
        list[str]: The arguments of `serving.embed_server()`.
    """
    return embed_server().args


def flag(args: list[str], name: str) -> str:
    """Return the value that follows a flag.

    Args:
        args (list[str]): A command line.
        name (str): The flag.

    Returns:
        str: The value after the flag.
    """
    return args[args.index(name) + 1]


class EmbedServerTest(unittest.TestCase):
    """An embedding must fit in one batch, or llama-server answers HTTP 500."""

    def test_batch_is_as_large_as_the_context(self) -> None:
        """An input that fits the context also fits one batch. The default batch stops at 512 tokens."""
        args = embed_args()
        size = str(config.EMBED_CTX)
        self.assertEqual(flag(args, "-c"), size)
        self.assertEqual(flag(args, "-b"), size)
        self.assertEqual(flag(args, "-ub"), size)


class WatchTest(unittest.TestCase):
    """The watcher restarts a server that stopped by accident, and only that server."""

    def setUp(self) -> None:
        """Keep markers, logs and events in a temporary folder, and build a server with no process."""
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        runtime = mock.patch.object(config, "RUNTIME", Path(folder.name))
        runtime.start()
        self.addCleanup(runtime.stop)
        self.server = embed_server()
        for name, value in (("_processes", []), ("running_model", None), ("healthy", False)):
            patcher = mock.patch.object(self.server, name, return_value=value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def check(self, **start) -> tuple[float, mock.MagicMock]:
        """Run one check of the watcher with `start` mocked.

        Args:
            **start: Arguments for the mock of `start`.

        Returns:
            tuple[float, mock.MagicMock]: The seconds to wait and the mock.
        """
        with mock.patch.object(self.server, "start", **start) as started:
            return self.server.restart_if_stopped(), started

    def test_stopped_by_accident_restarts_and_slows_down(self) -> None:
        """No process and no stop marker: start the server, then wait the longer time."""
        wait, started = self.check()
        started.assert_called_once()
        self.assertEqual(wait, config.SERVER_RETRY_S)
        self.assertEqual((serving.events.last("serving.restart") or {}).get("server"), "embed")

    def test_running_process_is_left_alone(self) -> None:
        """A process that is there may still be loading, so an unhealthy one is not restarted."""
        with mock.patch.object(self.server, "_processes", return_value=[mock.Mock()]):
            wait, started = self.check()
        started.assert_not_called()
        self.assertEqual(wait, config.SERVER_WATCH_S)

    def test_stopped_on_purpose_stays_stopped(self) -> None:
        """`stop` leaves a marker that the watcher obeys, also when the stop came from another process."""
        self.server.stop()
        wait, started = self.check()
        started.assert_not_called()
        self.assertEqual(wait, config.SERVER_WATCH_S)

    def test_start_clears_the_stop_marker(self) -> None:
        """After a later `start`, a stop by accident restarts the server again."""
        self.server.stop()
        with (
            mock.patch.object(self.server, "running_model", return_value=config.EMBED["model"]),
            mock.patch.object(self.server, "healthy", return_value=True),
        ):
            self.server.start()
        _, started = self.check()
        started.assert_called_once()

    def test_swap_keeps_the_marker_until_the_new_process_exists(self) -> None:
        """While one model replaces another, no moment has both no process and no marker."""
        seen = []
        process = mock.Mock()
        process.poll.return_value = None

        def popen(*_args, **_kwargs) -> mock.Mock:
            seen.append((config.RUNTIME / "llama-embed.stopped").exists())
            return process

        with (
            mock.patch.object(self.server, "running_model", return_value="other.gguf"),
            mock.patch.object(self.server, "healthy", return_value=True),
            mock.patch.object(serving.subprocess, "Popen", side_effect=popen),
            mock.patch.object(serving, "open", create=True),
        ):
            self.server.start()
        self.assertEqual(seen, [True])
        self.assertFalse((config.RUNTIME / "llama-embed.stopped").exists())

    def test_watch_follows_the_current_server(self) -> None:
        """A switch of tier replaces the server, so each check asks which server is current."""

        class Done(Exception):
            """Ends the loop that never returns."""

        first, second = mock.Mock(), mock.Mock()
        first.restart_if_stopped.return_value = second.restart_if_stopped.return_value = 1
        with mock.patch.object(serving.time, "sleep", side_effect=[None, Done]), self.assertRaises(Done):
            servers = iter([first, second])
            serving.watch(lambda: next(servers))
        first.restart_if_stopped.assert_called_once()
        second.restart_if_stopped.assert_called_once()

    def test_failed_restart_is_reported_and_the_watcher_survives(self) -> None:
        """A server that cannot start shows in the event log, and the next check waits the longer time."""
        wait, _ = self.check(side_effect=RuntimeError("boom"))
        self.assertEqual(wait, config.SERVER_RETRY_S)
        self.assertIn("boom", (serving.events.last("serving.error") or {}).get("message", ""))

    def test_log_of_the_stopped_server_is_kept(self) -> None:
        """`start` overwrites the log, so the restart keeps the old one for finding out why the server stopped."""
        (config.RUNTIME / "llama-embed.log").write_text("the reason")
        self.check()
        self.assertEqual((config.RUNTIME / "llama-embed.previous.log").read_text(), "the reason")


if __name__ == "__main__":
    unittest.main()
