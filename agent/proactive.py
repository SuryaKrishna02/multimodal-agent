"""Proactive output: the agent speaks up on its own, and says why (Session 3).

An alert has a key, a title, a message and a reason. `alert()` drops a repeat of the same key within
`config.ALERT_COOLDOWN_S`, then sends it three ways:

    the dashboard feed    event proactive.alert, with the reason (Always on tab)
    a notification        the OS plug-in's notify hook
    speech                agent/voice/speaker.py: the dashboard tab if one listens, else this machine

Two sources use it: due reminders from the reminders MCP server (Task 6, `watch_reminders`, run by the daemon), and
the screen watcher of Task 5. Each output that fails is reported and the others still run, so a machine without a
notification hook still speaks.
"""

import json
import threading
import time

from agent import config, events, platforms

_last: dict[str, float] = {}  # alert key -> when it was last sent, in this process
_lock = threading.Lock()


def _fresh(key: str, now: float) -> bool:
    """Return True and note the time if `key` was not sent within the cool-down."""
    with _lock:
        if now - _last.get(key, 0.0) < config.ALERT_COOLDOWN_S:
            return False
        _last[key] = now
        return True


def alert(
    key: str, title: str, message: str, reason: str, spoken: str | None = None, speak: bool = True
) -> dict | None:
    """Tell the user something without being asked: feed, notification and speech.

    Args:
        key (str): Same key within the cool-down = the same alert, sent once. For example "reminder:3".
        title (str): The notification title.
        message (str): One line for the notification and the feed.
        reason (str): Why the agent speaks up, shown in the feed.
        spoken (str | None, optional): What to say out loud. Defaults to `message`.
        speak (bool, optional): Also say it. Defaults to True.

    Returns:
        dict | None: The proactive.alert event, or None when the cool-down dropped it.
    """
    if not _fresh(key, time.time()):
        events.emit("proactive.skipped", key=key, reason="same alert within the cool-down")
        return None
    event = events.emit("proactive.alert", key=key, title=title, message=message, reason=reason)
    try:
        platforms.current().notify(title, message)
    except Exception as e:  # noqa: BLE001 - no notification hook: the feed and the voice still have it
        events.emit("proactive.warning", key=key, message=f"notification failed: {e}")
    if speak:
        from agent.voice import speaker

        try:
            speaker.speak(spoken or message, reason=reason)
        except Exception as e:  # noqa: BLE001 - no voice (model missing): the feed and the notification have it
            events.emit("proactive.warning", key=key, message=f"speech failed: {e}")
    return event


def _as_list(result) -> list:
    """An MCP result as a list: structured content arrives as a list, plain text as JSON."""
    return json.loads(result) if isinstance(result, str) else list(result)


def check_reminders() -> int:
    """Ask the reminders server which reminders are due, announce each one, then mark it done.

    The daemon (code, not the model) calls the server here, so no tool tier applies: tiers govern what the model may
    call. A reminder is marked done only after its alert went out.

    Returns:
        int: How many reminders were announced.
    """
    from agent import mcp_client

    server = config.REMINDERS_SERVER
    if server not in mcp_client.hub.connected():
        mcp_client.hub.connect(server)
    due = _as_list(mcp_client.hub.call(server, "due_reminders", {}))
    for r in due:
        alert(
            key=f"reminder:{r['id']}",
            title="Reminder",
            message=r["text"],
            reason=f"reminder #{r['id']} was due at {r['due'][11:16]}",
            spoken=f"Reminder: {r['text']}.",
        )
        mcp_client.hub.call(server, "complete_reminder", {"reminder_id": r["id"]})
    return len(due)


def watch_reminders(stop: threading.Event) -> None:
    """Check for due reminders every `config.REMINDER_POLL_S` until `stop` is set. The daemon runs this in a thread.

    Args:
        stop (threading.Event): Set it to end the loop.
    """
    from agent import mcp_client

    if config.REMINDERS_SERVER not in mcp_client.load()["mcpServers"]:
        return
    failing = False
    while not stop.wait(config.REMINDER_POLL_S):
        try:
            check_reminders()
            failing = False
        except Exception as e:  # noqa: BLE001 - report once, keep trying: the server may come back
            if not failing:
                events.emit("proactive.warning", key="reminders", message=f"reminders server: {e}")
            failing = True


class ScreenWatcher:
    """Task 5: look at the screen every `config.WATCH_EVERY_S` and speak up when the watched condition appears.

    A screen that did not change since the last check is skipped (the Session 1 sampler), so a static screen costs
    no model request. The dashboard runs it: on macOS the dashboard's terminal holds the Screen Recording permission,
    and the launchd daemon does not.
    """

    def __init__(self, condition: str = config.WATCH_CONDITION):
        """Prepare the watcher without starting it.

        Args:
            condition (str, optional): What to watch for. Defaults to `config.WATCH_CONDITION`.
        """
        from agent.senses import screen

        self.condition, self.sampler = condition, screen.Sampler()
        self.stop_flag = threading.Event()

    def start(self) -> None:
        """Start watching in a background thread."""
        threading.Thread(target=self._run, name="screen-watch", daemon=True).start()
        events.emit("task5.watch", state="on", condition=self.condition)

    def stop(self) -> None:
        """Stop watching."""
        self.stop_flag.set()
        events.emit("task5.watch", state="off")

    def check(self, img) -> dict | None:
        """Check one screenshot: skip it if unchanged, else ask the model and alert on a match.

        Args:
            img (PIL.Image.Image): The screenshot.

        Returns:
            dict | None: The model's answer, or None for a screen that did not change.
        """
        from agent import tasks

        if not self.sampler.offer(img):
            return None
        out = tasks.check_screen(img, self.condition)
        if out["alert"]:
            what = out["what"] or "the watched condition"
            alert(f"screen:{what.lower()}", "On your screen", what, f"the screen shows {self.condition}", out["say"])
        return out

    def _run(self) -> None:
        from agent.senses import screen

        while not self.stop_flag.wait(config.WATCH_EVERY_S):
            try:
                self.check(screen.capture())
            except Exception as e:  # noqa: BLE001 - one failed look must not stop the watcher
                events.emit("proactive.warning", key="screen", message=f"screen check failed: {e}")
