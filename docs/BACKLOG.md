# Backlog

This file lists planned work that nobody has started. Each item names the files it changes and the condition for done.

## Session 2

### Timed listening

Status: open. Do this after the Session 2 live session.

Today, always-on listening starts with `python cli.py listen on` or the dashboard switch. It stops only when the user turns it off. Add a way to stop it automatically:

- `python cli.py listen on --for 45m`: listening stops after the given duration. The dashboard gets the same option.
- A daily window in `agent/config.py`, for example 09:00 to 17:00. The daemon listens only inside the window.

Notes:

- The daemon reads the switch once each `DAEMON_TICK_S` in `Daemon._sync_listener()` (`agent/daemon.py`). `daemon.wants_listening()` can return `False` after a stored end time or outside the window.
- The mic pause switch (`audio.paused()`) must still override every other setting.
- `daemon.status()` and the dashboard show the end time or the window.

Done when:

- Both options work from `cli.py` and from the dashboard.
- A test in `tests/` checks that `wants_listening()` returns `False` after the end time and outside the window.
- `docs/SETUP.md` and `docs/architecture.md` describe the new options.

### Partial summaries of a conversation

Status: open. Do this after the Session 2 live session.

The scheduled `summarize` job summarizes only complete windows (`SUMMARY_WINDOW_S`). Two paths call `consolidate(force=True)`, which also summarizes the open window:

- The daemon trigger in `agent/daemon.py` runs when `TRANSCRIPTS_PER_SUMMARY` new transcripts wait.
- `python cli.py consolidate`.

During a long conversation, the trigger can fire before the window closes. The summary then covers only the first part of the window. The rows that arrive later form a new window and get a second summary. One stretch of conversation gets two summaries, and neither summary has the full context.

Options:

- The trigger summarizes only complete windows and does not pass `force`.
- A window closes at a silence gap, not only at `SUMMARY_WINDOW_S`.
- A later run updates the summary of a partial window when new rows arrive in the same window.

Done when:

- A conversation that continues while the trigger fires gets one summary for each window.
- A test in `tests/` feeds rows across a trigger and checks the number of summaries.
- `docs/architecture.md` describes when a window closes.
