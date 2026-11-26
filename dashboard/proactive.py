"""The Task 5 endpoints: watch the screen and speak up when a test fails.

    POST /api/task5/watch    start or stop the screen watcher ({"on": true})
    POST /api/task5/check    look at the screen once, now, and alert if the condition is there
    GET  /api/task5          is the watcher on, and what it watches for

The dashboard runs the watcher, because its terminal holds the Screen Recording permission (the launchd daemon does
not). An alert goes to the feed, a notification and the voice (agent/proactive.py).
"""

from fastapi import APIRouter
from fastapi.concurrency import run_in_threadpool

from agent import config, proactive

router = APIRouter(prefix="/api/task5")
watcher: dict = {"on": None}  # the running proactive.ScreenWatcher, if any


@router.get("")
def state() -> dict:
    """Describe the screen watcher.

    Returns:
        dict: `watching`, `condition` and `every_s`.
    """
    return {"watching": watcher["on"] is not None, "condition": config.WATCH_CONDITION, "every_s": config.WATCH_EVERY_S}


@router.post("/watch")
def watch(body: dict) -> dict:
    """Start or stop watching the screen.

    Args:
        body (dict): `on` (bool).

    Returns:
        dict: The new state.
    """
    if body.get("on") and watcher["on"] is None:
        watcher["on"] = proactive.ScreenWatcher()
        watcher["on"].start()
    elif not body.get("on") and watcher["on"] is not None:
        watcher["on"].stop()
        watcher["on"] = None
    return state()


@router.post("/check")
async def check() -> dict | None:
    """Look at the screen once now (a fresh watcher, so an unchanged screen is still checked).

    Returns:
        dict | None: The model's answer (`alert`, `what`, `say`, `screenshot`); None never happens here.
    """
    from agent.senses import screen

    return await run_in_threadpool(lambda: proactive.ScreenWatcher().check(screen.capture()))
