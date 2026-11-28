"""Global hotkey: press the combo in config.HOTKEY anywhere -> Task 2 (look at my screen and help).

How the key listener starts and which OS permissions it needs is the OS plug-in's job
(agent/platforms/<os>.py): macOS, for one, only delivers global key events to apps you trust.
"""

import threading

from agent import config, events, platforms


def listen(on_press):
    """Start listening in a background thread. on_press() runs in its own thread.

    The `hotkey.listening` event says whether the OS trusts this process, so the dashboard can clear the warning
    of an earlier start, which it replays from the event log.

    Args:
        on_press (Callable[[], None]): Runs each time the combo is pressed.

    Returns:
        object: The listener of the OS plug-in.
    """

    def fire():
        events.emit("hotkey.pressed", combo=config.HOTKEY)
        threading.Thread(target=on_press, daemon=True).start()

    plat = platforms.current()
    listener = plat.start_hotkey(config.HOTKEY, fire)
    warning = plat.hotkey_warning()
    if warning:
        events.emit("hotkey.warning", message=warning)
    events.emit("hotkey.listening", combo=config.HOTKEY, trusted=warning is None)
    return listener
