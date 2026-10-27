"""Global hotkey: press Ctrl+Option+Space anywhere -> Task 2 (look at my screen and help).

macOS only delivers global key events to apps you trust. Grant your terminal app
Accessibility and Input Monitoring (System Settings > Privacy & Security), then restart it.
"""
import threading

import HIServices          # installed with pynput (pyobjc)
from pynput import keyboard

from agent import config, events


def listen(on_press) -> keyboard.GlobalHotKeys:
    """Start listening in a background thread. on_press() runs in its own thread."""
    def fire():
        events.emit("hotkey.pressed", combo=config.HOTKEY)
        threading.Thread(target=on_press, daemon=True).start()

    listener = keyboard.GlobalHotKeys({config.HOTKEY: fire})
    listener.start()
    if not HIServices.AXIsProcessTrusted():
        events.emit("hotkey.warning", message="Hotkey not trusted by macOS: grant Accessibility and "
                                              "Input Monitoring to your terminal app, then restart it.")
    events.emit("hotkey.listening", combo=config.HOTKEY)
    return listener
