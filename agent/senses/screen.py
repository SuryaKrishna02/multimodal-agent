"""Seeing: screen capture and sampling.

Gemma 4 in llama.cpp takes images, not video. So "seeing the screen" means picking
which frames to send. The sampler looks every SCREEN_SAMPLE_EVERY_S seconds and keeps
a frame only when enough of the screen changed. Fewer frames = fewer tokens.
"""
import json
import time
from pathlib import Path

import mss
import numpy as np
from PIL import Image

from agent import config, events

CAPTURES = config.RUNTIME / "captures"


def capture(monitor: int = 1) -> Image.Image:
    """Grab one monitor, downscaled so the longest side is SCREEN_MAX_SIDE."""
    with mss.mss() as m:
        shot = m.grab(m.monitors[monitor])
    img = Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")
    img.thumbnail((config.SCREEN_MAX_SIDE, config.SCREEN_MAX_SIDE))
    if np.asarray(img).std() < 3:   # a flat image usually means macOS blocked the capture
        events.emit("senses.screen.warning",
                    message="Screenshot is blank: grant Screen Recording to your terminal app "
                            "(System Settings > Privacy & Security > Screen Recording).")
    return img


def save(img: Image.Image, name: str | None = None) -> Path:
    CAPTURES.mkdir(parents=True, exist_ok=True)
    path = CAPTURES / (name or f"screen-{time.strftime('%Y%m%d-%H%M%S')}.png")
    img.save(path)
    events.emit("senses.screen", path=str(path), size=img.size)
    return path


def changed_fraction(a: Image.Image, b: Image.Image) -> float:
    """Share of pixels that changed noticeably, compared on small grayscale thumbnails."""
    small = (160, 100)
    x = np.asarray(a.convert("L").resize(small), np.float32) / 255
    y = np.asarray(b.convert("L").resize(small), np.float32) / 255
    return float((np.abs(x - y) > 0.1).mean())


class Sampler:
    """Keep a frame when the screen changed by more than the threshold since the last kept frame."""

    def __init__(self, threshold: float = config.SCREEN_CHANGE_THRESHOLD):
        self.threshold = threshold
        self.last_kept: Image.Image | None = None

    def offer(self, frame: Image.Image) -> bool:
        if self.last_kept is None or changed_fraction(self.last_kept, frame) > self.threshold:
            self.last_kept = frame
            return True
        return False


def record_frames(out_dir: Path, seconds: float, fps: float = 2.0) -> int:
    """Save raw frames + timestamps, to build a gold set for the sampler eval."""
    out_dir.mkdir(parents=True, exist_ok=True)
    t0, n, index = time.time(), 0, []
    while time.time() - t0 < seconds:
        t = time.time() - t0
        capture().save(out_dir / f"{n:05d}.png")
        index.append({"frame": f"{n:05d}.png", "t": round(t, 2)})
        n += 1
        time.sleep(max(0.0, (n / fps) - (time.time() - t0)))
    (out_dir / "frames.json").write_text(json.dumps(index, indent=1))
    return n
