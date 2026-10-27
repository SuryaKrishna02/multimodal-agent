"""Seeing over time: record the screen as video and send it as one `input_video` part.

What llama-server does with it: runs ffmpeg to take VIDEO_FPS frames per second, adds a
text timestamp every 5 s, and feeds every frame to Gemma 4 as an image. The audio track is
ignored. So the cost is (seconds x fps) images: keep clips short, slow and small.

Measured on E4B (12 s clip, 1 fps): 1280x720 = 4,944 tokens / 31 s; 640x360 = 1,342 tokens / 9 s.
"""
import json
import shutil
import subprocess
import time
from pathlib import Path

from agent import config, events

CAPTURES = config.RUNTIME / "captures"


def ffmpeg_dir() -> str | None:
    """Folder holding ffmpeg + ffprobe (llama-server needs both to decode video)."""
    found = shutil.which("ffmpeg")
    return str(Path(found).parent) if found and shutil.which("ffprobe") else None


def probe(path: Path) -> dict:
    """Width, height and duration of a video file."""
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-of", "json",
                          "-show_entries", "stream=width,height:format=duration", str(path)],
                         capture_output=True, text=True, check=True)
    info = json.loads(out.stdout)
    s = info["streams"][0]
    return {"width": s["width"], "height": s["height"], "seconds": float(info["format"]["duration"])}


def shrink(src: Path, dst: Path | None = None) -> Path:
    """Downscale to VIDEO_MAX_SIDE, cut to VIDEO_MAX_S, drop audio. Frame size drives token cost."""
    dst = dst or src.with_name(src.stem + "-small.mp4")
    side = config.VIDEO_MAX_SIDE
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-t", str(config.VIDEO_MAX_S),
                    "-vf", f"scale='min({side},iw)':-2", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    str(dst)], check=True)
    return dst


def record(seconds: float) -> Path:
    """Record the main screen with macOS `screencapture`, then shrink it."""
    CAPTURES.mkdir(parents=True, exist_ok=True)
    raw = CAPTURES / f"screen-{time.strftime('%Y%m%d-%H%M%S')}.mov"
    events.emit("senses.video", state="recording", seconds=seconds)
    subprocess.run(["screencapture", "-v", "-x", f"-V{int(seconds)}", str(raw)], check=True)
    small = shrink(raw)
    raw.unlink()
    events.emit("senses.video", state="recorded", path=str(small), **probe(small))
    return small
