"""Smoke test: can the pinned llama-server + Gemma 4 handle text, image and audio?

    python scripts/smoke_test.py            # default tier (E4B), run with the .venv python
    python scripts/smoke_test.py --tier e2b

Makes its own inputs (a rendered error screenshot, a 4 s screen video made from rendered
frames, and speech from the OS plug-in's text-to-speech), so it runs anywhere, anytime.
Exit code 0 means every check passed.
It checks that each path WORKS, not how accurate it is: that is what evals/ is for.
"""

import argparse
import base64
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from agent import config, platforms, serving  # noqa: E402
from agent.senses import audio, video  # noqa: E402

SPOKEN = "Please buy oat milk and call the plumber on Monday."


def get_font():
    """Load the monospace font for rendered test images.

    Returns:
        ImageFont: The font from the platform plug-in, or the Pillow default.
    """
    path = platforms.current().monospace_font
    return ImageFont.truetype(path, 22) if path else ImageFont.load_default(size=22)


def make_video(tmp: Path, font) -> Path:
    """4 s at 4 fps: a prompt, then a DiskFullError appears at 2 s."""
    for n in range(16):
        img = Image.new("RGB", (640, 360), (30, 30, 30))
        text = "$ ./backup.sh" + ("\nDiskFullError: no space left on device" if n >= 8 else "")
        ImageDraw.Draw(img).text((20, 20), text, fill="white", font=font)
        img.save(tmp / f"f{n:02d}.png")
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            "4",
            "-i",
            str(tmp / "f%02d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(tmp / "screen.mp4"),
        ],
        check=True,
    )
    return tmp / "screen.mp4"


def make_inputs(tmp: Path) -> tuple[Path, Path]:
    """Render an error screenshot and synthesize speech.

    Args:
        tmp (Path): Folder for the files.

    Returns:
        tuple[Path, Path]: The PNG screenshot and the 16 kHz mono WAV.
    """
    img = Image.new("RGB", (900, 260), (30, 30, 30))
    font = get_font()
    ImageDraw.Draw(img).text(
        (30, 30),
        "$ python app.py\nTraceback (most recent call last):\nModuleNotFoundError: No module named 'requests'",
        fill="white",
        font=font,
    )
    img.save(tmp / "screen.png")
    speech = platforms.current().synthesize_speech(SPOKEN, tmp / "speech")
    (tmp / "speech.wav").write_bytes(audio.to_wav_bytes(audio.load_wav(speech)))  # 16 kHz mono WAV
    return tmp / "screen.png", tmp / "speech.wav"


def b64(path: Path) -> str:
    """Encode a file as base64 text.

    Args:
        path (Path): The file.

    Returns:
        str: Base64 text.
    """
    return base64.b64encode(path.read_bytes()).decode()


def check(name: str, content, must_contain: list[str], **params) -> bool:
    """Send one request and check that the reply has all the expected words.

    Args:
        name (str): Label for the report line.
        content: Message content, a string or a list of parts.
        must_contain (list[str]): Words that must appear in the reply or the tool calls, in any case.
        **params: Extra request parameters.

    Returns:
        bool: True if every word was found.
    """
    t0 = time.time()
    try:
        r = serving.chat([{"role": "user", "content": content}], max_tokens=200, **params)
    except Exception as e:  # noqa: BLE001 - a smoke test reports every failure the same way
        print(f"FAIL  {name:6s} {e}")
        return False
    msg = r["choices"][0]["message"]
    text = (msg.get("content") or "") + str(msg.get("tool_calls") or "")
    ok = all(w.lower() in text.lower() for w in must_contain)
    print(f"{'PASS' if ok else 'FAIL'}  {name:6s} {time.time() - t0:4.1f}s  {text.strip()[:110]!r}")
    return ok


def main() -> int:
    """Run all smoke checks.

    Returns:
        int: 0 if every check passed, else 1.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default=config.TIER, choices=list(config.TIERS))
    tier = ap.parse_args().tier
    server = serving.llm_server(tier)
    started_here = not server.healthy()
    print(
        f"llama.cpp {config.LLAMA_BUILD}, tier {tier}: starting server..."
        if started_here
        else f"Using the server already running on port {config.LLM_PORT}"
    )
    server.start()

    with tempfile.TemporaryDirectory() as d:
        png, wav = make_inputs(Path(d))
        results = [
            check("text", "Reply with exactly one word: ready", ["ready"]),
            check(
                "image",
                [
                    {"type": "text", "text": "Which Python module is missing in this screenshot?"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64(png)}},
                ],
                ["requests"],
            ),
            check(
                "audio",
                [
                    {"type": "text", "text": "Transcribe this audio."},
                    {"type": "input_audio", "input_audio": {"data": b64(wav), "format": "wav"}},
                ],
                ["milk", "plumber", "monday"],
            ),
            check(
                "tools",
                "What time is it right now?",
                ["get_time"],
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "get_time",
                            "description": "Current local time",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ],
            ),
        ]
        if video.ffmpeg_dir():
            mp4 = make_video(Path(d), get_font())
            results.append(
                check(
                    "video",
                    [
                        {"type": "input_video", "input_video": {"data": b64(mp4)}},
                        {"type": "text", "text": "What error appears in this screen recording?"},
                    ],
                    ["DiskFullError"],
                )
            )
        else:
            print(f"SKIP  video  ffmpeg/ffprobe not found ({platforms.current().install_hint('ffmpeg')})")
    print(f"llama-server resident memory: {server.rss_gb():.1f} GB")
    if started_here:
        server.stop()
    print("ALL PASS" if all(results) else "SOME CHECKS FAILED")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
