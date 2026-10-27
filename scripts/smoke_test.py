"""Smoke test: can the pinned llama-server + Gemma 4 handle text, image and audio?

    .venv/bin/python scripts/smoke_test.py            # default tier (E4B)
    .venv/bin/python scripts/smoke_test.py --tier e2b

Makes its own inputs (a rendered error screenshot, and speech from macOS `say`),
so it runs anywhere, anytime. Exit code 0 means every check passed.
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

from agent import config, serving  # noqa: E402

SPOKEN = "Please buy oat milk and call the plumber on Monday."


def make_inputs(tmp: Path) -> tuple[Path, Path]:
    img = Image.new("RGB", (900, 260), (30, 30, 30))
    font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 22)
    ImageDraw.Draw(img).text((30, 30), "$ python app.py\nTraceback (most recent call last):\n"
                             "ModuleNotFoundError: No module named 'requests'", fill="white", font=font)
    img.save(tmp / "screen.png")
    subprocess.run(["say", "-o", str(tmp / "speech.aiff"), SPOKEN], check=True)
    subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1",
                    str(tmp / "speech.aiff"), str(tmp / "speech.wav")], check=True)
    return tmp / "screen.png", tmp / "speech.wav"


def b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def check(name: str, content, must_contain: list[str], **params) -> bool:
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default=config.TIER, choices=list(config.TIERS))
    tier = ap.parse_args().tier
    server = serving.llm_server(tier)
    started_here = not server.healthy()
    print(f"llama.cpp {config.LLAMA_BUILD}, tier {tier}: starting server..." if started_here
          else f"Using the server already running on port {config.LLM_PORT}")
    server.start()

    with tempfile.TemporaryDirectory() as d:
        png, wav = make_inputs(Path(d))
        results = [
            check("text", "Reply with exactly one word: ready", ["ready"]),
            check("image", [{"type": "text", "text": "Which Python module is missing in this screenshot?"},
                            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64(png)}}],
                  ["requests"]),
            check("audio", [{"type": "text", "text": "Transcribe this audio."},
                            {"type": "input_audio", "input_audio": {"data": b64(wav), "format": "wav"}}],
                  ["milk", "plumber", "monday"]),
            check("tools", "What time is it right now?", ["get_time"],
                  tools=[{"type": "function", "function": {"name": "get_time", "description": "Current local time",
                                                           "parameters": {"type": "object", "properties": {}}}}]),
        ]
    print(f"llama-server resident memory: {server.rss_gb():.1f} GB")
    if started_here:
        server.stop()
    print("ALL PASS" if all(results) else "SOME CHECKS FAILED")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
