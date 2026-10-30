"""Synthetic screen cases: terminals, editors and dialogs drawn with Pillow, and videos made from them.

Nothing here captures your real screen. The font comes from the OS plug-in (`monospace_font`),
so the pictures look the same on every machine that has that font.
"""

import json
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from agent import platforms
from agent.senses import video

DARK, LIGHT = ((30, 30, 30), "white"), ((245, 245, 245), (20, 20, 20))
RED = (200, 40, 40)

# name -> (lines, has_problem, must_mention). must_mention is a list of groups; the answer needs one phrase from each.
SHOTS = {
    "module-missing": (
        [
            "$ python app.py",
            "Traceback (most recent call last):",
            '  File "app.py", line 1, in <module>',
            "    import requests",
            "ModuleNotFoundError: No module named 'requests'",
        ],
        True,
        [["requests"], ["pip install", "install"]],
    ),
    "port-in-use": (
        ["$ python -m http.server 8000", "OSError: [Errno 48] Address already in use"],
        True,
        [
            ["address already in use", "already in use", "8000"],
            ["kill", "another port", "different port", "free the port", "lsof"],
        ],
    ),
    "syntax-error": (
        [
            "$ python main.py",
            '  File "main.py", line 12',
            "    if count == 3",
            "                ^",
            "SyntaxError: expected ':'",
        ],
        True,
        [["syntaxerror", "syntax error", "syntax"], ["colon", "':'", "add a :"]],
    ),
    "file-missing": (
        ["$ python load.py", "FileNotFoundError: [Errno 2] No such file or directory: 'config.yaml'"],
        True,
        [["config.yaml"], ["create", "exist", "path", "missing"]],
    ),
    "merge-conflict": (
        [
            "$ git merge feature/login",
            "Auto-merging app.py",
            "CONFLICT (content): Merge conflict in app.py",
            "Automatic merge failed; fix conflicts and then commit the result.",
        ],
        True,
        [["conflict"], ["resolve", "fix the conflict", "edit", "fix conflicts"]],
    ),
    "test-failure": (
        ["$ pytest", "FAILED tests/test_math.py::test_add - assert 4 == 5", "1 failed, 12 passed in 0.31s"],
        True,
        [["test_add"], ["assert", "expected", "5", "wrong", "fix"]],
    ),
    "all-clean": (
        ["$ ls", "README.md  app.py  tests", "$ git status", "On branch main", "nothing to commit, working tree clean"],
        False,
        [],
    ),
    "tests-pass": (
        ["$ pytest", "============ 13 passed in 0.42s ============"],
        False,
        [],
    ),
}


def font(size: int = 20) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    """Load the monospace font of the OS plug-in, or the Pillow default."""
    path = platforms.current().monospace_font
    return ImageFont.truetype(path, size) if path else ImageFont.load_default(size=size)


def render(
    lines: list[str], size: tuple[int, int] = (900, 280), light: bool = False, cursor: bool = False
) -> Image.Image:
    """Draw text lines on a plain background, like a terminal or editor.

    Args:
        lines (list[str]): Text, one entry per row.
        size (tuple[int, int], optional): Width and height in pixels. Defaults to (900, 280).
        light (bool, optional): Dark text on a light background. Defaults to False.
        cursor (bool, optional): Draw a block cursor after the last line. Defaults to False.

    Returns:
        Image.Image: The picture.
    """
    bg, fg = LIGHT if light else DARK
    img = Image.new("RGB", size, bg)
    draw, f = ImageDraw.Draw(img), font()
    for i, line in enumerate(lines):
        draw.text((24, 20 + i * 28), line, fill=fg, font=f)
    if cursor:
        draw.rectangle((24, 20 + len(lines) * 28, 36, 20 + len(lines) * 28 + 22), fill=fg)
    return img


def with_dialog(img: Image.Image, title: str, message: str) -> Image.Image:
    """Put an error dialog over a picture.

    Args:
        img (Image.Image): The background. It is not changed.
        title (str): Dialog title.
        message (str): Dialog text.

    Returns:
        Image.Image: A copy with the dialog in the middle.
    """
    out = img.copy()
    w, h = out.size
    box = (w // 8, h // 4, w * 7 // 8, h * 3 // 4)
    draw = ImageDraw.Draw(out)
    draw.rectangle(box, fill=(250, 250, 250), outline=RED, width=4)
    draw.rectangle((box[0], box[1], box[2], box[1] + 36), fill=RED)
    draw.text((box[0] + 12, box[1] + 6), title, fill="white", font=font())
    draw.text((box[0] + 12, box[1] + 56), message, fill=(20, 20, 20), font=font())
    return out


def video_from_frames(frames: list[Image.Image], fps: int, out: Path) -> None:
    """Encode pictures as an H.264 video with ffmpeg.

    Args:
        frames (list[Image.Image]): The frames, all the same size.
        fps (int): Frames per second.
        out (Path): The MP4 file to write.
    """
    with tempfile.TemporaryDirectory() as d:
        for i, f in enumerate(frames):
            f.save(Path(d) / f"f{i:03d}.png")
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", f"{d}/f%03d.png"]
            + ["-c:v", "libx264", "-pix_fmt", "yuv420p", str(out)],
            check=True,
        )


def _video_cases() -> dict[str, tuple[list[Image.Image], bool, list[list[str]]]]:
    """Describe the two video cases: a disk-full error at 2 s, and a build that goes well."""
    size = (640, 360)
    before = render(["$ ./backup.sh", "copying files..."], size)
    error = render(["$ ./backup.sh", "copying files...", "DiskFullError: no space left on device"], size)
    steps = [render(["$ make", f"building... {p}%"], size) for p in (10, 40, 70, 100)]
    done = render(["$ make", "building... 100%", "Build finished, all tests passed"], size)
    return {
        "video-disk-full": (
            [before] * 8 + [error] * 12,
            True,
            [["disk", "space"], ["free", "delete", "clean", "remove"]],
        ),
        "video-build-ok": ([f for s in steps for f in [s] * 4] + [done] * 4, False, []),
    }


def build_task2(out: Path) -> None:
    """Write screenshots, and screen recordings when ffmpeg is installed, with what a correct answer must mention.

    Args:
        out (Path): The case folder of the `task2` eval.
    """
    out.mkdir(parents=True, exist_ok=True)
    for i, (name, (lines, has_problem, mention)) in enumerate(SHOTS.items()):
        render(lines, light=i % 3 == 2).save(out / f"{name}.png")
        (out / f"{name}.json").write_text(json.dumps({"has_problem": has_problem, "must_mention": mention}))
    if not video.ffmpeg_dir():
        return
    for name, (frames, has_problem, mention) in _video_cases().items():
        video_from_frames(frames, 4, out / f"{name}.mp4")
        (out / f"{name}.json").write_text(json.dumps({"has_problem": has_problem, "must_mention": mention}))


def _sequences() -> dict[str, tuple[Callable[[float], Image.Image], float, list[dict]]]:
    """Describe the screen sequences: how to draw frame at time t, how long, and the key moments."""
    size, log = (640, 360), ["$ python train.py", "epoch 1/10  loss 0.91", "epoch 2/10  loss 0.74"]

    def error_dialog(t: float) -> Image.Image:
        img = render(log, size, cursor=int(t * 2) % 2 == 0)
        return with_dialog(img, "Error", "Out of memory: training stopped") if 14.5 <= t < 24 else img

    def window_switch(t: float) -> Image.Image:
        if 8 <= t < 16:
            return render(["$ git log --oneline", "a1b2c3d fix login", "e4f5a6b add tests"], size)
        return render(["def add(a, b):", "    return a + b", "", "# TODO: write docs"], size, light=True)

    def idle(t: float) -> Image.Image:
        return render(log, size, cursor=int(t * 2) % 2 == 0)

    def log_lines(t: float) -> Image.Image:
        shown = log + [f"epoch {n + 3}/10  loss {0.7 - n * 0.05:.2f}" for n in range(int(t // 5))]
        if t >= 20:
            shown.append("ERROR: build failed")
        return render(shown, size)

    return {
        "error-dialog": (error_dialog, 30, [{"t": 14.5, "label": "error dialog appears"}]),
        "window-switch": (
            window_switch,
            20,
            [{"t": 8.0, "label": "terminal opens"}, {"t": 16.0, "label": "editor returns"}],
        ),
        "idle": (idle, 20, []),
        "log-lines": (log_lines, 30, [{"t": 20.0, "label": "build failed line appears"}]),
    }


def build_screen(out: Path) -> None:
    """Write frame sequences at 2 fps with the times of their key moments, as `record-screen` does.

    Args:
        out (Path): The case folder of the `screen` eval.
    """
    fps = 2
    for name, (draw, seconds, moments) in _sequences().items():
        folder = out / name
        folder.mkdir(parents=True, exist_ok=True)
        index = []
        for n in range(int(seconds * fps)):
            draw(n / fps).save(folder / f"{n:05d}.png")
            index.append({"frame": f"{n:05d}.png", "t": n / fps})
        (folder / "frames.json").write_text(json.dumps(index))
        (folder / "key_moments.json").write_text(json.dumps(moments))
