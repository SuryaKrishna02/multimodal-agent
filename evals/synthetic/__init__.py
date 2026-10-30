"""Generate the synthetic eval cases, so the evals run without any recording.

    python cli.py evals build            rebuild every component
    python cli.py evals build asr        rebuild one

The cases are written to `config.EVAL_SYNTHETIC/<component>/` in the format that the loader of
each eval reads. They are rebuilt when the code in this package changes. Synthetic cases are
labeled by construction (we know what the speaker said and where the error is), but they are
cleaner than real use: text-to-speech is easier than a real voice. Treat the scores as a
regression check, and add your own cases for the real picture (`python cli.py evals add`).
"""

import hashlib
import shutil
from collections.abc import Callable
from pathlib import Path

from agent import config
from evals.synthetic import audio, screens, structured, text

BUILDERS: dict[str, Callable[[Path], None]] = {
    "tools": text.build_tools,
    "tool_calls": text.build_tool_calls,
    "memory": text.build_memory,
    "asr": audio.build_asr,
    "vad": audio.build_vad,
    "task1": audio.build_task1,
    "task2": screens.build_task2,
    "screen": screens.build_screen,
    "structured": structured.build_structured,
}
MARKER = ".version"


def code_version() -> str:
    """Hash the code of this package, so a changed case table rebuilds the files.

    Returns:
        str: A short hex digest.
    """
    sources = sorted(Path(__file__).parent.glob("*.py"))
    return hashlib.sha256(b"".join(p.read_bytes() for p in sources)).hexdigest()[:12]


def is_current(component: str) -> bool:
    """Check whether the synthetic cases of a component exist and match the code.

    Args:
        component (str): A key of `BUILDERS`.

    Returns:
        bool: True if the folder was built by this version of the code.
    """
    marker = config.EVAL_SYNTHETIC / component / MARKER
    return marker.exists() and marker.read_text() == code_version()


def build(component: str) -> Path:
    """Build the synthetic cases of one component, replacing the old ones.

    The files are written to a temporary folder first, so a failure keeps the old cases.

    Args:
        component (str): A key of `BUILDERS`.

    Returns:
        Path: The folder with the cases.

    Raises:
        Unsupported: If the component needs speech and the OS plug-in has no speech hook.
    """
    final = config.EVAL_SYNTHETIC / component
    tmp = config.EVAL_SYNTHETIC / f".{component}.tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    try:
        BUILDERS[component](tmp)
        (tmp / MARKER).write_text(code_version())
        shutil.rmtree(final, ignore_errors=True)
        tmp.rename(final)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return final


def ensure(component: str | None = None) -> list[str]:
    """Build the synthetic cases that are missing or outdated.

    Args:
        component (str, optional): Only this component. Defaults to every component.

    Returns:
        list[str]: The components that were built.

    Raises:
        Unsupported: If a component needs speech and the OS plug-in has no speech hook.
    """
    names = [component] if component else list(BUILDERS)
    built = [c for c in names if c in BUILDERS and not is_current(c)]
    for c in built:
        build(c)
    return built
