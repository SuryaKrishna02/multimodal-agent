"""One-time setup on any OS: pinned llama.cpp build for your accelerator, pinned models, Python venv.

    python3 scripts/setup.py                     # detect OS + accelerator automatically
    python3 scripts/setup.py --only e2b vad      # skip the groups you do not need (e4b e2b embed vad)
    AGENT_ACCEL=cuda python3 scripts/setup.py    # force an accelerator (see: cli.py doctor)

Everything is checksum-verified; re-running skips files that are already correct.
Standard library only, because it runs before the venv exists.
"""

import argparse
import hashlib
import shutil
import subprocess
import sys
import tarfile
import urllib.error
import urllib.request
import venv
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from agent import accel, config, pins, platforms  # noqa: E402


def sha256(path: Path) -> str:
    """Compute the sha256 of a file.

    Args:
        path (Path): The file.

    Returns:
        str: Hex digest.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _download(url: str, part: Path, have: int) -> None:
    """Stream `url` into `part`, resuming after `have` bytes when the server accepts a range request."""
    req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
    with urllib.request.urlopen(req) as r:
        resumed = have and r.status == 206
        total = int(r.headers.get("Content-Length", 0)) + (have if resumed else 0)
        done = have if resumed else 0
        with open(part, "ab" if resumed else "wb") as f:
            while block := r.read(1 << 20):
                f.write(block)
                done += len(block)
                if total:
                    print(f"\r          {done / total:4.0%} of {total / 1e6:,.0f} MB", end="", flush=True)
    print()


def fetch(url: str, dest: Path, sha: str) -> None:
    """Download `url` to `dest` and verify its sha256. Skips the download when `dest` already matches."""
    if dest.exists() and sha256(dest) == sha:
        print(f"ok        {dest.relative_to(ROOT)}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    print(f"download  {dest.relative_to(ROOT)}")
    for attempt in (1, 2):  # attempt 2 restarts from zero if resuming failed
        have = part.stat().st_size if part.exists() and attempt == 1 else 0
        try:
            _download(url, part, have)
            break
        except urllib.error.HTTPError as e:
            if attempt == 2 or e.code != 416:  # 416: our partial file is already complete or stale
                raise
            part.unlink()
    if sha256(part) != sha:
        part.unlink()
        sys.exit(f"CHECKSUM MISMATCH: {dest.name} (deleted). If a pin was bumped, update its sha256.")
    part.replace(dest)


def install_llama(a: accel.Accelerator, plat: platforms.Platform) -> None:
    """Download, extract and prepare the pinned llama.cpp build.

    Args:
        a (accel.Accelerator): The accelerator plug-in.
        plat (platforms.Platform): The platform plug-in.
    """
    dl = a.download_for(plat.name)
    archive = config.CACHE / "downloads" / dl.url.rsplit("/", 1)[-1]
    fetch(dl.url, archive, dl.sha256)
    server = a.bin_dir(plat.name) / plat.exe("llama-server")
    if server.exists():
        print(f"ok        {server.relative_to(ROOT)}")
        return
    dest = a.install_dir()
    dest.mkdir(parents=True, exist_ok=True)
    if dl.archive == "zip":
        with zipfile.ZipFile(archive) as z:
            z.extractall(dest)
    else:
        with tarfile.open(archive) as t:
            if hasattr(tarfile, "data_filter"):
                t.extractall(dest, filter="data")
            else:
                t.extractall(dest)
    if not server.exists():
        sys.exit(f"{server} not found after extracting {archive.name}: fix bin_subdir in agent/accel/{a.name}.py")
    for f in server.parent.iterdir():  # zip does not keep the executable bit (harmless on Windows)
        if f.is_file() and not f.suffix:
            f.chmod(f.stat().st_mode | 0o111)


def setup_venv(plat: platforms.Platform) -> Path:
    """Create the virtual environment and install the pinned packages.

    Args:
        plat (platforms.Platform): The platform plug-in.

    Returns:
        Path: The Python executable of the virtual environment.
    """
    env = ROOT / ".venv"
    if not env.exists():
        venv.create(env, with_pip=True)
    py = plat.venv_python(env)
    lock = ROOT / "requirements" / f"lock-{plat.name}.txt"
    reqs = lock if lock.exists() else ROOT / "requirements" / "base.txt"
    if reqs.name == "base.txt":
        print(
            f"NOTE: no full lockfile for '{plat.name}' yet; installing the pinned direct dependencies "
            f"from requirements/base.txt. After it works, add requirements/lock-{plat.name}.txt "
            "(pip freeze) in your PR."
        )
    subprocess.run([str(py), "-m", "pip", "install", "-q", "--upgrade", "pip"], check=True)
    subprocess.run([str(py), "-m", "pip", "install", "-q", "-r", str(reqs)], check=True)
    return py


def main() -> None:
    """Run the one-time setup: llama.cpp, model files, virtual environment."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--only",
        nargs="+",
        choices=["e4b", "e2b", "embed", "vad"],
        metavar="GROUP",
        help="download only these groups: e4b e2b embed vad (default: all)",
    )
    only = ap.parse_args().only
    plat, a = platforms.current(), accel.current()
    print(f"platform: {plat.name}   accelerator: {a.name}   llama.cpp {config.LLAMA_BUILD}\n")

    install_llama(a, plat)
    for pin in pins.PINS:
        if not only or pin.group in only:
            fetch(pin.url, pins.path(pin), pin.sha256)
    py = setup_venv(plat)

    if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
        print(f"\nNOTE: ffmpeg not found. Video input is disabled until you: {plat.install_hint('ffmpeg')}")
    print(f"\nSetup done. Next:  {py} cli.py doctor   then   {py} scripts/smoke_test.py")


if __name__ == "__main__":
    main()
