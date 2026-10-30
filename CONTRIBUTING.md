# Contributing

The most useful contribution is making this run on your machine. That is one file for your OS, one
file for your GPU backend, and no edits to core code. Background and the support matrix:
[docs/PLATFORMS.md](docs/PLATFORMS.md).

First run `python cli.py doctor` (or `python3 scripts/doctor.py` before setup works). It tells you
which plug-ins it detected and which hooks are missing for your machine.

## Add a platform (an operating system)

1. `cp agent/platforms/_template.py agent/platforms/<name>.py` (`linux`, `windows`, `freebsd`...).
2. Set `name` (equal to the file name) and `matches()`.
3. Implement what you can: `record_screen`, `synthesize_speech`, and the optional overrides
   (`venv_python`, `install_hint`, `start_hotkey`, `hotkey_warning`, `permission_notes`, the facts).
   Delete the hooks you do not implement: the base class then raises a clear `Unsupported`.
   `synthesize_speech` also builds the synthetic audio cases of the evals; without it `asr`, `vad` and `task1` skip.
4. Import OS-specific packages **inside** the method that uses them. The file must import on every OS.
5. If the portable packages in `requirements/base.txt` are not enough, put OS-only packages in
   `requirements/lock-<name>.txt` (full `pip freeze` of an environment where everything works).

## Add an accelerator (a GPU or runtime)

1. `cp agent/accel/_template.py agent/accel/<name>.py` (`cuda`, `rocm`, `vulkan`, `cpu`...).
2. Set `name`, `priority` and `matches()` (standard library only).
3. Fill `downloads` for each OS you tested. Use the official llama.cpp release asset for the pinned
   build (`config.LLAMA_BUILD`). Add its sha256, the archive type, and the folder inside the archive that
   holds `llama-server`. Compute the checksum on the file you ran. Never copy one you did not verify.
4. Override `server_args()` if the backend needs flags, for example layer offload. Override
   `server_memory_gb()` if weights live in VRAM. The serving eval then measures the right thing.
5. If no prebuilt binary exists for your combination, say so in the PR. Building from source is a
   hook that does not exist yet: propose it in the PR by adding it to `agent/accel/base.py`.

## Before you open the PR

```bash
python -m unittest discover tests -v        # contract tests, standard library only
ruff check .                                # needs: pip install -r requirements/dev.txt
ruff format --check .
pyright
python cli.py doctor --exercise             # paste this output into the PR
python cli.py smoke                         # text, image, audio, tools, video: ALL PASS?
```

- Every public Python symbol needs a Google-style docstring, and `ruff check .` checks it.
- Add your row to the table in [docs/PLATFORMS.md](docs/PLATFORMS.md) with the hardware you tested on.
- The contract tests cannot test your GPU, so "verified" depends on your pasted output.
- Do not add recordings, screenshots, personal files or model weights. `data/evals/` is git-ignored.

## Changing core code

Core code (senses, serving, brain, tasks, cli, dashboard) must stay OS- and GPU-neutral. Suppose a command
differs by machine. Then add a hook to `agent/platforms/base.py` (or `agent/accel/base.py`). Implement it in
every existing plug-in, add it to `HOOKS`, and call it from core. The contract tests fail if core files
contain strings such as `screencapture` or `sys.platform`.
