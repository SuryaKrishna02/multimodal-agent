# Platforms and accelerators

The harness has two independent plug-in axes. Pick one of each. The agent finds both by file name, so
supporting a new machine means adding files, not editing core code.

| Axis | Folder | Answers | Examples |
|---|---|---|---|
| Operating system | `agent/platforms/` | How do I record the screen, speak, listen for a hotkey, install ffmpeg? | `macos`, `linux`, `windows` |
| Accelerator | `agent/accel/` | Which llama.cpp build, which flags, how do I measure its memory? | `metal`, `cuda`, `rocm`, `vulkan`, `cpu` |

Ubuntu + NVIDIA is `linux` + `cuda`. Windows + NVIDIA is `windows` + `cuda`. Windows without a
GPU is `windows` + `cpu`.

## How a machine is chosen

1. `AGENT_PLATFORM=<name>` / `AGENT_ACCEL=<name>` if set.
2. Otherwise the agent calls `matches()` on each plug-in. It tries the accelerator with the highest `priority` first.
3. No OS plug-in matches: a generic fallback runs. Portable parts work (mic, screenshots, hotkey via
   pynput, model serving) and OS hooks raise `Unsupported` naming the file to add.
4. No accelerator matches: setup stops and says so.

`python cli.py doctor` shows the result, the missing hooks and the installed files.

## Support matrix

Status values: **verified** means a maintainer ran `doctor --exercise` and the smoke test on real hardware.
**wanted** means no plug-in exists yet and PRs are welcome. **community** means the person named contributed
and tested it.
Add yourself to the table in your PR.

| OS | Accelerator | Status | Tested by | Notes |
|---|---|---|---|---|
| macOS (Apple Silicon) | metal | verified | Surya Guthikonda, M4 24 GB, macOS 27.0.1 | smoke test ALL PASS. First run asks for Microphone, Screen Recording, Accessibility and Input Monitoring |
| Linux (Ubuntu) | cuda | wanted | | |
| Linux | vulkan / rocm / cpu | wanted | | |
| Windows | cuda | wanted | | |
| Windows | vulkan / cpu | wanted | | |
| macOS (Intel) | cpu | wanted | | |

## What each OS hook must do

`agent/platforms/base.py` defines these hooks. `doctor` checks the first three:

| Hook | Contract |
|---|---|
| `record_screen(out, seconds)` | Record the main display to the file `out` (container: `raw_video_suffix`). ffmpeg is fine. |
| `synthesize_speech(text, out)` | Write spoken text to an audio file that soundfile can read, and return its path. The smoke test and the synthetic eval cases (asr, vad, task1, structured) use it; without it those evals skip. |
| `start_hotkey(combo, on_press)` | Default uses pynput. Override where pynput does not work, for example on Wayland. |
| `hotkey_warning()` | Return a message if the OS will silently drop key events. |
| `install_hint(tool)` | One line to install a tool such as ffmpeg (`brew install ffmpeg`, `sudo apt install ffmpeg`...). |
| `venv_python(venv)` | Path to the venv's Python. |
| facts | `exe_suffix`, `raw_video_suffix`, `monospace_font`, `screen_blank_hint`, `permission_notes()` |

## What each accelerator must do

Defined in `agent/accel/base.py`:

| Member | Contract |
|---|---|
| `matches()` | Detect the hardware with the standard library (`shutil.which("nvidia-smi")`...). |
| `downloads` | `{platform_name: Download(url, sha256, archive, bin_subdir)}` for the pinned llama.cpp build. The sha256 must come from the file you tested. |
| `server_args()` | Extra `llama-server` flags, for example `["-ngl", "99"]`. |
| `server_memory_gb(processes)` | Memory the servers hold. Discrete GPUs: read VRAM from the driver, because RSS does not include it. |
