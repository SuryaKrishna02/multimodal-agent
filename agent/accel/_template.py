"""TEMPLATE for a new accelerator plug-in. Copy to agent/accel/<name>.py, e.g. cuda.py or vulkan.py.

Files starting with _ are ignored by the loader. See CONTRIBUTING.md ("Add an accelerator").
"""

import shutil
import subprocess

from agent import config
from agent.accel.base import Accelerator, Download


class MyBackend(Accelerator):  # one Accelerator subclass per file
    name = "mybackend"  # must equal the file name
    description = "one line: which hardware and driver this needs"
    priority = 50  # higher wins auto-detect; metal/cuda use 90, cpu 0

    # One entry per OS this backend runs on. The sha256 must be real: compute it with
    # `shasum -a 256 file` (macOS/Linux) or `Get-FileHash file` (Windows) on the file you tested.
    downloads = {
        # "linux": Download(
        #     url=f"https://github.com/ggml-org/llama.cpp/releases/download/{config.LLAMA_BUILD}/<asset>",
        #     sha256="<64 hex characters>",
        #     archive="tar.gz",                 # or "zip"
        #     bin_subdir="<folder inside the archive holding llama-server>",
        # ),
    }

    @classmethod
    def matches(cls) -> bool:
        """Detect the hardware with stdlib only, e.g. shutil.which("nvidia-smi")."""
        return False

    # def server_args(self) -> list[str]:                # e.g. ["-ngl", "99"]
    # def server_memory_gb(self, processes) -> float:    # VRAM from the driver, not RSS
