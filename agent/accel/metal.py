"""Apple Silicon GPU via Metal. The official macOS arm64 build offloads to Metal by default."""

import platform
import sys

from agent import config
from agent.accel.base import Accelerator, Download


class Metal(Accelerator):
    """Apple Silicon GPU through the official macOS arm64 llama.cpp build."""

    name = "metal"
    description = "Apple Silicon GPU (Metal, unified memory)"
    priority = 90
    downloads = {
        "macos": Download(
            url=f"https://github.com/ggml-org/llama.cpp/releases/download/{config.LLAMA_BUILD}/"
            f"llama-{config.LLAMA_BUILD}-bin-macos-arm64.tar.gz",
            sha256="58e5a5e1d7c2bf66cedd9c29e0323f3e570660a70a1e2156d32eed08f4cc1e3c",  # b11960
            archive="tar.gz",
            bin_subdir=f"llama-{config.LLAMA_BUILD}",
        ),
    }

    @classmethod
    def matches(cls) -> bool:
        """Detect Apple Silicon.

        Returns:
            bool: True on macOS with an arm64 processor.
        """
        return sys.platform == "darwin" and platform.machine() == "arm64"
