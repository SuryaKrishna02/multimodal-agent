"""The contract every accelerator plug-in implements. One file per backend in this folder.

An accelerator is the llama.cpp build + flags that match your GPU: metal, cuda, rocm, vulkan, cpu.
It is separate from the OS (agent/platforms/) because one backend runs on several OSes
(CUDA on Linux and Windows) and one OS offers several backends (Windows: CUDA, Vulkan, CPU).

Core code (serving, setup, doctor) calls `accel.current().<hook>()` and never hardcodes a binary
name, a GPU flag or a download URL. Top-level imports: standard library and `agent` only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from agent import config
from agent.platforms.base import Unsupported


@dataclass(frozen=True)
class Download:
    """One prebuilt llama.cpp archive: where it is, what it must hash to, where llama-server sits in it."""

    url: str
    sha256: str  # copy from the release page or compute it yourself; never guess
    archive: str  # "tar.gz" or "zip"
    bin_subdir: str = ""  # folder inside the archive that holds llama-server ("" = archive root)


class Accelerator:
    """Base class of every accelerator plug-in, with the portable defaults.

    Attributes:
        name (str): Must equal the file name.
        description (str): One line shown by `cli.py doctor`.
        priority (int): Auto-detection tries the highest first.
        downloads (dict[str, Download]): Prebuilt llama.cpp archives, keyed by platform name.
    """

    name = "cpu"  # must equal the file name
    description = ""
    priority = 0  # auto-detect tries the highest first (cuda 90, metal 90, vulkan 50, cpu 0)
    downloads: dict[str, Download] = {}  # keyed by platform name: {"macos": ..., "linux": ...}

    @classmethod
    def matches(cls) -> bool:
        """True when this machine has the hardware/driver this backend needs."""
        return False

    # --- llama.cpp install ----------------------------------------------------
    def download_for(self, platform_name: str) -> Download:
        """Return the llama.cpp archive for an operating system.

        Args:
            platform_name (str): Name of a platform plug-in, for example "macos".

        Returns:
            Download: The pinned archive.

        Raises:
            Unsupported: If this accelerator has no download for that platform.
        """
        if platform_name not in self.downloads:
            have = ", ".join(self.downloads) or "none"
            raise Unsupported(
                f"Accelerator '{self.name}' has no llama.cpp download for platform "
                f"'{platform_name}' (has: {have}). Add one in agent/accel/{self.name}.py; "
                "see CONTRIBUTING.md."
            )
        return self.downloads[platform_name]

    def install_dir(self) -> Path:
        """Return the folder that holds this accelerator's llama.cpp build.

        Returns:
            Path: A folder under `config.CACHE`.
        """
        return config.CACHE / "llama.cpp" / config.LLAMA_BUILD / self.name

    def bin_dir(self, platform_name: str) -> Path:
        """Return the folder that holds the llama-server executable.

        Args:
            platform_name (str): Name of a platform plug-in.

        Returns:
            Path: The install folder plus the archive's `bin_subdir`.
        """
        return self.install_dir() / self.download_for(platform_name).bin_subdir

    # --- llama-server behaviour -----------------------------------------------
    def server_args(self) -> list[str]:
        """Extra llama-server flags for this backend, e.g. ["-ngl", "99"] to offload layers."""
        return []

    def server_memory_gb(self, processes: list) -> float:
        """Memory the model servers hold. Default: resident memory, right for unified-memory Metal.

        A discrete GPU keeps weights in VRAM, which RSS does not show: override with the
        driver's number (nvidia-smi, rocm-smi) or the serving eval under-reports.
        """
        return sum(p.memory_info().rss for p in processes) / 1e9
