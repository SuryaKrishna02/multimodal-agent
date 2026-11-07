"""Apple Silicon GPU via Metal. The official macOS arm64 build offloads to Metal by default."""

import platform
import sys

from agent import config
from agent.accel.base import Accelerator, Download
from agent.pins import HF, Pin

WHISPER_REV = "a4aaeec0636e6fef84abdcbe3544cb2bf7e9f6fb"  # mlx-community/whisper-large-v3-turbo
_WHISPER = f"{HF}/mlx-community/whisper-large-v3-turbo/resolve/{WHISPER_REV}"


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

    stt_pins = [
        Pin(
            f"{_WHISPER}/config.json",
            "models/whisper-large-v3-turbo/config.json",
            "b34fc29e4e11e0a25e812775dd67f4dd16fc2c8eb43d28ae25ff7d660ecb6379",
            "whisper",
        ),
        Pin(
            f"{_WHISPER}/weights.safetensors",
            "models/whisper-large-v3-turbo/weights.safetensors",
            "951ed3fc1203e6a62467abb2144a96ce7eafca8fa77e3704fdb8635ff3e7f8a6",
            "whisper",
        ),
    ]

    @classmethod
    def matches(cls) -> bool:
        """Detect Apple Silicon.

        Returns:
            bool: True on macOS with an arm64 processor.
        """
        return sys.platform == "darwin" and platform.machine() == "arm64"

    def transcribe(self, samples, language: str | None) -> str:
        """Transcribe audio with Whisper large-v3-turbo on MLX (mlx-whisper, installed without torch).

        Args:
            samples (np.ndarray): 16 kHz mono float32 audio.
            language (str | None): Language code such as "en", or None to detect it.

        Returns:
            str: The transcript.
        """
        import mlx_whisper  # macOS lock file only; installed with --no-deps (see requirements/nodeps-macos.txt)

        out = mlx_whisper.transcribe(
            samples,
            path_or_hf_repo=str(config.WHISPER_DIR),
            language=language,
            condition_on_previous_text=False,  # each clip stands alone: no hallucinated carry-over
            verbose=None,
        )
        return str(out["text"]).strip()
