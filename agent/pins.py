"""Pinned downloads that do not depend on the OS or the GPU: model weights and the VAD model.

Each file is verified by sha256 after download (scripts/setup.py, `cli.py doctor`). The llama.cpp
binaries are pinned per accelerator in agent/accel/<name>.py instead. Standard library only.
"""

from dataclasses import dataclass

from agent import config

HF = "https://huggingface.co"
E4B_REV = "a69f018d7678a2b9f5e705f8903d6fb29b9b25dc"  # ggml-org/gemma-4-E4B-it-GGUF
E2B_REV = "6938c58e698502037401a5f19f6905f2fb2f7dee"  # ggml-org/gemma-4-E2B-it-GGUF
EMB_REV = "58faf90940886df2d62be555c8765d00dac362c5"  # ggml-org/embeddinggemma-2-GGUF
SILERO_TAG = "v6.2.3"


@dataclass(frozen=True)
class Pin:
    """One pinned download.

    Attributes:
        url (str): Where to download the file.
        dest (str): Path of the file under `.cache/`.
        sha256 (str): Expected checksum of the file.
        group (str): "e4b", "e2b", "embed" or "vad", so that setup can fetch a subset.
    """

    url: str
    dest: str  # path under .cache/
    sha256: str
    group: str  # "e4b", "e2b", "embed" or "vad": setup can fetch a subset


PINS = [
    Pin(
        f"{HF}/ggml-org/gemma-4-E4B-it-GGUF/resolve/{E4B_REV}/gemma-4-E4B-it-Q4_0.gguf",
        "models/gemma-4-E4B-it-Q4_0.gguf",
        "a555b900214b477d8880e7832e0b8925e139b0159640036b09fe472b6f2097f2",
        "e4b",
    ),
    Pin(
        f"{HF}/ggml-org/gemma-4-E4B-it-GGUF/resolve/{E4B_REV}/mmproj-gemma-4-E4B-it-Q8_0.gguf",
        "models/mmproj-gemma-4-E4B-it-Q8_0.gguf",
        "197f49a93027f9843772bd24a6a9e0be2a32a788de5a3def330e9c585d86edd1",
        "e4b",
    ),
    Pin(
        f"{HF}/ggml-org/gemma-4-E2B-it-GGUF/resolve/{E2B_REV}/gemma-4-E2B-it-Q4_0.gguf",
        "models/gemma-4-E2B-it-Q4_0.gguf",
        "8e30dff3ac4c8434c49a7036fa15564bdbb6044e42bf04550bf1a096ad7e6a52",
        "e2b",
    ),
    Pin(
        f"{HF}/ggml-org/gemma-4-E2B-it-GGUF/resolve/{E2B_REV}/mmproj-gemma-4-E2B-it-Q8_0.gguf",
        "models/mmproj-gemma-4-E2B-it-Q8_0.gguf",
        "9406f99c16d68cda4f1f0552192dcc99021ea1fc6d2fd50b1dc3ccf30d04b292",
        "e2b",
    ),
    Pin(
        f"{HF}/ggml-org/embeddinggemma-2-GGUF/resolve/{EMB_REV}/embeddinggemma-2-Q8_0.gguf",
        "models/embeddinggemma-2-Q8_0.gguf",
        "2188ac1deca4b77dffefd603c2776a9d76d9d74ec01841392982ebb840b09135",
        "embed",
    ),
    Pin(
        f"{HF}/ggml-org/embeddinggemma-2-GGUF/resolve/{EMB_REV}/mmproj-embeddinggemma-2-Q8_0.gguf",
        "models/mmproj-embeddinggemma-2-Q8_0.gguf",
        "c4a8a52691ecef40618438928bdf9e68379b854e24166f292592353db0aab64f",
        "embed",
    ),
    Pin(
        f"https://raw.githubusercontent.com/snakers4/silero-vad/{SILERO_TAG}/src/silero_vad/data/silero_vad.onnx",
        "silero_vad.onnx",
        "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3",
        "vad",
    ),
]


def path(pin: Pin):
    """Return where a pinned file lives on disk.

    Args:
        pin (Pin): The pinned download.

    Returns:
        Path: Absolute path under `config.CACHE`.
    """
    return config.CACHE / pin.dest
