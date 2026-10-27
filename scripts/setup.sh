#!/usr/bin/env bash
# One-time setup: pinned llama.cpp build, pinned model weights, Python venv.
# Everything is checksum-verified. Re-running skips files that are already correct.
# Apple Silicon Mac only. Needs ~11 GB of disk (both tiers + embeddings).
set -euo pipefail
cd "$(dirname "$0")/.."

LLAMA_BUILD=b11669   # llama.cpp v0.7.0
LLAMA_SHA=dbed55493d72bc1ba2babc8770d669c2289b5bc4ee9fb2a4dfc158d45b7d13fd
E4B_REV=a69f018d7678a2b9f5e705f8903d6fb29b9b25dc     # ggml-org/gemma-4-E4B-it-GGUF
E2B_REV=6938c58e698502037401a5f19f6905f2fb2f7dee     # ggml-org/gemma-4-E2B-it-GGUF
EMB_REV=58faf90940886df2d62be555c8765d00dac362c5     # ggml-org/embeddinggemma-2-GGUF
SILERO_TAG=v6.2.3

fetch() {  # fetch <url> <dest> <sha256>
  local url=$1 dest=$2 sha=$3
  if [[ -f $dest ]] && [[ $(shasum -a 256 "$dest" | cut -d' ' -f1) == "$sha" ]]; then
    echo "ok       $dest"; return
  fi
  echo "download $dest"
  curl -fL --progress-bar -C - -o "$dest" "$url" || curl -fL --progress-bar -o "$dest" "$url"
  [[ $(shasum -a 256 "$dest" | cut -d' ' -f1) == "$sha" ]] || { echo "CHECKSUM MISMATCH: $dest"; exit 1; }
}

[[ $(uname -sm) == "Darwin arm64" ]] || { echo "This harness supports Apple Silicon macOS only."; exit 1; }
mkdir -p .cache/llama.cpp .cache/models

# 1. llama.cpp (prebuilt Metal binaries from the official GitHub release)
fetch "https://github.com/ggml-org/llama.cpp/releases/download/$LLAMA_BUILD/llama-$LLAMA_BUILD-bin-macos-arm64.tar.gz" \
  ".cache/llama.cpp/$LLAMA_BUILD.tar.gz" $LLAMA_SHA
if [[ ! -x .cache/llama.cpp/$LLAMA_BUILD/llama-$LLAMA_BUILD/llama-server ]]; then
  mkdir -p .cache/llama.cpp/$LLAMA_BUILD && tar -xzf .cache/llama.cpp/$LLAMA_BUILD.tar.gz -C .cache/llama.cpp/$LLAMA_BUILD
fi

# 2. Models (Q4_0 weights come from Google's QAT checkpoints; Q8_0 projectors for image + audio)
HF=https://huggingface.co
fetch $HF/ggml-org/gemma-4-E4B-it-GGUF/resolve/$E4B_REV/gemma-4-E4B-it-Q4_0.gguf \
  .cache/models/gemma-4-E4B-it-Q4_0.gguf a555b900214b477d8880e7832e0b8925e139b0159640036b09fe472b6f2097f2
fetch $HF/ggml-org/gemma-4-E4B-it-GGUF/resolve/$E4B_REV/mmproj-gemma-4-E4B-it-Q8_0.gguf \
  .cache/models/mmproj-gemma-4-E4B-it-Q8_0.gguf 197f49a93027f9843772bd24a6a9e0be2a32a788de5a3def330e9c585d86edd1
fetch $HF/ggml-org/gemma-4-E2B-it-GGUF/resolve/$E2B_REV/gemma-4-E2B-it-Q4_0.gguf \
  .cache/models/gemma-4-E2B-it-Q4_0.gguf 8e30dff3ac4c8434c49a7036fa15564bdbb6044e42bf04550bf1a096ad7e6a52
fetch $HF/ggml-org/gemma-4-E2B-it-GGUF/resolve/$E2B_REV/mmproj-gemma-4-E2B-it-Q8_0.gguf \
  .cache/models/mmproj-gemma-4-E2B-it-Q8_0.gguf 9406f99c16d68cda4f1f0552192dcc99021ea1fc6d2fd50b1dc3ccf30d04b292
fetch $HF/ggml-org/embeddinggemma-2-GGUF/resolve/$EMB_REV/embeddinggemma-2-Q8_0.gguf \
  .cache/models/embeddinggemma-2-Q8_0.gguf 2188ac1deca4b77dffefd603c2776a9d76d9d74ec01841392982ebb840b09135
fetch $HF/ggml-org/embeddinggemma-2-GGUF/resolve/$EMB_REV/mmproj-embeddinggemma-2-Q8_0.gguf \
  .cache/models/mmproj-embeddinggemma-2-Q8_0.gguf c4a8a52691ecef40618438928bdf9e68379b854e24166f292592353db0aab64f

# 3. Silero VAD (ONNX file only, run with onnxruntime, so no PyTorch install)
fetch https://raw.githubusercontent.com/snakers4/silero-vad/$SILERO_TAG/src/silero_vad/data/silero_vad.onnx \
  .cache/silero_vad.onnx 1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3

# 4. Python environment
if [[ ! -d .venv ]]; then python3 -m venv .venv; fi
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q -r requirements.lock
# 5. ffmpeg: llama-server calls ffmpeg + ffprobe to decode video input (optional)
if ! command -v ffmpeg >/dev/null || ! command -v ffprobe >/dev/null; then
  echo "NOTE: ffmpeg not found. Video input is disabled until you run:  brew install ffmpeg"
fi
echo
echo "Setup done. Next:  .venv/bin/python scripts/smoke_test.py"
