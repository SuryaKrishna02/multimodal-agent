"""Every knob in one place: model tier, paths, ports, thresholds.

Switch tier with one change:  AGENT_TIER=e2b python cli.py ...
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / ".cache"                # binaries + model weights (git-ignored)
RUNTIME = ROOT / "data" / "runtime"    # logs, sqlite, captures (git-ignored)
GOLD = ROOT / "data" / "gold"          # your recorded + labeled eval data
RESULTS = ROOT / "evals" / "results"   # cached eval results

# --- Serving (llama.cpp) -----------------------------------------------------
LLAMA_BUILD = "b11669"                 # = llama.cpp v0.7.0, verified with Gemma 4 audio
LLAMA_BIN = CACHE / "llama.cpp" / LLAMA_BUILD / f"llama-{LLAMA_BUILD}"
MODELS = CACHE / "models"

TIERS = {
    "e4b": {"model": "gemma-4-E4B-it-Q4_0.gguf", "mmproj": "mmproj-gemma-4-E4B-it-Q8_0.gguf"},
    "e2b": {"model": "gemma-4-E2B-it-Q4_0.gguf", "mmproj": "mmproj-gemma-4-E2B-it-Q8_0.gguf"},
}
TIER = os.environ.get("AGENT_TIER", "e4b")

LLM_PORT = 9931
LLM_CTX = 16384                        # tokens of context the server allocates
THINKING = False                       # Gemma 4 thinks by default; off = faster + shorter

EMBED = {"model": "embeddinggemma-2-Q8_0.gguf", "mmproj": "mmproj-embeddinggemma-2-Q8_0.gguf"}
EMBED_PORT = 9932
EMBED_DIM = 768

# --- Senses ------------------------------------------------------------------
SAMPLE_RATE = 16000                    # Gemma 4 audio + Silero VAD both expect 16 kHz mono
VAD_MODEL = CACHE / "silero_vad.onnx"
VAD_THRESHOLD = 0.5                    # speech probability per 32 ms frame
VAD_MIN_SILENCE_S = 0.6                # pause length that ends a speech segment
VAD_MIN_SPEECH_S = 0.25                # shorter blips are treated as noise
VAD_PAD_S = 0.2                        # keep a little audio around each segment
MAX_CHUNK_S = 28.0                     # Gemma 4 hard limit is 30 s per clip; stay under it
AUDIO_TOKENS_PER_S = 25                # ~750 tokens per 30 s clip

SCREEN_MAX_SIDE = 1280                 # downscale screenshots before sending
SCREEN_SAMPLE_EVERY_S = 1.0            # sampler: how often to look
SCREEN_CHANGE_THRESHOLD = 0.005        # sampler: share of pixels that must change (2 new lines of
                                       # terminal text = ~1%; a blinking cursor = ~0.01%)
IMAGE_TOKENS_ESTIMATE = 280            # Gemma 4 default vision budget per image

# --- Brain -------------------------------------------------------------------
MAX_OUTPUT_TOKENS = 1024
MAX_TOOLS_PER_REQUEST = 5              # small tool list per request: E4B degrades with many
TOOL_RETRIES = 2                       # re-ask the model after an invalid tool call
MEMORY_TOP_K = 3                       # memories injected into each prompt

# --- Memory ------------------------------------------------------------------
MEMORY_DB = RUNTIME / "memory.sqlite"
MIN_MEMORY_WORDS = 3                   # junk filter: too short to be useful
DUPLICATE_SIMILARITY = 0.95            # junk filter: near-copy of an existing memory

# --- Tools -------------------------------------------------------------------
WORKSPACE = Path(os.environ.get("AGENT_WORKSPACE", ROOT / "data" / "workspace"))

# --- Interfaces --------------------------------------------------------------
DASHBOARD_PORT = 8765
HOTKEY = "<ctrl>+<alt>+<space>"        # pynput syntax; "look at my screen and help"

# --- Eval pass thresholds (scorecard) ----------------------------------------
THRESHOLDS = {
    "serving.ttft_s": ("<=", 1.5),
    "serving.gen_tok_s": (">=", 15.0),
    "serving.peak_rss_gb": ("<=", 10.0),
    "vad.false_trigger_rate": ("<=", 0.10),
    "vad.missed_speech_rate": ("<=", 0.05),
    "asr.wer": ("<=", 0.15),
    "screen.key_moment_recall": (">=", 0.90),
    "structured.valid_rate": (">=", 0.95),
    "tool_calls.exact_rate": (">=", 0.80),
    "memory.recall_at_3": (">=", 0.80),
    "memory.junk_stored_rate": ("<=", 0.10),
    "tools.success_rate": (">=", 0.90),
    "tools.unsafe_blocked_rate": (">=", 1.00),
    "task1.precision": (">=", 0.80),
    "task1.recall": (">=", 0.80),
    "task2.correct_rate": (">=", 0.70),
}
