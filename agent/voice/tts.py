"""Text to speech with Kokoro-82M, one sentence at a time.

A reply is cut into sentences, so the first one can play while the next ones are written. Kokoro runs through
onnxruntime on the CPU (`config.TTS_PROVIDERS`), so the GPU stays free for Gemma. espeak-ng turns the text into
phonemes. The few words that espeak says wrong (`config.PRONUNCIATIONS`) go in as IPA instead.
"""

import re
import threading
import unicodedata
from functools import lru_cache

import numpy as np

from agent import config

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_CLAUSE_END = re.compile(r"(?<=[,;:])\s+")
_lock = threading.Lock()  # one synthesis at a time: two would only share the same CPU cores

# Markdown and symbols that espeak would read out ("asterisk", "hash"). Each pattern runs in order on the reply.
_MARKUP = [
    (re.compile(r"```.*?(```|$)", re.S), " "),  # a code block is not read out
    (re.compile(r"`([^`]*)`"), r"\1"),  # inline code: the code without the backticks
    (re.compile(r"!?\[([^\]]*)\]\([^)]*\)"), r"\1"),  # a link or an image: its text only
    (re.compile(r"(https?://|www\.)\S*[^\s.,;:!?)]"), "the link"),  # a full stop after the link stays
    # [ \t] and not \s below: a line end must stay, because it ends a list item.
    (re.compile(r"^[ \t]{0,3}#{1,6}[ \t]+", re.M), ""),  # heading marks
    (re.compile(r"^[ \t]*>[ \t]?", re.M), ""),  # quote marks
    (re.compile(r"^[ \t]*[-*+\u2022][ \t]+", re.M), ""),  # bullet marks
    (re.compile(r"^[ \t]*(\d+)[.)][ \t]+", re.M), r"\1: "),  # "1. Balboa Park" -> "1: Balboa Park"
    (re.compile(r"^[ \t]*\|?[ \t]*:?-{3,}.*$", re.M), ""),  # the |---|---| line of a table
    (re.compile(r"[ \t]*\|[ \t]*"), ", "),  # table cells
    (re.compile(r"(\*{1,3}|~~)"), ""),  # bold, italic, strike-through
    (re.compile(r"(?<!\w)_{1,3}|_{1,3}(?!\w)"), ""),  # _italic_ (an underscore inside a word stays for below)
    (re.compile(r"_"), " "),  # snake_case -> snake case
    (re.compile(r"#(?=\d)"), "number "),
    (re.compile(r"[ \t]*(->|=>|\u2192)[ \t]*"), " to "),
    (re.compile(r"[ \t]*&[ \t]*"), " and "),
    (re.compile(r"\u00b0\s*([CF])\b"), r" degrees \1"),
    (re.compile(r"[#*\\^{}<>=|]"), " "),  # stray symbols that are left
]
_LINE_END = re.compile(r"[.!?:;,]$")


@lru_cache(maxsize=1)
def engine():
    """Load Kokoro once per process.

    Returns:
        kokoro_onnx.Kokoro: The model with its voices.

    Raises:
        FileNotFoundError: The model files are missing.
    """
    import onnxruntime as ort
    from kokoro_onnx import Kokoro

    for f in (config.TTS_MODEL, config.TTS_VOICES):
        if not f.exists():
            raise FileNotFoundError(f"{f.name} is missing: run python3 scripts/setup.py --only tts")
    session = ort.InferenceSession(str(config.TTS_MODEL), providers=config.TTS_PROVIDERS)
    return Kokoro.from_session(session, str(config.TTS_VOICES))


def voices() -> list[str]:
    """List the voice names that the voices file holds.

    Returns:
        list[str]: For example "af_heart". The prefix is the accent and gender: a = US, b = UK, f / m.
    """
    return sorted(engine().get_voices())


def speakable(text: str) -> str:
    """Turn a reply written for the screen into text to say: no markdown, symbols or emoji.

    Each line that ends without punctuation gets a full stop, so a list is spoken as separate sentences with a pause
    after each item.

    Args:
        text (str): The reply, often with markdown from the model.

    Returns:
        str: Plain text on one line.
    """
    for pattern, repl in _MARKUP:
        text = pattern.sub(repl, text)
    # Emoji and other pictographs (Unicode "So"), and the joiners and variation selectors between them.
    text = "".join(c for c in text if unicodedata.category(c) not in ("So", "Cs", "Cf") and c != "\ufe0f")
    lines = [" ".join(line.split()).strip(" ,") for line in text.splitlines()]
    return " ".join(line if _LINE_END.search(line) else f"{line}." for line in lines if line)


def sentences(
    text: str, max_chars: int = config.TTS_MAX_CHUNK_CHARS, first_chars: int = config.TTS_FIRST_CHUNK_CHARS
) -> list[str]:
    """Cut text into the pieces that are spoken one by one.

    Args:
        text (str): The reply.
        max_chars (int, optional): A longer sentence is cut at commas. Defaults to `config.TTS_MAX_CHUNK_CHARS`.
        first_chars (int, optional): The same for the first piece, which sets the time to first audio. Defaults
            to `config.TTS_FIRST_CHUNK_CHARS`.

    Returns:
        list[str]: Sentences, or parts of long sentences, in order.
    """
    pieces = []
    for i, sentence in enumerate(_SENTENCE_END.split(" ".join(text.split()))):
        pieces.extend(_cut(sentence, first_chars if i == 0 else max_chars))
    return [p for p in pieces if p]


def _cut(sentence: str, max_chars: int) -> list[str]:
    """Join the clauses of one sentence into pieces of at most `max_chars`. A clause longer than that stays whole."""
    if len(sentence) <= max_chars:
        return [sentence]
    parts, current = [], ""
    for clause in _CLAUSE_END.split(sentence):
        if current and len(current) + 1 + len(clause) > max_chars:
            parts.append(current)
            current = clause
        else:
            current = f"{current} {clause}".strip()
    return [*parts, current]


def phonemes(text: str) -> str:
    """Turn text into Kokoro's phonemes, with the IPA of `config.PRONUNCIATIONS` for the words that it lists.

    Args:
        text (str): One piece of a reply.

    Returns:
        str: The phoneme string.
    """
    tokenizer = engine().tokenizer
    if not config.PRONUNCIATIONS:
        return tokenizer.phonemize(text, config.TTS_LANG)
    words = "|".join(re.escape(w) for w in config.PRONUNCIATIONS)
    out = []
    # With a group in the pattern, re.split keeps each matched word at the odd positions.
    for i, part in enumerate(re.split(rf"\b({words})\b", text, flags=re.IGNORECASE)):
        if i % 2:
            out.append(config.PRONUNCIATIONS[part.lower()])
        elif part.strip():
            out.append(tokenizer.phonemize(part, config.TTS_LANG))
    return " ".join(out)


def synthesize(text: str, voice: str | None = None) -> np.ndarray:
    """Speak one piece of text.

    Args:
        text (str): The text, best one sentence (see `sentences`).
        voice (str | None, optional): A Kokoro voice name. Defaults to `config.TTS_VOICE`.

    Returns:
        np.ndarray: Float32 mono audio at `config.TTS_SAMPLE_RATE`.
    """
    k = engine()
    with _lock:
        audio, _ = k.create(
            phonemes(text),
            voice=voice or config.TTS_VOICE,
            speed=config.TTS_SPEED,
            lang=config.TTS_LANG,
            is_phonemes=True,
        )
    return np.asarray(audio, dtype=np.float32)
