"""Local ASR on Apple Silicon via parakeet-mlx (verbatim output, word timestamps).

Verbatim matters: disfluencies are the signal used to find failed takes, so the
transcript must not be cleaned up.

parakeet-mlx returns SentencePiece sub-word tokens, not words: a leading space in a
token's raw text marks the start of a new word, tokens without one continue the
previous word (this is how punctuation such as a trailing comma stays attached to the
word it follows). `merge_tokens_to_words` reassembles tokens into words along that
boundary signal before anything downstream (phrase building, take detection, timeline
re-anchoring) ever sees them.
"""
from __future__ import annotations

import os
from functools import cache
from pathlib import Path
from typing import Protocol

from videoai.core.models import Word

DEFAULT_MODEL = os.getenv("PARAKEET_MODEL", "mlx-community/parakeet-tdt-0.6b-v3")


class _Token(Protocol):
    text: str
    start: float
    end: float


def merge_tokens_to_words(tokens: list[_Token]) -> list[Word]:
    """Merge SentencePiece sub-word tokens into words on the leading-space boundary.

    A token whose raw text starts with a space begins a new word; any other token
    (including punctuation-only tokens) continues the current word. Empty and
    whitespace-only tokens are dropped. Word start/end come from the first/last token
    that contributed to that word.
    """
    words: list[Word] = []
    pieces: list[str] = []
    start: float | None = None
    end: float | None = None
    for token in tokens:
        raw = token.text
        if not raw or not raw.strip():
            continue
        if raw.startswith(" ") and pieces:
            words.append(Word(text="".join(pieces).strip(), start=start, end=end))
            pieces = []
            start = None
        if start is None:
            start = float(token.start)
        pieces.append(raw)
        end = float(token.end)
    if pieces:
        words.append(Word(text="".join(pieces).strip(), start=start, end=end))
    return words


@cache
def _load_model(model_name: str):
    from parakeet_mlx import from_pretrained

    return from_pretrained(model_name)


class ParakeetASR:
    name = "parakeet"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        chunk_duration_seconds: float = 120.0,
        overlap_duration_seconds: float = 15.0,
    ) -> None:
        self.model_name = model_name
        self.chunk_duration_seconds = chunk_duration_seconds
        self.overlap_duration_seconds = overlap_duration_seconds

    def transcribe(self, audio_path: Path) -> list[Word]:
        model = _load_model(self.model_name)
        result = model.transcribe(
            str(audio_path),
            chunk_duration=self.chunk_duration_seconds,
            overlap_duration=self.overlap_duration_seconds,
        )
        words: list[Word] = []
        for sentence in result.sentences:
            words.extend(merge_tokens_to_words(sentence.tokens))
        return words
