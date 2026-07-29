"""Local ASR on Apple Silicon via parakeet-mlx (verbatim output, word timestamps).

Verbatim matters: disfluencies are the signal used to find failed takes, so the
transcript must not be cleaned up.
"""
from __future__ import annotations

import os
from functools import cache
from pathlib import Path

from videoai.core.models import Word

DEFAULT_MODEL = os.getenv("PARAKEET_MODEL", "mlx-community/parakeet-tdt-0.6b-v3")


@cache
def _load_model(model_name: str):
    from parakeet_mlx import from_pretrained

    return from_pretrained(model_name)


class ParakeetASR:
    name = "parakeet"

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self.model_name = model_name

    def transcribe(self, audio_path: Path) -> list[Word]:
        model = _load_model(self.model_name)
        result = model.transcribe(str(audio_path))
        words: list[Word] = []
        for sentence in result.sentences:
            for token in sentence.tokens:
                text = token.text.strip()
                if text:
                    words.append(Word(text=text, start=float(token.start), end=float(token.end)))
        return words
