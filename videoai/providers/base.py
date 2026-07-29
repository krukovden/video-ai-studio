"""Provider protocols and resolution. Providers are the only place external
services are touched; every one has a mock twin so the pipeline runs offline."""
from __future__ import annotations

from pathlib import Path
from typing import Protocol

from videoai.core.models import Word


class ASRProvider(Protocol):
    name: str

    def transcribe(self, audio_path: Path) -> list[Word]: ...


class LLMProvider(Protocol):
    name: str

    def complete_json(self, prompt: str, images: list[Path], timeout: int) -> dict: ...


def resolve_asr(
    name: str,
    chunk_duration_seconds: float | None = None,
    overlap_duration_seconds: float | None = None,
) -> ASRProvider:
    """`chunk_duration_seconds`/`overlap_duration_seconds` are the configured
    `transcribe.chunk_duration_seconds`/`transcribe.overlap_duration_seconds`;
    providers that have no chunking (the mock) ignore them."""
    if name == "mock":
        from videoai.providers.asr_mock import MockASR

        return MockASR()
    if name == "parakeet":
        from videoai.providers.asr_parakeet import ParakeetASR

        kwargs = {}
        if chunk_duration_seconds is not None:
            kwargs["chunk_duration_seconds"] = chunk_duration_seconds
        if overlap_duration_seconds is not None:
            kwargs["overlap_duration_seconds"] = overlap_duration_seconds
        return ParakeetASR(**kwargs)
    raise ValueError(f"unknown asr provider: {name}")


def resolve_llm(name: str, model: str | None = None) -> LLMProvider:
    """`model` is the configured `analyze.llm_model`; providers that have no model
    choice (the mock) ignore it."""
    if name == "mock":
        from videoai.providers.llm_mock import MockLLM

        return MockLLM()
    if name == "claude_cli":
        from videoai.providers.llm_claude_cli import ClaudeCliLLM

        return ClaudeCliLLM(model) if model else ClaudeCliLLM()
    raise ValueError(f"unknown llm provider: {name}")
