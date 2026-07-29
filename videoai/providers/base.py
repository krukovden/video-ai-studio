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


def resolve_asr(name: str) -> ASRProvider:
    if name == "mock":
        from videoai.providers.asr_mock import MockASR

        return MockASR()
    if name == "parakeet":
        from videoai.providers.asr_parakeet import ParakeetASR

        return ParakeetASR()
    raise ValueError(f"unknown asr provider: {name}")


def resolve_llm(name: str) -> LLMProvider:
    if name == "mock":
        from videoai.providers.llm_mock import MockLLM

        return MockLLM()
    if name == "claude_cli":
        from videoai.providers.llm_claude_cli import ClaudeCliLLM

        return ClaudeCliLLM()
    raise ValueError(f"unknown llm provider: {name}")
