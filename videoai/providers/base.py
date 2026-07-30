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
    # Whether this provider can hand the model actual video rather than stills.
    # A stage asks before submitting clips: a provider that silently dropped them
    # would return a text-quality answer while the caller believed it had paid
    # for video understanding.
    reads_video: bool

    def complete_json(
        self,
        prompt: str,
        images: list[Path],
        timeout: int,
        videos: list[Path] | None = None,
    ) -> dict: ...


def llm_system_preamble(name: str) -> str:
    """The fixed instruction a provider prepends to, or sends alongside, the prompt.

    It is part of what the model was asked, so it belongs in the fingerprint of any
    stage that calls the provider: editing Codex's preamble changes the analysis
    exactly as editing the stage's own prompt does. Read as a module constant so
    fingerprinting never constructs a provider or touches a CLI.
    """
    if name == "claude_cli":
        from videoai.providers.llm_claude_cli import SYSTEM_PROMPT

        return SYSTEM_PROMPT
    if name == "codex_cli":
        from videoai.providers.llm_codex_cli import SYSTEM_PROMPT

        return SYSTEM_PROMPT
    if name == "gemini_cli":
        from videoai.providers.llm_gemini_cli import SYSTEM_PROMPT

        return SYSTEM_PROMPT
    return ""


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
    if name == "codex_cli":
        from videoai.providers.llm_codex_cli import CodexCliLLM

        # Codex uses the model selected by the authenticated CLI profile. The
        # existing `analyze.llm_model` setting is Claude-specific.
        return CodexCliLLM()
    if name == "gemini_cli":
        from videoai.providers.llm_gemini_cli import GeminiCliLLM

        # `analyze.llm_model` names a Claude alias by default ("sonnet"), which
        # Gemini would reject, so a model is passed on only when it looks like a
        # Gemini one. Otherwise the CLI's own default is used.
        return GeminiCliLLM(model if model and model.startswith("gemini") else None)
    raise ValueError(f"unknown llm provider: {name}")
