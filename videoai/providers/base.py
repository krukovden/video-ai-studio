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


def split_llm_choice(choice: str) -> tuple[str, str | None]:
    """A provider selection, and the model it was told to answer as.

    `analyze.llm_model` is one field shared by every stage, and the providers are
    not one family: a config that watches the footage with Gemini while the cheap
    stages stay on the Claude CLI has one setting for two model namespaces, and
    whichever it holds is wrong for somebody. Writing
    "gemini_api:gemini-3.1-flash-lite" in `llm_by_stage` says it per stage, which
    is where the provider is already chosen per stage.
    """
    name, separator, model = choice.partition(":")
    return name.strip(), (model.strip() or None) if separator else None


def resolved_llm_model(choice: str, model: str | None = None) -> str:
    """The model that will actually answer, without constructing a provider.

    A stage's fingerprint has to carry this and not only the provider name.
    "gemini_api" is an alias for whichever weights `DEFAULT_MODEL` points at
    today: rotate that — Google's doing, or an edit to this repository — and
    every editorial decision in the pipeline is made by a different model while
    the cache goes on reporting that the analysis is up to date.

    Empty means the provider has no model to pin: the Codex CLI answers as
    whatever its authenticated profile selected, and the mock has no model at
    all. Pure and import-light on purpose — this runs for every stage on every
    run, and it must never start a CLI or open a socket to answer.
    """
    name, named_model = split_llm_choice(choice)
    model = named_model or model
    if name == "claude_cli":
        from videoai.providers.llm_claude_cli import DEFAULT_MODEL

        return model or DEFAULT_MODEL
    if name == "gemini_api":
        from videoai.providers.llm_gemini_api import DEFAULT_MODEL

        return model or DEFAULT_MODEL
    if name == "gemini_cli":
        # The CLI picks for itself when nothing is passed, and what it picks is
        # not visible from here.
        return model or ""
    return ""


def llm_cache_dir(work_dir: Path) -> Path:
    """Where a stage's replay cache lives: inside the project, so one project's
    answer can never be served for another's identically worded prompt."""
    from videoai.providers.llm_cache import CACHE_DIRNAME

    return work_dir / CACHE_DIRNAME


def llm_system_preamble(name: str) -> str:
    """The fixed instruction a provider prepends to, or sends alongside, the prompt.

    It is part of what the model was asked, so it belongs in the fingerprint of any
    stage that calls the provider: editing Codex's preamble changes the analysis
    exactly as editing the stage's own prompt does. Read as a module constant so
    fingerprinting never constructs a provider or touches a CLI.
    """
    name, _ = split_llm_choice(name)
    if name == "claude_cli":
        from videoai.providers.llm_claude_cli import SYSTEM_PROMPT

        return SYSTEM_PROMPT
    if name == "codex_cli":
        from videoai.providers.llm_codex_cli import SYSTEM_PROMPT

        return SYSTEM_PROMPT
    if name == "gemini_cli":
        from videoai.providers.llm_gemini_cli import SYSTEM_PROMPT

        return SYSTEM_PROMPT
    if name == "gemini_api":
        from videoai.providers.llm_gemini_api import SYSTEM_PROMPT

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


def _require_gemini_model(name: str, model: str) -> None:
    """Refuse a model a Gemini provider cannot be.

    The previous guard passed a model on only when it started with "gemini" and
    otherwise let the provider fall back to its own default. With the shipped
    `llm_model: sonnet` that guard fails on every run: the creator configures one
    model, an entirely different one answers, and nothing anywhere says so.
    Substitution has to be refused rather than made quieter — this setting picks
    the voice that scores the footage.
    """
    if model.startswith("gemini"):
        return
    raise ValueError(
        f"{name} cannot answer as '{model}': that is not a Gemini model. Set "
        "analyze.llm_model to one (e.g. gemini-3.1-flash-lite), or name it with "
        f"the provider so the other stages keep theirs: "
        f"llm_by_stage: {{analyze: {name}:gemini-3.1-flash-lite}}"
    )


def resolve_llm(
    name: str, model: str | None = None, cache_dir: Path | None = None
) -> LLMProvider:
    """`model` is the configured `analyze.llm_model`, unless the choice named one
    of its own (see `split_llm_choice`); providers that have no model choice (the
    mock, Codex) ignore it.

    `cache_dir` turns on the replay cache, which is what makes the CLI providers
    reproducible at all — none of them exposes a temperature. None means every
    call goes out.
    """
    name, named_model = split_llm_choice(name)
    model = named_model or model
    provider = _build_llm(name, model)
    if cache_dir is None or name == "mock":
        # The mock is already reproducible, and its fixtures answer a repeated
        # question differently on purpose (the auto-fix loop plans twice); a
        # cache in front of it would replay round one for ever.
        return provider

    from videoai.providers.llm_cache import CachedLLM, caching_enabled

    return CachedLLM(provider, cache_dir) if caching_enabled() else provider


def _build_llm(name: str, model: str | None) -> LLMProvider:
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

        if model:
            _require_gemini_model(name, model)
        return GeminiCliLLM(model or None)
    if name == "gemini_api":
        from videoai.providers.llm_gemini_api import GeminiApiLLM

        if model:
            _require_gemini_model(name, model)
        # The only provider that can be handed the footage itself; see its module
        # docstring for why that costs money and the CLI cannot do it.
        return GeminiApiLLM(model or None)
    raise ValueError(f"unknown llm provider: {name}")
