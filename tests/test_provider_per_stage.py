"""Choosing a different model for one stage without touching the others.

Models are interchangeable editorial voices, and they are not equally worth
paying for at every step. Watching the footage is worth a metered call; naming a
section title is not. So the provider is selectable per stage, and swapping one
out is a line of YAML rather than a code change.
"""
from __future__ import annotations

from pathlib import Path

import pytest

import videoai.stages  # noqa: F401  (registers every stage)
from videoai.config import Config, load_config
from videoai.core.registry import REGISTRY, StageContext
from videoai.core.runner import _fingerprint
from videoai.core.store import ArtifactStore


def _context(tmp_path: Path, config: Config) -> StageContext:
    work = tmp_path / "work"
    return StageContext(
        project_dir=tmp_path, input_dir=tmp_path, work_dir=work,
        output_dir=tmp_path / "output", config=config, store=ArtifactStore(work),
    )


def test_without_an_override_every_stage_uses_the_default():
    config = Config(providers={"asr": "parakeet", "llm": "claude_cli"})
    for stage_id in ("analyze", "plan", "visual_check"):
        assert config.llm_for(stage_id) == "claude_cli"


def test_an_override_applies_to_one_stage_only():
    config = Config(
        providers={"asr": "parakeet", "llm": "claude_cli"},
        llm_by_stage={"analyze": "gemini_api"},
    )
    assert config.llm_for("analyze") == "gemini_api"
    assert config.llm_for("plan") == "claude_cli"


def test_it_reads_from_yaml(tmp_path: Path):
    path = tmp_path / "config.yaml"
    path.write_text(
        "providers:\n  asr: parakeet\n  llm: claude_cli\n"
        "llm_by_stage:\n  analyze: gemini_api\n",
        encoding="utf-8",
    )
    config = load_config(path)
    assert config.llm_for("analyze") == "gemini_api"
    assert config.llm_for("visual_check") == "claude_cli"


def test_an_unknown_stage_name_is_refused():
    """A typo would otherwise be a silent no-op: the stage keeps the default and
    the creator believes they are paying for video where they are not."""
    with pytest.raises(ValueError, match="unknown stage"):
        Config(llm_by_stage={"analize": "gemini_api"})


def test_a_stage_with_no_model_call_cannot_be_overridden():
    with pytest.raises(ValueError, match="does not call a model"):
        Config(llm_by_stage={"ingest": "gemini_api"})


def test_overriding_one_stage_invalidates_only_that_stage(tmp_path: Path):
    base = _context(tmp_path, Config())
    overridden = _context(
        tmp_path, Config(llm_by_stage={"analyze": "gemini_api"})
    )
    moved = {
        spec.id
        for spec in REGISTRY.values()
        if _fingerprint(spec, base, "media", "brief")
        != _fingerprint(spec, overridden, "media", "brief")
    }
    assert moved == {"analyze"}


def test_a_stage_may_name_the_model_as_well_as_the_provider(tmp_path: Path):
    """There is one `analyze.llm_model` and the providers are not one family: a
    config that watches the footage with Gemini while the cheap stages stay on
    the Claude CLI has to say which model each of them means."""
    from videoai.providers.base import resolve_llm, resolved_llm_model

    path = tmp_path / "config.yaml"
    path.write_text(
        "providers:\n  asr: parakeet\n  llm: claude_cli\n"
        "llm_by_stage:\n  analyze: gemini_api:gemini-3.1-flash-lite\n"
        "analyze:\n  llm_model: sonnet\n",
        encoding="utf-8",
    )
    config = load_config(path)

    analyst = resolve_llm(config.llm_for("analyze"), config.analyze.llm_model)
    assert (analyst.name, analyst.model) == ("gemini_api", "gemini-3.1-flash-lite")
    assert resolve_llm(config.llm_for("plan"), config.analyze.llm_model).model == "sonnet"
    assert resolved_llm_model(config.llm_for("plan"), config.analyze.llm_model) == "sonnet"


def test_pointing_a_stage_at_gemini_without_a_gemini_model_is_refused(tmp_path: Path):
    """The shipped default is `llm_model: sonnet`, so this is what most people
    hit first: it used to answer as gemini_api's own default and say nothing."""
    from videoai.providers.base import resolve_llm

    config = Config(llm_by_stage={"analyze": "gemini_api"})

    with pytest.raises(ValueError, match="not a Gemini model"):
        resolve_llm(config.llm_for("analyze"), config.analyze.llm_model)


def test_every_llm_stage_can_be_named():
    """Whatever the pipeline grows, every model-calling stage stays swappable."""
    config = Config()
    for spec in REGISTRY.values():
        if spec.provider_key == "llm":
            assert config.llm_for(spec.id) == config.providers["llm"]
            # ...and naming it in the override is accepted.
            Config(llm_by_stage={spec.id: "mock"})
