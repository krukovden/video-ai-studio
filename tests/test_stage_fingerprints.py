"""Finding I1: a stage's fingerprint must include the settings and prompt text it
actually reads, or changing one leaves the stage cached against an input it no
longer matches (`sync.primary_camera` was the worst case: transcription kept using
the wrong camera forever)."""
import dataclasses
from pathlib import Path

import pytest

import videoai.stages  # noqa: F401  (imports register every stage)
from videoai.config import AnalyzeSettings, Config, RenderSettings, SyncSettings, TranscribeSettings
from videoai.core.registry import REGISTRY, StageContext
from videoai.core.runner import _fingerprint, config_value
from videoai.core.store import ArtifactStore


def _context(tmp_path: Path, config: Config) -> StageContext:
    work = tmp_path / "work"
    return StageContext(
        project_dir=tmp_path,
        input_dir=tmp_path,
        work_dir=work,
        output_dir=tmp_path / "output",
        config=config,
        store=ArtifactStore(work),
    )


def _fingerprints(tmp_path: Path, config: Config) -> dict[str, str]:
    ctx = _context(tmp_path, config)
    return {
        spec.id: _fingerprint(spec, ctx, "media-fp", "brief-fp") for spec in REGISTRY.values()
    }


def _stages_invalidated_by(tmp_path: Path, config: Config) -> set[str]:
    """Which stages a config change invalidates directly. The store is empty, so
    no artifact content is in play and only declared settings can differ."""
    base = _fingerprints(tmp_path, Config())
    changed = _fingerprints(tmp_path, config)
    return {stage_id for stage_id, value in base.items() if changed[stage_id] != value}


@pytest.mark.parametrize(
    "config, expected",
    [
        (Config(render=RenderSettings(draft_height=480)), {"ingest"}),
        (Config(sync=SyncSettings(primary_camera="cam-b")), {"sync"}),
        (
            Config(transcribe=TranscribeSettings(phrase_gap_seconds=0.9)),
            {"transcribe", "analyze"},
        ),
        (Config(transcribe=TranscribeSettings(max_words_per_phrase=12)), {"analyze"}),
        (Config(transcribe=TranscribeSettings(cut_padding_seconds=0.4)), {"plan"}),
        (Config(analyze=AnalyzeSettings(keyframes_per_phrase=0)), {"analyze"}),
        (Config(analyze=AnalyzeSettings(max_keyframes=10)), {"analyze"}),
        (Config(analyze=AnalyzeSettings(llm_model="opus")), {"analyze", "plan"}),
        (Config(render=RenderSettings(audio_fade_seconds=0.5)), {"render_draft"}),
        (Config(render=RenderSettings(draft_crf=30)), {"render_draft"}),
    ],
)
def test_changing_a_setting_invalidates_exactly_the_stages_that_read_it(
    tmp_path: Path, config: Config, expected: set[str]
):
    assert _stages_invalidated_by(tmp_path, config) == expected


def test_an_unrelated_setting_invalidates_nothing(tmp_path: Path):
    # The timeout does not change any stage's output, only how long it waits.
    assert _stages_invalidated_by(tmp_path, Config(analyze=AnalyzeSettings(llm_timeout_seconds=1))) == set()


@pytest.mark.parametrize("stage_id", ["analyze", "plan"])
def test_editing_a_prompt_invalidates_its_stage(tmp_path: Path, stage_id: str):
    ctx = _context(tmp_path, Config())
    spec = REGISTRY[stage_id]
    assert spec.prompt, f"{stage_id} must declare the prompt it sends"
    edited = dataclasses.replace(spec, prompt=spec.prompt + "\n- and one more rule")

    assert _fingerprint(spec, ctx, "media-fp", "brief-fp") != _fingerprint(
        edited, ctx, "media-fp", "brief-fp"
    )


def test_every_declared_config_key_exists(tmp_path: Path):
    for spec in REGISTRY.values():
        for key in spec.config_keys:
            config_value(Config(), key)


def test_unknown_config_key_is_rejected():
    with pytest.raises(KeyError, match="unknown config key"):
        config_value(Config(), "render.nope")
