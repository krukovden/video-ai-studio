"""Finding I1: a stage's fingerprint must include the settings and prompt text it
actually reads, or changing one leaves the stage cached against an input it no
longer matches (`sync.primary_camera` was the worst case: transcription kept using
the wrong camera forever)."""
import dataclasses
from pathlib import Path

import pytest

import videoai.stages  # noqa: F401  (imports register every stage)
from videoai.config import (
    AnalyzeSettings,
    Config,
    EffectsSettings,
    PlanSettings,
    PolishSettings,
    RenderSettings,
    SyncSettings,
    TranscribeSettings,
)
from videoai.core.registry import REGISTRY, StageContext
from videoai.core.runner import _fingerprint, config_value
from videoai.core.store import ArtifactStore
from videoai.core.models import ClipInfo, Manifest, SyncMap


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
        (
            Config(transcribe=TranscribeSettings(chunk_duration_seconds=60.0)),
            {"transcribe"},
        ),
        (
            Config(transcribe=TranscribeSettings(overlap_duration_seconds=5.0)),
            {"transcribe"},
        ),
        (Config(analyze=AnalyzeSettings(keyframes_per_phrase=0)), {"analyze"}),
        (Config(analyze=AnalyzeSettings(max_keyframes=10)), {"analyze"}),
        (
            Config(analyze=AnalyzeSettings(llm_model="opus")),
            {"analyze", "plan", "visual_check", "effects"},
        ),
        (
            Config(analyze=AnalyzeSettings(insert_max_words_per_second=0.2)),
            {"analyze"},
        ),
        (
            Config(analyze=AnalyzeSettings(describe_inserts=False)),
            {"analyze"},
        ),
        # Both stages cut their own segments now — the draft from the proxies and
        # the final from the originals — so both fade at each cut.
        (Config(render=RenderSettings(audio_fade_seconds=0.5)), {"render_draft", "polish"}),
        (Config(render=RenderSettings(draft_crf=30)), {"render_draft"}),
        (Config(plan=PlanSettings(exclude_phrases=["clip-01#004"])), {"plan"}),
        (Config(plan=PlanSettings(gain_db_by_beat={"Popping": -6.0})), {"plan"}),
        (Config(plan=PlanSettings(reject_adult_in_frame=False)), {"visual_check"}),
        (Config(plan=PlanSettings(reject_unusable_shots=False)), {"visual_check"}),
        # Effects are chosen by one stage and drawn by another, so switching them
        # off has to re-run both: the delivery has them composited into its picture.
        (Config(effects=EffectsSettings(enabled=False)), {"effects", "polish"}),
        (Config(effects=EffectsSettings(max_events=3)), {"effects"}),
        (Config(polish=PolishSettings(enabled=False)), {"polish"}),
        (Config(polish=PolishSettings(require_approval=True)), {"polish"}),
        (Config(polish=PolishSettings(intro_seconds=4.0)), {"polish"}),
        (Config(polish=PolishSettings(outro_seconds=4.0)), {"polish"}),
        (Config(polish=PolishSettings(outro_text="Subscribe!")), {"polish"}),
        (Config(polish=PolishSettings(title_seconds=3.0)), {"polish"}),
        (Config(polish=PolishSettings(captions_enabled=False)), {"polish"}),
        (Config(polish=PolishSettings(burn_captions=True)), {"polish"}),
        (Config(polish=PolishSettings(caption_words=6)), {"polish"}),
        (Config(polish=PolishSettings(music_gain_db=-18.0)), {"polish"}),
        (Config(polish=PolishSettings(music_duck_db=-6.0)), {"polish"}),
        (Config(polish=PolishSettings(transition_frames=2)), {"polish"}),
        (Config(polish=PolishSettings(music_track="bensound-energy.mp3")), {"polish"}),
        (Config(polish=PolishSettings(music_dir="assets/Other")), {"polish"}),
        # Delivery quality. These re-render the final and must touch nothing
        # upstream: the draft is a review copy and is not affected by them.
        (Config(polish=PolishSettings(output_height=2160)), {"polish"}),
        (Config(polish=PolishSettings(output_crf=14)), {"polish"}),
        (Config(polish=PolishSettings(lossless_intermediates=False)), {"polish"}),
        (Config(polish=PolishSettings(hardware_encode=False)), {"polish"}),
    ],
)
def test_changing_a_setting_invalidates_exactly_the_stages_that_read_it(
    tmp_path: Path, config: Config, expected: set[str]
):
    assert _stages_invalidated_by(tmp_path, config) == expected


def test_an_unrelated_setting_invalidates_nothing(tmp_path: Path):
    # The timeout does not change any stage's output, only how long it waits.
    assert _stages_invalidated_by(tmp_path, Config(analyze=AnalyzeSettings(llm_timeout_seconds=1))) == set()


@pytest.mark.parametrize("stage_id", ["analyze", "plan", "visual_check", "effects"])
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


def test_editing_the_production_contract_reruns_delivery(tmp_path: Path):
    """The contract defines what a valid `final.mp4` is, so a cached one that was
    validated against different rules is stale."""
    ctx = _context(tmp_path, Config())
    spec = REGISTRY["polish"]
    contract = tmp_path / "production-contract.yaml"
    contract.write_text(
        "version: 1\nrequired_output: {width: 1920, height: 1080}\n", encoding="utf-8"
    )
    before = _fingerprint(spec, ctx, "media-fp", "brief-fp")

    contract.write_text(
        "version: 1\nrequired_output: {width: 3840, height: 2160}\n", encoding="utf-8"
    )

    assert _fingerprint(spec, ctx, "media-fp", "brief-fp") != before


def _transcribe_context(tmp_path: Path) -> tuple[StageContext, ClipInfo]:
    ctx = _context(tmp_path, Config())
    clip = ClipInfo(
        clip_id="clip-01",
        path="original.mov",
        duration=10.0,
        width=1920,
        height=1080,
        fps=30.0,
        has_audio=True,
        camera="main",
        source_key="same-source",
        audio_path="audio.wav",
        proxy_path="proxy-540p.mp4",
    )
    ctx.store.write("01-manifest", Manifest(clips=[clip]), fingerprint="manifest")
    ctx.store.write("01b-sync", SyncMap(primary_camera="main"), fingerprint="sync")
    return ctx, clip


# Every field `transcribe` reads, and what changing it must do to the projection
# that stands in for the whole of 01-manifest and 01b-sync.
@pytest.mark.parametrize(
    "update, reruns",
    [
        ({"clip_id": "clip-02"}, True),
        ({"camera": "cam-b"}, True),
        ({"source_key": "a-different-file"}, True),
        # `audio_path` is read as well, but only for whether it is set, and ingest
        # sets it if and only if `has_audio` — so `has_audio` stands for it and the
        # disposable work/ path itself must not be an input.
        ({"has_audio": False, "audio_path": ""}, True),
        ({"audio_path": "/somewhere/else/audio.wav"}, False),
        ({"duration": 11.0}, True),
        # Read by nothing in this stage: the proxy is the draft's business.
        ({"proxy_path": "proxy-720p.mp4"}, False),
        ({"width": 1280, "height": 720}, False),
    ],
)
def test_the_transcribe_projection_covers_every_field_the_stage_reads(
    tmp_path: Path, update: dict, reruns: bool
):
    """A projection replaces the required artifacts' content hashes, so a field the
    stage reads but the projection omits leaves transcription cached against input
    it no longer matches — the bug that kept the wrong camera forever."""
    ctx, clip = _transcribe_context(tmp_path)
    before = _fingerprint(REGISTRY["transcribe"], ctx, "media-fp", "brief-fp")

    ctx.store.write(
        "01-manifest",
        Manifest(clips=[clip.model_copy(update=update)]),
        fingerprint="changed",
    )
    after = _fingerprint(REGISTRY["transcribe"], ctx, "media-fp", "brief-fp")

    assert (after != before) is reruns, update


def test_the_transcribe_projection_includes_the_sync_maps_primary_camera(tmp_path: Path):
    """`transcribe` transcribes the primary camera's clips and stubs the rest, so a
    re-detected primary camera has to re-transcribe — and the projection is all the
    fingerprint sees of `01b-sync`."""
    ctx, _ = _transcribe_context(tmp_path)
    before = _fingerprint(REGISTRY["transcribe"], ctx, "media-fp", "brief-fp")

    ctx.store.write("01b-sync", SyncMap(primary_camera="cam-b"), fingerprint="sync")

    assert _fingerprint(REGISTRY["transcribe"], ctx, "media-fp", "brief-fp") != before


def test_editing_a_providers_system_preamble_invalidates_the_stages_that_send_it(
    tmp_path: Path, monkeypatch
):
    """Codex prepends a fixed instruction to every prompt. It is part of what the
    model was asked, so it belongs in the fingerprint next to `spec.prompt`."""
    from videoai.providers import llm_codex_cli

    ctx = _context(tmp_path, Config(providers={"asr": "mock", "llm": "codex_cli"}))
    before = {
        stage_id: _fingerprint(REGISTRY[stage_id], ctx, "media-fp", "brief-fp")
        for stage_id in ("analyze", "plan", "visual_check")
    }

    monkeypatch.setattr(
        llm_codex_cli, "SYSTEM_PROMPT", llm_codex_cli.SYSTEM_PROMPT + " Be terse."
    )

    for stage_id, value in before.items():
        assert _fingerprint(REGISTRY[stage_id], ctx, "media-fp", "brief-fp") != value, stage_id


def test_transcription_fingerprint_ignores_disposable_proxy_path(tmp_path: Path):
    ctx = _context(tmp_path, Config())
    clip = ClipInfo(
        clip_id="clip-01",
        path="original.mov",
        duration=10.0,
        width=1920,
        height=1080,
        fps=30.0,
        has_audio=True,
        source_key="same-source",
        audio_path="audio.wav",
        proxy_path="proxy-540p.mp4",
    )
    ctx.store.write("01-manifest", Manifest(clips=[clip]), fingerprint="old")
    ctx.store.write("01b-sync", SyncMap(primary_camera="main"), fingerprint="sync")
    before = _fingerprint(REGISTRY["transcribe"], ctx, "media-fp", "brief-fp")

    ctx.store.write(
        "01-manifest",
        Manifest(clips=[clip.model_copy(update={"proxy_path": "proxy-720p.mp4"})]),
        fingerprint="new",
    )
    after = _fingerprint(REGISTRY["transcribe"], ctx, "media-fp", "brief-fp")

    assert after == before
