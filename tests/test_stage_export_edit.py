from pathlib import Path

import opentimelineio as otio
import pytest

from videoai.config import Config
from videoai.core.models import ClipInfo, ExportResult, Manifest, Timeline, TimelineClip
from videoai.core.registry import StageContext
from videoai.core.store import ArtifactStore
from videoai.stages.s07_export_edit import export_edit


def _context(tmp_path: Path) -> StageContext:
    for name in ("input", "work", "output"):
        (tmp_path / name).mkdir(exist_ok=True)
    return StageContext(
        project_dir=tmp_path,
        input_dir=tmp_path / "input",
        work_dir=tmp_path / "work",
        output_dir=tmp_path / "output",
        config=Config(),
        store=ArtifactStore(tmp_path / "work"),
    )


def _write_two_clip_fixture(ctx: StageContext, tmp_path: Path) -> None:
    source_a = tmp_path / "a.mov"
    source_a.write_bytes(b"fake video")
    source_b = tmp_path / "b.mov"
    source_b.write_bytes(b"fake video")
    ctx.store.write("01-manifest", Manifest(clips=[
        ClipInfo(clip_id="clip-01", path=str(source_a), duration=10.0, width=3840,
                 height=2160, fps=30.0, has_audio=True, proxy_path=str(tmp_path / "a-proxy.mp4")),
        ClipInfo(clip_id="clip-02", path=str(source_b), duration=10.0, width=3840,
                 height=2160, fps=30.0, has_audio=True, proxy_path=str(tmp_path / "b-proxy.mp4")),
    ]), fingerprint="fp")
    ctx.store.write("05-timeline", Timeline(fps=30.0, width=3840, height=2160, clips=[
        TimelineClip(src="clip-01", offset=1.0, dur=2.0, start=0.0,
                     beat="Hook", quote="hello everyone", reason="opens strong"),
        TimelineClip(src="clip-02", offset=3.0, dur=1.5, start=2.0,
                     beat="Body", is_insert=True, reason="visual insert"),
    ]), fingerprint="fp")


def test_export_writes_otio_and_edl_that_parse_back_with_matching_clips(tmp_path: Path):
    ctx = _context(tmp_path)
    _write_two_clip_fixture(ctx, tmp_path)

    result = export_edit(ctx)

    assert isinstance(result, ExportResult)
    otio_path = Path(result.otio_path)
    edl_path = Path(result.edl_path)
    assert otio_path.exists()
    assert edl_path.exists()

    parsed = otio.adapters.read_from_file(str(otio_path))
    clips = parsed.tracks[0].find_clips()
    assert len(clips) == 2
    assert result.clip_count == 2

    sources = [Path(c.media_reference.target_url).name for c in clips]
    assert sources == ["a.mov", "b.mov"]
    assert clips[0].source_range.start_time.to_seconds() == 1.0
    assert clips[0].source_range.duration.to_seconds() == 2.0
    assert clips[1].source_range.start_time.to_seconds() == 3.0
    assert clips[1].source_range.duration.to_seconds() == 1.5


def test_edl_is_non_empty_and_lists_expected_event_count(tmp_path: Path):
    ctx = _context(tmp_path)
    _write_two_clip_fixture(ctx, tmp_path)

    result = export_edit(ctx)

    edl_text = Path(result.edl_path).read_text(encoding="utf-8")
    assert edl_text.strip()
    event_lines = [
        line for line in edl_text.splitlines()
        if line[:3].strip().isdigit()
    ]
    assert len(event_lines) == 2


def test_media_references_are_absolute_originals_not_proxies(tmp_path: Path):
    ctx = _context(tmp_path)
    _write_two_clip_fixture(ctx, tmp_path)

    result = export_edit(ctx)

    parsed = otio.adapters.read_from_file(result.otio_path)
    for clip in parsed.tracks[0].find_clips():
        target = clip.media_reference.target_url
        assert Path(target).is_absolute()
        assert "proxy" not in target


def test_beat_quote_reason_survive_into_the_otio_document(tmp_path: Path):
    ctx = _context(tmp_path)
    _write_two_clip_fixture(ctx, tmp_path)

    result = export_edit(ctx)

    parsed = otio.adapters.read_from_file(result.otio_path)
    first, second = parsed.tracks[0].find_clips()
    assert first.metadata["videoai"]["beat"] == "Hook"
    assert first.metadata["videoai"]["quote"] == "hello everyone"
    assert first.metadata["videoai"]["reason"] == "opens strong"
    assert first.markers and "Hook" in first.markers[0].comment
    assert second.metadata["videoai"]["is_insert"] is True


def test_missing_source_file_is_recorded_not_raised(tmp_path: Path):
    ctx = _context(tmp_path)
    source_a = tmp_path / "a.mov"
    source_a.write_bytes(b"fake video")
    missing_path = tmp_path / "gone.mov"
    ctx.store.write("01-manifest", Manifest(clips=[
        ClipInfo(clip_id="clip-01", path=str(source_a), duration=10.0, width=3840,
                 height=2160, fps=30.0, has_audio=True),
        ClipInfo(clip_id="clip-02", path=str(missing_path), duration=10.0, width=3840,
                 height=2160, fps=30.0, has_audio=True),
    ]), fingerprint="fp")
    ctx.store.write("05-timeline", Timeline(fps=30.0, width=3840, height=2160, clips=[
        TimelineClip(src="clip-01", offset=0.0, dur=1.0, start=0.0),
        TimelineClip(src="clip-02", offset=0.0, dur=1.0, start=1.0),
    ]), fingerprint="fp")

    result = export_edit(ctx)

    assert str(missing_path.resolve()) in result.missing_media
    parsed = otio.adapters.read_from_file(result.otio_path)
    assert len(parsed.tracks[0].find_clips()) == 2


def test_artifact_records_both_output_paths(tmp_path: Path):
    ctx = _context(tmp_path)
    _write_two_clip_fixture(ctx, tmp_path)

    result = export_edit(ctx)

    assert Path(result.otio_path) == ctx.output_dir / "edit.otio"
    assert Path(result.edl_path) == ctx.output_dir / "edit.edl"


def test_empty_timeline_raises(tmp_path: Path):
    ctx = _context(tmp_path)
    source_a = tmp_path / "a.mov"
    source_a.write_bytes(b"fake video")
    ctx.store.write("01-manifest", Manifest(clips=[
        ClipInfo(clip_id="clip-01", path=str(source_a), duration=10.0, width=3840,
                 height=2160, fps=30.0, has_audio=True),
    ]), fingerprint="fp")
    ctx.store.write("05-timeline", Timeline(fps=30.0, width=3840, height=2160, clips=[]),
                    fingerprint="fp")

    with pytest.raises(RuntimeError, match="empty timeline"):
        export_edit(ctx)


def test_missing_cmx_3600_adapter_fails_with_a_clear_message_naming_the_package(
    tmp_path: Path, monkeypatch
):
    """Free Resolve is never scripted here; the only external dependency this
    stage has is the OTIO adapter plugin for the EDL fallback. If it is not
    installed the failure must name exactly what to `uv add`, not surface a
    bare adapter-not-found error from deep inside OTIO."""
    import videoai.stages.s07_export_edit as export_edit_module

    class _EmptyManifest:
        adapters: list = []

    monkeypatch.setattr(
        export_edit_module.otio.plugins, "ActiveManifest", lambda: _EmptyManifest()
    )

    ctx = _context(tmp_path)
    _write_two_clip_fixture(ctx, tmp_path)

    with pytest.raises(RuntimeError, match="otio-cmx3600-adapter"):
        export_edit(ctx)
