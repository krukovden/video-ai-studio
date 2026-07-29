from pathlib import Path

from videoai.config import Config
from videoai.core.ffmpeg import probe
from videoai.core.models import ClipInfo, DraftResult, Manifest, Timeline, TimelineClip
from videoai.core.registry import StageContext
from videoai.core.store import ArtifactStore
from videoai.stages.s06_render_draft import render_draft


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


def test_draft_is_rendered_from_two_segments(tmp_path: Path, make_clip):
    ctx = _context(tmp_path)
    source = make_clip("a.mp4", seconds=6.0)
    ctx.store.write("01-manifest", Manifest(clips=[
        ClipInfo(clip_id="clip-01", path=str(source), duration=6.0, width=320,
                 height=240, fps=30.0, has_audio=True, proxy_path=str(source)),
    ]), fingerprint="fp")
    ctx.store.write("05-timeline", Timeline(fps=30.0, width=320, height=240, clips=[
        TimelineClip(src="clip-01", offset=0.5, dur=1.5, start=0.0),
        TimelineClip(src="clip-01", offset=3.0, dur=2.0, start=1.5),
    ]), fingerprint="fp")

    result = render_draft(ctx)

    assert isinstance(result, DraftResult)
    assert result.segment_count == 2
    output = Path(result.path)
    assert output.exists()
    assert 3.0 < probe(output).duration < 4.2


def test_draft_output_lands_in_output_directory(tmp_path: Path, make_clip):
    ctx = _context(tmp_path)
    source = make_clip("a.mp4", seconds=4.0)
    ctx.store.write("01-manifest", Manifest(clips=[
        ClipInfo(clip_id="clip-01", path=str(source), duration=4.0, width=320,
                 height=240, fps=30.0, has_audio=True, proxy_path=str(source)),
    ]), fingerprint="fp")
    ctx.store.write("05-timeline", Timeline(fps=30.0, width=320, height=240, clips=[
        TimelineClip(src="clip-01", offset=0.5, dur=1.0, start=0.0),
    ]), fingerprint="fp")

    result = render_draft(ctx)

    assert Path(result.path).parent == ctx.output_dir
    assert Path(result.path).name == "draft.mp4"


def test_empty_timeline_raises(tmp_path: Path, make_clip):
    ctx = _context(tmp_path)
    source = make_clip("a.mp4", seconds=2.0)
    ctx.store.write("01-manifest", Manifest(clips=[
        ClipInfo(clip_id="clip-01", path=str(source), duration=2.0, width=320,
                 height=240, fps=30.0, has_audio=True, proxy_path=str(source)),
    ]), fingerprint="fp")
    ctx.store.write("05-timeline", Timeline(fps=30.0, width=320, height=240, clips=[]),
                    fingerprint="fp")

    try:
        render_draft(ctx)
    except RuntimeError as error:
        assert "empty timeline" in str(error)
    else:
        raise AssertionError("expected RuntimeError")
