import subprocess
from pathlib import Path

from videoai.config import Config
from videoai.core.models import ClipInfo, Manifest, QualityReport
from videoai.core.registry import StageContext
from videoai.core.store import ArtifactStore
from videoai.stages.s01_ingest import ingest
from videoai.stages.s02_quality import quality


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


def _blurred_clip(path: Path, seconds: float = 2.0) -> Path:
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", f"testsrc=size=320x240:rate=30:duration={seconds}",
            "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
            "-vf", "boxblur=10:1",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
            str(path),
        ],
        check=True,
    )
    return path


def _black_clip(path: Path, seconds: float = 2.0) -> Path:
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", f"color=c=black:size=320x240:rate=30:duration={seconds}",
            "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
            str(path),
        ],
        check=True,
    )
    return path


def test_blurred_clip_scores_higher_blur_than_sharp(tmp_path: Path, make_clip):
    ctx = _context(tmp_path)
    make_clip("a.mp4", seconds=2.0).rename(ctx.input_dir / "a.mp4")
    _blurred_clip(ctx.input_dir / "b.mp4", seconds=2.0)
    ctx.store.write("01-manifest", ingest(ctx), fingerprint="fp")

    report = quality(ctx)

    assert isinstance(report, QualityReport)
    sharp = report.by_id("clip-01")
    blurred = report.by_id("clip-02")
    assert blurred.blur > sharp.blur


def test_black_clip_is_flagged_unusable(tmp_path: Path):
    ctx = _context(tmp_path)
    _black_clip(ctx.input_dir / "a.mp4", seconds=2.0)
    ctx.store.write("01-manifest", ingest(ctx), fingerprint="fp")

    report = quality(ctx)

    assert report.by_id("clip-01").black_ratio > 0.5
    assert report.by_id("clip-01").usable is False


def test_every_clip_appears_in_report(tmp_path: Path, make_clip):
    ctx = _context(tmp_path)
    make_clip("a.mp4", seconds=2.0).rename(ctx.input_dir / "a.mp4")
    make_clip("b.mp4", seconds=2.0).rename(ctx.input_dir / "b.mp4")
    manifest: Manifest = ingest(ctx)
    ctx.store.write("01-manifest", manifest, fingerprint="fp")

    report = quality(ctx)

    assert {c.clip_id for c in report.clips} == {c.clip_id for c in manifest.clips}


def test_scores_are_rebuilt_when_the_proxy_came_from_another_encoder(tmp_path: Path):
    """Every number here is measured off the proxy's pixels, and libx264 and
    VideoToolbox do not produce the same ones. A manifest naming a different
    encoder therefore describes footage these scores were never taken from."""
    import videoai.stages  # noqa: F401  (imports register every stage)
    from videoai.core.registry import REGISTRY
    from videoai.core.runner import _fingerprint

    ctx = _context(tmp_path)
    clip = ClipInfo(
        clip_id="clip-01",
        path="video/a.mp4",
        duration=2.0,
        width=1920,
        height=1080,
        fps=30.0,
        has_audio=True,
        source_key="same-source",
        proxy_path="work/media/a-proxy-720p-libx264.mp4",
        proxy_encoder="libx264",
    )
    ctx.store.write("01-manifest", Manifest(clips=[clip]), fingerprint="fp")
    before = _fingerprint(REGISTRY["quality"], ctx, "media-fp", "brief-fp")

    ctx.store.write(
        "01-manifest",
        Manifest(
            clips=[
                clip.model_copy(
                    update={
                        "proxy_path": "work/media/a-proxy-720p-h264_videotoolbox.mp4",
                        "proxy_encoder": "h264_videotoolbox",
                    }
                )
            ]
        ),
        fingerprint="fp",
    )

    assert _fingerprint(REGISTRY["quality"], ctx, "media-fp", "brief-fp") != before


def test_undecodable_proxy_is_marked_not_scored_not_raised(tmp_path: Path):
    ctx = _context(tmp_path)
    bad_proxy = ctx.work_dir / "garbage.mp4"
    bad_proxy.write_bytes(b"this is not a video file")
    manifest = Manifest(
        clips=[
            ClipInfo(
                clip_id="clip-01",
                path=str(bad_proxy),
                duration=2.0,
                width=320,
                height=240,
                fps=30.0,
                has_audio=False,
                audio_path=None,
                proxy_path=str(bad_proxy),
            )
        ]
    )
    ctx.store.write("01-manifest", manifest, fingerprint="fp")

    report = quality(ctx)

    assert len(report.clips) == 1
    clip = report.by_id("clip-01")
    assert clip.scored is False
    assert clip.usable is False
