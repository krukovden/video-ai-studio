from pathlib import Path

from videoai.config import Config
from videoai.core.models import ClipInfo, Manifest, SyncMap
from videoai.core.registry import StageContext
from videoai.core.store import ArtifactStore
from videoai.stages.s02b_sync import sync


def _context(tmp_path: Path) -> StageContext:
    (tmp_path / "work").mkdir(parents=True, exist_ok=True)
    (tmp_path / "output").mkdir(parents=True, exist_ok=True)
    return StageContext(
        project_dir=tmp_path,
        input_dir=tmp_path,
        work_dir=tmp_path / "work",
        output_dir=tmp_path / "output",
        config=Config(),
        store=ArtifactStore(tmp_path / "work"),
    )


def test_sync_stage_places_every_clip_on_the_timeline(tmp_path: Path):
    ctx = _context(tmp_path)
    ctx.store.write("01-manifest", Manifest(clips=[
        ClipInfo(clip_id="clip-01", path="/tmp/a.mov", duration=10.0, width=1920, height=1080,
                 fps=30.0, has_audio=False, camera="main", recorded_at=1000.0),
        ClipInfo(clip_id="clip-02", path="/tmp/b.mov", duration=10.0, width=1920, height=1080,
                 fps=30.0, has_audio=False, camera="main", recorded_at=1020.0),
    ]), fingerprint="fp")

    result = sync(ctx)

    assert isinstance(result, SyncMap)
    assert result.by_id("clip-01").global_start == 0.0
    assert result.by_id("clip-02").global_start == 20.0


def test_sync_stage_tolerates_missing_audio_files(tmp_path: Path):
    ctx = _context(tmp_path)
    ctx.store.write("01-manifest", Manifest(clips=[
        ClipInfo(clip_id="clip-01", path="/tmp/a.mov", duration=10.0, width=1920, height=1080,
                 fps=30.0, has_audio=True, audio_path="/tmp/does-not-exist.wav",
                 camera="cam-a", recorded_at=1000.0),
        ClipInfo(clip_id="clip-02", path="/tmp/b.mov", duration=10.0, width=1920, height=1080,
                 fps=30.0, has_audio=True, audio_path="/tmp/also-missing.wav",
                 camera="cam-b", recorded_at=1000.0),
    ]), fingerprint="fp")

    result = sync(ctx)

    assert len(result.clips) == 2
    assert all(clip.method in {"metadata", "sequential"} for clip in result.clips)
