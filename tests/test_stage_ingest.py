from pathlib import Path

from videoai.config import Config
from videoai.core.models import Manifest
from videoai.core.registry import StageContext
from videoai.core.store import ArtifactStore
from videoai.stages.s01_ingest import ingest


def _context(project: Path) -> StageContext:
    (project / "work").mkdir(parents=True, exist_ok=True)
    (project / "output").mkdir(parents=True, exist_ok=True)
    return StageContext(
        project_dir=project,
        input_dir=project,
        work_dir=project / "work",
        output_dir=project / "output",
        config=Config(),
        store=ArtifactStore(project / "work"),
    )


def test_ingest_indexes_clips_from_video_folder(tmp_path: Path, make_clip):
    project = tmp_path / "project"
    clips = project / "video"
    clips.mkdir(parents=True)
    make_clip("b.mp4", seconds=2.0).rename(clips / "b.mp4")
    make_clip("a.mp4", seconds=2.0).rename(clips / "a.mp4")

    manifest = ingest(_context(project))

    assert isinstance(manifest, Manifest)
    assert [clip.clip_id for clip in manifest.clips] == ["clip-01", "clip-02"]
    assert [Path(clip.path).name for clip in manifest.clips] == ["a.mp4", "b.mp4"]
    assert all(clip.duration > 1.5 for clip in manifest.clips)


def test_ingest_creates_audio_and_proxy_files(tmp_path: Path, make_clip):
    project = tmp_path / "project"
    clips = project / "video"
    clips.mkdir(parents=True)
    make_clip("a.mp4", seconds=2.0).rename(clips / "a.mp4")

    manifest = ingest(_context(project))

    clip = manifest.clips[0]
    assert Path(clip.audio_path).exists()
    assert Path(clip.proxy_path).exists()


def test_ingest_ignores_non_video_and_macos_metadata(tmp_path: Path, make_clip):
    project = tmp_path / "project"
    clips = project / "video"
    clips.mkdir(parents=True)
    make_clip("a.mp4", seconds=2.0).rename(clips / "a.mp4")
    (clips / ".DS_Store").write_bytes(b"junk")
    (clips / "IMG_8195.JPG").write_bytes(b"\xff\xd8\xff")
    (project / "project.yaml").write_text("title: test\n", encoding="utf-8")

    manifest = ingest(_context(project))

    assert len(manifest.clips) == 1


def test_ingest_is_idempotent_and_reuses_existing_media(tmp_path: Path, make_clip):
    project = tmp_path / "project"
    clips = project / "video"
    clips.mkdir(parents=True)
    make_clip("a.mp4", seconds=2.0).rename(clips / "a.mp4")
    ctx = _context(project)

    first = ingest(ctx)
    proxy = Path(first.clips[0].proxy_path)
    marker = proxy.stat().st_mtime_ns

    second = ingest(ctx)

    assert second.clips[0].proxy_path == first.clips[0].proxy_path
    assert Path(second.clips[0].proxy_path).stat().st_mtime_ns == marker


def test_ingest_raises_when_no_clips_found(tmp_path: Path):
    project = tmp_path / "project"
    (project / "video").mkdir(parents=True)

    try:
        ingest(_context(project))
    except RuntimeError as error:
        assert "no video files" in str(error)
    else:
        raise AssertionError("expected RuntimeError")
