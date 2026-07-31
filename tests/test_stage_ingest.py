import json
import os
import shutil
import subprocess
from pathlib import Path

import videoai.core.ffmpeg as ffmpeg
import videoai.stages.s01_ingest as s01_ingest
from videoai.config import Config, RenderSettings
from videoai.core.ffmpeg import proxy_encoder
from videoai.core.models import Manifest
from videoai.core.registry import StageContext
from videoai.core.store import ArtifactStore
from videoai.stages.s01_ingest import _proxy_filename, ingest


def _context(project: Path, *, draft_height: int | None = None) -> StageContext:
    (project / "work").mkdir(parents=True, exist_ok=True)
    (project / "output").mkdir(parents=True, exist_ok=True)
    config = Config() if draft_height is None else Config(render=RenderSettings(draft_height=draft_height))
    return StageContext(
        project_dir=project,
        input_dir=project,
        work_dir=project / "work",
        output_dir=project / "output",
        config=config,
        store=ArtifactStore(project / "work"),
    )


def _video_height(path: Path) -> int:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=height", "-print_format", "json", str(path),
        ],
        capture_output=True, text=True, check=True,
    )
    return json.loads(result.stdout)["streams"][0]["height"]


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


def test_ingest_handles_clip_without_audio_stream(tmp_path: Path):
    project = tmp_path / "project"
    clips = project / "video"
    clips.mkdir(parents=True)
    video_only = clips / "a.mp4"
    # make_clip always adds a tone track, so build a video-only source directly.
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", "testsrc=size=320x240:rate=30:duration=2",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            str(video_only),
        ],
        check=True,
    )

    manifest = ingest(_context(project))

    clip = manifest.clips[0]
    assert clip.has_audio is False
    assert clip.audio_path is None
    assert Path(clip.proxy_path).exists()


def test_audio_path_is_set_exactly_when_the_clip_has_audio(tmp_path: Path, make_clip):
    """`transcribe`'s fingerprint projection leaves `audio_path` out and relies on
    `has_audio` standing for it. That is only sound while this invariant holds, so
    it is asserted here rather than assumed there."""
    project = tmp_path / "project"
    clips = project / "video"
    clips.mkdir(parents=True)
    make_clip("with-audio.mp4", seconds=1.0).rename(clips / "with-audio.mp4")
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", "testsrc=size=320x240:rate=30:duration=1",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", str(clips / "silent.mp4"),
        ],
        check=True,
    )

    manifest = ingest(_context(project))

    assert len(manifest.clips) == 2
    for clip in manifest.clips:
        assert bool(clip.audio_path) is clip.has_audio, clip.clip_id


def test_ingest_raises_when_no_clips_found(tmp_path: Path):
    project = tmp_path / "project"
    (project / "video").mkdir(parents=True)

    try:
        ingest(_context(project))
    except RuntimeError as error:
        assert "no video files" in str(error)
    else:
        raise AssertionError("expected RuntimeError")


def _media_duration(path: Path) -> float:
    """Duration of any media file, including the audio-only WAVs (`probe` insists
    on a video stream)."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(json.loads(result.stdout)["format"]["duration"])


# --- Finding C1: derived media must be keyed by source identity, not by position ---


def test_derived_media_follows_its_source_when_a_clip_sorts_in_front(tmp_path: Path, make_clip):
    """Adding a clip that sorts first renumbers every clip id. Positional cache
    names would then serve the first clip's audio and proxy under the second
    clip's entry, and ASR would transcribe the wrong footage."""
    project = tmp_path / "project"
    clips = project / "video"
    clips.mkdir(parents=True)
    make_clip("b_second.mp4", seconds=4.0, tone_hz=440).rename(clips / "b_second.mp4")
    ctx = _context(project)

    first = ingest(ctx)
    assert [Path(clip.path).name for clip in first.clips] == ["b_second.mp4"]

    make_clip("a_first.mp4", seconds=1.5, tone_hz=880).rename(clips / "a_first.mp4")
    second = ingest(ctx)

    assert [clip.clip_id for clip in second.clips] == ["clip-01", "clip-02"]
    assert [Path(clip.path).name for clip in second.clips] == ["a_first.mp4", "b_second.mp4"]
    for clip in second.clips:
        source_duration = _media_duration(Path(clip.path))
        assert abs(_media_duration(Path(clip.audio_path)) - source_duration) < 0.3, clip.path
        assert abs(_media_duration(Path(clip.proxy_path)) - source_duration) < 0.3, clip.path


def test_derived_media_is_rebuilt_when_the_source_file_changes(tmp_path: Path, make_clip):
    project = tmp_path / "project"
    clips = project / "video"
    clips.mkdir(parents=True)
    make_clip("a.mp4", seconds=4.0).rename(clips / "a.mp4")
    ctx = _context(project)

    first = ingest(ctx)

    # Same name, different footage: a re-export of the same take.
    (clips / "a.mp4").unlink()
    make_clip("a.mp4", seconds=1.5).rename(clips / "a.mp4")
    second = ingest(ctx)

    assert second.clips[0].proxy_path != first.clips[0].proxy_path
    assert abs(_media_duration(Path(second.clips[0].proxy_path)) - 1.5) < 0.3


def test_ingest_records_a_stable_source_key_per_clip(tmp_path: Path, make_clip):
    project = tmp_path / "project"
    clips = project / "video"
    clips.mkdir(parents=True)
    make_clip("a.mp4", seconds=2.0).rename(clips / "a.mp4")
    make_clip("b.mp4", seconds=3.0).rename(clips / "b.mp4")
    ctx = _context(project)

    manifest = ingest(ctx)

    keys = [clip.source_key for clip in manifest.clips]
    assert all(keys)
    assert len(set(keys)) == 2
    assert [clip.source_key for clip in ingest(ctx).clips] == keys


# --- Finding 1: the proxy filename must fold in its build height, or a height
# change reuses the old, wrongly-sized proxy file and the pipeline never re-runs ---


def test_proxy_is_rebuilt_when_draft_height_changes_but_audio_path_is_stable(
    tmp_path: Path, make_clip
):
    project = tmp_path / "project"
    clips = project / "video"
    clips.mkdir(parents=True)
    make_clip("a.mp4", seconds=2.0, size="640x360").rename(clips / "a.mp4")

    first = ingest(_context(project, draft_height=240))
    assert _video_height(Path(first.clips[0].proxy_path)) == 240

    second = ingest(_context(project, draft_height=480))

    assert _video_height(Path(second.clips[0].proxy_path)) == 480
    assert second.clips[0].proxy_path != first.clips[0].proxy_path
    assert second.clips[0].audio_path == first.clips[0].audio_path


# --- a cached proxy is only the right file for the encoder that wrote it ---


def test_ingest_records_the_encoder_that_built_each_proxy(tmp_path: Path, make_clip):
    project = tmp_path / "project"
    clips = project / "video"
    clips.mkdir(parents=True)
    make_clip("a.mp4", seconds=1.0).rename(clips / "a.mp4")

    clip = ingest(_context(project)).clips[0]

    assert clip.proxy_encoder == proxy_encoder()
    assert clip.proxy_encoder in Path(clip.proxy_path).name


def test_proxy_filename_separates_the_two_encoders():
    assert _proxy_filename("a-key", 720, "libx264") != _proxy_filename(
        "a-key", 720, "h264_videotoolbox"
    )


def _pin_encoder(monkeypatch, name: str) -> list[str]:
    """Pin the encoder ingest believes it has, and record what it asked to build.

    The pixels stay software-encoded whichever name is pinned, so the
    VideoToolbox case runs on a machine whose media engine cannot open a session
    — what is under test is the cache key, not the encoder.
    """
    requested: list[str] = []
    monkeypatch.setattr(s01_ingest, "proxy_encoder", lambda: name)

    def build(src: Path, dst: Path, height: int, encoder: str) -> str:
        requested.append(encoder)
        return ffmpeg.make_proxy(src, dst, height, encoder="libx264")

    monkeypatch.setattr(s01_ingest, "make_proxy", build)
    return requested


def test_a_proxy_built_by_one_encoder_is_not_reused_by_the_other(
    tmp_path: Path, make_clip, monkeypatch
):
    """A run while the media engine was busy writes libx264 pixels. A later
    VideoToolbox run reusing them re-scores every clip's blur off frames its own
    encoder never produced, and the analyze prompt — so the edit — moves."""
    project = tmp_path / "project"
    clips = project / "video"
    clips.mkdir(parents=True)
    make_clip("a.mp4", seconds=1.0).rename(clips / "a.mp4")
    ctx = _context(project)

    _pin_encoder(monkeypatch, "libx264")
    first = ingest(ctx)

    requested = _pin_encoder(monkeypatch, "h264_videotoolbox")
    second = ingest(ctx)

    assert requested == ["h264_videotoolbox"]
    assert second.clips[0].proxy_path != first.clips[0].proxy_path
    assert second.clips[0].proxy_encoder == "h264_videotoolbox"
    # The software run's proxy stays where it is, ready for the next busy session.
    assert Path(first.clips[0].proxy_path).exists()


def test_the_same_encoder_still_reuses_its_own_proxy(
    tmp_path: Path, make_clip, monkeypatch
):
    project = tmp_path / "project"
    clips = project / "video"
    clips.mkdir(parents=True)
    make_clip("a.mp4", seconds=1.0).rename(clips / "a.mp4")
    ctx = _context(project)

    _pin_encoder(monkeypatch, "libx264")
    first = ingest(ctx)
    marker = Path(first.clips[0].proxy_path).stat().st_mtime_ns

    requested = _pin_encoder(monkeypatch, "libx264")
    second = ingest(ctx)

    assert requested == []
    assert Path(second.clips[0].proxy_path).stat().st_mtime_ns == marker


# --- Finding: derived media must survive a transfer and a move, or every one of
# them is rebuilt and every metered model call is paid for twice ---


def test_derived_media_is_reused_after_a_transfer_restamps_mtime(
    tmp_path: Path, make_clip
):
    project = tmp_path / "project"
    clips = project / "video"
    clips.mkdir(parents=True)
    make_clip("a.mp4", seconds=1.0).rename(clips / "a.mp4")
    ctx = _context(project)

    first = ingest(ctx)
    marker = Path(first.clips[0].proxy_path).stat().st_mtime_ns

    # What an rsync, a restore or a cloud sync leaves behind: same bytes, new date.
    os.utime(clips / "a.mp4", (1_600_000_000, 1_600_000_000))
    second = ingest(ctx)

    assert second.clips[0].source_key == first.clips[0].source_key
    assert second.clips[0].proxy_path == first.clips[0].proxy_path
    assert Path(second.clips[0].proxy_path).stat().st_mtime_ns == marker


def test_derived_media_is_reused_after_the_project_folder_moves(
    tmp_path: Path, make_clip
):
    """`work/` travels with the project, so a move must leave every proxy and every
    source key exactly as they were — only the folder above them changed."""
    first_home = tmp_path / "Desktop" / "toy-review"
    (first_home / "video").mkdir(parents=True)
    make_clip("a.mp4", seconds=1.0).rename(first_home / "video" / "a.mp4")

    first = ingest(_context(first_home))
    marker = Path(first.clips[0].proxy_path).stat().st_mtime_ns

    second_home = tmp_path / "Archive" / "toy-review"
    second_home.parent.mkdir(parents=True)
    shutil.move(str(first_home), str(second_home))
    second = ingest(_context(second_home))

    assert second.clips[0].source_key == first.clips[0].source_key
    assert Path(second.clips[0].proxy_path).name == Path(first.clips[0].proxy_path).name
    assert Path(second.clips[0].proxy_path).stat().st_mtime_ns == marker


def test_ingest_labels_clips_with_their_camera(tmp_path: Path, make_clip):
    project = tmp_path / "project"
    (project / "video" / "cam-a").mkdir(parents=True)
    (project / "video" / "cam-b").mkdir(parents=True)
    make_clip("a.mp4", seconds=2.0).rename(project / "video" / "cam-a" / "a.mp4")
    make_clip("b.mp4", seconds=2.0).rename(project / "video" / "cam-b" / "b.mp4")

    manifest = ingest(_context(project))

    assert {clip.camera for clip in manifest.clips} == {"cam-a", "cam-b"}
    assert [clip.clip_id for clip in manifest.clips] == ["clip-01", "clip-02"]
