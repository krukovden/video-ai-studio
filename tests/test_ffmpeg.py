import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pytest

from videoai.core.ffmpeg import (
    extract_audio,
    extract_frame,
    list_video_files,
    make_proxy,
    probe,
)


def test_probe_reads_stream_properties(make_clip):
    clip = make_clip("a.mp4", seconds=3.0)
    result = probe(clip)
    assert 2.8 < result.duration < 3.3
    assert (result.width, result.height) == (320, 240)
    assert 29.0 < result.fps < 31.0
    assert result.has_audio is True


def test_probe_created_at_is_none_when_creation_time_absent(make_clip):
    clip = make_clip("a.mp4", seconds=1.0)
    assert probe(clip).created_at is None


def test_probe_reads_creation_time_tag_as_epoch_seconds(make_clip, tmp_path: Path):
    clip = make_clip("a.mp4", seconds=1.0)
    tagged = tmp_path / "tagged.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error", "-i", str(clip),
            "-c", "copy", "-metadata", "creation_time=2024-01-15T10:30:00Z",
            str(tagged),
        ],
        check=True,
    )

    result = probe(tagged)

    expected = datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc).timestamp()
    assert result.created_at == pytest.approx(expected)


def test_probe_creation_time_that_does_not_parse_is_none(make_clip, tmp_path: Path):
    clip = make_clip("a.mp4", seconds=1.0)
    tagged = tmp_path / "tagged.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error", "-i", str(clip),
            "-c", "copy", "-metadata", "creation_time=not-a-timestamp",
            str(tagged),
        ],
        check=True,
    )

    assert probe(tagged).created_at is None


def test_extract_audio_writes_wav(make_clip, tmp_path: Path):
    clip = make_clip("a.mp4", seconds=2.0)
    wav = tmp_path / "out" / "a.wav"
    extract_audio(clip, wav)
    assert wav.exists() and wav.stat().st_size > 1000


def test_make_proxy_scales_height(make_clip, tmp_path: Path):
    clip = make_clip("a.mp4", seconds=2.0, size="640x480")
    proxy = tmp_path / "out" / "a-proxy.mp4"
    make_proxy(clip, proxy, height=240)
    assert probe(proxy).height == 240


def test_make_proxy_keeps_audio(make_clip, tmp_path: Path):
    clip = make_clip("a.mp4", seconds=2.0, size="640x480")
    proxy = tmp_path / "out" / "a-proxy.mp4"
    make_proxy(clip, proxy, height=240)
    assert probe(proxy).has_audio is True


def test_extract_frame_writes_image(make_clip, tmp_path: Path):
    clip = make_clip("a.mp4", seconds=3.0)
    frame = tmp_path / "frames" / "f.jpg"
    extract_frame(clip, at=1.0, dst=frame, height=180)
    assert frame.exists() and frame.stat().st_size > 500


def test_list_video_files_sorts_and_filters(tmp_path: Path, make_clip):
    make_clip("b.MOV", seconds=1.0).rename(tmp_path / "b.MOV")
    make_clip("a.mp4", seconds=1.0).rename(tmp_path / "a.mp4")
    (tmp_path / ".DS_Store").write_bytes(b"junk")
    (tmp_path / "._a.mp4").write_bytes(b"junk")
    (tmp_path / "notes.md").write_text("hello", encoding="utf-8")
    (tmp_path / "sub").mkdir()

    found = list_video_files(tmp_path)

    assert [path.name for path in found] == ["a.mp4", "b.MOV"]


def test_list_video_files_on_missing_directory_returns_empty(tmp_path: Path):
    assert list_video_files(tmp_path / "nope") == []


def test_probe_raises_when_duration_metadata_is_missing(tmp_path: Path):
    # A raw H.264 elementary stream (no container) has no format.duration; ffprobe
    # reports width/height/r_frame_rate but omits "duration" from the format section.
    raw = tmp_path / "raw.h264"
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", "testsrc=size=320x240:rate=30:duration=1",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-f", "h264",
            str(raw),
        ],
        check=True,
    )

    with pytest.raises(RuntimeError, match="duration"):
        probe(raw)


def test_extract_audio_failure_leaves_no_partial_or_temp_file(tmp_path: Path):
    bad_input = tmp_path / "not-a-video.mp4"
    bad_input.write_text("this is not video data", encoding="utf-8")
    dst = tmp_path / "out" / "a.wav"

    with pytest.raises(RuntimeError):
        extract_audio(bad_input, dst)

    assert not dst.exists()
    assert list(dst.parent.iterdir()) == []


def test_extract_audio_success_leaves_only_the_destination_file(make_clip, tmp_path: Path):
    clip = make_clip("a.mp4", seconds=2.0)
    dst = tmp_path / "out" / "a.wav"

    extract_audio(clip, dst)

    assert dst.exists() and dst.stat().st_size > 1000
    assert [path.name for path in dst.parent.iterdir()] == [dst.name]


def test_make_proxy_failure_leaves_no_partial_or_temp_file(tmp_path: Path):
    bad_input = tmp_path / "not-a-video.mp4"
    bad_input.write_text("this is not video data", encoding="utf-8")
    dst = tmp_path / "out" / "a-proxy.mp4"

    with pytest.raises(RuntimeError):
        make_proxy(bad_input, dst, height=240)

    assert not dst.exists()
    assert list(dst.parent.iterdir()) == []


def test_extract_frame_failure_leaves_no_partial_or_temp_file(tmp_path: Path):
    bad_input = tmp_path / "not-a-video.mp4"
    bad_input.write_text("this is not video data", encoding="utf-8")
    dst = tmp_path / "frames" / "f.jpg"

    with pytest.raises(RuntimeError):
        extract_frame(bad_input, at=1.0, dst=dst, height=180)

    assert not dst.exists()
    assert list(dst.parent.iterdir()) == []
