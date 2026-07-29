from pathlib import Path

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
