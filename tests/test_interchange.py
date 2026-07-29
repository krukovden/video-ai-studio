"""Pure-logic tests for the Timeline -> OpenTimelineIO conversion, independent of
ffmpeg or real media (the same shape as `test_timeline.py`)."""
from pathlib import Path

import opentimelineio as otio

from videoai.core.models import ClipInfo, Manifest, Timeline, TimelineClip
from videoai.logic.interchange import build_otio_timeline


def _manifest(tmp_path: Path, second_path: str | None = None) -> Manifest:
    first = tmp_path / "a.mov"
    first.write_bytes(b"fake")
    return Manifest(clips=[
        ClipInfo(clip_id="clip-01", path=str(first), duration=10.0, width=3840,
                 height=2160, fps=30.0, has_audio=True, proxy_path=str(tmp_path / "a-proxy.mp4")),
        ClipInfo(clip_id="clip-02", path=second_path or str(tmp_path / "b.mov"),
                 duration=10.0, width=3840, height=2160, fps=30.0, has_audio=True,
                 proxy_path=str(tmp_path / "b-proxy.mp4")),
    ])


def _two_clip_timeline() -> Timeline:
    return Timeline(fps=30.0, width=3840, height=2160, clips=[
        TimelineClip(src="clip-01", offset=1.0, dur=2.0, start=0.0,
                     beat="Hook", quote="hello everyone", reason="opens strong"),
        TimelineClip(src="clip-02", offset=3.0, dur=1.5, start=2.0,
                     beat="Body", is_insert=True, reason="visual insert"),
    ])


def test_one_otio_clip_per_timeline_clip_in_order(tmp_path: Path):
    manifest = _manifest(tmp_path)
    (tmp_path / "b.mov").write_bytes(b"fake")
    otio_timeline, missing = build_otio_timeline(_two_clip_timeline(), manifest, tmp_path)

    track = otio_timeline.tracks[0]
    clips = list(track.find_clips())
    assert len(clips) == 2
    assert missing == []
    assert [c.name.split(" ")[0] for c in clips] == ["clip-01", "clip-02"]


def test_source_in_out_points_use_offset_and_dur_at_timeline_fps(tmp_path: Path):
    manifest = _manifest(tmp_path)
    (tmp_path / "b.mov").write_bytes(b"fake")
    otio_timeline, _ = build_otio_timeline(_two_clip_timeline(), manifest, tmp_path)

    clips = list(otio_timeline.tracks[0].find_clips())
    first, second = clips
    assert first.source_range.start_time.to_seconds() == 1.0
    assert first.source_range.duration.to_seconds() == 2.0
    assert second.source_range.start_time.to_seconds() == 3.0
    assert second.source_range.duration.to_seconds() == 1.5
    assert first.source_range.start_time.rate == 30.0


def test_timeline_position_matches_clip_start(tmp_path: Path):
    manifest = _manifest(tmp_path)
    (tmp_path / "b.mov").write_bytes(b"fake")
    otio_timeline, _ = build_otio_timeline(_two_clip_timeline(), manifest, tmp_path)

    track = otio_timeline.tracks[0]
    clips = list(track.find_clips())
    first_range = track.range_of_child(clips[0])
    second_range = track.range_of_child(clips[1])
    assert first_range.start_time.to_seconds() == 0.0
    assert second_range.start_time.to_seconds() == 2.0


def test_a_gap_between_clips_is_inserted_when_start_leaves_room(tmp_path: Path):
    """`build_timeline()` never produces gaps itself, but a hand-edited
    `05-timeline.json` could — position must come from `start`, not from
    trusting sequential append."""
    manifest = _manifest(tmp_path)
    (tmp_path / "b.mov").write_bytes(b"fake")
    timeline = Timeline(fps=30.0, width=3840, height=2160, clips=[
        TimelineClip(src="clip-01", offset=0.0, dur=1.0, start=0.0),
        TimelineClip(src="clip-02", offset=0.0, dur=1.0, start=2.0),
    ])
    otio_timeline, _ = build_otio_timeline(timeline, manifest, tmp_path)

    track = otio_timeline.tracks[0]
    clips = list(track.find_clips())
    second_range = track.range_of_child(clips[1])
    assert second_range.start_time.to_seconds() == 2.0


def test_media_references_are_absolute_and_point_at_originals_not_proxies(tmp_path: Path):
    manifest = _manifest(tmp_path)
    (tmp_path / "b.mov").write_bytes(b"fake")
    otio_timeline, _ = build_otio_timeline(_two_clip_timeline(), manifest, tmp_path)

    for clip in otio_timeline.tracks[0].find_clips():
        target = clip.media_reference.target_url
        assert Path(target).is_absolute()
        assert "proxy" not in target


def test_relative_manifest_paths_resolve_against_project_dir(tmp_path: Path):
    """The real manifest stores paths relative to wherever the pipeline was run
    from (e.g. `assets/.../video/IMG_5665.MOV`); the interchange file must not
    depend on the creator's cwd matching that."""
    (tmp_path / "video").mkdir()
    (tmp_path / "video" / "a.mov").write_bytes(b"fake")
    manifest = Manifest(clips=[
        ClipInfo(clip_id="clip-01", path="video/a.mov", duration=10.0, width=3840,
                 height=2160, fps=30.0, has_audio=True),
    ])
    timeline = Timeline(fps=30.0, width=3840, height=2160, clips=[
        TimelineClip(src="clip-01", offset=0.0, dur=1.0, start=0.0),
    ])

    otio_timeline, missing = build_otio_timeline(timeline, manifest, tmp_path)

    clip = otio_timeline.tracks[0].find_clips()[0]
    assert clip.media_reference.target_url == str((tmp_path / "video" / "a.mov").resolve())
    assert missing == []


def test_beat_quote_reason_and_insert_flag_survive_as_metadata_and_marker(tmp_path: Path):
    manifest = _manifest(tmp_path)
    (tmp_path / "b.mov").write_bytes(b"fake")
    otio_timeline, _ = build_otio_timeline(_two_clip_timeline(), manifest, tmp_path)

    first, second = otio_timeline.tracks[0].find_clips()
    assert first.metadata["videoai"]["beat"] == "Hook"
    assert first.metadata["videoai"]["quote"] == "hello everyone"
    assert first.metadata["videoai"]["reason"] == "opens strong"
    assert first.metadata["videoai"]["is_insert"] is False
    assert len(first.markers) == 1
    assert "Hook" in first.markers[0].comment
    assert "hello everyone" in first.markers[0].comment

    assert second.metadata["videoai"]["is_insert"] is True
    assert len(second.markers) == 1
    assert "visual insert" in second.markers[0].comment.lower()


def test_a_missing_source_is_recorded_not_raised(tmp_path: Path):
    manifest = _manifest(tmp_path, second_path=str(tmp_path / "gone.mov"))
    otio_timeline, missing = build_otio_timeline(_two_clip_timeline(), manifest, tmp_path)

    assert len(list(otio_timeline.tracks[0].find_clips())) == 2
    assert str((tmp_path / "gone.mov").resolve()) in missing
