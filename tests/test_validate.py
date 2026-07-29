from videoai.core.models import (
    ClipInfo,
    ClipTranscript,
    Manifest,
    Timeline,
    TimelineClip,
    Transcript,
    Word,
)
from videoai.logic.validate import validate_timeline


def _manifest() -> Manifest:
    return Manifest(clips=[ClipInfo(clip_id="clip-01", path="/tmp/a.mp4", duration=30.0,
                                    width=1920, height=1080, fps=30.0, has_audio=True)])


def _transcript() -> Transcript:
    return Transcript(provider="mock", clips=[ClipTranscript(
        clip_id="clip-01",
        words=[Word(text="hello", start=2.0, end=2.5), Word(text="world", start=2.6, end=3.0)],
    )])


def _timeline(*clips: TimelineClip) -> Timeline:
    return Timeline(fps=30.0, width=1920, height=1080, clips=list(clips))


def test_valid_timeline_has_no_violations():
    timeline = _timeline(
        TimelineClip(src="clip-01", offset=1.8, dur=1.4, start=0.0, quote="hello world")
    )
    assert validate_timeline(timeline, _manifest(), _transcript()) == []


def test_gap_between_clips_is_reported():
    timeline = _timeline(
        TimelineClip(src="clip-01", offset=1.8, dur=1.4, start=0.0, quote="hello world"),
        TimelineClip(src="clip-01", offset=5.0, dur=1.0, start=2.0, quote=""),
    )
    violations = validate_timeline(timeline, _manifest(), _transcript())
    assert any("contiguous" in v for v in violations)


def test_segment_beyond_source_duration_is_reported():
    timeline = _timeline(TimelineClip(src="clip-01", offset=29.0, dur=5.0, start=0.0, quote=""))
    violations = validate_timeline(timeline, _manifest(), _transcript())
    assert any("exceeds source duration" in v for v in violations)


def test_unknown_source_is_reported():
    timeline = _timeline(TimelineClip(src="clip-99", offset=0.0, dur=1.0, start=0.0, quote=""))
    violations = validate_timeline(timeline, _manifest(), _transcript())
    assert any("unknown source" in v for v in violations)


def test_too_short_segment_is_reported():
    timeline = _timeline(TimelineClip(src="clip-01", offset=2.0, dur=0.1, start=0.0, quote=""))
    violations = validate_timeline(timeline, _manifest(), _transcript())
    assert any("shorter than" in v for v in violations)


def test_cut_inside_a_word_is_reported():
    timeline = _timeline(TimelineClip(src="clip-01", offset=2.2, dur=0.6, start=0.0, quote=""))
    violations = validate_timeline(timeline, _manifest(), _transcript())
    assert any("inside word" in v for v in violations)


def test_quote_not_present_in_segment_is_reported():
    timeline = _timeline(
        TimelineClip(src="clip-01", offset=1.8, dur=1.4, start=0.0, quote="totally invented")
    )
    violations = validate_timeline(timeline, _manifest(), _transcript())
    assert any("quote not found" in v for v in violations)


def test_negative_offset_is_reported():
    timeline = _timeline(TimelineClip(src="clip-01", offset=-0.5, dur=1.0, start=0.0, quote=""))
    violations = validate_timeline(timeline, _manifest(), _transcript())
    assert any("negative offset" in v for v in violations)
