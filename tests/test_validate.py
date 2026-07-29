from videoai.core.models import (
    Analysis,
    ClipInfo,
    ClipTranscript,
    Manifest,
    PlanSection,
    SegmentAnalysis,
    StoryPlan,
    Timeline,
    TimelineClip,
    Transcript,
    Word,
)
from videoai.logic.timeline import build_timeline
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
        TimelineClip(src="clip-01", offset=1.8, dur=1.4, start=0.0, quote="hello world",
                     core_dur=1.0)
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


# --- Finding 1: normalisation must not doubled-space on punctuation next to a space ---


def test_quote_differing_only_by_punctuation_still_validates():
    transcript = Transcript(provider="mock", clips=[ClipTranscript(
        clip_id="clip-01",
        words=[
            Word(text="Hello,", start=2.0, end=2.5),
            Word(text="everyone!", start=2.6, end=3.0),
            Word(text="welcome", start=3.1, end=3.5),
            Word(text="back", start=3.6, end=3.9),
        ],
    )])
    timeline = _timeline(
        TimelineClip(src="clip-01", offset=1.8, dur=2.25, start=0.0,
                     quote="hello everyone welcome back", core_dur=1.9)
    )
    assert validate_timeline(timeline, _manifest(), transcript) == []


def test_quote_differing_only_by_capitalisation_and_spacing_still_validates():
    timeline = _timeline(
        TimelineClip(src="clip-01", offset=1.8, dur=1.4, start=0.0,
                     quote="  HELLO   World  ", core_dur=1.0)
    )
    assert validate_timeline(timeline, _manifest(), _transcript()) == []


def test_genuinely_absent_quote_is_still_rejected_after_normalisation_fix():
    timeline = _timeline(
        TimelineClip(src="clip-01", offset=1.8, dur=1.4, start=0.0,
                     quote="totally, invented!", core_dur=1.0)
    )
    violations = validate_timeline(timeline, _manifest(), _transcript())
    assert any("quote not found" in v for v in violations)


# --- Finding 2: MIN_SPEECH_SECONDS must judge the real speech length, not the
# padded total, which the padding default alone can inflate past MIN_SEGMENT_SECONDS ---


def test_zero_length_core_segment_is_rejected():
    timeline = _timeline(
        TimelineClip(src="clip-01", offset=1.8, dur=1.4, start=0.0, quote="", core_dur=0.0)
    )
    violations = validate_timeline(timeline, _manifest(), _transcript())
    assert any("core speech shorter than" in v for v in violations)


def test_short_core_segment_is_rejected():
    timeline = _timeline(
        TimelineClip(src="clip-01", offset=1.8, dur=1.4, start=0.0, quote="", core_dur=0.1)
    )
    violations = validate_timeline(timeline, _manifest(), _transcript())
    assert any("core speech shorter than" in v for v in violations)


def test_core_segment_of_0_4_seconds_passes():
    timeline = _timeline(
        TimelineClip(src="clip-01", offset=1.8, dur=1.4, start=0.0, quote="", core_dur=0.4)
    )
    violations = validate_timeline(timeline, _manifest(), _transcript())
    assert not any("core speech" in v for v in violations)


def test_padding_clamped_at_source_boundary_is_still_judged_on_core_duration():
    """Reproduces the dormant defect: padding alone (0.15s each side) can inflate
    a clamped, nearly-zero-length segment's total duration to exactly
    MIN_SEGMENT_SECONDS, which the old total-duration rule alone would accept.
    core_dur exposes the real speech length regardless of that clamp."""
    manifest = Manifest(clips=[ClipInfo(clip_id="clip-01", path="/tmp/a.mp4", duration=30.0,
                                        width=1920, height=1080, fps=30.0, has_audio=True)])
    analysis = Analysis(provider="mock", segments=[
        SegmentAnalysis(phrase_id="clip-01#001", clip_id="clip-01", start=29.85, end=29.95,
                        text="uh", content="noise", delivery_score=1),
    ])
    plan = StoryPlan(
        sections=[PlanSection(name="Noise", goal="x", phrase_ids=["clip-01#001"])],
        title="T", description="D", tags=[],
    )
    timeline = build_timeline(plan, analysis, manifest, padding=0.15, fps=30.0)
    clip = timeline.clips[0]
    # The clamp at the 30s source boundary produces exactly the old minimum, which
    # would have passed the total-duration rule alone.
    assert abs(clip.dur - 0.30) < 1e-6
    assert abs(clip.core_dur - 0.10) < 1e-6

    violations = validate_timeline(timeline, manifest, Transcript(provider="mock", clips=[]))
    assert any("core speech shorter than" in v for v in violations)


# --- Finding I3: mixed clip geometry silently corrupts the draft. Proxies scale
# to a fixed height, so a portrait clip becomes 405x720 beside 1280x720 landscape
# ones, and the concat demuxer's `-c copy` accepts that without an error ---


def _mixed_manifest() -> Manifest:
    return Manifest(clips=[
        ClipInfo(clip_id="clip-01", path="/tmp/landscape.mp4", duration=30.0,
                 width=1920, height=1080, fps=30.0, has_audio=True),
        ClipInfo(clip_id="clip-02", path="/tmp/portrait.mp4", duration=30.0,
                 width=1080, height=1920, fps=30.0, has_audio=True),
    ])


def _empty_transcript() -> Transcript:
    return Transcript(provider="mock", clips=[])


def test_timeline_mixing_portrait_and_landscape_sources_is_reported():
    timeline = _timeline(
        TimelineClip(src="clip-01", offset=1.0, dur=1.0, start=0.0, quote="", core_dur=1.0),
        TimelineClip(src="clip-02", offset=1.0, dur=1.0, start=1.0, quote="", core_dur=1.0),
    )

    violations = validate_timeline(timeline, _mixed_manifest(), _empty_transcript())

    geometry = [v for v in violations if "mixed source geometry" in v]
    assert len(geometry) == 1
    assert "clip-01" in geometry[0]
    assert "clip-02" in geometry[0]


def test_uniform_timeline_reports_no_geometry_violation():
    manifest = Manifest(clips=[
        ClipInfo(clip_id="clip-01", path="/tmp/a.mp4", duration=30.0,
                 width=3840, height=2160, fps=30.0, has_audio=True),
        # Different pixel size, same aspect ratio: both proxies come out identical.
        ClipInfo(clip_id="clip-02", path="/tmp/b.mp4", duration=30.0,
                 width=1920, height=1080, fps=30.0, has_audio=True),
    ])
    timeline = _timeline(
        TimelineClip(src="clip-01", offset=1.0, dur=1.0, start=0.0, quote="", core_dur=1.0),
        TimelineClip(src="clip-02", offset=1.0, dur=1.0, start=1.0, quote="", core_dur=1.0),
    )

    violations = validate_timeline(timeline, manifest, _empty_transcript())

    assert not any("mixed source geometry" in v for v in violations)


def test_mixed_frame_rate_is_reported():
    manifest = Manifest(clips=[
        ClipInfo(clip_id="clip-01", path="/tmp/a.mp4", duration=30.0,
                 width=1920, height=1080, fps=30.0, has_audio=True),
        ClipInfo(clip_id="clip-02", path="/tmp/b.mp4", duration=30.0,
                 width=1920, height=1080, fps=60.0, has_audio=True),
    ])
    timeline = _timeline(
        TimelineClip(src="clip-01", offset=1.0, dur=1.0, start=0.0, quote="", core_dur=1.0),
        TimelineClip(src="clip-02", offset=1.0, dur=1.0, start=1.0, quote="", core_dur=1.0),
    )

    violations = validate_timeline(timeline, manifest, _empty_transcript())

    assert any("mixed source geometry" in v and "60.00fps" in v for v in violations)


def test_each_offending_source_is_reported_once_however_often_it_is_cut_in():
    timeline = _timeline(
        TimelineClip(src="clip-01", offset=1.0, dur=1.0, start=0.0, quote="", core_dur=1.0),
        TimelineClip(src="clip-02", offset=1.0, dur=1.0, start=1.0, quote="", core_dur=1.0),
        TimelineClip(src="clip-02", offset=3.0, dur=1.0, start=2.0, quote="", core_dur=1.0),
    )

    violations = validate_timeline(timeline, _mixed_manifest(), _empty_transcript())

    assert len([v for v in violations if "mixed source geometry" in v]) == 1


def test_single_source_timeline_reports_no_geometry_violation():
    timeline = _timeline(
        TimelineClip(src="clip-01", offset=1.8, dur=1.4, start=0.0, quote="hello world",
                     core_dur=1.0)
    )
    assert validate_timeline(timeline, _manifest(), _transcript()) == []
