from videoai.core.models import (
    Analysis,
    ClipInfo,
    ClipTranscript,
    Manifest,
    PlanSection,
    SegmentAnalysis,
    StoryPlan,
    Transcript,
    Word,
)
from videoai.logic.timeline import build_timeline
from videoai.logic.validate import validate_timeline


def _manifest() -> Manifest:
    return Manifest(clips=[
        ClipInfo(clip_id="clip-01", path="/tmp/a.mp4", duration=30.0, width=1920,
                 height=1080, fps=30.0, has_audio=True),
        ClipInfo(clip_id="clip-02", path="/tmp/b.mp4", duration=30.0, width=1920,
                 height=1080, fps=30.0, has_audio=True),
    ])


def _analysis() -> Analysis:
    return Analysis(provider="mock", segments=[
        SegmentAnalysis(phrase_id="clip-01#001", clip_id="clip-01", start=2.0, end=5.0,
                        text="hello everyone", content="intro", delivery_score=9),
        SegmentAnalysis(phrase_id="clip-02#001", clip_id="clip-02", start=10.0, end=14.0,
                        text="look at the wheels", content="wheels", delivery_score=7),
    ])


def _plan() -> StoryPlan:
    return StoryPlan(
        sections=[
            PlanSection(name="Hook", goal="open strong", phrase_ids=["clip-01#001"]),
            PlanSection(name="Body", goal="show details", phrase_ids=["clip-02#001"]),
        ],
        title="T", description="D", tags=["toy"],
    )


def _transcript(**words_by_clip: list[Word]) -> Transcript:
    """An empty-words transcript unless a clip's words are given by keyword,
    e.g. _transcript(**{"clip-01": [...]})."""
    return Transcript(provider="mock", clips=[
        ClipTranscript(clip_id="clip-01", words=words_by_clip.get("clip-01", [])),
        ClipTranscript(clip_id="clip-02", words=words_by_clip.get("clip-02", [])),
    ])


def test_timeline_clips_follow_plan_order():
    timeline = build_timeline(_plan(), _analysis(), _manifest(), _transcript(), padding=0.15,
                               fps=30.0)
    assert [clip.src for clip in timeline.clips] == ["clip-01", "clip-02"]


def test_timeline_positions_are_contiguous():
    timeline = build_timeline(_plan(), _analysis(), _manifest(), _transcript(), padding=0.15,
                               fps=30.0)
    assert timeline.clips[0].start == 0.0
    expected = timeline.clips[0].start + timeline.clips[0].dur
    assert abs(timeline.clips[1].start - expected) < 1e-6


def test_padding_extends_each_segment_on_both_sides():
    timeline = build_timeline(_plan(), _analysis(), _manifest(), _transcript(), padding=0.15,
                               fps=30.0)
    first = timeline.clips[0]
    assert abs(first.offset - 1.85) < 1e-6
    assert abs(first.dur - 3.3) < 1e-6


def test_no_neighbouring_words_within_padding_window_gets_full_padding():
    # Neighbours exist but sit well outside the 0.15s padding window (0.5s away
    # on each side), so they must not constrain the edges at all.
    transcript = _transcript(**{
        "clip-01": [
            Word(text="before", start=1.0, end=1.5),
            Word(text="after", start=5.5, end=6.0),
        ],
    })
    timeline = build_timeline(_plan(), _analysis(), _manifest(), transcript, padding=0.15,
                               fps=30.0)
    first = timeline.clips[0]
    assert abs(first.offset - 1.85) < 1e-6
    assert abs(first.offset + first.dur - 5.15) < 1e-6


def test_padding_is_clamped_to_clip_bounds():
    analysis = Analysis(provider="mock", segments=[
        SegmentAnalysis(phrase_id="clip-01#001", clip_id="clip-01", start=0.05, end=29.98,
                        text="whole clip", content="all", delivery_score=8),
    ])
    plan = StoryPlan(
        sections=[PlanSection(name="All", goal="everything", phrase_ids=["clip-01#001"])],
        title="T", description="D", tags=[],
    )
    timeline = build_timeline(plan, analysis, _manifest(), _transcript(), padding=0.5, fps=30.0)
    clip = timeline.clips[0]
    assert clip.offset == 0.0
    assert clip.offset + clip.dur <= 30.0


def test_left_edge_stops_at_close_neighbour_word_end():
    # The neighbour word ends 0.05s before the segment starts, inside the 0.15s
    # padding window: the left edge must stop there, not push a full 0.15s back
    # into that word.
    transcript = _transcript(**{
        "clip-02": [Word(text="Some", start=9.5, end=9.95)],
    })
    timeline = build_timeline(_plan(), _analysis(), _manifest(), transcript, padding=0.15,
                               fps=30.0)
    second = timeline.clips[1]
    assert abs(second.offset - 9.95) < 1e-6


def test_right_edge_stops_at_close_neighbour_word_start():
    # The next word starts 0.05s after the segment ends, inside the padding
    # window: the right edge must stop there, not overrun into that word.
    transcript = _transcript(**{
        "clip-02": [Word(text="next", start=14.05, end=14.4)],
    })
    timeline = build_timeline(_plan(), _analysis(), _manifest(), transcript, padding=0.15,
                               fps=30.0)
    second = timeline.clips[1]
    assert abs((second.offset + second.dur) - 14.05) < 1e-6


def test_segment_at_start_of_clip_still_clamps_to_zero():
    analysis = Analysis(provider="mock", segments=[
        SegmentAnalysis(phrase_id="clip-01#001", clip_id="clip-01", start=0.05, end=2.0,
                        text="start", content="x", delivery_score=8),
    ])
    plan = StoryPlan(
        sections=[PlanSection(name="Start", goal="x", phrase_ids=["clip-01#001"])],
        title="T", description="D", tags=[],
    )
    timeline = build_timeline(plan, analysis, _manifest(), _transcript(), padding=0.15, fps=30.0)
    assert timeline.clips[0].offset == 0.0


def test_segment_at_end_of_clip_still_clamps_to_duration():
    analysis = Analysis(provider="mock", segments=[
        SegmentAnalysis(phrase_id="clip-01#001", clip_id="clip-01", start=28.0, end=29.98,
                        text="end", content="x", delivery_score=8),
    ])
    plan = StoryPlan(
        sections=[PlanSection(name="End", goal="x", phrase_ids=["clip-01#001"])],
        title="T", description="D", tags=[],
    )
    timeline = build_timeline(plan, analysis, _manifest(), _transcript(), padding=0.15, fps=30.0)
    clip = timeline.clips[0]
    assert abs((clip.offset + clip.dur) - 30.0) < 1e-6


def test_clip_carries_provenance_fields():
    timeline = build_timeline(_plan(), _analysis(), _manifest(), _transcript(), padding=0.15,
                               fps=30.0)
    first = timeline.clips[0]
    assert first.quote == "hello everyone"
    assert first.beat == "Hook"
    assert "intro" in first.reason


def test_resolution_comes_from_first_source_clip():
    timeline = build_timeline(_plan(), _analysis(), _manifest(), _transcript(), padding=0.15,
                               fps=30.0)
    assert (timeline.width, timeline.height) == (1920, 1080)
    assert timeline.fps == 30.0


def test_core_dur_is_the_unpadded_segment_length():
    timeline = build_timeline(_plan(), _analysis(), _manifest(), _transcript(), padding=0.15,
                               fps=30.0)
    first, second = timeline.clips
    assert abs(first.core_dur - 3.0) < 1e-6  # 5.0 - 2.0, no padding
    assert abs(second.core_dur - 4.0) < 1e-6  # 14.0 - 10.0


def test_core_dur_is_unaffected_by_padding_clamped_at_source_bounds():
    analysis = Analysis(provider="mock", segments=[
        SegmentAnalysis(phrase_id="clip-01#001", clip_id="clip-01", start=0.05, end=29.98,
                        text="whole clip", content="all", delivery_score=8),
    ])
    plan = StoryPlan(
        sections=[PlanSection(name="All", goal="everything", phrase_ids=["clip-01#001"])],
        title="T", description="D", tags=[],
    )
    timeline = build_timeline(plan, analysis, _manifest(), _transcript(), padding=0.5, fps=30.0)
    clip = timeline.clips[0]
    # The clamp shrinks the padded total (clip.dur) but core_dur is the raw
    # segment span, computed before padding and before the source clamp.
    assert abs(clip.core_dur - 29.93) < 1e-6


def test_core_dur_is_unaffected_by_neighbour_word_clamping():
    transcript = _transcript(**{
        "clip-02": [
            Word(text="Some", start=9.5, end=9.95),
            Word(text="next", start=14.05, end=14.4),
        ],
    })
    timeline = build_timeline(_plan(), _analysis(), _manifest(), transcript, padding=0.15,
                               fps=30.0)
    second = timeline.clips[1]
    assert abs(second.core_dur - 4.0) < 1e-6  # 14.0 - 10.0, unchanged by clamping


def test_unknown_phrase_id_in_plan_raises():
    plan = StoryPlan(
        sections=[PlanSection(name="Hook", goal="x", phrase_ids=["clip-09#001"])],
        title="T", description="D", tags=[],
    )
    try:
        build_timeline(plan, _analysis(), _manifest(), _transcript(), padding=0.15, fps=30.0)
    except KeyError as error:
        assert "clip-09#001" in str(error)
    else:
        raise AssertionError("expected KeyError")


def test_build_timeline_never_produces_a_timeline_its_own_validator_rejects():
    # Round-trip contract: over tightly packed continuous speech (neighbour
    # words closer than the padding window on both sides of both segments),
    # build_timeline must not pad a cut edge into the middle of a word.
    manifest = Manifest(clips=[
        ClipInfo(clip_id="clip-04", path="/tmp/c.mp4", duration=60.0, width=1920,
                 height=1080, fps=30.0, has_audio=True),
    ])
    words = [
        Word(text="and", start=49.80, end=50.02),
        Word(text="Some", start=50.10, end=50.45),
        Word(text="people", start=50.45, end=50.90),
        Word(text="say", start=50.90, end=51.20),
        Word(text="that", start=51.20, end=51.50),
        Word(text="the", start=54.90, end=55.10),
        Word(text="wheels", start=55.10, end=55.60),
        Word(text="spin", start=55.60, end=55.95),
        Word(text="fast", start=55.95, end=56.20),
        Word(text="really", start=56.25, end=56.60),
    ]
    transcript = Transcript(provider="mock", clips=[
        ClipTranscript(clip_id="clip-04", words=words),
    ])
    analysis = Analysis(provider="mock", segments=[
        SegmentAnalysis(phrase_id="clip-04#001", clip_id="clip-04", start=50.10, end=51.50,
                        text="Some people say that", content="hook", delivery_score=8),
        SegmentAnalysis(phrase_id="clip-04#002", clip_id="clip-04", start=54.90, end=55.60,
                        text="the wheels", content="wheels", delivery_score=7),
    ])
    plan = StoryPlan(
        sections=[
            PlanSection(name="Hook", goal="x", phrase_ids=["clip-04#001"]),
            PlanSection(name="Body", goal="y", phrase_ids=["clip-04#002"]),
        ],
        title="T", description="D", tags=[],
    )

    timeline = build_timeline(plan, analysis, manifest, transcript, padding=0.15, fps=30.0)
    violations = validate_timeline(timeline, manifest, transcript)
    assert violations == []
