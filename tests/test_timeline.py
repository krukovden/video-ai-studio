from videoai.core.models import (
    Analysis,
    ClipInfo,
    Manifest,
    PlanSection,
    SegmentAnalysis,
    StoryPlan,
)
from videoai.logic.timeline import build_timeline


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


def test_timeline_clips_follow_plan_order():
    timeline = build_timeline(_plan(), _analysis(), _manifest(), padding=0.15, fps=30.0)
    assert [clip.src for clip in timeline.clips] == ["clip-01", "clip-02"]


def test_timeline_positions_are_contiguous():
    timeline = build_timeline(_plan(), _analysis(), _manifest(), padding=0.15, fps=30.0)
    assert timeline.clips[0].start == 0.0
    expected = timeline.clips[0].start + timeline.clips[0].dur
    assert abs(timeline.clips[1].start - expected) < 1e-6


def test_padding_extends_each_segment_on_both_sides():
    timeline = build_timeline(_plan(), _analysis(), _manifest(), padding=0.15, fps=30.0)
    first = timeline.clips[0]
    assert abs(first.offset - 1.85) < 1e-6
    assert abs(first.dur - 3.3) < 1e-6


def test_padding_is_clamped_to_clip_bounds():
    analysis = Analysis(provider="mock", segments=[
        SegmentAnalysis(phrase_id="clip-01#001", clip_id="clip-01", start=0.05, end=29.98,
                        text="whole clip", content="all", delivery_score=8),
    ])
    plan = StoryPlan(
        sections=[PlanSection(name="All", goal="everything", phrase_ids=["clip-01#001"])],
        title="T", description="D", tags=[],
    )
    timeline = build_timeline(plan, analysis, _manifest(), padding=0.5, fps=30.0)
    clip = timeline.clips[0]
    assert clip.offset == 0.0
    assert clip.offset + clip.dur <= 30.0


def test_clip_carries_provenance_fields():
    timeline = build_timeline(_plan(), _analysis(), _manifest(), padding=0.15, fps=30.0)
    first = timeline.clips[0]
    assert first.quote == "hello everyone"
    assert first.beat == "Hook"
    assert "intro" in first.reason


def test_resolution_comes_from_first_source_clip():
    timeline = build_timeline(_plan(), _analysis(), _manifest(), padding=0.15, fps=30.0)
    assert (timeline.width, timeline.height) == (1920, 1080)
    assert timeline.fps == 30.0


def test_unknown_phrase_id_in_plan_raises():
    plan = StoryPlan(
        sections=[PlanSection(name="Hook", goal="x", phrase_ids=["clip-09#001"])],
        title="T", description="D", tags=[],
    )
    try:
        build_timeline(plan, _analysis(), _manifest(), padding=0.15, fps=30.0)
    except KeyError as error:
        assert "clip-09#001" in str(error)
    else:
        raise AssertionError("expected KeyError")
