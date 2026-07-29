"""The visual gate: nothing upstream ever looks at what a SELECTED speech segment
shows, so an adult filling the frame reaches the draft unseen. These tests cover
the gate itself; `test_cli_end_to_end.py` covers the plan/check/re-plan loop."""
import json
from pathlib import Path

import pytest

from videoai.config import Config, PlanSettings
from videoai.core.models import (
    Analysis,
    ClipInfo,
    Manifest,
    RejectedPhrases,
    SegmentAnalysis,
    Timeline,
    TimelineClip,
    VisualCheck,
)
from videoai.core.registry import StageContext
from videoai.core.store import ArtifactStore, source_key
from videoai.stages.s05b_visual_check import (
    _segment_frames,
    build_visual_check_prompt,
    parse_findings,
    visual_check,
)

CLEAN = {"adult_prominent": False, "child_visible": True, "unusable": False,
         "note": "the child holds the toy"}
ADULT = {"adult_prominent": True, "child_visible": False, "unusable": False,
         "note": "an adult's head fills half the frame"}
BLURRED = {"adult_prominent": False, "child_visible": False, "unusable": True,
           "note": "out of focus and pointing at the ceiling"}


def _timeline() -> Timeline:
    return Timeline(fps=30.0, width=320, height=240, clips=[
        TimelineClip(src="clip-01", offset=0.35, dur=1.30, start=0.0, quote="hello everyone"),
        TimelineClip(src="clip-01", offset=2.85, dur=1.25, start=1.30, quote="look here"),
    ])


def _analysis() -> Analysis:
    return Analysis(provider="mock", segments=[
        SegmentAnalysis(phrase_id="clip-01#001", clip_id="clip-01", start=0.5, end=1.5,
                        text="hello everyone"),
        SegmentAnalysis(phrase_id="clip-01#002", clip_id="clip-01", start=3.0, end=4.0,
                        text="look here"),
    ])


def _context(
    tmp_path: Path,
    monkeypatch,
    clip_path: Path,
    payload: dict,
    config: Config | None = None,
    timeline: Timeline | None = None,
) -> StageContext:
    for name in ("input", "work", "output"):
        (tmp_path / name).mkdir(parents=True, exist_ok=True)
    payload_path = tmp_path / "llm.json"
    payload_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setenv("VIDEOAI_MOCK_LLM", str(payload_path))
    ctx = StageContext(
        project_dir=tmp_path,
        input_dir=tmp_path / "input",
        work_dir=tmp_path / "work",
        output_dir=tmp_path / "output",
        config=config or Config(providers={"asr": "mock", "llm": "mock"}),
        store=ArtifactStore(tmp_path / "work"),
    )
    ctx.store.write("01-manifest", Manifest(clips=[ClipInfo(
        clip_id="clip-01", path=str(clip_path), duration=6.0, width=320, height=240,
        fps=30.0, has_audio=True, source_key=source_key(clip_path),
    )]), fingerprint="fp")
    ctx.store.write("05-timeline", timeline or _timeline(), fingerprint="fp")
    ctx.store.write("04-analysis", _analysis(), fingerprint="fp")
    return ctx


def _both_flags_off() -> Config:
    return Config(
        providers={"asr": "mock", "llm": "mock"},
        plan=PlanSettings(reject_adult_in_frame=False, reject_unusable_shots=False),
    )


def test_adult_in_frame_raises_and_records_the_phrase_id(tmp_path: Path, make_clip, monkeypatch):
    """The failure this whole stage exists for: a chosen segment whose frames show
    an adult. It must stop the render and name the phrase, so the next plan can
    avoid it instead of picking the neighbouring phrase with the same problem."""
    clip = make_clip("a.mp4", seconds=6.0)
    ctx = _context(tmp_path, monkeypatch, clip, {"0": CLEAN, "1": ADULT})

    with pytest.raises(RuntimeError) as error:
        visual_check(ctx)

    message = str(error.value)
    assert "clip-01#002" in message
    assert "an adult's head fills half the frame" in message
    assert "2.85-4.10s" in message
    assert "clip-01#001" not in message

    rejected = json.loads((ctx.work_dir / "05c-rejected.json").read_text(encoding="utf-8"))
    assert rejected["phrase_ids"] == ["clip-01#002"]


def test_unusable_shot_raises_when_that_flag_is_on(tmp_path: Path, make_clip, monkeypatch):
    clip = make_clip("a.mp4", seconds=6.0)
    ctx = _context(tmp_path, monkeypatch, clip, {"0": CLEAN, "1": BLURRED})

    with pytest.raises(RuntimeError) as error:
        visual_check(ctx)

    assert "unusable" in str(error.value)
    assert ctx.store.read("05c-rejected", RejectedPhrases).phrase_ids == ["clip-01#002"]


def test_findings_are_recorded_without_rejecting_when_both_flags_are_off(
    tmp_path: Path, make_clip, monkeypatch
):
    """With the gate switched off the same reply must still be reported: the
    creator who turned it off gets the findings, just not the refusal."""
    clip = make_clip("a.mp4", seconds=6.0)
    ctx = _context(tmp_path, monkeypatch, clip, {"0": CLEAN, "1": ADULT},
                   config=_both_flags_off())

    result = visual_check(ctx)

    assert isinstance(result, VisualCheck)
    assert [entry.adult_prominent for entry in result.segments] == [False, True]
    assert [entry.checked for entry in result.segments] == [True, True]
    assert result.segments[1].note == "an adult's head fills half the frame"
    assert not (ctx.work_dir / "05c-rejected.json").exists()


def test_a_segment_the_model_ignored_is_not_approved(tmp_path: Path, make_clip, monkeypatch):
    """A silent gap in the reply must never read as approval — that is exactly how
    an unseen shot would slip through a gate that only looks for `true` flags."""
    clip = make_clip("a.mp4", seconds=6.0)
    ctx = _context(tmp_path, monkeypatch, clip, {"0": CLEAN})

    with pytest.raises(RuntimeError) as error:
        visual_check(ctx)

    assert "said nothing" in str(error.value)
    assert "clip-01#002" in str(error.value)

    unchecked = _context(tmp_path / "off", monkeypatch, clip, {"0": CLEAN},
                         config=_both_flags_off())
    result = visual_check(unchecked)
    assert [entry.checked for entry in result.segments] == [True, False]
    assert result.segments[1].adult_prominent is False
    assert result.segments[1].child_visible is False


def test_findings_are_recorded_for_the_run_that_refused_them(
    tmp_path: Path, make_clip, monkeypatch
):
    clip = make_clip("a.mp4", seconds=6.0)
    ctx = _context(tmp_path, monkeypatch, clip, {"0": CLEAN, "1": ADULT})

    with pytest.raises(RuntimeError):
        visual_check(ctx)

    recorded = ctx.store.read("05b-visual", VisualCheck)
    assert recorded.segments[1].adult_prominent is True


def test_rejections_accumulate_across_rounds(tmp_path: Path, make_clip, monkeypatch):
    """Round two must not forget round one, or the planner is offered a shot it
    has already been refused and the loop never converges."""
    clip = make_clip("a.mp4", seconds=6.0)
    ctx = _context(tmp_path, monkeypatch, clip, {"0": CLEAN, "1": ADULT})
    with pytest.raises(RuntimeError):
        visual_check(ctx)

    later = _context(tmp_path, monkeypatch, clip, {"0": ADULT, "1": CLEAN})
    with pytest.raises(RuntimeError):
        visual_check(later)

    assert later.store.read("05c-rejected", RejectedPhrases).phrase_ids == [
        "clip-01#002", "clip-01#001",
    ]


def test_an_insert_is_checked_and_rejected_by_its_reference(
    tmp_path: Path, make_clip, monkeypatch
):
    """Inserts reach the timeline without a phrase, so they are named by the
    reference the planner uses to place them."""
    clip = make_clip("a.mp4", seconds=6.0)
    timeline = Timeline(fps=30.0, width=320, height=240, clips=[
        TimelineClip(src="clip-01", offset=1.0, dur=2.0, start=0.0, is_insert=True),
    ])
    ctx = _context(tmp_path, monkeypatch, clip, {"0": ADULT}, timeline=timeline)

    with pytest.raises(RuntimeError):
        visual_check(ctx)

    assert ctx.store.read("05c-rejected", RejectedPhrases).phrase_ids == ["insert:clip-01"]


def test_frames_are_sampled_inside_the_segment_span(tmp_path: Path, make_clip, monkeypatch):
    """The cut is what ships: frames taken from the middle of the source clip say
    nothing about the seconds the timeline actually uses."""
    clip = make_clip("a.mp4", seconds=6.0)
    ctx = _context(tmp_path, monkeypatch, clip, {"0": CLEAN, "1": CLEAN})
    manifest = ctx.store.read("01-manifest", Manifest)
    timeline = ctx.store.read("05-timeline", Timeline)

    frames = _segment_frames(ctx, manifest, timeline)

    assert sorted(frames) == [0, 1]
    assert all(len(paths) == 3 for paths in frames.values())
    key = manifest.clips[0].source_key
    # 0.35 + 1.30 * 0.15 / 0.50 / 0.85 in milliseconds.
    assert [path.name for path in frames[0]] == [
        f"{key}-00000545.jpg", f"{key}-00001000.jpg", f"{key}-00001455.jpg",
    ]
    assert set(frames[0]).isdisjoint(frames[1])


def test_frames_already_extracted_are_not_re_extracted(tmp_path: Path, make_clip, monkeypatch):
    clip = make_clip("a.mp4", seconds=6.0)
    ctx = _context(tmp_path, monkeypatch, clip, {"0": CLEAN, "1": CLEAN})
    manifest = ctx.store.read("01-manifest", Manifest)
    timeline = ctx.store.read("05-timeline", Timeline)

    first = _segment_frames(ctx, manifest, timeline)

    import videoai.stages.s05b_visual_check as s05b

    calls: list[object] = []
    real_extract_frame = s05b.extract_frame

    def _counting_extract_frame(*args, **kwargs):
        calls.append((args, kwargs))
        return real_extract_frame(*args, **kwargs)

    monkeypatch.setattr(s05b, "extract_frame", _counting_extract_frame)

    second = _segment_frames(ctx, manifest, timeline)

    assert second == first
    assert calls == []  # every frame was already cached on disk


def test_prompt_lists_every_frame_path_and_index(tmp_path: Path, make_clip, monkeypatch):
    clip = make_clip("a.mp4", seconds=6.0)
    ctx = _context(tmp_path, monkeypatch, clip, {"0": CLEAN, "1": CLEAN})
    manifest = ctx.store.read("01-manifest", Manifest)
    timeline = ctx.store.read("05-timeline", Timeline)
    frames = _segment_frames(ctx, manifest, timeline)

    prompt = build_visual_check_prompt(timeline, frames)

    for paths in frames.values():
        for path in paths:
            assert str(path) in prompt
    assert "0 - clip-01 0.35s to 1.65s" in prompt
    assert "1 - clip-01 2.85s to 4.10s" in prompt
    assert "adult_prominent" in prompt


def test_findings_accept_a_wrapped_reply_and_ignore_unknown_keys():
    assert parse_findings({"0": CLEAN}) == {0: CLEAN}
    assert parse_findings({"visual": {"1": CLEAN}, "title": "T"}) == {1: CLEAN}
    assert parse_findings({"notes": "nothing to report"}) == {}


def test_a_segment_whose_frames_could_not_be_extracted_is_refused(
    tmp_path: Path, make_clip, monkeypatch
):
    """A source that cannot be read was never shown to the model at all, so it is
    refused for that reason rather than reported as the model's silence."""
    clip = make_clip("a.mp4", seconds=6.0)
    ctx = _context(tmp_path, monkeypatch, clip, {"0": CLEAN, "1": CLEAN})
    ctx.store.write("01-manifest", Manifest(clips=[ClipInfo(
        clip_id="clip-01", path=str(tmp_path / "gone.mp4"), duration=6.0, width=320,
        height=240, fps=30.0, has_audio=True, source_key="missing",
    )]), fingerprint="fp")

    with pytest.raises(RuntimeError) as error:
        visual_check(ctx)

    assert "no frames could be extracted" in str(error.value)
