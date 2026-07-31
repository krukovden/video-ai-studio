"""Where an accent goes is measured, not guessed.

The model chooses which moment deserves an accent and what it should express.
It has never seen the frame, so its screen position is a guess — on this
project's first delivery, one of seven landed where anything was happening. The
placement stage looks.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from videoai.core.models import (
    ClipEvent,
    ClipInfo,
    ClipNote,
    ClipNotes,
    EffectEvent,
    Manifest,
    Timeline,
    TimelineClip,
)
from videoai.stages.s05d_effects import place_on_motion


def _timeline() -> Timeline:
    return Timeline(fps=30, width=1920, height=1080, clips=[
        TimelineClip(src="clip-01", offset=0.0, dur=10.0, start=0.0),
    ])


def _manifest(path: str = "/nowhere/clip.mp4") -> Manifest:
    return Manifest(clips=[ClipInfo(
        clip_id="clip-01", path=path, proxy_path=path, duration=10.0,
        width=1920, height=1080, fps=30.0, has_audio=True)])


def _event(cell: str = "top-left") -> EffectEvent:
    return EffectEvent(at_seconds=2.0, effect_name="badge_boom",
                       screen_position=cell, scale="medium", reason="r")


def test_a_measured_cell_replaces_the_guess(monkeypatch):
    monkeypatch.setattr(
        "videoai.stages.s05d_effects.busiest_cell", lambda path, at: "bottom-right"
    )
    placed = place_on_motion([_event("top-left")], _timeline(), _manifest(), lambda p: p)
    assert placed[0].screen_position == "bottom-right"


def test_the_guess_survives_when_nothing_is_moving(monkeypatch):
    """A still shot has nowhere to point at, and moving the accent to a cell
    picked at random would be the very thing this replaces."""
    monkeypatch.setattr(
        "videoai.stages.s05d_effects.busiest_cell", lambda path, at: None
    )
    placed = place_on_motion([_event("top-left")], _timeline(), _manifest(), lambda p: p)
    assert placed[0].screen_position == "top-left"


def test_unreadable_media_stops_the_run_rather_than_keeping_the_guess(monkeypatch):
    """Measuring is the entire point of this step. Swallowing the error and
    keeping the model's cell is how seven of seven accents landed over a door, an
    empty wall and a paper towel — and nothing said so."""
    def explode(path, at):
        raise OSError("codec")

    monkeypatch.setattr("videoai.stages.s05d_effects.busiest_cell", explode)
    with pytest.raises(OSError, match="codec"):
        place_on_motion([_event("center")], _timeline(), _manifest(), lambda p: p)


def _notes(at: float, where: str) -> ClipNotes:
    return ClipNotes(notes=[ClipNote(
        clip_id="clip-01", source_key="k", duration=10.0,
        events=[ClipEvent(at=at, what="the pop", where=where)],
    )])


def test_a_still_shot_falls_back_to_where_the_clip_was_watched_happening(monkeypatch):
    """Nothing measurably moves, so there is nothing to subtract — but a model
    that actually watched this clip wrote down where the action was, and it has
    seen more of the frame than the effects model, which has seen none of it."""
    monkeypatch.setattr(
        "videoai.stages.s05d_effects.busiest_cell", lambda path, at: None
    )
    placed = place_on_motion(
        [_event("top-left")], _timeline(), _manifest(), lambda p: p,
        _notes(2.2, "bottom-right"),
    )
    assert placed[0].screen_position == "bottom-right"


def test_a_measurement_still_beats_what_was_merely_watched(monkeypatch):
    monkeypatch.setattr(
        "videoai.stages.s05d_effects.busiest_cell", lambda path, at: "middle-left"
    )
    placed = place_on_motion(
        [_event("top-left")], _timeline(), _manifest(), lambda p: p,
        _notes(2.0, "bottom-right"),
    )
    assert placed[0].screen_position == "middle-left"


def test_a_watched_moment_too_far_from_the_accent_is_not_about_it(monkeypatch):
    monkeypatch.setattr(
        "videoai.stages.s05d_effects.busiest_cell", lambda path, at: None
    )
    placed = place_on_motion(
        [_event("top-left")], _timeline(), _manifest(), lambda p: p,
        _notes(7.0, "bottom-right"),
    )
    assert placed[0].screen_position == "top-left"


def test_a_watched_moment_with_no_position_says_nothing(monkeypatch):
    monkeypatch.setattr(
        "videoai.stages.s05d_effects.busiest_cell", lambda path, at: None
    )
    placed = place_on_motion(
        [_event("top-left")], _timeline(), _manifest(), lambda p: p, _notes(2.0, ""),
    )
    assert placed[0].screen_position == "top-left"


def test_everything_else_about_the_event_is_untouched(monkeypatch):
    monkeypatch.setattr(
        "videoai.stages.s05d_effects.busiest_cell", lambda path, at: "center"
    )
    original = EffectEvent(at_seconds=3.0, effect_name="badge_star",
                           screen_position="top-left", scale="large",
                           text="WOW", reason="the reveal")
    placed = place_on_motion([original], _timeline(), _manifest(), lambda p: p)[0]
    assert (placed.effect_name, placed.scale, placed.text, placed.at_seconds) == (
        "badge_star", "large", "WOW", 3.0,
    )


def test_an_event_outside_the_timeline_is_left_where_it_was(monkeypatch):
    monkeypatch.setattr(
        "videoai.stages.s05d_effects.busiest_cell", lambda path, at: "center"
    )
    late = EffectEvent(at_seconds=999.0, effect_name="badge_boom",
                       screen_position="top-left", scale="medium", reason="r")
    assert place_on_motion([late], _timeline(), _manifest(), lambda p: p)[0].screen_position == "top-left"


# --- Order of operations: the position must be measured at the time the accent
# will actually play, not at the time the model first guessed ---


def _effects_context(tmp_path: Path, reply: dict, monkeypatch) -> "StageContext":
    from videoai.config import Config
    from videoai.core.models import Analysis, Observation, StoryPlan, Transcript
    from videoai.core.registry import StageContext
    from videoai.core.store import ArtifactStore

    llm = tmp_path / "llm.json"
    llm.write_text(json.dumps(reply), encoding="utf-8")
    monkeypatch.setenv("VIDEOAI_MOCK_LLM", str(llm))
    # Never opened: the measurement itself is stubbed out below. It has to exist
    # because `place_on_motion` no longer treats missing media as a still shot.
    (tmp_path / "clip-01.mp4").write_bytes(b"")

    work = tmp_path / "work"
    ctx = StageContext(
        project_dir=tmp_path, input_dir=tmp_path, work_dir=work,
        output_dir=tmp_path / "output",
        config=Config(providers={"asr": "mock", "llm": "mock"}),
        store=ArtifactStore(work),
    )
    ctx.store.write("05-proposal", Timeline(
        fps=30, width=1920, height=1080,
        clips=[TimelineClip(ref="clip-01#001", src="clip-01", offset=0.0, dur=10.0,
                            start=0.0, quote="watch this")],
    ), fingerprint="proposal")
    ctx.store.write("05a-storyplan", StoryPlan(title="T"), fingerprint="story")
    ctx.store.write("03-transcript", Transcript(provider="mock"), fingerprint="transcript")
    ctx.store.write("04-analysis", Analysis(
        provider="mock",
        observations=[Observation(clip_id="clip-01", at=4.5, reel_at=0.0, what="it pops")],
    ), fingerprint="analysis")
    ctx.store.write("01-manifest", Manifest(clips=[ClipInfo(
        clip_id="clip-01", path=str(tmp_path / "clip-01.mp4"), duration=10.0,
        width=1920, height=1080, fps=30.0, has_audio=True)]), fingerprint="manifest")
    return ctx


def test_the_frame_is_measured_at_the_moment_the_accent_ends_up_on(
    tmp_path: Path, monkeypatch
):
    """Placement used to run before snapping, so the busiest cell was read from a
    frame up to 0.6s away from where the accent finally landed. On the fast action
    these accents exist for, half a second is a different part of the frame."""
    from videoai.stages.s05d_effects import effects

    asked: list[float] = []

    def record(path, at):
        asked.append(at)
        return "bottom-right"

    monkeypatch.setattr("videoai.stages.s05d_effects.busiest_cell", record)
    ctx = _effects_context(tmp_path, {"events": [{
        "at_seconds": 4.1, "effect_name": "badge_boom",
        "screen_position": "top-left", "scale": "medium", "reason": "it pops",
    }]}, monkeypatch)

    plan = effects(ctx)

    assert plan.events[0].at_seconds == 4.5
    assert asked == [4.5]
    assert plan.events[0].screen_position == "bottom-right"
