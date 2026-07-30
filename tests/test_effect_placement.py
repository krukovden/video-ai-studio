"""Where an accent goes is measured, not guessed.

The model chooses which moment deserves an accent and what it should express.
It has never seen the frame, so its screen position is a guess — on this
project's first delivery, one of seven landed where anything was happening. The
placement stage looks.
"""
from __future__ import annotations

from videoai.core.models import EffectEvent, Manifest, ClipInfo, Timeline, TimelineClip
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


def test_unreadable_media_leaves_the_plan_alone(monkeypatch):
    def explode(path, at):
        raise OSError("codec")

    monkeypatch.setattr("videoai.stages.s05d_effects.busiest_cell", explode)
    placed = place_on_motion([_event("center")], _timeline(), _manifest(), lambda p: p)
    assert placed[0].screen_position == "center"


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
