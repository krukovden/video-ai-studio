"""Accents land on moments somebody watched, not on times a model chose.

The model picks WHICH moment deserves an accent; the clock belongs to the
pipeline. When the analysis was made by a model that could see the footage,
every chosen time is pulled onto the nearest observed moment — and when it was
not, nothing moves and the behaviour is exactly as before.
"""
from __future__ import annotations

from videoai.core.models import EffectEvent, Observation, Timeline, TimelineClip
from videoai.stages.s05d_effects import SNAP_SECONDS, snap_to_observed


def _timeline() -> Timeline:
    return Timeline(fps=30, width=1920, height=1080, clips=[
        TimelineClip(src="clip-01", offset=10.0, dur=6.0, start=0.0),
    ])


def _seen(at: float, what: str = "it gives way") -> Observation:
    return Observation(clip_id="clip-01", at=at, reel_at=0.0, kind="action", what=what)


def _event(at: float) -> EffectEvent:
    return EffectEvent(
        at_seconds=at, effect_name="comic_starburst",
        screen_position="center", scale="medium", reason="the pop",
    )


def test_a_nearby_choice_is_pulled_onto_the_observed_moment():
    # Observed at 12.4s of the clip, which is 2.4s into the edit.
    events = snap_to_observed([_event(2.7)], [_seen(12.4)], _timeline())
    assert events[0].at_seconds == 2.4


def test_a_choice_far_from_any_observation_is_left_alone():
    """Not every accent marks a physical event; a reaction beat has no onset."""
    events = snap_to_observed([_event(5.0)], [_seen(12.4)], _timeline())
    assert events[0].at_seconds == 5.0


def test_the_nearest_observation_wins():
    events = snap_to_observed(
        [_event(2.5)], [_seen(12.4, "closer"), _seen(12.9, "further")], _timeline()
    )
    assert events[0].at_seconds == 2.4


def test_nothing_moves_without_observations():
    """A text-only analysis leaves the stage behaving exactly as it did."""
    events = snap_to_observed([_event(2.7)], [], _timeline())
    assert events[0].at_seconds == 2.7


def test_an_observation_the_edit_dropped_cannot_attract_anything():
    # 30.0s of the clip is outside the 10-16s the edit uses.
    events = snap_to_observed([_event(2.7)], [_seen(30.0)], _timeline())
    assert events[0].at_seconds == 2.7


def test_the_tolerance_is_a_real_boundary():
    at = 2.4 + SNAP_SECONDS + 0.05
    assert snap_to_observed([_event(at)], [_seen(12.4)], _timeline())[0].at_seconds == at


def test_every_other_field_survives_the_move():
    original = EffectEvent(
        at_seconds=2.7, effect_name="sparkles", screen_position="top-left",
        scale="small", text="WOW", reason="the reveal",
    )
    moved = snap_to_observed([original], [_seen(12.4)], _timeline())[0]
    assert moved.at_seconds == 2.4
    assert (moved.effect_name, moved.screen_position, moved.scale, moved.text) == (
        "sparkles", "top-left", "small", "WOW",
    )
