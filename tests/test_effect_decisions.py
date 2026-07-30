"""The creator's decisions have to reach the picture, or approving is theatre.

Three things come back from the approval page: an accent turned down, one
dragged somewhere exact, and one swapped for a different drawing. Each has to
survive into the render.
"""
from __future__ import annotations

from pathlib import Path

from videoai.core.models import EffectEvent, Timeline, TimelineClip
from videoai.logic.effects import load_library
from videoai.stages.s08_polish import build_effect_overlays

LIBRARY_DIR = Path("/Users/denyskriukov/Documents/TestDocAI/VideoAI/assets/effects")


def _timeline() -> Timeline:
    return Timeline(fps=30, width=1920, height=1080, clips=[
        TimelineClip(src="clip-01", offset=0.0, dur=10.0, start=0.0),
    ])


def _event(**changes) -> EffectEvent:
    base = dict(at_seconds=2.0, effect_name="comic_starburst",
                screen_position="bottom-right", scale="medium", reason="r")
    return EffectEvent(**{**base, **changes})


def _overlays(events, tmp_path: Path):
    return build_effect_overlays(
        events, load_library(LIBRARY_DIR), _timeline(), [0.0], 0.0,
        (1920, 1080), tmp_path,
    )


def test_an_accent_the_creator_turned_down_is_not_rendered(tmp_path: Path):
    events = [_event(keep=False), _event(at_seconds=5.0, keep=True)]
    overlays = _overlays(events, tmp_path)
    assert len(overlays) == 1
    assert overlays[0].start == 5.0


def test_everything_is_rendered_when_nothing_was_turned_down(tmp_path: Path):
    assert len(_overlays([_event(), _event(at_seconds=5.0)], tmp_path)) == 2


def test_a_dragged_position_is_carried_through(tmp_path: Path):
    overlay = _overlays([_event(x=0.42, y=0.31)], tmp_path)[0]
    assert overlay.point == (0.42, 0.31)


def test_an_untouched_accent_keeps_its_grid_cell(tmp_path: Path):
    overlay = _overlays([_event()], tmp_path)[0]
    assert overlay.point is None
    assert overlay.cell == "bottom-right"


def test_a_swapped_drawing_is_the_one_that_gets_loaded(tmp_path: Path):
    overlay = _overlays([_event(effect_name="sparkle_stars")], tmp_path)[0]
    assert overlay.name == "sparkle_stars"
