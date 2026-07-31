"""The creator's decisions have to reach the picture, or approving is theatre.

Three things come back from the approval page: an accent turned down, one
dragged somewhere exact, and one swapped for a different drawing. Each has to
survive into the render.
"""
from __future__ import annotations

from pathlib import Path

from videoai.core.models import EffectEvent, EffectPlan, Timeline, TimelineClip
from videoai.logic.decisions import reproject_events
from videoai.logic.effects import default_library_dir, load_library
from videoai.logic.timeline import renumber
from videoai.stages.s08_polish import build_effect_overlays

# Resolved from the package, never a literal path: a literal one here read a
# different checkout of this repository, so these tests exercised that tree's
# sprite library instead of the one they sit beside.
LIBRARY_DIR = default_library_dir()


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


# --- and the shot it was about is what it is composited over ------------------


def _two_shots(order: list[str]) -> Timeline:
    clips = {
        "hook": TimelineClip(ref="clip-01#001", src="clip-01", offset=0.0, dur=4.0,
                             start=0.0, beat="Hook"),
        "closing": TimelineClip(ref="clip-01#002", src="clip-01", offset=20.0, dur=6.0,
                                start=0.0, beat="Closing"),
    }
    timeline = Timeline(fps=30, width=1920, height=1080,
                        clips=[clips[name] for name in order])
    return renumber(timeline)


def test_an_accent_is_composited_over_the_shot_it_is_anchored_to(tmp_path: Path):
    """The two shots swap places, so the closing shot now starts at 0.0s. Its
    accent has to move with it rather than stay on the second it used to occupy."""
    event = _event(clip_ref="clip-01#002", at_in_clip=1.5, at_seconds=5.5)
    edit = _two_shots(["closing", "hook"])

    overlay = build_effect_overlays(
        [event], load_library(LIBRARY_DIR), edit,
        [clip.start for clip in edit.clips], 0.0, (1920, 1080), tmp_path,
    )[0]

    assert overlay.start == 1.5


def test_an_accent_whose_shot_is_not_in_the_edit_is_never_drawn(tmp_path: Path):
    """`reproject_events` removes it; this is the second line of defence, so a
    stale `at_seconds` cannot put it over whatever moved underneath."""
    plan = EffectPlan(events=[_event(clip_ref="clip-01#404", at_in_clip=1.0, at_seconds=5.5)])
    edit = _two_shots(["hook", "closing"])

    projected, dropped = reproject_events(plan, edit)

    assert projected.events == []
    assert len(dropped) == 1 and "clip-01#404" in dropped[0]
