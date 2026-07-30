"""The motions an accent can have, as arithmetic.

Every frame of a composited accent comes from this function and nothing else, so
each motion is checked at the moments that decide whether it reads as animation
or as a cut: where it starts, what it does in the middle, and where it ends.
"""
from __future__ import annotations

import pytest

from videoai.logic.effects import ANIMATIONS, animation_transform


@pytest.mark.parametrize("animation", ANIMATIONS)
def test_every_motion_starts_and_ends_invisible(animation: str):
    """An accent that appears at full opacity reads as a glitch, and one that is
    still solid on its last frame reads as a dropped shot."""
    assert animation_transform(animation, 0.0).alpha < 0.2, animation
    assert animation_transform(animation, 1.0).alpha < 0.2, animation


@pytest.mark.parametrize("animation", ANIMATIONS)
def test_every_motion_is_fully_visible_somewhere_in_its_life(animation: str):
    """Not at any particular instant: drift-up deliberately fades as it rises, so
    it is brightest early rather than halfway."""
    brightest = max(animation_transform(animation, i / 20).alpha for i in range(21))
    assert brightest > 0.8, animation


@pytest.mark.parametrize("animation", ANIMATIONS)
def test_no_motion_ever_explodes_or_vanishes_while_visible(animation: str):
    """A pop-in legitimately starts at zero size — that is the pop. What must
    never happen is a sprite that is visible and has no size, or one that grows
    past the frame."""
    for step in range(21):
        t = animation_transform(animation, step / 20)
        assert t.scale_x < 3.0 and t.scale_y < 3.0, (animation, step)
        if t.alpha > 0.1:
            assert t.scale_x > 0.05 and t.scale_y > 0.05, (animation, step)


@pytest.mark.parametrize("animation", ANIMATIONS)
def test_progress_outside_the_window_is_clamped_not_extrapolated(animation: str):
    assert animation_transform(animation, -5.0) == animation_transform(animation, 0.0)
    assert animation_transform(animation, 5.0) == animation_transform(animation, 1.0)


def test_pop_in_overshoots_its_final_size():
    """Arriving at exactly the final size reads as a cut, not as a pop."""
    peak = max(animation_transform("pop-in", i / 40).scale_x for i in range(41))
    assert peak > 1.1
    assert animation_transform("pop-in", 0.9).scale_x == pytest.approx(1.0, abs=0.02)


def test_squash_stretches_one_axis_against_the_other():
    """Squash and stretch is the whole difference between a cartoon accent and a
    sticker that changes size."""
    wide = [animation_transform("squash", i / 40) for i in range(41)]
    assert any(t.scale_x > t.scale_y * 1.15 for t in wide)
    assert any(t.scale_y > t.scale_x * 1.15 for t in wide)


def test_bounce_comes_down_and_settles():
    start = animation_transform("bounce", 0.05).dy
    end = animation_transform("bounce", 0.95).dy
    assert start < -0.1, "it should start above where it lands"
    assert abs(end) < 0.05, "and finish at rest"


def test_spin_in_turns_and_stops():
    assert abs(animation_transform("spin-in", 0.05).rotation) > 40
    assert abs(animation_transform("spin-in", 0.95).rotation) < 5


def test_swing_crosses_upright_rather_than_leaning_one_way():
    angles = [animation_transform("swing", i / 40).rotation for i in range(41)]
    assert max(angles) > 4 and min(angles) < -4


def test_drift_up_rises_and_fades():
    assert animation_transform("drift-up", 0.9).dy < animation_transform("drift-up", 0.1).dy
