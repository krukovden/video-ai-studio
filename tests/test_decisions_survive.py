"""A creator's decisions outlive the plan they were made against.

Re-planning throws away the model's previous proposal, which is the point. It
must not throw away what a person decided about it: an accent they turned down
comes straight back, a badge they dragged into place jumps to a corner, and
nothing says so. That is the same silent loss as the browser tab, one layer
down.
"""
from __future__ import annotations

from videoai.core.models import EffectEvent, EffectPlan
from videoai.logic.decisions import carry_decisions


def _event(at: float, name: str, **changes) -> EffectEvent:
    base = dict(at_seconds=at, effect_name=name, screen_position="center",
                scale="medium", reason="r")
    return EffectEvent(**{**base, **changes})


def test_a_refusal_is_carried_onto_the_new_plan():
    old = EffectPlan(events=[_event(10.0, "badge_boom", keep=False)])
    fresh = EffectPlan(events=[_event(10.0, "badge_boom")])
    carried, lost = carry_decisions(fresh, old)
    assert carried.events[0].keep is False
    assert lost == 0


def test_a_dragged_position_is_carried():
    old = EffectPlan(events=[_event(10.0, "badge_boom", x=0.3, y=0.7)])
    fresh = EffectPlan(events=[_event(10.0, "badge_boom")])
    carried, _ = carry_decisions(fresh, old)
    assert (carried.events[0].x, carried.events[0].y) == (0.3, 0.7)


def test_a_moment_that_moved_slightly_still_matches():
    """The model re-reads the same timeline and lands a few frames away; that is
    the same moment, not a new one."""
    old = EffectPlan(events=[_event(10.0, "badge_boom", keep=False)])
    fresh = EffectPlan(events=[_event(10.4, "badge_boom")])
    carried, lost = carry_decisions(fresh, old)
    assert carried.events[0].keep is False and lost == 0


def test_a_different_moment_does_not_inherit_someone_elses_decision():
    old = EffectPlan(events=[_event(10.0, "badge_boom", keep=False)])
    fresh = EffectPlan(events=[_event(40.0, "badge_boom")])
    carried, lost = carry_decisions(fresh, old)
    assert carried.events[0].keep is True
    assert lost == 1, "and the loss is reported rather than hidden"


def test_a_decision_with_nowhere_to_go_is_counted_not_swallowed():
    old = EffectPlan(events=[
        _event(10.0, "badge_boom", keep=False),
        _event(20.0, "badge_star", x=0.2, y=0.2),
    ])
    fresh = EffectPlan(events=[_event(10.0, "badge_boom")])
    carried, lost = carry_decisions(fresh, old)
    assert carried.events[0].keep is False
    assert lost == 1


def test_a_plan_nobody_touched_carries_nothing_and_loses_nothing():
    old = EffectPlan(events=[_event(10.0, "badge_boom")])
    fresh = EffectPlan(events=[_event(10.0, "badge_boom")])
    carried, lost = carry_decisions(fresh, old)
    assert carried.events[0].keep is True and lost == 0


def test_one_decision_cannot_be_spent_twice():
    old = EffectPlan(events=[_event(10.0, "badge_boom", keep=False)])
    fresh = EffectPlan(events=[_event(10.0, "badge_boom"), _event(10.2, "badge_boom")])
    carried, _ = carry_decisions(fresh, old)
    assert [e.keep for e in carried.events] == [False, True]
