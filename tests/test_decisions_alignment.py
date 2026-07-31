"""A decision belongs to a moment, never to a position in a list.

Matching by index looked fine until the plan changed underneath it: every
decision slid one place along, so a badge chosen for "he spots the syringe"
landed on the box being opened, and the last one fell off the end. The names all
matched, which is what made it convincing and wrong.
"""
from __future__ import annotations

from pathlib import Path

from videoai.core.models import EffectEvent, EffectPlan
from videoai.core.store import ArtifactStore
from videoai.logic.approval import apply_decisions


def _event(at: float, name: str) -> EffectEvent:
    return EffectEvent(at_seconds=at, effect_name=name, screen_position="center",
                       scale="medium", reason="r")


def _store(tmp_path: Path, *events: EffectEvent) -> ArtifactStore:
    store = ArtifactStore(tmp_path / "work")
    store.write("05d-effects", EffectPlan(events=list(events)), fingerprint="f")
    return store


def _decision(index: int, at: float, name: str, **extra) -> dict:
    return {"index": index, "at": at, "effect_name": name, "keep": True, **extra}


def test_a_decision_lands_on_its_own_moment_not_its_old_slot(tmp_path: Path):
    # The plan lost an earlier accent, so everything after it shifted place.
    store = _store(tmp_path, _event(40.0, "sparkle_stars"), _event(75.0, "badge_paint"))
    apply_decisions(store, {"events": [
        _decision(1, 40.0, "sparkle_stars", x=0.1, y=0.1, moved=True),
        _decision(2, 75.0, "badge_paint", x=0.9, y=0.9, moved=True),
    ]})
    plan = store.read("05d-effects", EffectPlan)
    assert (plan.events[0].x, plan.events[0].y) == (0.1, 0.1)
    assert (plan.events[1].x, plan.events[1].y) == (0.9, 0.9)


def test_a_decision_for_a_moment_that_is_gone_is_not_applied_to_a_neighbour(tmp_path: Path):
    store = _store(tmp_path, _event(75.0, "badge_paint"))
    result = apply_decisions(store, {"events": [
        _decision(0, 19.0, "badge_thinking", x=0.7, y=0.2, moved=True),
    ]})
    assert store.read("05d-effects", EffectPlan).events[0].x is None
    assert result.unmatched == 1


def test_a_few_frames_of_drift_is_still_the_same_moment(tmp_path: Path):
    store = _store(tmp_path, _event(40.3, "sparkle_stars"))
    apply_decisions(store, {"events": [
        _decision(0, 40.0, "sparkle_stars", x=0.4, y=0.4, moved=True),
    ]})
    assert store.read("05d-effects", EffectPlan).events[0].x == 0.4


def test_a_swap_is_matched_on_where_it_came_from(tmp_path: Path):
    """The page reports the NEW name; the plan still holds the old one."""
    store = _store(tmp_path, _event(163.0, "badge_boom"))
    apply_decisions(store, {"events": [
        {"index": 0, "at": 163.0, "effect_name": "badge_blush",
         "from_name": "badge_boom", "keep": True, "swapped": True},
    ]})
    assert store.read("05d-effects", EffectPlan).events[0].effect_name == "badge_blush"


def test_an_old_decisions_file_without_times_still_works(tmp_path: Path):
    store = _store(tmp_path, _event(40.0, "sparkle_stars"))
    result = apply_decisions(store, {"events": [
        {"index": 0, "keep": False, "effect_name": "sparkle_stars"},
    ]})
    assert store.read("05d-effects", EffectPlan).events[0].keep is False
    assert result.dropped == 1
