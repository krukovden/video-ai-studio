"""A decision belongs to a moment in a shot, never to a position in a list.

Matching by index looked fine until the plan changed underneath it: every
decision slid one place along, so a badge chosen for "he spots the syringe"
landed on the box being opened, and the last one fell off the end. The names all
matched, which is what made it convincing and wrong.

What comes back from the page is now filed under the accent's own anchor in
`05e-overrides` — the creator's artifact — rather than written back into the
model's plan.
"""
from __future__ import annotations

from pathlib import Path

from videoai.core.models import (
    EffectEvent,
    EffectPlan,
    Overrides,
    Timeline,
    TimelineClip,
)
from videoai.core.store import ArtifactStore
from videoai.logic.approval import apply_decisions
from videoai.logic.decisions import OVERRIDES_ARTIFACT, apply_effect_overrides


def _event(at: float, name: str, ref: str = "", at_in_clip: float = 0.0) -> EffectEvent:
    return EffectEvent(at_seconds=at, effect_name=name, screen_position="center",
                       scale="medium", reason="r", clip_ref=ref, at_in_clip=at_in_clip)


def _store(tmp_path: Path, *events: EffectEvent) -> ArtifactStore:
    store = ArtifactStore(tmp_path / "work")
    store.write("05d-effects", EffectPlan(events=list(events)), fingerprint="f")
    return store


def _decision(index: int, at: float, name: str, **extra) -> dict:
    return {"index": index, "at": at, "effect_name": name, "keep": True, **extra}


def _overrides(store: ArtifactStore) -> Overrides:
    return store.read(OVERRIDES_ARTIFACT, Overrides)


def _applied(store: ArtifactStore) -> EffectPlan:
    """The plan as the renderer will see it: the model's proposal, overridden."""
    plan, _ = apply_effect_overrides(
        store.read("05d-effects", EffectPlan), _overrides(store)
    )
    return plan


def test_a_decision_lands_on_its_own_moment_not_its_old_slot(tmp_path: Path):
    # The plan lost an earlier accent, so everything after it shifted place.
    store = _store(tmp_path, _event(40.0, "sparkle_stars"), _event(75.0, "badge_paint"))
    apply_decisions(store, {"events": [
        _decision(1, 40.0, "sparkle_stars", x=0.1, y=0.1, moved=True),
        _decision(2, 75.0, "badge_paint", x=0.9, y=0.9, moved=True),
    ]})
    events = _applied(store).events
    assert (events[0].x, events[0].y) == (0.1, 0.1)
    assert (events[1].x, events[1].y) == (0.9, 0.9)


def test_a_decision_for_a_moment_that_is_gone_is_not_applied_to_a_neighbour(tmp_path: Path):
    store = _store(tmp_path, _event(75.0, "badge_paint"))
    result = apply_decisions(store, {"events": [
        _decision(0, 19.0, "badge_thinking", x=0.7, y=0.2, moved=True),
    ]})
    assert _applied(store).events[0].x is None
    assert result.unmatched == 1


def test_a_few_frames_of_drift_is_still_the_same_moment(tmp_path: Path):
    store = _store(tmp_path, _event(40.3, "sparkle_stars"))
    apply_decisions(store, {"events": [
        _decision(0, 40.0, "sparkle_stars", x=0.4, y=0.4, moved=True),
    ]})
    assert _applied(store).events[0].x == 0.4


def test_a_swap_is_matched_on_where_it_came_from(tmp_path: Path):
    """The page reports the NEW name; the plan still holds the old one."""
    store = _store(tmp_path, _event(163.0, "badge_boom"))
    apply_decisions(store, {"events": [
        {"index": 0, "at": 163.0, "effect_name": "badge_blush",
         "from_name": "badge_boom", "keep": True, "swapped": True},
    ]})
    assert _applied(store).events[0].effect_name == "badge_blush"


def test_an_old_decisions_file_without_times_still_works(tmp_path: Path):
    store = _store(tmp_path, _event(40.0, "sparkle_stars"))
    result = apply_decisions(store, {"events": [
        {"index": 0, "keep": False, "effect_name": "sparkle_stars"},
    ]})
    assert _applied(store).events[0].keep is False
    assert result.dropped == 1


# --- The decision is filed under the shot, so the edit may move under it -------


def _anchored_project(tmp_path: Path) -> ArtifactStore:
    """Two shots, one accent in each, both anchored to the shot they mark."""
    store = ArtifactStore(tmp_path / "work")
    store.write("05-timeline", Timeline(fps=30, width=1920, height=1080, clips=[
        TimelineClip(ref="clip-01#001", src="clip-01", offset=0.0, dur=10.0, start=0.0),
        TimelineClip(ref="clip-01#002", src="clip-01", offset=20.0, dur=6.0, start=10.0),
    ]), fingerprint="f")
    store.write("05d-effects", EffectPlan(events=[
        _event(4.0, "badge_boom", ref="clip-01#001", at_in_clip=4.0),
        _event(12.0, "sparkle_stars", ref="clip-01#002", at_in_clip=2.0),
    ]), fingerprint="f")
    return store


def test_a_decision_is_filed_under_the_shot_it_was_made_about(tmp_path: Path):
    store = _anchored_project(tmp_path)
    apply_decisions(store, {"events": [
        _decision(1, 12.0, "sparkle_stars", x=0.8, y=0.2, moved=True),
    ]})
    decided = _overrides(store).effects
    assert [(item.clip_ref, item.at_in_clip) for item in decided] == [("clip-01#002", 2.0)]


def test_a_decision_never_moves_to_another_shot_when_the_edit_is_reordered(tmp_path: Path):
    """The two shots swap places, so the second accent's absolute second now
    belongs to the FIRST shot. Nothing may be re-applied on the strength of that."""
    store = _anchored_project(tmp_path)
    apply_decisions(store, {"events": [
        _decision(1, 12.0, "sparkle_stars", x=0.8, y=0.2, moved=True),
    ]})
    store.write("05-timeline", Timeline(fps=30, width=1920, height=1080, clips=[
        TimelineClip(ref="clip-01#002", src="clip-01", offset=20.0, dur=6.0, start=0.0),
        TimelineClip(ref="clip-01#001", src="clip-01", offset=0.0, dur=10.0, start=6.0),
    ]), fingerprint="f")

    events = _applied(store).events

    placed = {event.effect_name: (event.x, event.y) for event in events}
    assert placed["sparkle_stars"] == (0.8, 0.2)
    assert placed["badge_boom"] == (None, None)


def test_a_second_visit_to_the_page_does_not_undo_the_first(tmp_path: Path):
    """The page is drawn from the overridden plan, so saving one accent must not
    reset the seven the creator is not looking at."""
    store = _anchored_project(tmp_path)
    apply_decisions(store, {"events": [
        _decision(0, 4.0, "badge_boom", x=0.1, y=0.1, moved=True),
    ]})
    apply_decisions(store, {"events": [
        _decision(1, 12.0, "sparkle_stars", keep=False),
    ]})

    events = _applied(store).events

    assert (events[0].x, events[0].y) == (0.1, 0.1)
    assert events[1].keep is False
    assert len(_overrides(store).effects) == 2, "one entry per accent, not per save"
