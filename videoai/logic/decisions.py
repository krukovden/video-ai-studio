"""Carrying what a person decided across a plan they did not ask to be redone.

Re-planning discards the model's previous proposal, which is the point of
re-planning. It must not also discard what a creator decided about it — an
accent they turned down reappearing, or a badge they placed by hand jumping back
to a corner, with nothing said. That is the same silent loss as a closed browser
tab, one layer down, and it is worse there because nobody is watching.

A decision belongs to a MOMENT IN A SHOT, not to a position in a list and not to
a second of the edit. Indices renumber whenever the model returns a different
number of accents; absolute times move whenever the running order does. So the
one thing matched on is the anchor: this shot, this far into it.

There is exactly one matcher here, and every caller uses it. There used to be
two — one for the approval page, one for the re-plan — with subtly different
algorithms, which meant the answer to "does my decision survive?" depended on
which door it came through.

Matching is scoped to a single `clip_ref`. That is what makes the failure mode
safe: two accents can no longer be confused across shots however the edit moves,
so the worst case is a decision that matches nothing and is REPORTED, never a
decision applied to footage it was not about. A decision that quietly evaporates
is the failure this module exists to prevent, so it is never allowed to
evaporate quietly.
"""
from __future__ import annotations

from dataclasses import dataclass

from videoai.core.models import (
    EffectEvent,
    EffectOverride,
    EffectPlan,
    Overrides,
    Timeline,
)
from videoai.core.store import ArtifactStore

OVERRIDES_ARTIFACT = "05e-overrides"

# How far a re-planned accent may drift and still be the same moment. The model
# re-reads the same shot, so it lands within a few frames of where it landed
# before, and `snap_to_observed` may move one by up to 0.6s; a second covers both
# and is still far short of the two-second gap the placement prompt requires
# between accents.
SAME_MOMENT_SECONDS = 1.0


@dataclass(frozen=True)
class Moment:
    """One accent's identity for matching: which shot, how far in, what it draws.

    `at_seconds` is carried alongside only so an artifact written before refs
    existed still has something to be matched on. When both sides know their
    `clip_ref` it is never consulted.
    """

    clip_ref: str
    at_in_clip: float
    at_seconds: float
    effect_name: str = ""


def override_key(decided: EffectOverride) -> str:
    """The identity `EffectEvent.anchor_key` gives, spelled for the override side.

    Both sides have to agree on what "the same accent" is filed under, and the
    model's own property is the definition — so this reproduces it rather than
    inventing a second one. An override authored against an artifact from before
    refs existed carries the absolute time in `at_in_clip`, because that is the
    only clock such an accent has.
    """
    if not decided.clip_ref:
        return f"@{decided.at_in_clip:.2f}"
    return f"{decided.clip_ref}@{decided.at_in_clip:.2f}"


def moment_of(event: EffectEvent) -> Moment:
    return Moment(
        clip_ref=event.clip_ref,
        at_in_clip=event.at_in_clip,
        at_seconds=event.at_seconds,
        effect_name=event.effect_name,
    )


def _distance(target: Moment, candidate: Moment) -> float | None:
    """How far apart two moments are, or None when they cannot be the same one."""
    if target.effect_name and candidate.effect_name:
        if target.effect_name != candidate.effect_name:
            return None
    if target.clip_ref and candidate.clip_ref:
        if target.clip_ref != candidate.clip_ref:
            return None
        return abs(target.at_in_clip - candidate.at_in_clip)
    # One side predates refs. Absolute time is all it has, so it is compared on
    # absolute time exactly as it was before — the old behaviour, kept for old
    # artifacts and for nothing else.
    return abs(target.at_seconds - candidate.at_seconds)


def nearest_moment(
    target: Moment, candidates: list[Moment], spent: set[int]
) -> int | None:
    """Index of the unspent candidate `target` is about, or None.

    Nearest wins, ties go to the earlier candidate, and nothing beyond
    `SAME_MOMENT_SECONDS` matches at all. `spent` is the caller's: one decision
    can only be applied once, or a plan that repeats an accent would collect the
    same placement twice.
    """
    best_index, best_distance = None, SAME_MOMENT_SECONDS
    for index, candidate in enumerate(candidates):
        if index in spent:
            continue
        distance = _distance(target, candidate)
        if distance is not None and distance <= best_distance:
            best_index, best_distance = index, distance
    return best_index


def _was_decided(event: EffectEvent) -> bool:
    """Whether a person touched this one at all."""
    return (not event.keep) or event.x is not None or event.y is not None


def carry_decisions(fresh: EffectPlan, previous: EffectPlan) -> tuple[EffectPlan, int]:
    """Apply an older plan's human decisions to a newly planned one.

    Only for decisions that live inside the plan itself, which is where they were
    kept before `05e-overrides` existed. Returns the plan and how many decisions
    had nowhere to land, so the caller can say so out loud.
    """
    decided = [event for event in previous.events if _was_decided(event)]
    if not decided:
        return fresh, 0

    anchors = [moment_of(event) for event in decided]
    spent: set[int] = set()
    updated: list[EffectEvent] = []
    for event in fresh.events:
        match_index = nearest_moment(moment_of(event), anchors, spent)
        if match_index is None:
            updated.append(event)
            continue
        spent.add(match_index)
        old = decided[match_index]
        changes: dict = {"keep": old.keep}
        if old.x is not None and old.y is not None:
            changes["x"] = old.x
            changes["y"] = old.y
        updated.append(event.model_copy(update=changes))

    return fresh.model_copy(update={"events": updated}), len(decided) - len(spent)


def read_overrides(store: ArtifactStore) -> Overrides:
    """What the creator decided, or an empty answer when they have not looked yet.

    No stage produces this artifact, so its absence is the ordinary case and not
    a missing dependency: an untouched project delivers the plan as proposed.
    """
    if not store.exists(OVERRIDES_ARTIFACT):
        return Overrides()
    return store.read(OVERRIDES_ARTIFACT, Overrides)


def reproject_events(
    plan: EffectPlan, timeline: Timeline
) -> tuple[EffectPlan, list[str]]:
    """Put every accent back on the current edit's clock, and drop the homeless.

    `at_seconds` is a cache of "where this anchor currently falls", so it is
    recomputed here rather than trusted: switch one clip off and every later clip
    slides several seconds earlier, which would leave a badge either outside its
    own shot or over whatever footage moved under it.

    An accent whose shot is no longer in the edit is removed and named. An accent
    from before refs existed has no anchor to project, so it keeps the only time
    it has ever had.
    """
    starts = {clip.ref: timeline.start_of(clip.ref) for clip in timeline.clips if clip.ref}
    spans = {clip.ref: clip.dur for clip in timeline.clips if clip.ref}
    projected: list[EffectEvent] = []
    notes: list[str] = []
    for event in plan.events:
        if not event.clip_ref:
            projected.append(event)
            continue
        if event.clip_ref not in starts:
            notes.append(
                f"{event.effect_name} was about {event.clip_ref}, which is not in "
                "this edit"
            )
            continue
        inside = min(max(0.0, event.at_in_clip), max(0.0, spans[event.clip_ref]))
        projected.append(
            event.model_copy(
                update={"at_seconds": round(starts[event.clip_ref] + inside, 3)}
            )
        )
    return plan.model_copy(update={"events": projected}), notes


def apply_effect_overrides(
    plan: EffectPlan, overrides: Overrides
) -> tuple[EffectPlan, int]:
    """The creator's per-accent decisions, laid over the model's proposal.

    Every field of an override except its anchor is optional and means
    "unchanged", so a creator who only dragged a badge does not thereby freeze
    the sprite the planner chose for it. Returns the plan and how many overrides
    matched nothing — reported, because matching within one shot means an
    unmatched decision is a decision lost, never one applied elsewhere.
    """
    if not overrides.effects:
        return plan, 0

    anchors = [
        Moment(
            clip_ref=decided.clip_ref,
            at_in_clip=decided.at_in_clip,
            at_seconds=decided.at_in_clip,
            effect_name="",
        )
        for decided in overrides.effects
    ]
    spent: set[int] = set()
    updated: list[EffectEvent] = []
    for event in plan.events:
        match_index = nearest_moment(moment_of(event), anchors, spent)
        if match_index is None:
            updated.append(event)
            continue
        spent.add(match_index)
        decided = overrides.effects[match_index]
        changes: dict = {"keep": decided.keep}
        if decided.effect_name:
            changes["effect_name"] = decided.effect_name
        if decided.x is not None and decided.y is not None:
            changes["x"] = decided.x
            changes["y"] = decided.y
        if decided.scale_factor is not None:
            changes["scale_factor"] = decided.scale_factor
        if decided.rotation is not None:
            changes["rotation"] = decided.rotation
        updated.append(event.model_copy(update=changes))

    return plan.model_copy(update={"events": updated}), len(anchors) - len(spent)
