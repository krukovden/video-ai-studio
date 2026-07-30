"""Carrying what a person decided across a plan they did not ask to be redone.

Re-planning discards the model's previous proposal, which is the point of
re-planning. It must not also discard what a creator decided about it — an
accent they turned down reappearing, or a badge they placed by hand jumping back
to a corner, with nothing said. That is the same silent loss as a closed browser
tab, one layer down, and it is worse there because nobody is watching.

A decision belongs to a MOMENT, not to a position in a list. Indices renumber
whenever the model returns a different number of accents, so they are matched by
what they are about: the same drawing at close to the same second.

What cannot be matched is counted and reported. A decision that quietly
evaporates is the failure this module exists to prevent, so it is never allowed
to evaporate quietly.
"""
from __future__ import annotations

from videoai.core.models import EffectEvent, EffectPlan

# How far a re-planned accent may drift and still be the same moment. The model
# re-reads the same printed timeline, so it lands within a few frames of where it
# landed before; a second is generous and still far short of the two-second gap
# the placement prompt requires between accents.
SAME_MOMENT_SECONDS = 1.0


def _was_decided(event: EffectEvent) -> bool:
    """Whether a person touched this one at all."""
    return (not event.keep) or event.x is not None or event.y is not None


def carry_decisions(fresh: EffectPlan, previous: EffectPlan) -> tuple[EffectPlan, int]:
    """Apply the old plan's human decisions to a newly planned one.

    Returns the plan and how many decisions had nowhere to land, so the caller
    can say so out loud.
    """
    decided = [event for event in previous.events if _was_decided(event)]
    if not decided:
        return fresh, 0

    spent: set[int] = set()
    updated: list[EffectEvent] = []
    for event in fresh.events:
        match_index = None
        best = SAME_MOMENT_SECONDS
        for index, old in enumerate(decided):
            if index in spent or old.effect_name != event.effect_name:
                continue
            distance = abs(old.at_seconds - event.at_seconds)
            if distance <= best:
                match_index, best = index, distance
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
