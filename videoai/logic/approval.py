"""Getting the creator's decisions out of the browser and into the plan.

A page cannot write to disk — the browser forbids it — so the approval page's
edits lived only in the tab, and the handoff was a manual download followed by a
command. Somebody spent an afternoon placing badges and none of it reached the
video, which is a design failure rather than a user error.

So the page posts to a small server that runs only while the approval is open,
on the loopback interface, guarded by a token minted for that one session. Press
Save and the decisions are on disk and in the plan before the button stops
looking pressed.

The applying itself lives here rather than in the CLI so the server and the
`apply-effects` command cannot drift apart: there is one definition of what a
decision does.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from videoai.core.models import EffectPlan
from videoai.core.store import ArtifactStore

# How far a moment may drift and still be the same one. The plan is re-read from
# the same timeline, so an accent lands within a few frames of where it was.
SAME_MOMENT_SECONDS = 1.0


@dataclass(frozen=True)
class ApplyResult:
    """What a set of decisions did to the plan, for telling the creator."""

    kept: int
    dropped: int
    moved: int
    swapped: int
    # Decisions whose moment is no longer in the plan. Reported rather than
    # quietly dropped onto whatever happened to be nearby.
    unmatched: int = 0

    def summary(self) -> str:
        text = (
            f"{self.kept} kept, {self.dropped} dropped, "
            f"{self.moved} moved, {self.swapped} swapped"
        )
        if self.unmatched:
            text += f", {self.unmatched} with no matching moment"
        return text


def apply_decisions(
    store: ArtifactStore, document: dict, stage_fingerprint: str | None = None
) -> ApplyResult:
    """Write the creator's choices into the effect plan.

    `stage_fingerprint` is what stops the work being undone. An artifact stored
    under a made-up fingerprint looks stale to the runner, which re-plans the
    stage and discards the very decisions just applied — twice this destroyed a
    creator's placements minutes after they made them. Storing the fingerprint
    the stage would itself have written says, truthfully, that this plan is
    current and needs no redoing.

    A dropped accent is marked rather than deleted. A decision that vanishes is
    one the next re-plan will cheerfully make again, and the record of what was
    looked at and turned down is worth as much as the record of what was kept.

    Anything the document does not mention is left exactly as it was, so a
    partial answer is safe: saving after editing two accents does not silently
    reset the other six.
    """
    plan = store.read("05d-effects", EffectPlan)
    choices = [item for item in document.get("events", []) if isinstance(item, dict)]

    # Matched by MOMENT, not by position. Indices are only meaningful against the
    # exact plan the page was drawn from, and a plan that has been re-made since
    # slides every decision one place along: a badge picked for "he spots the
    # syringe" lands on the box opening, every name still matches, and nothing
    # looks wrong. Timed decisions are matched on the second they were made
    # about; older files without times fall back to the index they carry.
    spent: set[int] = set()

    def pick(index: int, event) -> dict | None:
        best, best_distance = None, SAME_MOMENT_SECONDS
        for position, choice in enumerate(choices):
            if position in spent or "at" not in choice:
                continue
            # A swap reports the new name, so match on where it came from.
            origin = str(choice.get("from_name") or choice.get("effect_name") or "")
            if origin and origin != event.effect_name:
                continue
            distance = abs(float(choice["at"]) - event.at_seconds)
            if distance <= best_distance:
                best, best_distance = position, distance
        if best is not None:
            spent.add(best)
            return choices[best]
        untimed = [
            position for position, choice in enumerate(choices)
            if position not in spent and "at" not in choice
            and int(choice.get("index", -1)) == index
        ]
        if untimed:
            spent.add(untimed[0])
            return choices[untimed[0]]
        return None

    updated, dropped, moved, swapped = [], 0, 0, 0
    for index, event in enumerate(plan.events):
        choice = pick(index, event)
        if choice is None:
            updated.append(event)
            continue
        changes: dict = {"keep": bool(choice.get("keep", True))}
        if not changes["keep"]:
            dropped += 1
        name = str(choice.get("effect_name") or event.effect_name)
        if name != event.effect_name:
            changes["effect_name"] = name
            swapped += 1
        if choice.get("moved") and choice.get("x") is not None:
            changes["x"] = float(choice["x"])
            changes["y"] = float(choice["y"])
            moved += 1
        updated.append(event.model_copy(update=changes))

    store.write(
        "05d-effects",
        plan.model_copy(update={"events": updated}),
        fingerprint=stage_fingerprint or "creator-approved",
    )
    return ApplyResult(
        kept=sum(1 for event in updated if event.keep),
        dropped=dropped,
        moved=moved,
        swapped=swapped,
        unmatched=len(choices) - len(spent),
    )


def save_decisions_copy(work_dir: Path, document: dict) -> Path:
    """Keep the raw decisions beside the plan they produced.

    The plan records what was decided; this records what the creator actually
    sent, which is what you want when the two disagree.
    """
    target = work_dir / "effects-decisions.json"
    target.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return target
