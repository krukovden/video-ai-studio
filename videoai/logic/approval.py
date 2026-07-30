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


@dataclass(frozen=True)
class ApplyResult:
    """What a set of decisions did to the plan, for telling the creator."""

    kept: int
    dropped: int
    moved: int
    swapped: int

    def summary(self) -> str:
        return (
            f"{self.kept} kept, {self.dropped} dropped, "
            f"{self.moved} moved, {self.swapped} swapped"
        )


def apply_decisions(store: ArtifactStore, document: dict) -> ApplyResult:
    """Write the creator's choices into the effect plan.

    A dropped accent is marked rather than deleted. A decision that vanishes is
    one the next re-plan will cheerfully make again, and the record of what was
    looked at and turned down is worth as much as the record of what was kept.

    Anything the document does not mention is left exactly as it was, so a
    partial answer is safe: saving after editing two accents does not silently
    reset the other six.
    """
    plan = store.read("05d-effects", EffectPlan)
    by_index = {
        int(item["index"]): item
        for item in document.get("events", [])
        if isinstance(item, dict) and "index" in item
    }

    updated, dropped, moved, swapped = [], 0, 0, 0
    for index, event in enumerate(plan.events):
        choice = by_index.get(index)
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
        fingerprint="creator-approved",
    )
    return ApplyResult(
        kept=sum(1 for event in updated if event.keep),
        dropped=dropped,
        moved=moved,
        swapped=swapped,
    )


def save_decisions_copy(work_dir: Path, document: dict) -> Path:
    """Keep the raw decisions beside the plan they produced.

    The plan records what was decided; this records what the creator actually
    sent, which is what you want when the two disagree.
    """
    target = work_dir / "effects-decisions.json"
    target.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return target
