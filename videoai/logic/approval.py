"""Getting the creator's decisions out of the browser and onto disk.

A page cannot write to disk — the browser forbids it — so the approval page's
edits lived only in the tab, and the handoff was a manual download followed by a
command. Somebody spent an afternoon placing badges and none of it reached the
video, which is a design failure rather than a user error.

So the page posts to a small server that runs only while the approval is open,
on the loopback interface, guarded by a token minted for that one session. Press
Save and the decisions are on disk and in the edit before the button stops
looking pressed.

What they are written INTO is `05e-overrides`, the creator's own artifact —
never the effect plan. Creator intent and model output sharing one file is what
forced the CLI to recompute and forge the effects stage's fingerprint so the
runner would not treat a human's work as stale: a person's decision had to
impersonate a model's reply to survive. Kept apart, each file has exactly one
author, and the runner has no reason to overwrite either.

The applying itself lives here rather than in the CLI so the server and the
`apply-effects` command cannot drift apart: there is one definition of what a
decision does.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from videoai.core.models import ClipOverride, EffectOverride, EffectPlan, Overrides, Timeline
from videoai.core.store import ArtifactStore
from videoai.logic.decisions import (
    OVERRIDES_ARTIFACT,
    Moment,
    moment_of,
    nearest_moment,
    override_key,
    read_overrides,
    reproject_events,
)


@dataclass(frozen=True)
class ApplyResult:
    """What a set of decisions did to the edit, for telling the creator."""

    kept: int
    dropped: int
    moved: int
    swapped: int
    resized: int = 0
    rotated: int = 0
    # What the running order came back saying. Counted separately from the
    # accents because they are separate decisions: a creator can move a shot
    # without touching a badge, and telling them "0 kept, 0 dropped" after they
    # have rearranged the whole video reads as nothing having happened.
    shots: int = 0
    shots_off: int = 0
    reordered: bool = False
    # Decisions whose moment is no longer in the plan. Reported rather than
    # quietly dropped onto whatever happened to be nearby.
    unmatched: int = 0

    def summary(self) -> str:
        text = (
            f"{self.kept} kept, {self.dropped} dropped, "
            f"{self.moved} moved, {self.swapped} swapped"
        )
        if self.resized:
            text += f", {self.resized} resized"
        if self.rotated:
            text += f", {self.rotated} turned"
        if self.reordered:
            text += ", running order changed"
        if self.shots_off:
            text += f", {self.shots_off} shot(s) switched off"
        if self.unmatched:
            text += f", {self.unmatched} with no matching moment"
        return text


def _shown_plan(store: ArtifactStore) -> EffectPlan:
    """The accents as the approval page drew them.

    The page works on the current edit, so its times are the reprojected ones;
    matching what comes back against the plan's own stored times would compare
    two different clocks the moment the creator has reordered anything.
    """
    plan = store.read("05d-effects", EffectPlan)
    if not store.exists("05-timeline"):
        return plan
    projected, _ = reproject_events(plan, store.read("05-timeline", Timeline))
    return projected


def _anchor_of(event) -> tuple[str, float]:
    """The (clip_ref, at_in_clip) an override about `event` is filed under.

    An event from before refs existed has no shot to name, so it is filed under
    its absolute time and matched back on that — which is the old behaviour,
    exactly, for the artifacts that are all that behaviour was ever safe for.
    """
    if event.clip_ref:
        return event.clip_ref, round(event.at_in_clip, 3)
    return "", round(event.at_seconds, 3)


def _clip_overrides(document: dict, existing: list[ClipOverride]) -> list[ClipOverride]:
    """The running order the document names, or the one already on file.

    Position in the list is position in the edit, so an absent "clips" key is
    "the creator did not touch the order", not "the creator wants no order".
    """
    raw = document.get("clips")
    if not isinstance(raw, list):
        return existing
    return [
        ClipOverride(
            ref=str(item.get("ref") or ""),
            enabled=bool(item.get("enabled", True)),
            note=str(item.get("note") or ""),
        )
        for item in raw
        if isinstance(item, dict) and item.get("ref")
    ]


def apply_decisions(store: ArtifactStore, document: dict) -> ApplyResult:
    """Write the creator's choices into `05e-overrides`.

    A dropped accent is recorded as `keep: false` rather than deleted. A decision
    that vanishes is one the next re-plan will cheerfully make again, and the
    record of what was looked at and turned down is worth as much as the record
    of what was kept.

    Anything the document does not mention is left exactly as it was, so a
    partial answer is safe: saving after editing two accents does not silently
    reset the other six.

    This used to take a `stage_fingerprint` — the value the CLI forged out of
    `runner._fingerprint` so the runner would see a current plan rather than
    re-plan over the top of the creator's work. Overrides being their own
    artifact is what made a human's edit stop having to impersonate a model's
    reply, and the parameter went with it.
    """
    shown = _shown_plan(store)
    choices = [item for item in document.get("events", []) if isinstance(item, dict)]
    existing = read_overrides(store)
    by_anchor = {override_key(decided): decided for decided in existing.effects}

    # Matched by MOMENT, not by position. Indices are only meaningful against the
    # exact plan the page was drawn from, and a plan that has been re-made since
    # slides every decision one place along: a badge picked for "he spots the
    # syringe" lands on the box opening, every name still matches, and nothing
    # looks wrong. Older decision files carry no time at all and fall back to the
    # index they do carry.
    spent: set[int] = set()
    timed: list[tuple[int, Moment]] = [
        (
            position,
            Moment(
                clip_ref="",
                at_in_clip=0.0,
                at_seconds=float(choice["at"]),
                # A swap reports the NEW name, so the origin is what identifies
                # the accent the decision was made about.
                effect_name=str(
                    choice.get("from_name") or choice.get("effect_name") or ""
                ),
            ),
        )
        for position, choice in enumerate(choices)
        if "at" in choice
    ]

    def pick(index: int, event) -> dict | None:
        blocked = {slot for slot, (position, _) in enumerate(timed) if position in spent}
        best = nearest_moment(
            moment_of(event), [moment for _, moment in timed], blocked
        )
        if best is not None:
            spent.add(timed[best][0])
            return choices[timed[best][0]]
        untimed = [
            position for position, choice in enumerate(choices)
            if position not in spent and "at" not in choice
            and int(choice.get("index", -1)) == index
        ]
        if untimed:
            spent.add(untimed[0])
            return choices[untimed[0]]
        return None

    dropped, moved, swapped, resized, rotated = 0, 0, 0, 0, 0
    for index, event in enumerate(shown.events):
        choice = pick(index, event)
        if choice is None:
            continue
        clip_ref, at_in_clip = _anchor_of(event)
        # Keyed by what the override itself will be filed under, not by the
        # event's own `anchor_key`, so a second visit updates the entry the first
        # visit wrote instead of adding a near-identical twin beside it.
        key = override_key(EffectOverride(clip_ref=clip_ref, at_in_clip=at_in_clip))
        previous = by_anchor.get(key)
        keep = bool(choice.get("keep", True))
        if not keep:
            dropped += 1
        name = str(choice.get("effect_name") or event.effect_name)
        swapped_name = previous.effect_name if previous else ""
        if name != event.effect_name:
            swapped_name = name
            swapped += 1
        x, y = (previous.x, previous.y) if previous else (None, None)
        if choice.get("moved") and choice.get("x") is not None:
            x, y = float(choice["x"]), float(choice["y"])
            moved += 1
        # Size and tilt follow the same rule as the dragged point: the page says
        # whether the creator touched the control, and an untouched control
        # leaves whatever was already on file. Reading the number alone would
        # make every save overwrite a size the creator set on an earlier visit
        # with the default the page happened to open at.
        scale_factor = previous.scale_factor if previous else None
        if choice.get("resized") and choice.get("scale_factor") is not None:
            scale_factor = float(choice["scale_factor"])
            resized += 1
        rotation = previous.rotation if previous else None
        if choice.get("rotated") and choice.get("rotation") is not None:
            rotation = float(choice["rotation"])
            rotated += 1
        by_anchor[key] = EffectOverride(
            clip_ref=clip_ref,
            at_in_clip=at_in_clip,
            effect_name=swapped_name,
            x=x,
            y=y,
            scale_factor=scale_factor,
            rotation=rotation,
            keep=keep,
        )

    clips = _clip_overrides(document, existing.clips)
    overrides = Overrides(
        timeline_hash=store.content_hash("05-timeline") or "",
        clips=clips,
        effects=list(by_anchor.values()),
    )
    store.write(OVERRIDES_ARTIFACT, overrides, fingerprint="creator")
    return ApplyResult(
        kept=sum(1 for decided in overrides.effects if decided.keep),
        dropped=dropped,
        moved=moved,
        swapped=swapped,
        resized=resized,
        rotated=rotated,
        shots=len(clips),
        shots_off=sum(1 for decided in clips if not decided.enabled),
        reordered=_reordered(store, clips),
        unmatched=len(choices) - len(spent),
    )


def _reordered(store: ArtifactStore, clips: list[ClipOverride]) -> bool:
    """Whether the creator's order differs from the edit as it currently stands.

    Compared against the assembled timeline rather than the planner's proposal,
    because the timeline is what the page drew: a creator who moved nothing on
    their second visit should not be told they reordered the video merely
    because their first visit did.
    """
    if not clips or not store.exists("05-timeline"):
        return False
    current = [clip.ref for clip in store.read("05-timeline", Timeline).clips if clip.ref]
    return current != [decided.ref for decided in clips if decided.enabled]


def save_decisions_copy(work_dir: Path, document: dict) -> Path:
    """Keep the raw decisions beside the overrides they produced.

    The overrides record what was decided; this records what the creator actually
    sent, which is what you want when the two disagree.
    """
    target = work_dir / "effects-decisions.json"
    target.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return target
