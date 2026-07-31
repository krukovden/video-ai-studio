"""Turn an editorial plan into an exact timeline, then into the creator's edit.

The model chooses phrases; geometry is computed here so timestamps are always
arithmetic on real transcript data rather than numbers a model wrote down. What
the creator then does to that proposal — reorder it, switch a shot off — is
applied here too, as a pure function of the plan and the overrides, so the two
authorships never have to share a file.
"""
from __future__ import annotations

from videoai.core.models import (
    Analysis,
    Manifest,
    Overrides,
    StoryPlan,
    Timeline,
    TimelineClip,
    Transcript,
    Word,
)
from videoai.logic.inserts import is_insert_ref, resolve_insert_ref

# What a repeated reference is renamed to. Not `#`, which phrase ids already use:
# `clip-01#004#2` reads as a phrase id and `clip-01#004~2` cannot be mistaken for
# one.
OCCURRENCE_SEPARATOR = "~"


def _left_neighbour_limit(words: list[Word], segment_start: float) -> float | None:
    """End time of the closest word that finishes at or before the segment starts.

    Words that overlap the segment itself (end after segment_start) are not
    neighbours and must not constrain the left edge.
    """
    ends = [word.end for word in words if word.end <= segment_start]
    return max(ends) if ends else None


def _right_neighbour_limit(words: list[Word], segment_end: float) -> float | None:
    """Start time of the closest word that begins at or after the segment ends."""
    starts = [word.start for word in words if word.start >= segment_end]
    return min(starts) if starts else None


def _unique_ref(reference: str, taken: set[str]) -> str:
    """`reference`, or the next free `~2`, `~3` … spelling of it.

    The planner is told never to repeat a phrase id, but nothing stops it cutting
    one silent insert in twice, and both really are separate shots in the edit.
    Two clips answering to the same name would make every lookup by ref ambiguous
    — which is the one thing a ref exists to prevent — so the second occurrence
    gets a name of its own.
    """
    candidate = reference
    occurrence = 1
    while candidate in taken:
        occurrence += 1
        candidate = f"{reference}{OCCURRENCE_SEPARATOR}{occurrence}"
    taken.add(candidate)
    return candidate


def renumber(timeline: Timeline) -> Timeline:
    """The same clips with `start` recomputed from the running order.

    Every `start` is a cached sum of the durations before it, so the moment
    anything is reordered or switched off, every later one is a lie. Recomputing
    is cheaper than deciding which of them can be trusted.
    """
    position = 0.0
    clips: list[TimelineClip] = []
    for clip in timeline.clips:
        clips.append(clip.model_copy(update={"start": position}))
        position += clip.dur
    return timeline.model_copy(update={"clips": clips})


def build_timeline(
    plan: StoryPlan,
    analysis: Analysis,
    manifest: Manifest,
    transcript: Transcript,
    padding: float,
    fps: float,
    gain_db_by_beat: dict[str, float] | None = None,
) -> Timeline:
    first_clip = manifest.clips[0]
    timeline = Timeline(fps=fps, width=first_clip.width, height=first_clip.height)
    known_transcripts = {clip.clip_id for clip in transcript.clips}
    gain_db_by_beat = gain_db_by_beat or {}

    position = 0.0
    taken_refs: set[str] = set()
    for section in plan.sections:
        for phrase_id in section.phrase_ids:
            if is_insert_ref(phrase_id):
                # A silent visual insert: no words, so no quote to anchor and no
                # word-boundary padding to apply — the requested span is the cut.
                clip_id, insert_start, insert_end = resolve_insert_ref(phrase_id, manifest)
                insert_duration = insert_end - insert_start
                timeline.clips.append(
                    TimelineClip(
                        ref=_unique_ref(phrase_id, taken_refs),
                        src=clip_id,
                        offset=insert_start,
                        dur=insert_duration,
                        start=position,
                        quote="",
                        reason=(
                            f"visual insert: {section.goal}"
                            if section.goal
                            else "visual insert"
                        ),
                        beat=section.name,
                        core_dur=insert_duration,
                        gain_db=gain_db_by_beat.get(section.name, 0.0),
                        is_insert=True,
                    )
                )
                position += insert_duration
                continue

            segment = analysis.by_phrase(phrase_id)
            source = manifest.by_id(segment.clip_id)
            # Unpadded, unclamped: the real speech length, independent of padding
            # and of any clamp applied below at a source or neighbour-word boundary.
            core_duration = max(0.0, segment.end - segment.start)

            words = (
                transcript.by_id(segment.clip_id).words
                if segment.clip_id in known_transcripts
                else []
            )

            # Pad into silence, never into speech: an edge may move by up to
            # `padding`, but stops early if the neighbouring word is closer than
            # that, so the padded cut never lands inside the adjacent word.
            offset = segment.start - padding
            left_limit = _left_neighbour_limit(words, segment.start)
            if left_limit is not None:
                offset = max(offset, left_limit)
            offset = max(0.0, offset)

            end = segment.end + padding
            right_limit = _right_neighbour_limit(words, segment.end)
            if right_limit is not None:
                end = min(end, right_limit)
            end = min(source.duration, end)

            duration = max(0.0, end - offset)
            if duration <= 0:
                continue
            timeline.clips.append(
                TimelineClip(
                    ref=_unique_ref(phrase_id, taken_refs),
                    src=segment.clip_id,
                    offset=offset,
                    dur=duration,
                    start=position,
                    quote=segment.text,
                    reason=segment.content or section.goal,
                    beat=section.name,
                    core_dur=core_duration,
                    gain_db=gain_db_by_beat.get(section.name, 0.0),
                )
            )
            position += duration
    return timeline


def apply_clip_overrides(
    timeline: Timeline, overrides: Overrides
) -> tuple[Timeline, list[str]]:
    """The creator's running order, applied to the planner's proposal.

    Position in `overrides.clips` IS position in the edit, so reordering a shot
    and switching one off are the same operation on the same list and neither
    costs a model call. Returns the edit and one line per decision that could not
    be honoured — a ref the re-plan no longer contains, say. Those are reported
    rather than raised: a decision outliving the plan it was made against is the
    entire point of keeping it in a file of its own.

    A ref the list does not mention at all is a shot the plan grew since the
    creator last looked. It keeps its planned place — immediately after the
    nearest earlier clip that survived — rather than being appended, because
    appending a new hook after the sign-off is not "leaving it where it was".
    """
    if not overrides.clips:
        return timeline, []

    by_ref = {clip.ref: clip for clip in timeline.clips if clip.ref}
    mentioned = {decided.ref for decided in overrides.clips}
    order = [ref for ref in overrides.order if ref in by_ref]
    kept = set(order)

    notes = [
        f"the creator's order names {ref!r}, which this plan no longer contains"
        for ref in overrides.order
        if ref not in by_ref
    ]

    unseen: dict[str | None, list[TimelineClip]] = {}
    anchor: str | None = None
    for clip in timeline.clips:
        if clip.ref in mentioned:
            if clip.ref in kept:
                anchor = clip.ref
            continue
        unseen.setdefault(anchor, []).append(clip)

    arranged = list(unseen.get(None, []))
    for ref in order:
        arranged.append(by_ref[ref])
        arranged.extend(unseen.get(ref, []))

    new_clips = [clip.ref or clip.src for group in unseen.values() for clip in group]
    if new_clips:
        notes.append(
            "kept in their planned places, because the creator's order predates "
            "them: " + ", ".join(new_clips)
        )
    return renumber(timeline.model_copy(update={"clips": arranged})), notes
