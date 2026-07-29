"""Turn an editorial plan into an exact timeline.

The model chooses phrases; geometry is computed here so timestamps are always
arithmetic on real transcript data rather than numbers a model wrote down.
"""
from __future__ import annotations

from videoai.core.models import (
    Analysis,
    Manifest,
    StoryPlan,
    Timeline,
    TimelineClip,
    Transcript,
    Word,
)
from videoai.logic.inserts import is_insert_ref, resolve_insert_ref


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
    for section in plan.sections:
        for phrase_id in section.phrase_ids:
            if is_insert_ref(phrase_id):
                # A silent visual insert: no words, so no quote to anchor and no
                # word-boundary padding to apply — the requested span is the cut.
                clip_id, insert_start, insert_end = resolve_insert_ref(phrase_id, manifest)
                insert_duration = insert_end - insert_start
                timeline.clips.append(
                    TimelineClip(
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
