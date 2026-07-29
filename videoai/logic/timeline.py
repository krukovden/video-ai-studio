"""Turn an editorial plan into an exact timeline.

The model chooses phrases; geometry is computed here so timestamps are always
arithmetic on real transcript data rather than numbers a model wrote down.
"""
from __future__ import annotations

from videoai.core.models import Analysis, Manifest, StoryPlan, Timeline, TimelineClip


def build_timeline(
    plan: StoryPlan,
    analysis: Analysis,
    manifest: Manifest,
    padding: float,
    fps: float,
) -> Timeline:
    first_clip = manifest.clips[0]
    timeline = Timeline(fps=fps, width=first_clip.width, height=first_clip.height)

    position = 0.0
    for section in plan.sections:
        for phrase_id in section.phrase_ids:
            segment = analysis.by_phrase(phrase_id)
            source = manifest.by_id(segment.clip_id)
            # Unpadded, unclamped: the real speech length, independent of padding
            # and of any clamp applied below at a source boundary.
            core_duration = max(0.0, segment.end - segment.start)
            offset = max(0.0, segment.start - padding)
            end = min(source.duration, segment.end + padding)
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
                )
            )
            position += duration
    return timeline
