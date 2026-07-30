"""Carrying what the model saw through the edit.

An observation is a fact about a source clip: at 41.0s of clip-01, the thing
gave way. The edit keeps a few seconds of that clip and discards the rest, so
before such a moment can be offered as somewhere to put an accent it has to
survive two questions — is it still in the video at all, and where in the
finished cut does it now sit.

Both are arithmetic on the timeline, which is why they live here rather than in
a prompt. The model is asked what happened; the pipeline works out where that
ended up.
"""
from __future__ import annotations

from videoai.core.models import Observation, Timeline


def observed_moments(
    observations: list[Observation], timeline: Timeline
) -> list[tuple[float, Observation]]:
    """Every observation that survived the cut, as (time on the edit clock, what).

    A moment the edit dropped is not returned: an accent fired over a shot the
    viewer never sees is worse than no accent. A source second that was cut in
    twice is returned twice, because both really are places in the finished
    video.

    Times are on the edit's own clock — the concatenated segments, before the
    intro card is prepended. That is the same clock the effects stage reasons
    on; the delivery render shifts everything by the intro and by its measured
    segment starts, exactly as it already does for section titles.
    """
    found: list[tuple[float, Observation]] = []
    for clip in timeline.clips:
        end = clip.offset + clip.dur
        for seen in observations:
            if seen.clip_id != clip.src:
                continue
            # The start of a cut is inside it; its end is the first frame the
            # edit dropped.
            if not (clip.offset <= seen.at < end):
                continue
            found.append((round(clip.start + (seen.at - clip.offset), 3), seen))
    return sorted(found, key=lambda item: item[0])


def observed_moment_lines(moments: list[tuple[float, Observation]]) -> list[str]:
    """The menu a model chooses from, so it never has to invent a timestamp.

    Offering measured moments and asking which deserve an accent keeps the
    editorial judgement with the model and the clock with the pipeline — the
    split the production contract asks for.
    """
    return [
        f"{at:.2f}s [{seen.kind}] {seen.what}"
        for at, seen in moments
    ]
