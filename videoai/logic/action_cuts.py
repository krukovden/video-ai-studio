"""Keeping a cut out of the middle of something the viewer is watching happen.

The planner cuts where the words end, because words are what it selects by. The
picture does not agree: a hand can be halfway into a glove, a bubble halfway
through bursting, and the shot ends anyway. Watching it, the effect is of the
video losing interest at the exact moment you did not.

The clip notes say when each action starts. That is enough to notice a cut
landing just after one and to move it — back to before the action began, or on
far enough for it to finish. Which of those is right is decided by whichever
moves the boundary less, because the smaller change is the one that keeps the
edit the creator already approved.

Only the OUT point moves. Pulling an IN point earlier would drag in the tail of
whatever came before, which is a different kind of wrong.
"""
from __future__ import annotations

from videoai.core.models import ClipNotes, Timeline, TimelineClip

# How soon after an action starts a cut still counts as interrupting it. Beyond
# this the action has had time to read and ending is a choice rather than an
# accident.
INTERRUPT_WINDOW = 1.5

# How long to let an action run when nothing described follows it. Most of the
# actions in this footage — a squeeze, a pop, a lid coming off — are done inside
# this.
ACTION_SECONDS = 2.0

# A cut moved back must still leave a segment long enough to read as a shot.
MIN_SEGMENT_SECONDS = 0.8

# Clear of the action rather than exactly on its first frame.
BEFORE_MARGIN = 0.15


def _events_for(notes: ClipNotes, clip_id: str) -> list[float]:
    for note in notes.notes:
        if note.clip_id == clip_id:
            return sorted(event.at for event in note.events)
    return []


def _source_duration(notes: ClipNotes, clip_id: str) -> float | None:
    for note in notes.notes:
        if note.clip_id == clip_id:
            return note.duration
    return None


def _mend(clip: TimelineClip, starts: list[float], limit: float | None) -> float:
    """The out point for this segment: unchanged, or moved clear of an action.

    Returns the new duration.
    """
    end = clip.offset + clip.dur
    interrupted = [
        at for at in starts if end - INTERRUPT_WINDOW < at < end
    ]
    if not interrupted:
        return clip.dur

    # The last action to begin before the cut is the one being cut through.
    action = max(interrupted)

    # Option one: stop before it started.
    before = action - BEFORE_MARGIN - clip.offset
    # Option two: run on until it has finished — the next described moment, or a
    # default if it is the last thing in the clip.
    following = [at for at in starts if at > action]
    finish = min(following[0], action + ACTION_SECONDS) if following else action + ACTION_SECONDS
    after = finish - clip.offset

    if limit is not None:
        after = min(after, limit - clip.offset)

    can_shorten = before >= MIN_SEGMENT_SECONDS
    can_lengthen = after > clip.dur

    if can_shorten and can_lengthen:
        # Whichever disturbs the approved edit less.
        return before if (clip.dur - before) <= (after - clip.dur) else after
    if can_shorten:
        return before
    if can_lengthen:
        return after
    # Neither is available: a very short segment at the very end of its source.
    # Cutting through the action is the least-bad answer, and saying nothing
    # about it would be worse than leaving it.
    return clip.dur


def avoid_mid_action_cuts(timeline: Timeline, notes: ClipNotes) -> Timeline:
    """The same edit, with no cut landing inside an action that just began."""
    if not notes.notes or not timeline.clips:
        return timeline

    mended: list[TimelineClip] = []
    start = 0.0
    for clip in timeline.clips:
        starts = _events_for(notes, clip.src)
        duration = (
            _mend(clip, starts, _source_duration(notes, clip.src)) if starts else clip.dur
        )
        duration = max(MIN_SEGMENT_SECONDS, round(duration, 3))
        mended.append(clip.model_copy(update={"dur": duration, "start": round(start, 3)}))
        start += duration

    return timeline.model_copy(update={"clips": mended})
