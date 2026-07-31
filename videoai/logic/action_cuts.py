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

The times in those notes are numbers a model wrote down, and everything below
turns one into a clip duration — so on the face of it this is a model setting a
timestamp, which the production contract does not allow. It is not, as long as
the caller passes a `locate`: no model time then reaches the arithmetic, because
each one is first put back onto the frame where the picture measurably changed.
"""
from __future__ import annotations

from typing import Callable

from videoai.core.models import ClipNotes, Timeline, TimelineClip, Transcript
from videoai.logic.motion import ONSET_SEARCH_SECONDS, ONSET_STEP_SECONDS

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

# Takes (clip id, the second a model reported) and answers with the second the
# picture there was measured to change, or None when it was measured not to.
Locator = Callable[[str, float], float | None]


def _snapper(locate: Locator | None):
    """Every model timestamp, replaced by something measured — and remembered.

    A described moment that `locate` cannot find in the picture does not move a
    cut: the only evidence for it was a number, and a number is exactly what must
    not decide where a segment ends. Answers are cached because measuring is a
    video decode and the same moment is consulted from both sides of a cut.

    With no locator at all — a caller that has no access to the media — the time
    is rounded onto the grid the measurement itself resolves to, so both paths
    agree about what counts as one moment. That collapses a model rephrasing
    18.14 as 18.1 and nothing wider: 18.1 and 18.4 are still two different cuts,
    which is why every caller that can measure does.
    """
    seen: dict[tuple[str, float], float | None] = {}

    def snap(clip_id: str, at: float) -> float | None:
        if locate is None:
            return round(round(at / ONSET_STEP_SECONDS) * ONSET_STEP_SECONDS, 3)
        key = (clip_id, at)
        if key not in seen:
            measured = locate(clip_id, at)
            seen[key] = None if measured is None else round(measured, 3)
        return seen[key]

    return snap


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


def _inside_a_word(at: float, words: list[tuple[float, float]]) -> bool:
    """Whether a cut here would land mid-syllable.

    The edit already forbids this and validates it, so dodging an action must
    not walk straight into it: both rules are real, and a plan that trades one
    for the other is refused before it renders.
    """
    return any(start + 1e-3 < at < end - 1e-3 for start, end in words)


def _mend(
    clip: TimelineClip,
    starts: list[float],
    limit: float | None,
    words: list[tuple[float, float]],
    snap,
) -> float:
    """The out point for this segment: unchanged, or moved clear of an action.

    Returns the new duration.
    """
    end = clip.offset + clip.dur
    # Only the described moments that could possibly move THIS cut are measured:
    # snapping is a video decode, and a moment ten seconds away cannot reach the
    # arithmetic below. The window is the one used further down, widened by how
    # far a measurement is allowed to move a moment.
    nearby = sorted(
        at
        for at in (
            snap(clip.src, moment)
            for moment in starts
            if end - INTERRUPT_WINDOW - ONSET_SEARCH_SECONDS
            < moment
            < end + ACTION_SECONDS + ONSET_SEARCH_SECONDS
        )
        if at is not None
    )
    interrupted = [
        at for at in nearby if end - INTERRUPT_WINDOW < at < end
    ]
    if not interrupted:
        return clip.dur

    # The last action to begin before the cut is the one being cut through.
    action = max(interrupted)

    # Option one: stop before it started.
    before = action - BEFORE_MARGIN - clip.offset
    # Option two: run on until it has finished — the next described moment, or a
    # default if it is the last thing in the clip.
    following = [at for at in nearby if at > action]
    finish = min(following[0], action + ACTION_SECONDS) if following else action + ACTION_SECONDS
    after = finish - clip.offset

    if limit is not None:
        after = min(after, limit - clip.offset)

    can_shorten = before >= MIN_SEGMENT_SECONDS and not _inside_a_word(
        clip.offset + before, words
    )
    can_lengthen = after > clip.dur and not _inside_a_word(clip.offset + after, words)

    if can_shorten and can_lengthen:
        # Whichever disturbs the approved edit less.
        return before if (clip.dur - before) <= (after - clip.dur) else after
    if can_shorten:
        return before
    if can_lengthen:
        return after
    # Neither landing is available — the segment is too short to trim, it is at
    # the end of its source, or speech covers every alternative. Cutting through
    # the action is then the least-bad answer: an imperfect cut beats one the
    # edit would refuse.
    return clip.dur


def avoid_mid_action_cuts(
    timeline: Timeline,
    notes: ClipNotes,
    transcript: Transcript | None = None,
    locate: Locator | None = None,
) -> Timeline:
    """The same edit, with no cut landing inside an action that just began.

    `transcript` is what keeps this honest: a boundary moved off an action must
    still fall in silence, because cutting mid-word is a rule the edit already
    has and already checks. `locate` is what keeps it deterministic: it puts each
    described moment back onto the frame where the picture measurably changed, so
    a re-run in which the model rephrases the same action never re-renders the
    video. A caller holding the media should always pass one.
    """
    if not notes.notes or not timeline.clips:
        return timeline

    snap = _snapper(locate)

    spoken: dict[str, list[tuple[float, float]]] = {}
    for clip_transcript in (transcript.clips if transcript else []):
        spoken[clip_transcript.clip_id] = [
            (word.start, word.end) for word in clip_transcript.words
        ]

    mended: list[TimelineClip] = []
    start = 0.0
    for clip in timeline.clips:
        starts = _events_for(notes, clip.src)
        duration = (
            _mend(
                clip,
                starts,
                _source_duration(notes, clip.src),
                spoken.get(clip.src, []),
                snap,
            )
            if starts
            else clip.dur
        )
        duration = max(MIN_SEGMENT_SECONDS, round(duration, 3))
        mended.append(clip.model_copy(update={"dur": duration, "start": round(start, 3)}))
        start += duration

    return timeline.model_copy(update={"clips": mended})
