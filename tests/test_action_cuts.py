"""Not cutting through something the viewer is watching happen.

The planner cuts on phrase boundaries, because that is where the words end. The
camera does not care: a hand is halfway into a glove, a bubble is halfway
through bursting. The described actions say when those start, so a cut that
lands inside one can be moved — either back before it began, or forward far
enough to let it finish.
"""
from __future__ import annotations

from videoai.core.models import ClipEvent, ClipNote, ClipNotes, Timeline, TimelineClip
from videoai.logic.action_cuts import avoid_mid_action_cuts


def _notes(*events: tuple[float, str]) -> ClipNotes:
    return ClipNotes(notes=[ClipNote(
        clip_id="clip-01", source_key="k", duration=120.0,
        events=[ClipEvent(at=at, what=what) for at, what in events],
    )])


def _timeline(*spans: tuple[float, float]) -> Timeline:
    clips, start = [], 0.0
    for offset, dur in spans:
        clips.append(TimelineClip(src="clip-01", offset=offset, dur=dur, start=start))
        start += dur
    return Timeline(fps=30, width=1920, height=1080, clips=clips)


def test_a_cut_just_after_an_action_starts_is_moved():
    """The complaint that produced this: the shot ends while he is still putting
    the glove on."""
    timeline = _timeline((30.0, 7.67))          # ends at 37.67
    fixed = avoid_mid_action_cuts(timeline, _notes((36.40, "he squeezes the toy")))
    end = fixed.clips[0].offset + fixed.clips[0].dur
    assert abs(end - 37.67) > 0.1, "the cut should have moved"
    # Either before the action, or far enough past it to have finished.
    assert end <= 36.40 or end >= 36.40 + 1.5


def test_it_prefers_whichever_move_is_smaller():
    # The action starts 0.2s before the cut: pulling back is a much smaller
    # change than running on for two more seconds.
    timeline = _timeline((30.0, 6.2))           # ends at 36.2
    fixed = avoid_mid_action_cuts(timeline, _notes((36.0, "a squeeze")))
    end = fixed.clips[0].offset + fixed.clips[0].dur
    assert end <= 36.0


def test_it_runs_on_when_the_action_started_long_before():
    # 1.3s in, so pulling back would throw away most of the action; letting it
    # finish is the smaller change.
    timeline = _timeline((30.0, 7.3))           # ends at 37.3
    fixed = avoid_mid_action_cuts(timeline, _notes((36.0, "a squeeze")))
    end = fixed.clips[0].offset + fixed.clips[0].dur
    assert end >= 37.5


def test_a_cut_nowhere_near_an_action_is_left_alone():
    timeline = _timeline((30.0, 8.0))
    fixed = avoid_mid_action_cuts(timeline, _notes((10.0, "long before")))
    assert fixed.clips[0].dur == 8.0


def test_a_segment_is_never_shortened_into_nothing():
    """Better a cut through an action than a segment too short to read."""
    timeline = _timeline((30.0, 0.9))           # ends at 30.9
    fixed = avoid_mid_action_cuts(timeline, _notes((30.4, "something")))
    assert fixed.clips[0].dur >= 0.8


def test_it_never_runs_past_the_end_of_the_source():
    timeline = _timeline((118.0, 1.6))          # source is 120s long
    fixed = avoid_mid_action_cuts(timeline, _notes((119.2, "right at the end")))
    clip = fixed.clips[0]
    assert clip.offset + clip.dur <= 120.0


def test_a_clip_with_no_notes_is_untouched():
    timeline = _timeline((30.0, 5.0))
    assert avoid_mid_action_cuts(timeline, ClipNotes()).clips[0].dur == 5.0


def test_every_segment_keeps_its_place_in_the_running_order():
    """Moving one boundary must not leave the next segment starting where the
    previous one used to end."""
    timeline = _timeline((30.0, 7.3), (50.0, 4.0))
    fixed = avoid_mid_action_cuts(timeline, _notes((36.0, "a squeeze")))
    assert fixed.clips[1].start == fixed.clips[0].dur
    assert fixed.clips[0].start == 0.0
