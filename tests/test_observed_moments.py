"""Which observed moments survived the edit, and where they landed in it.

An observation is a fact about a source clip. The edit keeps some of those
seconds and throws away most of them, so before a moment can be offered as a
place to put an accent it has to be asked: is it still in the video, and when?
"""
from __future__ import annotations

from videoai.core.models import Observation, Timeline, TimelineClip
from videoai.logic.observations import observed_moments


def _timeline() -> Timeline:
    # Two cuts from one clip, and the seconds between them are not in the edit.
    return Timeline(fps=30, width=1920, height=1080, clips=[
        TimelineClip(src="clip-01", offset=10.0, dur=4.0, start=0.0),
        TimelineClip(src="clip-01", offset=40.0, dur=2.0, start=4.0),
        TimelineClip(src="clip-02", offset=5.0, dur=3.0, start=6.0),
    ])


def _seen(clip_id: str, at: float, what: str = "something", kind: str = "action"):
    return Observation(clip_id=clip_id, at=at, reel_at=0.0, kind=kind, what=what)


def test_a_moment_inside_a_used_span_lands_on_the_edit_clock():
    moments = observed_moments([_seen("clip-01", 11.5)], _timeline())
    assert [round(at, 3) for at, _ in moments] == [1.5]


def test_a_moment_in_the_second_cut_accounts_for_the_first():
    moments = observed_moments([_seen("clip-01", 41.0)], _timeline())
    assert [round(at, 3) for at, _ in moments] == [5.0]


def test_a_moment_the_edit_cut_out_is_dropped():
    """Between 14s and 40s of clip-01 is on the floor; an accent there would fire
    over a shot the viewer never sees."""
    assert observed_moments([_seen("clip-01", 25.0)], _timeline()) == []


def test_a_moment_from_a_clip_that_never_made_the_edit_is_dropped():
    assert observed_moments([_seen("clip-09", 1.0)], _timeline()) == []


def test_a_moment_used_twice_appears_once_per_use():
    """The same source second can be cut into the edit more than once, and each
    appearance is a real place in the finished video."""
    timeline = Timeline(fps=30, width=1920, height=1080, clips=[
        TimelineClip(src="clip-01", offset=10.0, dur=2.0, start=0.0),
        TimelineClip(src="clip-01", offset=10.0, dur=2.0, start=2.0),
    ])
    moments = observed_moments([_seen("clip-01", 11.0)], timeline)
    assert [round(at, 3) for at, _ in moments] == [1.0, 3.0]


def test_moments_come_back_in_edit_order():
    seen = [_seen("clip-02", 6.0, "later"), _seen("clip-01", 11.0, "earlier")]
    assert [obs.what for _, obs in observed_moments(seen, _timeline())] == [
        "earlier", "later",
    ]


def test_the_boundary_of_a_cut_belongs_to_it():
    moments = observed_moments([_seen("clip-01", 10.0)], _timeline())
    assert [round(at, 3) for at, _ in moments] == [0.0]


def test_the_end_of_a_cut_is_not_inside_it():
    """14.0s is the first frame the edit dropped, not the last it kept."""
    assert observed_moments([_seen("clip-01", 14.0)], _timeline()) == []
