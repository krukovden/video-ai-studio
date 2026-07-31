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


def _transcript(*words: tuple[float, float, str]):
    from videoai.core.models import ClipTranscript, Transcript, Word

    return Transcript(provider="t", clips=[ClipTranscript(
        clip_id="clip-01",
        words=[Word(text=text, start=start, end=end) for start, end, text in words],
    )])


def test_a_moved_cut_never_lands_inside_a_spoken_word():
    """The edit already forbids cutting mid-word. Dodging an action must not
    walk straight into that: both rules are real."""
    timeline = _timeline((30.0, 7.3))                     # ends at 37.3
    speech = _transcript((36.9, 38.2, "and"), (38.6, 39.0, "then"))
    fixed = avoid_mid_action_cuts(
        timeline, _notes((36.0, "a squeeze")), transcript=speech
    )
    end = fixed.clips[0].offset + fixed.clips[0].dur
    assert not (36.9 < end < 38.2), f"cut landed inside 'and' at {end}"


def test_it_still_dodges_the_action_when_speech_allows():
    timeline = _timeline((30.0, 7.3))
    speech = _transcript((20.0, 21.0, "earlier"))          # nothing near the cut
    fixed = avoid_mid_action_cuts(
        timeline, _notes((36.0, "a squeeze")), transcript=speech
    )
    end = fixed.clips[0].offset + fixed.clips[0].dur
    assert end <= 36.0 or end >= 37.5


def test_with_no_safe_landing_the_cut_is_left_where_it_was():
    """Better an imperfect cut than an invalid one."""
    timeline = _timeline((30.0, 7.3))
    speech = _transcript((34.0, 39.5, "unbroken"))         # speech covers every option
    fixed = avoid_mid_action_cuts(
        timeline, _notes((36.0, "a squeeze")), transcript=speech
    )
    assert fixed.clips[0].dur == 7.3


# --- The model does not get to set a timestamp. A described moment is measured
# against the footage before any of the arithmetic above touches it ---


def test_the_measured_onset_is_what_moves_the_cut_not_the_reported_one():
    timeline = _timeline((30.0, 6.2))                      # ends at 36.2
    measured = avoid_mid_action_cuts(
        timeline, _notes((36.0, "a squeeze")), locate=lambda clip_id, at: 35.6
    )
    # Pulled back clear of 35.6, the frame the picture really changed on, rather
    # than of the 36.0 a model wrote down.
    end = measured.clips[0].offset + measured.clips[0].dur
    assert abs(end - (35.6 - 0.15)) < 1e-6


def test_two_runs_that_report_the_same_action_differently_produce_one_edit():
    """The whole reason for measuring: an LLM reporting 36.0 on one run and 36.3
    on the next used to mean two clip durations, two timeline hashes and a full
    re-render of an edit that nobody changed."""
    timeline = _timeline((30.0, 6.2))

    def locate(clip_id: str, at: float) -> float:
        return 35.6

    first = avoid_mid_action_cuts(timeline, _notes((36.0, "a squeeze")), locate=locate)
    second = avoid_mid_action_cuts(timeline, _notes((36.3, "a squeeze")), locate=locate)

    assert first.clips[0].dur == second.clips[0].dur


def test_a_moment_nothing_was_measured_at_never_moves_a_cut():
    """The picture did not change there. The only evidence for the moment was a
    number, and a number must not decide where a segment ends."""
    timeline = _timeline((30.0, 6.2))
    fixed = avoid_mid_action_cuts(
        timeline, _notes((36.0, "a squeeze")), locate=lambda clip_id, at: None
    )
    assert fixed.clips[0].dur == 6.2


def test_a_described_moment_nowhere_near_a_cut_is_never_measured():
    """Measuring is a video decode. A moment twenty seconds from every boundary
    cannot move one, so it must not cost anything to have been described."""
    timeline = _timeline((30.0, 6.2))
    asked: list[float] = []

    def locate(clip_id: str, at: float) -> float | None:
        asked.append(at)
        return at

    avoid_mid_action_cuts(
        timeline, _notes((12.0, "much earlier"), (36.0, "a squeeze")), locate=locate
    )

    assert asked == [36.0]


def test_the_same_moment_is_measured_once_however_often_it_is_consulted():
    timeline = _timeline((30.0, 6.2), (30.0, 6.2))
    calls: list[tuple[str, float]] = []

    def locate(clip_id: str, at: float) -> float | None:
        calls.append((clip_id, at))
        return at

    avoid_mid_action_cuts(timeline, _notes((36.0, "a squeeze")), locate=locate)

    assert calls == [("clip-01", 36.0)]


def test_without_a_locator_a_reported_time_is_rounded_onto_the_measurement_grid():
    """A caller with no access to the media cannot do better than collapse the
    jitter of a model rephrasing itself; the grid is the one the measurement
    itself resolves to."""
    timeline = _timeline((30.0, 6.2))
    coarse = avoid_mid_action_cuts(timeline, _notes((36.04, "a squeeze")))
    exact = avoid_mid_action_cuts(timeline, _notes((36.0, "a squeeze")))
    assert coarse.clips[0].dur == exact.clips[0].dur
