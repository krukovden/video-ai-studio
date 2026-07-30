"""Planning the review reel: which seconds of footage a video model is shown.

Submitting video is billed by the second, so what gets sent is a cost decision
as much as an editorial one. The planner is pure arithmetic — which spans, in
what order, and where each phrase lands in the finished reel — so the rules can
be tested without encoding anything.
"""
from __future__ import annotations

from videoai.core.models import Phrase, PhraseIndex
from videoai.logic.reel import plan_reel, reel_seconds


def _index(*spans: tuple[str, str, float, float]) -> PhraseIndex:
    return PhraseIndex(phrases=[
        Phrase(phrase_id=pid, clip_id=clip, start=start, end=end,
               text="words", word_start=0, word_end=1)
        for pid, clip, start, end in spans
    ])


def test_only_the_spans_with_speech_are_planned():
    index = _index(
        ("clip-01#001", "clip-01", 10.0, 12.0),
        ("clip-01#002", "clip-01", 40.0, 41.0),
    )
    spans = plan_reel(index, padding=0.0, max_seconds=None)
    # Two spans, three seconds total — not the whole clip.
    assert [(s.clip_id, s.start, s.end) for s in spans] == [
        ("clip-01", 10.0, 12.0), ("clip-01", 40.0, 41.0),
    ]
    assert reel_seconds(spans) == 3.0


def test_padding_widens_each_span():
    index = _index(("clip-01#001", "clip-01", 10.0, 12.0))
    span = plan_reel(index, padding=0.5, max_seconds=None)[0]
    assert (span.start, span.end) == (9.5, 12.5)


def test_padding_cannot_reach_before_the_start_of_a_clip():
    index = _index(("clip-01#001", "clip-01", 0.2, 1.0))
    assert plan_reel(index, padding=0.5, max_seconds=None)[0].start == 0.0


def test_neighbouring_phrases_become_one_span():
    """Two cuts a fifth of a second apart are one moment, and cutting them
    separately would pay the encoder twice and lose the beat between them."""
    index = _index(
        ("clip-01#001", "clip-01", 10.0, 12.0),
        ("clip-01#002", "clip-01", 12.2, 14.0),
    )
    spans = plan_reel(index, padding=0.0, max_seconds=None)
    assert len(spans) == 1
    assert (spans[0].start, spans[0].end) == (10.0, 14.0)
    assert [entry.phrase_id for entry in spans[0].entries] == [
        "clip-01#001", "clip-01#002",
    ]


def test_phrases_from_different_clips_never_merge():
    index = _index(
        ("clip-01#001", "clip-01", 10.0, 12.0),
        ("clip-02#001", "clip-02", 12.1, 14.0),
    )
    assert len(plan_reel(index, padding=0.0, max_seconds=None)) == 2


def test_each_phrase_knows_where_it_lands_in_the_reel():
    """The model watches one concatenated file, so it can only name a moment by
    its position in that file — the mapping back to phrase ids is this."""
    index = _index(
        ("clip-01#001", "clip-01", 10.0, 12.0),
        ("clip-02#001", "clip-02", 30.0, 33.0),
    )
    spans = plan_reel(index, padding=0.0, max_seconds=None)
    first, second = spans[0].entries[0], spans[1].entries[0]
    assert (first.reel_start, first.reel_end) == (0.0, 2.0)
    # The second span starts where the first ended.
    assert (second.reel_start, second.reel_end) == (2.0, 5.0)


def test_a_phrase_inside_a_merged_span_keeps_its_own_position():
    index = _index(
        ("clip-01#001", "clip-01", 10.0, 12.0),
        ("clip-01#002", "clip-01", 12.0, 14.0),
    )
    entries = plan_reel(index, padding=0.0, max_seconds=None)[0].entries
    assert (entries[0].reel_start, entries[0].reel_end) == (0.0, 2.0)
    assert (entries[1].reel_start, entries[1].reel_end) == (2.0, 4.0)


def test_a_budget_stops_the_reel_rather_than_overspending():
    """Video is billed by the second, so the cap is a real limit, not advice."""
    index = _index(
        ("clip-01#001", "clip-01", 0.0, 10.0),
        ("clip-01#002", "clip-01", 30.0, 40.0),
        ("clip-01#003", "clip-01", 60.0, 70.0),
    )
    spans = plan_reel(index, padding=0.0, max_seconds=15.0)
    assert reel_seconds(spans) <= 15.0
    # Whole spans only: half a moment is worse than no moment.
    assert reel_seconds(spans) == 10.0


def test_no_phrases_means_no_reel():
    assert plan_reel(PhraseIndex(), padding=0.5, max_seconds=None) == []
