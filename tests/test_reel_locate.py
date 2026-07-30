"""Putting a moment the model saw back where it came from.

The model watches a reel and names times on the reel's clock. Everything
downstream — the plan, the effects, the cut — works in source-clip time. This
mapping is what makes an observation usable, and it is arithmetic the pipeline
owns rather than something the model is asked to get right.
"""
from __future__ import annotations

from videoai.logic.reel import ReelEntry, ReelSpan, locate_in_source


def _spans() -> list[ReelSpan]:
    # Two spans laid end to end: 0.0-2.0s of the reel is clip-01 from 10s,
    # 2.0-5.0s is clip-02 from 30s.
    return [
        ReelSpan("clip-01", 10.0, 12.0, [ReelEntry("clip-01#001", 0.0, 2.0)]),
        ReelSpan("clip-02", 30.0, 33.0, [ReelEntry("clip-02#001", 2.0, 5.0)]),
    ]


def test_a_time_in_the_first_span_maps_to_its_clip():
    assert locate_in_source(_spans(), 0.5) == ("clip-01", 10.5)


def test_a_time_in_a_later_span_accounts_for_everything_before_it():
    assert locate_in_source(_spans(), 3.5) == ("clip-02", 31.5)


def test_the_first_frame_of_the_reel_maps_to_the_first_span():
    assert locate_in_source(_spans(), 0.0) == ("clip-01", 10.0)


def test_a_boundary_belongs_to_the_span_it_starts():
    """2.0s is the first frame of the second span, not the last of the first."""
    assert locate_in_source(_spans(), 2.0) == ("clip-02", 30.0)


def test_a_time_past_the_end_of_the_reel_is_refused():
    """A model naming a moment that is not in the file it watched is describing
    something it did not see; guessing which span was meant would invent data."""
    assert locate_in_source(_spans(), 9.0) is None


def test_a_negative_time_is_refused():
    assert locate_in_source(_spans(), -0.5) is None


def test_an_empty_reel_locates_nothing():
    assert locate_in_source([], 1.0) is None
