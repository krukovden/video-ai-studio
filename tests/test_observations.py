"""What the model saw, put back on the source clock.

Observations are the one place a model's timestamps enter the pipeline, and they
enter as evidence rather than as instructions: the model says what happened and
when it saw it, the pipeline decides where that lands and whether to believe it.
"""
from __future__ import annotations

from videoai.core.models import Observation
from videoai.logic.reel import ReelEntry, ReelSpan
from videoai.stages.s04_analyze import parse_observations


def _spans() -> list[ReelSpan]:
    return [
        ReelSpan("clip-01", 10.0, 12.0, [ReelEntry("clip-01#001", 0.0, 2.0)]),
        ReelSpan("clip-02", 30.0, 33.0, [ReelEntry("clip-02#001", 2.0, 5.0)]),
    ]


def test_an_observation_is_placed_back_in_its_source_clip():
    parsed = parse_observations(
        {"observations": [
            {"at": 0.5, "kind": "action", "what": "the lid comes off"},
        ]},
        _spans(),
    )
    assert len(parsed) == 1
    seen = parsed[0]
    assert seen.clip_id == "clip-01"
    assert seen.at == 10.5
    assert seen.reel_at == 0.5
    assert seen.kind == "action"
    assert seen.what == "the lid comes off"


def test_a_later_observation_accounts_for_the_spans_before_it():
    parsed = parse_observations(
        {"observations": [{"at": 3.5, "kind": "emotion", "what": "his face lights up"}]},
        _spans(),
    )
    assert (parsed[0].clip_id, parsed[0].at) == ("clip-02", 31.5)


def test_an_observation_outside_the_reel_is_dropped():
    """The model naming a moment the file does not contain saw nothing there."""
    parsed = parse_observations(
        {"observations": [
            {"at": 99.0, "kind": "action", "what": "invented"},
            {"at": 1.0, "kind": "action", "what": "real"},
        ]},
        _spans(),
    )
    assert [item.what for item in parsed] == ["real"]


def test_observations_come_back_in_time_order():
    parsed = parse_observations(
        {"observations": [
            {"at": 3.0, "kind": "action", "what": "second"},
            {"at": 0.2, "kind": "action", "what": "first"},
        ]},
        _spans(),
    )
    assert [item.what for item in parsed] == ["first", "second"]


def test_an_unusable_entry_is_skipped_rather_than_failing_the_stage():
    parsed = parse_observations(
        {"observations": [
            {"kind": "action", "what": "no time at all"},
            {"at": "banana", "kind": "action", "what": "unreadable time"},
            {"at": 1.0, "what": ""},
            {"at": 1.5, "kind": "action", "what": "keeps its place"},
        ]},
        _spans(),
    )
    assert [item.what for item in parsed] == ["keeps its place"]


def test_an_unknown_kind_falls_back_rather_than_raising():
    parsed = parse_observations(
        {"observations": [{"at": 1.0, "kind": "vibes", "what": "something"}]},
        _spans(),
    )
    assert parsed[0].kind == "action"


def test_no_observations_is_a_valid_answer():
    assert parse_observations({"segments": []}, _spans()) == []


def test_the_model_round_trips():
    seen = Observation(clip_id="clip-01", at=10.5, reel_at=0.5,
                       kind="emotion", what="a grin")
    assert Observation.model_validate_json(seen.model_dump_json()) == seen
