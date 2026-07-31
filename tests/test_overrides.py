"""The creator's running order, and what survives it.

An accent used to be keyed on its second of the edit, and a timeline clip had no
name at all. So "put this shot third" could not be said, and when the edit did
change, an accent whose absolute time now landed near a DIFFERENT clip's moment
was silently rebound to footage it was never about.

Every test here moves the edit under the decisions and checks that each decision
went with its own shot — or, when its shot is gone, that it was reported rather
than re-applied to whatever moved into its place.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from videoai.config import Config, PolishSettings
from videoai.core.models import (
    Analysis,
    Approval,
    ClipInfo,
    ClipOverride,
    ClipTranscript,
    DraftResult,
    EffectEvent,
    EffectOverride,
    EffectPlan,
    Manifest,
    Overrides,
    PlanSection,
    SegmentAnalysis,
    StoryPlan,
    Timeline,
    Transcript,
    Word,
)
from videoai.core.registry import StageContext
from videoai.core.store import ArtifactStore, hash_file, hash_parts
from videoai.logic.contract import has_closing_beat
from videoai.logic.decisions import (
    OVERRIDES_ARTIFACT,
    apply_effect_overrides,
    reproject_events,
)
from videoai.logic.timeline import apply_clip_overrides, build_timeline
from videoai.logic.validate import validate_timeline
from videoai.stages.s05_plan import assemble
from videoai.stages.s08_polish import (
    approval_is_current,
    build_captions,
    build_section_titles,
    section_changes,
)


# --------------------------------------------------------------------------- #
# Fixtures: a three-shot edit with a real phrase behind each shot
# --------------------------------------------------------------------------- #


BEATS = ["Hook", "Filling", "Closing"]


def _manifest(tmp_path: Path) -> Manifest:
    return Manifest(clips=[
        ClipInfo(clip_id="clip-01", path=str(tmp_path / "a.mp4"), duration=30.0,
                 width=1920, height=1080, fps=30.0, has_audio=True),
    ])


def _transcript() -> Transcript:
    return Transcript(provider="mock", clips=[ClipTranscript(clip_id="clip-01", words=[
        Word(text="hello", start=1.0, end=1.4),
        Word(text="everyone", start=1.5, end=2.0),
        Word(text="filling", start=6.0, end=6.5),
        Word(text="it", start=6.6, end=7.0),
        Word(text="thanks", start=11.0, end=11.5),
        Word(text="bye", start=11.6, end=12.0),
    ])])


def _analysis() -> Analysis:
    return Analysis(provider="mock", segments=[
        SegmentAnalysis(phrase_id="clip-01#001", clip_id="clip-01", start=1.0, end=2.0,
                        text="hello everyone", content="the greeting"),
        SegmentAnalysis(phrase_id="clip-01#002", clip_id="clip-01", start=6.0, end=7.0,
                        text="filling it", content="the syringe goes in"),
        SegmentAnalysis(phrase_id="clip-01#003", clip_id="clip-01", start=11.0, end=12.0,
                        text="thanks bye", content="the sign-off"),
    ])


def _story() -> StoryPlan:
    return StoryPlan(title="A Review", sections=[
        PlanSection(name=beat, goal=beat, phrase_ids=[f"clip-01#00{index + 1}"])
        for index, beat in enumerate(BEATS)
    ])


def _proposal(tmp_path: Path) -> Timeline:
    return build_timeline(
        _story(), _analysis(), _manifest(tmp_path), _transcript(),
        padding=0.15, fps=30.0,
    )


def _plan_of(timeline: Timeline) -> EffectPlan:
    """One accent per shot, each anchored half a second into its own shot."""
    return EffectPlan(events=[
        EffectEvent(
            clip_ref=clip.ref, at_in_clip=0.5,
            at_seconds=round(timeline.start_of(clip.ref) + 0.5, 3),
            effect_name=f"badge_{index}", screen_position="center", scale="medium",
            reason=clip.beat,
        )
        for index, clip in enumerate(timeline.clips)
    ])


# --------------------------------------------------------------------------- #
# Refs: every shot has a name, and no two share one
# --------------------------------------------------------------------------- #


def test_every_planned_shot_carries_the_reference_it_was_built_from(tmp_path: Path):
    refs = [clip.ref for clip in _proposal(tmp_path).clips]
    assert refs == ["clip-01#001", "clip-01#002", "clip-01#003"]


def test_one_insert_cut_in_twice_gets_two_distinct_refs(tmp_path: Path):
    manifest = Manifest(clips=[
        ClipInfo(clip_id="clip-01", path=str(tmp_path / "a.mp4"), duration=30.0,
                 width=1920, height=1080, fps=30.0, has_audio=True),
        ClipInfo(clip_id="clip-09", path=str(tmp_path / "b.mp4"), duration=8.0,
                 width=1920, height=1080, fps=30.0, has_audio=False),
    ])
    story = StoryPlan(sections=[PlanSection(
        name="Filling", goal="show it",
        phrase_ids=["insert:clip-09@1-4", "clip-01#002", "insert:clip-09@1-4"],
    )])

    refs = [clip.ref for clip in build_timeline(
        story, _analysis(), manifest, _transcript(), padding=0.15, fps=30.0
    ).clips]

    assert refs == ["insert:clip-09@1-4", "clip-01#002", "insert:clip-09@1-4~2"]
    assert len(set(refs)) == 3


def test_a_duplicate_ref_is_a_validation_failure(tmp_path: Path):
    timeline = _proposal(tmp_path)
    timeline.clips[1] = timeline.clips[1].model_copy(update={"ref": "clip-01#001"})

    violations = validate_timeline(timeline, _manifest(tmp_path), _transcript())

    assert any("duplicate ref" in violation for violation in violations)


def test_a_timeline_written_before_refs_existed_still_validates(tmp_path: Path):
    """The fallback the constraint asks for: no refs at all is an old artifact,
    not a broken one. Half of them missing is the state that is refused."""
    timeline = _proposal(tmp_path)
    bare = timeline.model_copy(update={
        "clips": [clip.model_copy(update={"ref": ""}) for clip in timeline.clips]
    })
    assert validate_timeline(bare, _manifest(tmp_path), _transcript()) == []

    half = timeline.model_copy(update={
        "clips": [timeline.clips[0].model_copy(update={"ref": ""}), *timeline.clips[1:]]
    })
    assert any(
        "has no ref" in violation
        for violation in validate_timeline(half, _manifest(tmp_path), _transcript())
    )


# --------------------------------------------------------------------------- #
# Reordering and disabling
# --------------------------------------------------------------------------- #


def _reordered(tmp_path: Path) -> tuple[Timeline, Timeline]:
    """The proposal, and the same shots with the sign-off moved to the front."""
    proposal = _proposal(tmp_path)
    overrides = Overrides(clips=[
        ClipOverride(ref="clip-01#003"),
        ClipOverride(ref="clip-01#001"),
        ClipOverride(ref="clip-01#002"),
    ])
    edit, notes = apply_clip_overrides(proposal, overrides)
    assert notes == []
    return proposal, edit


def test_a_reorder_puts_the_shots_in_the_creator_s_order(tmp_path: Path):
    _, edit = _reordered(tmp_path)
    assert [clip.ref for clip in edit.clips] == [
        "clip-01#003", "clip-01#001", "clip-01#002",
    ]


def test_a_reorder_recomputes_every_start(tmp_path: Path):
    _, edit = _reordered(tmp_path)
    position = 0.0
    for clip in edit.clips:
        assert clip.start == pytest.approx(position)
        position += clip.dur


def test_an_accent_stays_with_its_shot_when_the_shot_moves(tmp_path: Path):
    proposal, edit = _reordered(tmp_path)
    plan = _plan_of(proposal)

    projected, dropped = reproject_events(plan, edit)

    assert dropped == []
    by_name = {event.effect_name: event for event in projected.events}
    # badge_2 was the sign-off's accent and the sign-off now opens the video.
    assert by_name["badge_2"].clip_ref == "clip-01#003"
    assert by_name["badge_2"].at_seconds == pytest.approx(0.5, abs=0.01)
    for event in projected.events:
        assert event.at_seconds == pytest.approx(
            edit.start_of(event.clip_ref) + 0.5, abs=0.01
        )


def test_a_disabled_shot_takes_its_own_accent_with_it_and_nobody_else_s(tmp_path: Path):
    proposal = _proposal(tmp_path)
    plan = _plan_of(proposal)
    edit, _ = apply_clip_overrides(proposal, Overrides(clips=[
        ClipOverride(ref="clip-01#001"),
        ClipOverride(ref="clip-01#002", enabled=False, note="an adult walks through"),
        ClipOverride(ref="clip-01#003"),
    ]))

    projected, dropped = reproject_events(plan, edit)

    assert [clip.ref for clip in edit.clips] == ["clip-01#001", "clip-01#003"]
    assert [event.effect_name for event in projected.events] == ["badge_0", "badge_2"]
    assert len(dropped) == 1 and "clip-01#002" in dropped[0]
    # The sign-off's accent slid earlier with its own shot, and is still on it.
    assert projected.events[1].at_seconds == pytest.approx(
        edit.start_of("clip-01#003") + 0.5, abs=0.01
    )


def test_a_decision_is_never_re_applied_to_different_footage(tmp_path: Path):
    """The middle shot is dropped, so the third shot now plays over the second's
    old seconds. The decision made about the second must not land on it."""
    proposal = _proposal(tmp_path)
    plan = _plan_of(proposal)
    decided_at = plan.events[1].at_seconds
    overrides = Overrides(
        clips=[
            ClipOverride(ref="clip-01#001"),
            ClipOverride(ref="clip-01#002", enabled=False),
            ClipOverride(ref="clip-01#003"),
        ],
        effects=[EffectOverride(clip_ref="clip-01#002", at_in_clip=0.5, x=0.9, y=0.1)],
    )
    edit, _ = apply_clip_overrides(proposal, overrides)

    overridden, unmatched = apply_effect_overrides(plan, overrides)
    projected, _ = reproject_events(overridden, edit)

    # The shot that now occupies that second is the sign-off, and its accent is
    # untouched.
    survivor = next(
        event for event in projected.events
        if abs(event.at_seconds - decided_at) < 1.0
    )
    assert survivor.clip_ref == "clip-01#003"
    assert (survivor.x, survivor.y) == (None, None)
    # The decision was matched inside the plan, and then lost with its shot —
    # both facts are visible rather than silent.
    assert unmatched == 0
    assert all(event.clip_ref != "clip-01#002" for event in projected.events)


def test_a_shot_the_creator_never_saw_keeps_its_planned_place(tmp_path: Path):
    """A re-plan added a shot after the creator wrote their order. Appending it
    would put new material after the sign-off."""
    proposal = _proposal(tmp_path)
    edit, notes = apply_clip_overrides(proposal, Overrides(clips=[
        ClipOverride(ref="clip-01#001"),
        ClipOverride(ref="clip-01#003"),
    ]))

    assert [clip.ref for clip in edit.clips] == [
        "clip-01#001", "clip-01#002", "clip-01#003",
    ]
    assert any("clip-01#002" in note for note in notes)


def test_an_order_naming_a_shot_the_re_plan_dropped_says_so(tmp_path: Path):
    _, notes = apply_clip_overrides(_proposal(tmp_path), Overrides(clips=[
        ClipOverride(ref="clip-01#001"),
        ClipOverride(ref="clip-01#404"),
    ]))
    assert any("clip-01#404" in note for note in notes)


# --------------------------------------------------------------------------- #
# What the delivery derives from clip order
# --------------------------------------------------------------------------- #


def test_a_reorder_moves_the_section_titles_with_the_beats(tmp_path: Path):
    proposal, edit = _reordered(tmp_path)
    durations = [clip.dur for clip in edit.clips]
    starts = [clip.start for clip in edit.clips]

    titles = build_section_titles(edit, starts, durations, 0.0, 2.0)

    # Index 0 never fires a title, so the reordered edit names the two beats that
    # now differ from the clip before them — not the two the proposal named.
    assert [title.text for title in titles] == ["Hook", "Filling"]
    assert section_changes(edit) == [1, 2]
    assert [
        title.text
        for title in build_section_titles(
            proposal, [clip.start for clip in proposal.clips],
            [clip.dur for clip in proposal.clips], 0.0, 2.0,
        )
    ] == ["Filling", "Closing"]


def test_a_reorder_maps_the_captions_through_the_new_cuts(tmp_path: Path):
    _, edit = _reordered(tmp_path)
    starts = [clip.start for clip in edit.clips]

    captions = build_captions(edit, _transcript(), starts, [], 0.0, 0.0, 4)

    # "thanks bye" is spoken in the shot that now opens the video.
    assert captions[0].text == "thanks bye"
    assert captions[0].start < 1.0


def test_disabling_the_last_shot_removes_the_closing_beat_the_contract_requires(
    tmp_path: Path,
):
    proposal = _proposal(tmp_path)
    assert has_closing_beat(proposal)

    edit, _ = apply_clip_overrides(proposal, Overrides(clips=[
        ClipOverride(ref="clip-01#001"),
        ClipOverride(ref="clip-01#002"),
        ClipOverride(ref="clip-01#003", enabled=False),
    ]))

    assert not has_closing_beat(edit)


# --------------------------------------------------------------------------- #
# The assemble stage, and what a reorder does to an approval
# --------------------------------------------------------------------------- #


def _context(tmp_path: Path) -> StageContext:
    work = tmp_path / "work"
    return StageContext(
        project_dir=tmp_path, input_dir=tmp_path, work_dir=work,
        output_dir=tmp_path / "output",
        config=Config(polish=PolishSettings(require_approval=True)),
        store=ArtifactStore(work),
    )


def _seed(ctx: StageContext, tmp_path: Path) -> None:
    ctx.store.write("01-manifest", _manifest(tmp_path), fingerprint="fp")
    ctx.store.write("03-transcript", _transcript(), fingerprint="fp")
    ctx.store.write("05-proposal", _proposal(tmp_path), fingerprint="fp")


def test_assemble_delivers_the_proposal_untouched_when_nobody_has_decided_anything(
    tmp_path: Path,
):
    ctx = _context(tmp_path)
    _seed(ctx, tmp_path)

    assert assemble(ctx) == _proposal(tmp_path)


def test_assemble_applies_the_creator_s_order(tmp_path: Path):
    ctx = _context(tmp_path)
    _seed(ctx, tmp_path)
    ctx.store.write(OVERRIDES_ARTIFACT, Overrides(clips=[
        ClipOverride(ref="clip-01#003"),
        ClipOverride(ref="clip-01#002"),
        ClipOverride(ref="clip-01#001"),
    ]), fingerprint="creator")

    assert [clip.ref for clip in assemble(ctx).clips] == [
        "clip-01#003", "clip-01#002", "clip-01#001",
    ]


def test_a_reorder_invalidates_the_approval(tmp_path: Path):
    """Approval binds to the exact edit. Moving a shot is a different video, and
    a different video has not been reviewed."""
    ctx = _context(tmp_path)
    _seed(ctx, tmp_path)
    ctx.store.write("05-timeline", assemble(ctx), fingerprint="fp")

    draft_file = tmp_path / "output" / "draft.mp4"
    draft_file.parent.mkdir(parents=True, exist_ok=True)
    draft_file.write_bytes(b"a draft")
    draft = DraftResult(
        path=str(draft_file), duration=3.0, segment_count=3,
        timeline_hash=ctx.store.content_hash("05-timeline") or "",
    )
    ctx.store.write("06-approval", Approval(
        timeline_hash=ctx.store.content_hash("05-timeline") or "",
        draft_hash=hash_file(draft_file),
        config_hash=hash_parts(ctx.config.model_dump_json()),
        approved_at=datetime.now(timezone.utc).isoformat(),
    ), fingerprint="manual-approval")
    approval_is_current(ctx, draft)

    ctx.store.write(OVERRIDES_ARTIFACT, Overrides(clips=[
        ClipOverride(ref="clip-01#003"),
        ClipOverride(ref="clip-01#001"),
        ClipOverride(ref="clip-01#002"),
    ]), fingerprint="creator")
    ctx.store.write("05-timeline", assemble(ctx), fingerprint="fp")

    with pytest.raises(RuntimeError, match="approval is stale for: timeline"):
        approval_is_current(ctx, draft)
