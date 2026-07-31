"""The page is about the edit, not only about the badges on it.

A creator can rearrange the running order, take a shot out without losing it,
and swap, drag, resize or turn every accent. All of that has to arrive in
`05e-overrides` as one document, and what the browser remembers between visits
has to be keyed on the thing it is about — the project, and each accent's own
anchor — rather than on a timestamp and an array index, which is how two
unrelated projects came to share one another's saved work.
"""
from __future__ import annotations

import json
from pathlib import Path

from videoai.cli import _shown_order
from videoai.core.models import (
    ClipOverride,
    EffectEvent,
    EffectPlan,
    EffectOverride,
    Overrides,
    Timeline,
    TimelineClip,
)
from videoai.core.store import ArtifactStore
from videoai.logic.approval import apply_decisions
from videoai.logic.decisions import OVERRIDES_ARTIFACT, apply_effect_overrides
from videoai.logic.effects_preview import (
    EffectProposal,
    ShotCard,
    render_preview_html,
)

REFS = ["clip-01#001", "clip-01#002", "clip-01#003"]
BEATS = ["Hook", "Filling", "Closing"]


def _timeline(refs: list[str]) -> Timeline:
    clips, start = [], 0.0
    for ref in refs:
        index = REFS.index(ref)
        clips.append(TimelineClip(
            ref=ref, src="clip-01", offset=index * 10.0, dur=4.0, start=start,
            beat=BEATS[index], quote=f"line {index + 1}",
        ))
        start += 4.0
    return Timeline(fps=30, width=1920, height=1080, clips=clips)


def _plan() -> EffectPlan:
    return EffectPlan(events=[
        EffectEvent(clip_ref=ref, at_in_clip=1.0, at_seconds=1.0 + position * 4.0,
                    effect_name=f"badge_{position}", screen_position="center",
                    scale="medium", reason="r")
        for position, ref in enumerate(REFS)
    ])


def _project(tmp_path: Path) -> ArtifactStore:
    store = ArtifactStore(tmp_path / "work")
    store.write("05-proposal", _timeline(REFS), fingerprint="f")
    store.write("05-timeline", _timeline(REFS), fingerprint="f")
    store.write("05d-effects", _plan(), fingerprint="f")
    return store


def _overrides(store: ArtifactStore) -> Overrides:
    return store.read(OVERRIDES_ARTIFACT, Overrides)


def _accent(position: int, **extra) -> dict:
    return {
        "index": position, "at": 1.0 + position * 4.0,
        "clip_ref": REFS[position], "at_in_clip": 1.0,
        "from_name": f"badge_{position}", "effect_name": f"badge_{position}",
        "keep": True, **extra,
    }


# --------------------------------------------------------------------------- #
# The running order
# --------------------------------------------------------------------------- #


def test_a_reordered_running_order_reaches_the_overrides(tmp_path: Path):
    store = _project(tmp_path)

    apply_decisions(store, {"version": 2, "events": [], "clips": [
        {"ref": "clip-01#003", "enabled": True},
        {"ref": "clip-01#001", "enabled": True},
        {"ref": "clip-01#002", "enabled": True},
    ]})

    assert _overrides(store).order == ["clip-01#003", "clip-01#001", "clip-01#002"]


def test_a_reorder_is_reported_rather_than_summarised_as_nothing(tmp_path: Path):
    """"0 kept, 0 dropped" after rearranging the whole video reads as a failure."""
    store = _project(tmp_path)

    result = apply_decisions(store, {"events": [], "clips": [
        {"ref": "clip-01#002", "enabled": True},
        {"ref": "clip-01#001", "enabled": True},
        {"ref": "clip-01#003", "enabled": True},
    ]})

    assert result.reordered is True
    assert "running order changed" in result.summary()


def test_saving_the_same_order_twice_does_not_claim_a_second_reorder(tmp_path: Path):
    store = _project(tmp_path)
    order = [{"ref": ref, "enabled": True} for ref in REFS]

    result = apply_decisions(store, {"events": [], "clips": order})

    assert result.reordered is False


def test_a_switched_off_shot_is_recorded_in_place_rather_than_deleted(tmp_path: Path):
    """It has to stay in the list, or the next page has nothing to draw greyed —
    and a shot that vanished could not be switched back on."""
    store = _project(tmp_path)

    result = apply_decisions(store, {"events": [], "clips": [
        {"ref": "clip-01#001", "enabled": True},
        {"ref": "clip-01#002", "enabled": False},
        {"ref": "clip-01#003", "enabled": True},
    ]})

    decided = _overrides(store).clips
    assert [(item.ref, item.enabled) for item in decided] == [
        ("clip-01#001", True), ("clip-01#002", False), ("clip-01#003", True),
    ]
    assert _overrides(store).order == ["clip-01#001", "clip-01#003"]
    assert result.shots_off == 1
    assert "1 shot(s) switched off" in result.summary()


def test_a_later_save_about_only_the_accents_leaves_the_order_alone(tmp_path: Path):
    store = _project(tmp_path)
    apply_decisions(store, {"events": [], "clips": [
        {"ref": "clip-01#003", "enabled": True},
        {"ref": "clip-01#002", "enabled": False},
        {"ref": "clip-01#001", "enabled": True},
    ]})

    apply_decisions(store, {"events": [_accent(0, keep=False)]})

    assert [(item.ref, item.enabled) for item in _overrides(store).clips] == [
        ("clip-01#003", True), ("clip-01#002", False), ("clip-01#001", True),
    ]


def test_the_page_keeps_a_switched_off_shot_in_the_row(tmp_path: Path):
    """`05-timeline` no longer contains it — assembly took it out — so the page
    is built from the proposal and the creator's own list instead."""
    store = _project(tmp_path)
    store.write("05-timeline", _timeline(["clip-01#001", "clip-01#003"]), fingerprint="f")
    store.write(OVERRIDES_ARTIFACT, Overrides(clips=[
        ClipOverride(ref="clip-01#003"),
        ClipOverride(ref="clip-01#002", enabled=False),
        ClipOverride(ref="clip-01#001"),
    ]), fingerprint="creator")

    shown = _shown_order(store, store.read("05-timeline", Timeline))

    assert [(clip.ref, enabled) for clip, enabled in shown] == [
        ("clip-01#003", True), ("clip-01#002", False), ("clip-01#001", True),
    ]


# --------------------------------------------------------------------------- #
# Size and tilt
# --------------------------------------------------------------------------- #


def test_a_resize_and_a_turn_reach_the_overrides(tmp_path: Path):
    store = _project(tmp_path)

    result = apply_decisions(store, {"events": [
        _accent(1, scale_factor=1.6, rotation=-20.0, resized=True, rotated=True),
    ]})

    decided = _overrides(store).effects[0]
    assert (decided.clip_ref, decided.scale_factor, decided.rotation) == (
        "clip-01#002", 1.6, -20.0,
    )
    assert result.resized == 1 and result.rotated == 1
    assert "1 resized" in result.summary() and "1 turned" in result.summary()


def test_a_resize_and_a_turn_reach_the_plan_the_render_reads(tmp_path: Path):
    store = _project(tmp_path)
    apply_decisions(store, {"events": [
        _accent(1, scale_factor=0.7, rotation=35.0, resized=True, rotated=True),
    ]})

    plan, unmatched = apply_effect_overrides(
        store.read("05d-effects", EffectPlan), _overrides(store)
    )

    assert unmatched == 0
    assert (plan.events[1].scale_factor, plan.events[1].rotation) == (0.7, 35.0)
    # And nobody else's badge grew or tilted.
    assert [event.scale_factor for event in plan.events] == [1.0, 0.7, 1.0]
    assert [event.rotation for event in plan.events] == [0.0, 35.0, 0.0]


def test_an_untouched_size_control_does_not_reset_a_size_from_an_earlier_visit(
    tmp_path: Path,
):
    """The page opens showing the size already on file, so a save that only
    dragged a badge must not write the default back over it."""
    store = _project(tmp_path)
    apply_decisions(store, {"events": [
        _accent(0, scale_factor=1.8, rotation=12.0, resized=True, rotated=True),
    ]})

    apply_decisions(store, {"events": [
        _accent(0, x=0.3, y=0.4, moved=True, scale_factor=1.0, rotation=0.0),
    ]})

    decided = _overrides(store).effects[0]
    assert (decided.scale_factor, decided.rotation) == (1.8, 12.0)
    assert (decided.x, decided.y) == (0.3, 0.4)


def test_a_size_dragged_back_to_normal_is_still_a_decision(tmp_path: Path):
    store = _project(tmp_path)
    apply_decisions(store, {"events": [_accent(0, scale_factor=1.8, resized=True)]})

    apply_decisions(store, {"events": [_accent(0, scale_factor=1.0, resized=True)]})

    assert _overrides(store).effects[0].scale_factor == 1.0


def test_one_document_carries_the_order_the_shots_and_every_accent(tmp_path: Path):
    """The whole point of the page: one Save, one artifact, everything in it."""
    store = _project(tmp_path)

    apply_decisions(store, {
        "version": 2,
        "clips": [
            {"ref": "clip-01#003", "enabled": True},
            {"ref": "clip-01#002", "enabled": False},
            {"ref": "clip-01#001", "enabled": True},
        ],
        "events": [
            _accent(0, effect_name="badge_star", swapped=True),
            _accent(1, keep=False),
            _accent(2, x=0.8, y=0.2, moved=True,
                    scale_factor=1.5, resized=True, rotation=-45.0, rotated=True),
        ],
    })

    written = _overrides(store)
    assert written.order == ["clip-01#003", "clip-01#001"]
    by_ref = {item.clip_ref: item for item in written.effects}
    assert by_ref["clip-01#001"].effect_name == "badge_star"
    assert by_ref["clip-01#002"].keep is False
    assert (by_ref["clip-01#003"].x, by_ref["clip-01#003"].y) == (0.8, 0.2)
    assert by_ref["clip-01#003"].scale_factor == 1.5
    assert by_ref["clip-01#003"].rotation == -45.0


# --------------------------------------------------------------------------- #
# The page itself
# --------------------------------------------------------------------------- #


def _shots() -> list[ShotCard]:
    return [
        ShotCard(ref=ref, beat=BEATS[index], quote=f"line {index + 1}",
                 duration=4.0, enabled=(ref != "clip-01#002"), thumb_jpeg=b"jpeg")
        for index, ref in enumerate(REFS)
    ]


def _proposal(position: int, **changes) -> EffectProposal:
    base = dict(clip_ref=REFS[position], at_in_clip=1.0,
                at_seconds=1.0 + position * 4.0, effect_name=f"badge_{position}",
                screen_position="center", scale="medium", reason="r")
    return EffectProposal(
        index=position, event=EffectEvent(**{**base, **changes}),
        clip_ref=REFS[position], motion_xy=None, motion_cell=None,
        frame_jpeg=b"jpeg", sprite_png=b"png",
        sprite_width=0.1, sprite_height=0.2, start_x=0.4, start_y=0.4,
    )


def _page(**changes) -> str:
    settings = dict(
        shots=_shots(), proposals=[_proposal(0), _proposal(1)], title="A Review",
        sprite_choices=[{"name": "badge_0", "aspect": 1.0, "height_frac": 0.22,
                         "takes_text": False, "data": "png"}],
        project_key="abc123",
    )
    settings.update(changes)
    return render_preview_html(**settings)


def _embedded(page: str, name: str):
    """The JSON the page was built with, read back out of its own script."""
    prefix = f"const {name} = "
    line = next(row for row in page.splitlines() if row.startswith(prefix))
    return json.loads(line[len(prefix):].rstrip(";").replace("<\\/", "</"))


def test_every_shot_is_on_the_page_in_running_order():
    shots = _embedded(_page(), "SHOTS")
    assert [shot["ref"] for shot in shots] == REFS
    assert [shot["beat"] for shot in shots] == BEATS
    assert [shot["dur"] for shot in shots] == [4.0, 4.0, 4.0]
    assert [shot["enabled"] for shot in shots] == [True, False, True]


def test_each_accent_names_the_shot_it_is_nested_under():
    """This is what makes a moved shot visibly carry its badges: the page groups
    on it, so there is no second list to keep in step."""
    accents = _embedded(_page(), "DATA")
    assert [accent["shot"] for accent in accents] == REFS[:2]


def test_an_accent_carries_the_anchor_the_pipeline_files_it_under():
    accents = _embedded(_page(), "DATA")
    assert [accent["anchor"] for accent in accents] == [
        "clip-01#001@1.00", "clip-01#002@1.00",
    ]


def test_the_creators_own_size_and_tilt_open_where_they_were_left():
    page = _page(proposals=[_proposal(0, scale_factor=1.4, rotation=-30.0)])
    accent = _embedded(page, "DATA")[0]
    assert (accent["factor"], accent["rot"]) == (1.4, -30.0)


def test_what_the_browser_remembers_is_scoped_to_the_project():
    """Keying on the first accent's timestamp meant two projects whose first
    accent shared a second shared each other's work, and a re-plan that moved it
    orphaned every saved edit without a word."""
    page = _page(project_key="deadbeef")
    assert "videoai-edit-deadbeef" in page
    assert "'videoai-effects-' + " not in page
    assert "DATA[0].at" not in page


def test_a_restored_edit_is_matched_on_its_anchor_and_never_on_a_list_position():
    page = _page()
    assert "state.find(item => item.anchor === row.anchor)" in page
    assert "item.index === row.index" not in page


def test_a_served_page_tells_a_restored_creator_to_press_save():
    served = _page(save_url="/save", token="t")
    offline = _page()
    assert "Press <b>Save</b> to write them into the edit." in served
    assert "Copy as text" in offline


def test_a_quote_cannot_end_the_script_it_is_embedded_in():
    shots = _shots()
    shots[0] = ShotCard(ref=shots[0].ref, beat=shots[0].beat,
                        quote="</script><img src=x onerror=alert(1)>",
                        duration=4.0, enabled=True, thumb_jpeg=b"jpeg")
    page = _page(shots=shots)
    assert "</script><img" not in page
    assert _embedded(page, "SHOTS")[0]["quote"].startswith("</script>")


def test_a_calm_review_with_no_accents_is_still_an_edit_to_arrange():
    page = _page(proposals=[])
    assert _embedded(page, "DATA") == []
    assert [shot["ref"] for shot in _embedded(page, "SHOTS")] == REFS


def test_the_size_control_offers_exactly_the_range_the_render_will_honour():
    from videoai.logic.effects import MAX_SCALE_FACTOR, MIN_SCALE_FACTOR

    page = _page()
    assert f"const MIN_FACTOR = {MIN_SCALE_FACTOR};" in page
    assert f"const MAX_FACTOR = {MAX_SCALE_FACTOR};" in page


def test_the_page_reaches_for_nothing_outside_itself():
    """It is opened off a disk with no network, and the frames are of a child."""
    page = _page()
    for scheme in ("http://", "https://", "//cdn", "@import"):
        assert scheme not in page.replace("http://127.0.0.1", "")
