"""Machine enforcement of production-contract.yaml.

The rule this file exists to defend: a line in `production-contract.yaml` reads as
a guarantee, so it must either be measured or not be there. An unenforced rule is
worse than a missing one, because everybody downstream believes it.
"""
from pathlib import Path

import pytest
import yaml

from videoai.core.models import Timeline, TimelineClip
from videoai.logic.contract import (
    contract_hash,
    contract_path,
    estimated_delivery_bytes,
    fallback_output_name,
    free_bytes,
    has_closing_beat,
    load_contract,
    preflight,
    validate_production_report,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

# Every leaf rule the shipped contract is allowed to contain, and therefore every
# rule something in this repository measures. Adding a rule to the yaml without
# adding its check fails this test on purpose.
ENFORCED_RULES = {
    "version",
    "required_output.width",
    "required_output.height",
    "required_output.full_decode",
    "required_features.intro",
    "required_features.outro",
    "required_features.section_titles",
    "required_features.captions",
    "required_features.transitions",
    "required_features.music",
    "required_features.music_ducking.minimum_db",
    "required_features.closing_beat",
    "quality.source",
    "quality.maximum_lossy_video_generations",
    "failure_policy.fallback_output_name",
}


def _leaf_paths(data, prefix: str = "") -> set[str]:
    if not isinstance(data, dict):
        return {prefix}
    paths: set[str] = set()
    for key, value in data.items():
        paths |= _leaf_paths(value, f"{prefix}.{key}" if prefix else str(key))
    return paths


def _passing_report() -> dict:
    return {
        "contract_version": 1,
        "status": "passed",
        "output": "output/final.mp4",
        "duration": 190.0,
        "width": 1920,
        "height": 1080,
        "features": {
            "intro": True,
            "outro": True,
            "section_titles": 3,
            "captions": 74,
            "soft_captions": 74,
            "burned_in_captions": False,
            "music": "bensound-funday.mp3",
            "music_ducking": 9.4,
            "transitions": 12,
            "section_boundaries": 5,
            "closing_beat": True,
            "full_decode": True,
        },
        "quality": {
            "source": "originals",
            "segment_inputs": ["/footage/clip-01.MOV"],
            "proxy_inputs": [],
            "lossy_video_generations": 1,
        },
    }


def test_every_rule_in_the_shipped_contract_is_enforced():
    """The whole point of the file: no decorative rules."""
    assert _leaf_paths(load_contract(REPO_ROOT)) == ENFORCED_RULES


def test_the_shipped_contract_passes_a_report_that_meets_it(tmp_path: Path):
    validate_production_report(_passing_report(), REPO_ROOT)


@pytest.mark.parametrize(
    "mutate, expected",
    [
        (lambda r: r.update(width=1280), "width=1280"),
        (lambda r: r.update(height=720), "height=720"),
        (lambda r: r["features"].update(full_decode=False), "did not decode end to end"),
        (lambda r: r["features"].update(intro=False), "required feature missing: intro"),
        (lambda r: r["features"].update(outro=False), "required feature missing: outro"),
        (
            lambda r: r["features"].update(section_titles=0),
            "required feature missing: section_titles",
        ),
        (lambda r: r["features"].update(captions=0), "required feature missing: captions"),
        (
            lambda r: r["features"].update(transitions=0),
            "required feature missing: transitions",
        ),
        (lambda r: r["features"].update(music=""), "required feature missing: music"),
        (
            lambda r: r["features"].update(closing_beat=False),
            "required feature missing: closing_beat",
        ),
        (
            lambda r: r["features"].update(music_ducking=1.2),
            "music_ducking measured 1.20 dB",
        ),
        (
            lambda r: r["features"].update(music_ducking=0),
            "required feature missing: music_ducking",
        ),
        (
            lambda r: r["quality"].update(source="proxies"),
            "not built from original sources",
        ),
        (
            lambda r: r["quality"].update(proxy_inputs=["/work/media/clip-proxy-540p.mp4"]),
            "cut from proxies rather than originals",
        ),
        (
            lambda r: r["quality"].update(lossy_video_generations=2),
            "lossy video generations=2",
        ),
    ],
)
def test_each_rule_actually_rejects_a_report_that_breaks_it(mutate, expected: str):
    report = _passing_report()
    mutate(report)

    with pytest.raises(RuntimeError, match=expected):
        validate_production_report(report, REPO_ROOT)


def test_a_rule_set_to_false_requires_nothing(tmp_path: Path):
    (tmp_path / "production-contract.yaml").write_text(
        "version: 1\n"
        "required_output: {width: 1920, height: 1080}\n"
        "required_features: {captions: false}\n",
        encoding="utf-8",
    )
    report = _passing_report()
    report["features"]["captions"] = 0

    validate_production_report(report, tmp_path)


def test_a_project_contract_overrides_the_repository_one(tmp_path: Path):
    local = tmp_path / "production-contract.yaml"
    local.write_text(
        "version: 1\nrequired_output: {width: 3840, height: 2160}\n", encoding="utf-8"
    )

    assert contract_path(tmp_path) == local
    with pytest.raises(RuntimeError, match="width=1920"):
        validate_production_report(_passing_report(), tmp_path)


def test_a_contract_without_a_version_is_refused(tmp_path: Path):
    (tmp_path / "production-contract.yaml").write_text("required_output: {}\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="invalid production contract"):
        load_contract(tmp_path)


def test_the_contract_hash_follows_its_content(tmp_path: Path):
    local = tmp_path / "production-contract.yaml"
    local.write_text("version: 1\nrequired_output: {width: 1920}\n", encoding="utf-8")
    before = contract_hash(tmp_path)

    local.write_text("version: 1\nrequired_output: {width: 3840}\n", encoding="utf-8")

    assert contract_hash(tmp_path) != before
    assert contract_hash(tmp_path) == contract_hash(tmp_path)


def test_the_fallback_name_comes_from_the_contract():
    contract = load_contract(REPO_ROOT)

    assert fallback_output_name(contract) == "preview-fallback.mp4"
    assert fallback_output_name({}) == "preview-fallback.mp4"
    assert "final" not in fallback_output_name(contract).removesuffix(".mp4")


# --- preflight ---------------------------------------------------------------


def _preflight(**overrides) -> None:
    arguments = dict(
        frame=(1920, 1080),
        output_height=1080,
        music_dir=Path("/music"),
        track_count=3,
        closing_beat=True,
        estimated_bytes=1_000,
        available_bytes=10_000,
    )
    arguments.update(overrides)
    preflight(load_contract(REPO_ROOT), **arguments)


def test_preflight_passes_a_delivery_the_contract_can_accept():
    _preflight()


def test_preflight_rejects_a_delivery_height_the_contract_forbids():
    with pytest.raises(RuntimeError, match="1080px-tall delivery but polish.output_height=720"):
        _preflight(frame=(1280, 720), output_height=720)


def test_preflight_rejects_an_aspect_the_contract_width_cannot_match():
    """A 4:3 source at 1080p tall is 1440 wide, not 1920. Discovering that after the
    render is an hour lost to arithmetic that took a single probe."""
    with pytest.raises(RuntimeError, match="1440x1080"):
        _preflight(frame=(1440, 1080))


def test_preflight_rejects_a_missing_music_library_by_name():
    with pytest.raises(RuntimeError, match="/music holds no playable track"):
        _preflight(track_count=0)


def test_preflight_rejects_a_timeline_that_never_signs_off():
    with pytest.raises(RuntimeError, match="closing story beat"):
        _preflight(closing_beat=False)


def test_preflight_rejects_a_disk_that_cannot_hold_the_intermediates():
    with pytest.raises(RuntimeError, match="GB of scratch space"):
        _preflight(estimated_bytes=40_000_000_000, available_bytes=1_000_000_000)


def test_preflight_reports_every_problem_at_once():
    """One run, one list. Fixing four things one render at a time is the failure
    mode preflight exists to remove."""
    with pytest.raises(RuntimeError) as failure:
        _preflight(frame=(1280, 720), output_height=720, track_count=0, closing_beat=False)

    message = str(failure.value)
    assert message.count("\n- ") == 4


def test_the_disk_estimate_grows_with_the_frame_and_the_length():
    small = estimated_delivery_bytes((1280, 720), 30.0, 60.0)
    tall = estimated_delivery_bytes((3840, 2160), 30.0, 60.0)
    long = estimated_delivery_bytes((1280, 720), 30.0, 600.0)

    assert small > 0
    assert tall > small * 5
    assert long == pytest.approx(small * 10, rel=0.01)


def test_free_bytes_walks_up_to_a_directory_that_exists(tmp_path: Path):
    """The polish stage asks about `work/`, which the runner may not have made yet."""
    assert free_bytes(tmp_path / "work" / "delivery" / "nested") > 0


# --- the closing beat -------------------------------------------------------


@pytest.mark.parametrize(
    "beat, quote, closing",
    [
        ("Closing", "", True),
        ("Wrap Up", "", True),
        ("Verdict", "", True),
        ("Middle", "Thanks for watching!", True),
        ("Middle", "and then it popped", False),
        ("", "", False),
    ],
)
def test_has_closing_beat_reads_the_last_sections_language(beat, quote, closing):
    timeline = Timeline(fps=30, width=1920, height=1080, clips=[
        TimelineClip(src="clip-01", offset=0, dur=2, start=0, beat="Hook"),
        TimelineClip(src="clip-01", offset=2, dur=2, start=2, beat=beat, quote=quote),
    ])

    assert has_closing_beat(timeline) is closing


def test_the_shipped_contract_is_valid_yaml_and_names_its_version():
    raw = yaml.safe_load(contract_path(REPO_ROOT).read_text(encoding="utf-8"))

    assert raw["version"] == 1
