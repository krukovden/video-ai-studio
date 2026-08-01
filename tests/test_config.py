from pathlib import Path

import pytest
from pydantic import ValidationError

from videoai.config import Config, DescribeSettings, load_config


def test_load_config_returns_defaults_when_file_missing(tmp_path: Path):
    config = load_config(tmp_path / "nope.yaml")
    assert config.providers["asr"] == "parakeet"
    assert config.providers["llm"] == "claude_cli"
    assert config.transcribe.phrase_gap_seconds == 0.5


def test_load_config_overrides_from_yaml(tmp_path: Path):
    path = tmp_path / "config.yaml"
    path.write_text(
        "providers:\n  asr: mock\n  llm: mock\ntranscribe:\n  phrase_gap_seconds: 0.9\n",
        encoding="utf-8",
    )
    config = load_config(path)
    assert config.providers["asr"] == "mock"
    assert config.providers["llm"] == "mock"
    assert config.transcribe.phrase_gap_seconds == 0.9
    assert config.render.draft_height == 720


def test_unknown_provider_key_is_rejected(tmp_path: Path):
    path = tmp_path / "config.yaml"
    path.write_text('providers:\n  telepathy: "yes"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="unknown provider key"):
        load_config(path)


def test_config_is_immutable():
    config = Config()
    with pytest.raises(Exception):
        config.providers = {}


def test_load_config_default_sync_primary_camera_is_none(tmp_path: Path):
    config = load_config(tmp_path / "nope.yaml")
    assert config.sync.primary_camera is None


def test_load_config_overrides_sync_primary_camera(tmp_path: Path):
    path = tmp_path / "config.yaml"
    path.write_text("sync:\n  primary_camera: cam-a\n", encoding="utf-8")
    config = load_config(path)
    assert config.sync.primary_camera == "cam-a"


def test_load_config_default_plan_settings_are_empty(tmp_path: Path):
    config = load_config(tmp_path / "nope.yaml")
    assert config.plan.exclude_phrases == []
    assert config.plan.gain_db_by_beat == {}


def test_load_config_overrides_plan_exclude_phrases(tmp_path: Path):
    path = tmp_path / "config.yaml"
    path.write_text(
        "plan:\n  exclude_phrases: [clip-01#004, clip-02#009]\n", encoding="utf-8"
    )
    config = load_config(path)
    assert config.plan.exclude_phrases == ["clip-01#004", "clip-02#009"]


def test_load_config_overrides_plan_gain_db_by_beat(tmp_path: Path):
    path = tmp_path / "config.yaml"
    path.write_text("plan:\n  gain_db_by_beat:\n    Popping: -6\n", encoding="utf-8")
    config = load_config(path)
    assert config.plan.gain_db_by_beat == {"Popping": -6.0}


def test_plan_settings_are_immutable():
    from videoai.config import PlanSettings

    settings = PlanSettings()
    with pytest.raises(Exception):
        settings.exclude_phrases = ["x"]


def test_load_config_default_insert_threshold(tmp_path: Path):
    config = load_config(tmp_path / "nope.yaml")
    assert config.analyze.insert_max_words_per_second == 0.5


def test_load_config_overrides_insert_threshold(tmp_path: Path):
    path = tmp_path / "config.yaml"
    path.write_text("analyze:\n  insert_max_words_per_second: 0.2\n", encoding="utf-8")
    config = load_config(path)
    assert config.analyze.insert_max_words_per_second == 0.2


def test_load_config_default_output_snapshots_is_on(tmp_path: Path):
    config = load_config(tmp_path / "nope.yaml")
    assert config.output_snapshots is True


def test_load_config_overrides_output_snapshots(tmp_path: Path):
    path = tmp_path / "config.yaml"
    path.write_text("output_snapshots: false\n", encoding="utf-8")
    config = load_config(path)
    assert config.output_snapshots is False


# --------------------------------------------------------------------------- #
# free / paid: one codebase, one switch
# --------------------------------------------------------------------------- #


def test_a_free_run_keeps_every_stage_on_the_default_provider():
    config = Config(mode="free")

    assert config.llm_for("analyze") == "claude_cli"
    assert config.llm_for("describe") == "claude_cli"
    assert config.llm_for("plan") == "claude_cli"


def test_a_paid_run_moves_only_the_stages_worth_paying_for():
    """Watching the footage is worth a metered call. Naming a section title is
    not, so `plan` and `effects` stay where they are."""
    config = Config(mode="paid")

    assert config.llm_for("analyze") == "gemini_api"
    assert config.llm_for("describe") == "gemini_api"
    assert config.llm_for("plan") == "claude_cli"
    assert config.llm_for("effects") == "claude_cli"


def test_an_explicit_pin_beats_the_mode_in_both_directions():
    """A mode is a statement about the run; pinning a stage is a statement about
    that stage, and the more specific one wins."""
    cheap = Config(mode="paid", llm_by_stage={"analyze": "claude_cli"})
    assert cheap.llm_for("analyze") == "claude_cli"

    dear = Config(mode="free", llm_by_stage={"analyze": "gemini_api"})
    assert dear.llm_for("analyze") == "gemini_api"


def test_the_mode_decides_whether_describe_has_anything_that_can_watch():
    """Describing a clip means watching it. Whether the stage does anything is
    settled by the provider its mode resolves to, not by a second switch — the
    stage itself skips a provider that cannot watch, and refuses only one that
    was pinned deliberately."""
    enabled = DescribeSettings(enabled=True)

    assert Config(mode="free", describe=enabled).llm_for("describe") == "claude_cli"
    assert Config(mode="paid", describe=enabled).llm_for("describe") == "gemini_api"


def test_an_unknown_mode_is_refused_by_name():
    with pytest.raises(ValidationError, match="unknown mode: 'cheap'"):
        Config(mode="cheap")


def test_load_config_reads_the_mode_and_what_paid_means(tmp_path: Path):
    path = tmp_path / "config.yaml"
    path.write_text(
        "mode: paid\npaid:\n  llm: gemini_api\n  stages: [analyze]\n", encoding="utf-8"
    )

    config = load_config(path)

    assert config.mode == "paid"
    assert config.llm_for("analyze") == "gemini_api"
    # `describe` was not listed, so the mode leaves it alone.
    assert config.llm_for("describe") == "claude_cli"


def test_load_config_defaults_to_free(tmp_path: Path):
    """Constructing a config without saying must cost nothing."""
    assert load_config(tmp_path / "nope.yaml").mode == "free"
