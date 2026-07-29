from pathlib import Path

import pytest

from videoai.config import Config, load_config


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
