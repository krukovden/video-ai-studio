from pathlib import Path

import pytest
from pydantic import BaseModel

from videoai.core.store import ArtifactStore, hash_parts


class Sample(BaseModel):
    name: str
    value: int


def test_write_then_read_roundtrip(tmp_path: Path):
    store = ArtifactStore(tmp_path)
    store.write("01-sample", Sample(name="a", value=1), fingerprint="fp1")
    loaded = store.read("01-sample", Sample)
    assert loaded.name == "a"
    assert loaded.value == 1


def test_artifact_is_human_readable_json(tmp_path: Path):
    store = ArtifactStore(tmp_path)
    store.write("01-sample", Sample(name="привет", value=2), fingerprint="fp1")
    text = (tmp_path / "01-sample.json").read_text(encoding="utf-8")
    assert "привет" in text
    assert "\n  " in text


def test_fingerprint_is_recorded_and_returned(tmp_path: Path):
    store = ArtifactStore(tmp_path)
    assert store.fingerprint("01-sample") is None
    store.write("01-sample", Sample(name="a", value=1), fingerprint="fp1")
    assert store.fingerprint("01-sample") == "fp1"


def test_exists_is_false_before_write(tmp_path: Path):
    store = ArtifactStore(tmp_path)
    assert store.exists("01-sample") is False
    store.write("01-sample", Sample(name="a", value=1), fingerprint="fp1")
    assert store.exists("01-sample") is True


def test_read_missing_artifact_raises(tmp_path: Path):
    store = ArtifactStore(tmp_path)
    with pytest.raises(FileNotFoundError):
        store.read("01-sample", Sample)


def test_hash_parts_is_stable_and_order_sensitive():
    assert hash_parts("a", "b") == hash_parts("a", "b")
    assert hash_parts("a", "b") != hash_parts("b", "a")
    assert len(hash_parts("a")) == 16
