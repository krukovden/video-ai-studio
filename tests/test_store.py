from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import BaseModel, ValidationError

from videoai.core.store import ArtifactStore, hash_parts, source_key


class Sample(BaseModel):
    name: str
    value: int


class SampleWithTimestamp(BaseModel):
    name: str
    timestamp: float
    value: float


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


# --- Finding C3: the store has to expose artifact CONTENT so a stage can chain
# on what its inputs actually say, not on how they were fingerprinted ---


def test_content_hash_is_none_before_write(tmp_path: Path):
    assert ArtifactStore(tmp_path).content_hash("01-sample") is None


def test_content_hash_tracks_content_not_fingerprint(tmp_path: Path):
    store = ArtifactStore(tmp_path)
    store.write("01-sample", Sample(name="a", value=1), fingerprint="fp1")
    same_content = store.content_hash("01-sample")

    # Same content, different fingerprint: the content hash must not move.
    store.write("01-sample", Sample(name="a", value=1), fingerprint="fp2")
    assert store.content_hash("01-sample") == same_content

    # Different content, same fingerprint: the content hash must move.
    store.write("01-sample", Sample(name="a", value=2), fingerprint="fp2")
    assert store.content_hash("01-sample") != same_content


def test_content_hash_sees_a_hand_edited_artifact(tmp_path: Path):
    store = ArtifactStore(tmp_path)
    store.write("01-sample", Sample(name="a", value=1), fingerprint="fp1")
    written = store.content_hash("01-sample")

    store.path("01-sample").write_text(
        Sample(name="a", value=99).model_dump_json(indent=2), encoding="utf-8"
    )

    assert store.content_hash("01-sample") != written
    # The sidecar still reports the hash as of the last write, which is what makes
    # the hand edit detectable.
    assert store.recorded_content_hash("01-sample") == written


def test_source_key_changes_with_size_and_stays_stable_otherwise(tmp_path: Path):
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"aaa")
    first = source_key(path)

    assert source_key(path) == first
    path.write_bytes(b"aaaa")
    assert source_key(path) != first
    assert source_key(tmp_path / "clip.mp4") != source_key(_copy(path, tmp_path / "other.mp4"))


def _copy(source: Path, target: Path) -> Path:
    target.write_bytes(source.read_bytes())
    return target


def test_float_round_trip_fidelity(tmp_path: Path):
    """Verify that float fields maintain precision across serialization."""
    store = ArtifactStore(tmp_path)
    original = SampleWithTimestamp(
        name="test", timestamp=1234567890.123456, value=0.1
    )
    store.write("float-sample", original, fingerprint="fp1")
    loaded = store.read("float-sample", SampleWithTimestamp)
    assert loaded.timestamp == original.timestamp
    assert loaded.value == original.value
    assert loaded.name == original.name


def test_crash_between_writes_leaves_no_stale_fingerprint(tmp_path: Path):
    """Verify that fingerprint is absent if write is interrupted between artifact and meta.

    Simulates the scenario: if a crash occurs after artifact is written but before
    the meta sidecar, the fingerprint should be absent (not stale).
    """
    store = ArtifactStore(tmp_path)

    # First, write successfully to establish a baseline.
    store.write("01-sample", Sample(name="a", value=1), fingerprint="fp1")
    assert store.fingerprint("01-sample") == "fp1"

    # Now simulate a crash: write the artifact but fail before writing meta.
    # Use monkeypatch to break os.replace for meta writes only.
    original_replace = __import__("os").replace
    replace_count = [0]

    def patched_replace(src, dst):
        replace_count[0] += 1
        # Allow the first replace (artifact), but fail on the second (meta).
        if replace_count[0] == 2:
            raise OSError("Simulated crash during meta write")
        return original_replace(src, dst)

    with patch("os.replace", side_effect=patched_replace):
        with pytest.raises(OSError, match="Simulated crash"):
            store.write("01-sample", Sample(name="b", value=2), fingerprint="fp2")

    # After the crash, the artifact file exists but fingerprint should still be "fp1"
    # (the old one), proving we didn't write an orphaned fp2 that doesn't match.
    assert store.fingerprint("01-sample") == "fp1"


def test_read_with_mismatched_model_raises_validation_error(tmp_path: Path):
    """Verify that reading JSON that doesn't match the model raises ValidationError."""
    store = ArtifactStore(tmp_path)
    store.write("01-sample", Sample(name="a", value=1), fingerprint="fp1")

    # Try to read it as a different model that expects a float field.
    with pytest.raises(ValidationError):
        store.read("01-sample", SampleWithTimestamp)


def test_no_orphaned_temp_files_on_replace_failure(tmp_path: Path):
    """Verify that temp files are cleaned up if os.replace() fails.

    Simulates a failure during os.replace (e.g., disk full, permissions error) and
    verifies that the temp file is unlinked and does not accumulate in the directory.
    """
    store = ArtifactStore(tmp_path)
    original_replace = __import__("os").replace

    # First write succeeds.
    store.write("01-sample", Sample(name="a", value=1), fingerprint="fp1")

    # Now make os.replace fail on the meta write (second call).
    replace_count = [0]

    def failing_replace(src, dst):
        replace_count[0] += 1
        # Allow the first replace (artifact), fail on the second (meta).
        if replace_count[0] == 2:
            raise OSError("Simulated disk full during meta replace")
        return original_replace(src, dst)

    with patch("os.replace", side_effect=failing_replace):
        with pytest.raises(OSError, match="Simulated disk full"):
            store.write("01-sample", Sample(name="b", value=2), fingerprint="fp2")

    # Verify no orphaned temp files in work_dir.
    tmp_files_work = list(tmp_path.glob("tmp*"))
    assert len(tmp_files_work) == 0, f"Orphaned temp files in work_dir: {tmp_files_work}"

    # Verify no orphaned temp files in meta_dir.
    tmp_files_meta = list((tmp_path / ".meta").glob("tmp*"))
    assert len(tmp_files_meta) == 0, f"Orphaned temp files in .meta: {tmp_files_meta}"
