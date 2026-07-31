import os
import shutil
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import BaseModel, ValidationError

from videoai.core.store import ArtifactStore, hash_file_ends, hash_parts, source_key


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


# --- derived media has to be cacheable by the source's identity, and neither
# `mtime` nor an absolute path is one: both call the same footage new ---


def test_source_key_survives_a_transfer_that_only_restamps_mtime(tmp_path: Path):
    """An rsync, a Time Machine restore and a cloud-sync round trip all put the
    same bytes back under a new modification time. Keying on that re-transcribed
    the shoot and re-uploaded it to a model billed by the second."""
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"frames" * 4096)
    before = source_key(path, tmp_path)

    os.utime(path, (1_600_000_000, 1_600_000_000))

    assert source_key(path, tmp_path) == before


def test_source_key_survives_moving_the_project_folder(tmp_path: Path):
    first_home = tmp_path / "Desktop" / "toy-review"
    (first_home / "video").mkdir(parents=True)
    (first_home / "video" / "a.mp4").write_bytes(b"frames" * 4096)
    before = source_key(first_home / "video" / "a.mp4", first_home)

    second_home = tmp_path / "Archive" / "2026" / "toy-review"
    second_home.parent.mkdir(parents=True)
    shutil.move(str(first_home), str(second_home))

    assert source_key(second_home / "video" / "a.mp4", second_home) == before


def test_source_key_notices_a_re_export_of_the_same_length(tmp_path: Path):
    """Same name, same size, same second: the take was re-exported. The old
    path/size/mtime key could not see this at all."""
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"take-one" + bytes(4096))
    before = source_key(path, tmp_path)

    path.write_bytes(b"take-two" + bytes(4096))

    assert source_key(path, tmp_path) != before


def test_two_cameras_may_hold_the_same_filename(tmp_path: Path):
    """Two cards from a two-camera shoot both start at C0001.MP4, and the second
    camera's clip is not the first camera's proxy."""
    project = tmp_path / "project"
    for camera in ("cam-a", "cam-b"):
        (project / "video" / camera).mkdir(parents=True)
        (project / "video" / camera / "C0001.MP4").write_bytes(b"identical bytes")

    keys = {
        source_key(project / "video" / camera / "C0001.MP4", project)
        for camera in ("cam-a", "cam-b")
    }

    assert len(keys) == 2


def test_hash_file_ends_notices_either_end_and_the_length(tmp_path: Path):
    path = tmp_path / "clip.mp4"
    body = bytearray(512)
    path.write_bytes(bytes(body))
    plain = hash_file_ends(path, sample=16)

    head = bytearray(body)
    head[0] = 1
    path.write_bytes(bytes(head))
    assert hash_file_ends(path, sample=16) != plain

    tail = bytearray(body)
    tail[-1] = 1
    path.write_bytes(bytes(tail))
    assert hash_file_ends(path, sample=16) != plain

    path.write_bytes(bytes(body) + b"\x00")
    assert hash_file_ends(path, sample=16) != plain


def test_hash_file_ends_reads_a_file_shorter_than_one_sample(tmp_path: Path):
    """The head and the tail overlap completely here; the digest must still be
    stable and still tell two short files apart."""
    short = tmp_path / "short.mp4"
    short.write_bytes(b"tiny")

    assert hash_file_ends(short, sample=16) == hash_file_ends(short, sample=16)

    other = tmp_path / "other.mp4"
    other.write_bytes(b"tin!")
    assert hash_file_ends(other, sample=16) != hash_file_ends(short, sample=16)


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
