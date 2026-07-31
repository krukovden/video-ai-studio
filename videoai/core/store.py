"""Artifact persistence: JSON files plus a sidecar fingerprint used for caching."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def hash_parts(*parts: str) -> str:
    """Short stable digest over ordered string parts."""
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()[:16]


def hash_file(path: Path) -> str:
    """Stable streaming digest for media and other non-text artifacts."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()[:16]


# A megabyte of each end of the file. Enough to cover a container's header and
# index and the first and last frames, which is where two different recordings
# differ, and small enough to read off a 4K source without noticing.
SAMPLE_BYTES = 1024 * 1024


def hash_file_ends(path: Path, sample: int = SAMPLE_BYTES) -> str:
    """Digest of a file's size plus its first and last `sample` bytes.

    `hash_file` is the honest answer and too slow to spend on every clip of every
    run: the first real project is 44 minutes of 4K HEVC, and reading all of it
    costs more than the ingest this is meant to make cacheable. Size plus both
    ends catches everything that actually happens to source footage — another
    take, a re-export, a re-encode, a truncated or still-copying transfer — since
    all of those move the size or rewrite the header. A file edited only in the
    middle and left byte-for-byte the same length would slip through; nothing
    between a camera and an NLE produces one.
    """
    size = path.stat().st_size
    digest = hashlib.sha256()
    digest.update(str(size).encode("utf-8"))
    with path.open("rb") as source:
        digest.update(source.read(sample))
        if size > sample:
            # Start the tail after the head so a file just over one sample long
            # does not have its overlap counted twice.
            source.seek(max(sample, size - sample))
            digest.update(source.read(sample))
    return digest.hexdigest()[:16]


def _project_relative_name(path: Path, root: Path | None) -> str:
    """Where a source file sits inside the project, as a key may see it.

    A caller that cannot name the project root gets the bare filename rather than
    the absolute path. It still tells two unrelated sources apart, and it does not
    put back the dependency on where the footage happens to be mounted today.
    """
    if root is not None:
        try:
            return path.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            pass
    return path.name


def source_key(path: Path, root: Path | None = None) -> str:
    """Short digest identifying a source file by its content and its place in the project.

    Derived media (audio, proxies, keyframes) and every metered model call are
    keyed by this rather than by a positional `clip-NN` id: clip ids are assigned
    by sort order, so adding a clip that sorts earlier renumbers everything and
    would otherwise hand one clip's cached audio and proxy to a different source.

    Content addressed rather than `mtime` addressed, and relative rather than
    absolute, because both of the alternatives call the same footage new. An
    rsync, a Time Machine restore or a cloud-sync round trip re-stamps every
    modification time without touching a frame, and moving the project folder
    changes every resolved path; either one would re-transcribe the whole shoot
    and re-upload it to a model billed by the second.
    """
    return hash_parts(_project_relative_name(path, root), hash_file_ends(path))


class ArtifactStore:
    def __init__(self, work_dir: Path) -> None:
        self.work_dir = work_dir
        self.meta_dir = work_dir / ".meta"
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.meta_dir.mkdir(parents=True, exist_ok=True)

    def path(self, name: str) -> Path:
        return self.work_dir / f"{name}.json"

    def _meta_path(self, name: str) -> Path:
        return self.meta_dir / f"{name}.json"

    def exists(self, name: str) -> bool:
        return self.path(name).exists()

    def write(self, name: str, model: BaseModel, fingerprint: str) -> Path:
        target = self.path(name)
        meta_target = self._meta_path(name)
        payload = model.model_dump_json(indent=2)

        # Write artifact atomically: temp file + os.replace() ensures durability.
        # If a crash occurs before both files are in place, the fingerprint will be absent
        # (stale), not pointing at current content.
        tmp_artifact = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", dir=self.work_dir, delete=False, encoding="utf-8"
            ) as tmp:
                tmp.write(payload)
                tmp_artifact = tmp.name
            os.replace(tmp_artifact, target)
        finally:
            if tmp_artifact is not None:
                Path(tmp_artifact).unlink(missing_ok=True)

        # Write metadata sidecar atomically after artifact is durably in place.
        tmp_meta = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", dir=self.meta_dir, delete=False, encoding="utf-8"
            ) as tmp:
                json.dump(
                    {"fingerprint": fingerprint, "content_hash": hash_parts(payload)}, tmp
                )
                tmp_meta = tmp.name
            os.replace(tmp_meta, meta_target)
        finally:
            if tmp_meta is not None:
                Path(tmp_meta).unlink(missing_ok=True)

        return target

    def read(self, name: str, model_cls: type[T]) -> T:
        target = self.path(name)
        if not target.exists():
            raise FileNotFoundError(f"artifact not found: {target}")
        return model_cls.model_validate_json(target.read_text(encoding="utf-8"))

    def fingerprint(self, name: str) -> str | None:
        meta = self._meta_path(name)
        if not meta.exists():
            return None
        return json.loads(meta.read_text(encoding="utf-8")).get("fingerprint")

    def content_hash(self, name: str) -> str | None:
        """Digest of the artifact as it currently sits on disk.

        Read from the file rather than from the sidecar written at `write` time,
        so that hand-editing an artifact — which this project's file-based
        workflow invites — invalidates everything downstream of it. The sidecar
        keeps the hash as of the last write, which is what tells a reader whether
        the file has been edited since.
        """
        target = self.path(name)
        if not target.exists():
            return None
        return hash_parts(target.read_text(encoding="utf-8"))

    def recorded_content_hash(self, name: str) -> str | None:
        """The content hash recorded when this artifact was last written by a stage."""
        meta = self._meta_path(name)
        if not meta.exists():
            return None
        return json.loads(meta.read_text(encoding="utf-8")).get("content_hash")
