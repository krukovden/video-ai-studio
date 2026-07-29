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


def source_key(path: Path) -> str:
    """Short digest identifying a source file by path, size and mtime.

    Derived media (audio, proxies, keyframes) must be keyed by this rather than
    by a positional `clip-NN` id: clip ids are assigned by sort order, so adding
    a clip that sorts earlier renumbers everything and would otherwise hand one
    clip's cached audio and proxy to a different source.
    """
    stat = path.stat()
    return hash_parts(str(path.resolve()), str(stat.st_size), str(int(stat.st_mtime)))


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
