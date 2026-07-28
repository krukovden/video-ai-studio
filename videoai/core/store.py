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

        # Write artifact atomically: temp file + os.replace() ensures durability.
        # If a crash occurs before both files are in place, the fingerprint will be absent
        # (stale), not pointing at current content.
        with tempfile.NamedTemporaryFile(
            mode="w", dir=self.work_dir, delete=False, encoding="utf-8"
        ) as tmp:
            tmp.write(model.model_dump_json(indent=2))
            tmp_artifact = tmp.name
        os.replace(tmp_artifact, target)

        # Write metadata sidecar atomically after artifact is durably in place.
        with tempfile.NamedTemporaryFile(
            mode="w", dir=self.meta_dir, delete=False, encoding="utf-8"
        ) as tmp:
            json.dump({"fingerprint": fingerprint}, tmp)
            tmp_meta = tmp.name
        os.replace(tmp_meta, meta_target)

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
