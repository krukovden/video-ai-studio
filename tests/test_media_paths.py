"""Finding the media a manifest points at, from wherever the run happens.

The manifest records the path ingest was handed, which is relative whenever the
CLI was invoked with a relative project directory — so it is relative to that
run's working directory, not to anything the file itself knows. A later run from
somewhere else has to be able to find the same footage.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from videoai.cli import _media_fingerprint
from videoai.core.project import resolve_media_path


def test_an_absolute_path_is_used_as_recorded(tmp_path: Path):
    media = tmp_path / "clip.mov"
    media.write_bytes(b"x")
    assert resolve_media_path(tmp_path, str(media)) == media


def test_a_path_relative_to_the_project_is_found(tmp_path: Path):
    project = tmp_path / "project"
    (project / "video").mkdir(parents=True)
    media = project / "video" / "clip.mov"
    media.write_bytes(b"x")
    assert resolve_media_path(project, "video/clip.mov") == media


def test_a_path_relative_to_a_parent_of_the_project_is_found(tmp_path: Path):
    """The real case: ingest ran from the repository root, so the manifest holds
    'assets/<project>/video/clip.mov' — which resolves nowhere near the project
    folder itself."""
    root = tmp_path / "repo"
    project = root / "assets" / "toy"
    (project / "video").mkdir(parents=True)
    media = project / "video" / "clip.mov"
    media.write_bytes(b"x")
    assert resolve_media_path(project, "assets/toy/video/clip.mov") == media


def test_a_relative_path_still_resolves_after_the_project_folder_moves(tmp_path: Path):
    """Derived media is content addressed, so moving a project keeps every proxy
    and every cached model call. This is the other half of that: the manifest's
    own references have to survive the move too."""
    project = tmp_path / "Archive" / "toy"
    (project / "video").mkdir(parents=True)
    media = project / "video" / "clip.mov"
    media.write_bytes(b"x")

    assert resolve_media_path(project, "video/clip.mov") == media


def test_an_absolute_recorded_path_is_not_searched_for_when_it_moves(tmp_path: Path):
    """Pinned, not endorsed: ingest records the path it was handed, and only a
    relative one can be re-rooted onto the project. A manifest written from an
    absolute project path does not survive the folder moving, even though its
    derived media now does."""
    project = tmp_path / "moved"
    (project / "video").mkdir(parents=True)
    (project / "video" / "clip.mov").write_bytes(b"x")

    with pytest.raises(FileNotFoundError, match="clip.mov"):
        resolve_media_path(project, str(tmp_path / "before" / "video" / "clip.mov"))


def test_a_missing_file_names_everywhere_it_looked(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="gone.mov"):
        resolve_media_path(tmp_path, "video/gone.mov")


def _project_with_clip(root: Path, body: bytes = b"\x00" * 4096) -> Path:
    project = root / "shoot"
    (project / "video").mkdir(parents=True)
    (project / "video" / "clip.mov").write_bytes(body)
    return project


def test_the_media_fingerprint_survives_a_restamped_mtime(tmp_path: Path):
    """The value this guards is mixed into EVERY stage's fingerprint, so keying
    it on mtime made an rsync or a Time Machine restore re-run the whole
    pipeline — and re-upload the entire shoot to a metered video model — over
    footage that had not changed by a byte."""
    project = _project_with_clip(tmp_path)
    before = _media_fingerprint(project)

    os.utime(project / "video" / "clip.mov", (1_000_000, 1_000_000))

    assert _media_fingerprint(project) == before


def test_the_media_fingerprint_survives_the_project_moving(tmp_path: Path):
    project = _project_with_clip(tmp_path)
    before = _media_fingerprint(project)

    moved = tmp_path / "elsewhere"
    project.rename(moved)

    assert _media_fingerprint(moved) == before


def test_the_media_fingerprint_notices_replaced_footage(tmp_path: Path):
    project = _project_with_clip(tmp_path)
    before = _media_fingerprint(project)

    (project / "video" / "clip.mov").write_bytes(b"\x01" * 4096)

    assert _media_fingerprint(project) != before
