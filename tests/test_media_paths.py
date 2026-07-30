"""Finding the media a manifest points at, from wherever the run happens.

The manifest records the path ingest was handed, which is relative whenever the
CLI was invoked with a relative project directory — so it is relative to that
run's working directory, not to anything the file itself knows. A later run from
somewhere else has to be able to find the same footage.
"""
from __future__ import annotations

from pathlib import Path

import pytest

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


def test_a_missing_file_names_everywhere_it_looked(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="gone.mov"):
        resolve_media_path(tmp_path, "video/gone.mov")
