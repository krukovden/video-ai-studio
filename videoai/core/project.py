"""Project folder conventions.

The creator's folders came first, so the pipeline adapts to them: clips may sit
in `input/` or `video/`, and the brief is whatever prose lives in `description/`.
"""
from __future__ import annotations

from pathlib import Path

BRIEF_SUFFIXES = {".md", ".txt", ".docx"}

# Folders the pipeline itself creates. A flat project (clips directly in the
# project folder) would otherwise ingest its own `output/draft.mp4` on the second
# run and treat `work/`'s proxies and segments as extra cameras.
DERIVED_DIRS = {"work", "output"}

LOOSE_CAMERA = "main"


def resolve_clip_dir(project_dir: Path) -> Path:
    for name in ("input", "video"):
        candidate = project_dir / name
        if candidate.is_dir():
            return candidate
    return project_dir


def list_camera_clips(clip_dir: Path) -> dict[str, list[Path]]:
    """Camera name to its clips.

    Subdirectories are cameras, which is how a two-camera shoot is handed over;
    files sitting loose in `clip_dir` are a single camera called `main`. Both are
    merged: a folder holding both loose clips and a camera subfolder is a real
    layout, and silently dropping either half loses footage. `work/` and
    `output/` are the pipeline's own folders and are never cameras.
    """
    from videoai.core.ffmpeg import list_video_files

    cameras: dict[str, list[Path]] = {}
    loose = list_video_files(clip_dir)
    if loose:
        cameras[LOOSE_CAMERA] = loose

    for entry in sorted(clip_dir.iterdir()) if clip_dir.is_dir() else []:
        if not entry.is_dir() or entry.name.startswith(".") or entry.name in DERIVED_DIRS:
            continue
        files = list_video_files(entry)
        if not files:
            continue
        # A subfolder literally called `main` alongside loose clips: merge rather
        # than let one shadow the other.
        cameras[entry.name] = sorted(cameras.get(entry.name, []) + files)
    return cameras


def _read_docx(path: Path) -> str:
    import docx

    document = docx.Document(str(path))
    return "\n".join(paragraph.text for paragraph in document.paragraphs)


def _read_one(path: Path) -> str:
    if path.suffix.lower() == ".docx":
        try:
            return _read_docx(path)
        except Exception:
            return f"[could not read {path.name}]"
    try:
        return path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return f"[could not read {path.name}]"


def read_brief(project_dir: Path) -> str:
    """Everything the creator wrote about this video, concatenated."""
    parts: list[str] = []
    for name in ("project.yaml", "notes.md"):
        path = project_dir / name
        if path.is_file():
            parts.append(_read_one(path))

    description = project_dir / "description"
    if description.is_dir():
        for path in sorted(description.iterdir()):
            if (
                path.is_file()
                and not path.name.startswith(".")
                and path.suffix.lower() in BRIEF_SUFFIXES
            ):
                parts.append(_read_one(path))

    return "\n\n".join(part.strip() for part in parts if part.strip())
