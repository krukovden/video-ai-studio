"""Project folder conventions.

The creator's folders came first, so the pipeline adapts to them: clips may sit
in `input/` or `video/`, and the brief is whatever prose lives in `description/`.
"""
from __future__ import annotations

from pathlib import Path

BRIEF_SUFFIXES = {".md", ".txt", ".docx"}


def resolve_clip_dir(project_dir: Path) -> Path:
    for name in ("input", "video"):
        candidate = project_dir / name
        if candidate.is_dir():
            return candidate
    return project_dir


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
