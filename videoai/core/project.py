"""Project folder conventions.

The creator's folders came first, so the pipeline adapts to them: clips may sit
in `input/` or `video/`, and the brief is whatever prose lives in `description/`.
"""
from __future__ import annotations

import os
import shutil
from datetime import datetime
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


def snapshot_output(output_dir: Path, now: datetime | None = None) -> Path | None:
    """Archive output/'s current deliverables into a timestamped subfolder.

    Only regular files sitting directly in `output_dir` are snapshotted — existing
    timestamped folders (and any other subdirectory) are never touched or
    recursed into, so history never becomes part of itself. Returns None (and
    creates nothing) when `output_dir` holds no files, since a "nothing to do"
    run must leave no trace.

    Linked with `os.link` so an unchanged file costs no extra disk on a
    copy-on-write filesystem (APFS); a plain copy is the fallback when linking
    fails, e.g. the two paths are on different devices.
    """
    files = sorted(path for path in output_dir.iterdir() if path.is_file())
    if not files:
        return None

    stamp = (now or datetime.now()).strftime("%d_%b_%H_%M")
    target = output_dir / stamp
    suffix = 2
    while target.exists():
        target = output_dir / f"{stamp}_{suffix}"
        suffix += 1
    target.mkdir(parents=True)

    for source in files:
        dest = target / source.name
        try:
            os.link(source, dest)
        except OSError:
            shutil.copy2(source, dest)
    return target


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


def resolve_media_path(project_dir: Path, raw_path: str) -> Path:
    """The file a manifest entry points at, found from wherever this run happens.

    A manifest records the path ingest was handed. When the CLI was invoked with a
    relative project directory — the normal case — that path is relative to *that
    run's* working directory, which nothing in the file records. So a later run
    from a different directory (a worktree, a scheduled job, another checkout)
    cannot simply trust it.

    Tried in order: as recorded, under the project folder, and then under each
    parent of the project folder. The last of those is what actually finds
    `assets/<project>/video/clip.mov` when ingest ran from the repository root.

    A miss raises and names every place it looked. Falling back to "no footage"
    is how a run silently produces a worse result than the one that was asked
    for.
    """
    recorded = Path(raw_path)
    if recorded.is_absolute():
        candidates = [recorded]
    else:
        candidates = [recorded, project_dir / recorded]
        candidates += [parent / recorded for parent in project_dir.resolve().parents]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    looked = ", ".join(str(candidate) for candidate in candidates[:4])
    raise FileNotFoundError(f"media not found: {raw_path} (looked in {looked}, ...)")
