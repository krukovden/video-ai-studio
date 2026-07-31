"""Project folder conventions.

The creator's folders came first, so the pipeline adapts to them: clips may sit
in `input/` or `video/`, and the brief is whatever prose lives in `description/`.
"""
from __future__ import annotations

import os
import shutil
from datetime import datetime
from pathlib import Path

from videoai.core.models import Brief

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
    """One file of the brief, or an error naming it.

    This used to answer `[could not read x.docx]` and hand that string to the
    model as though it were part of what the creator wrote. A brief that cannot
    be read is not a shorter brief: the arc, the must-include list and the
    description of the toy are all silently gone, and every stage downstream
    plans as if the creator had asked for nothing.
    """
    try:
        if path.suffix.lower() == ".docx":
            return _read_docx(path)
        return path.read_text(encoding="utf-8")
    except Exception as error:
        raise RuntimeError(
            f"cannot read the brief file {path}: {error}. Fix or remove it — "
            "planning from a brief with a hole in it is worse than not planning."
        ) from error


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


# Which keys in `project.yaml` fill which field of the brief. Spelled as a list
# because the documented example says `toy_name` and a creator reviewing
# something that is not a toy will reasonably write `product` or `subject`; a key
# nobody parses is a key nobody honours, and the failure is silent.
BRIEF_TEXT_KEYS: dict[str, tuple[str, ...]] = {
    "title": ("title",),
    "subject": ("subject", "toy_name", "toy", "product", "product_name"),
    "subject_description": (
        "subject_description", "toy_description", "product_description",
        "about_the_toy",
    ),
    "summary": ("summary", "video_description", "about", "description"),
    "style": ("style",),
    "language": ("language", "lang"),
    "notes": ("notes",),
}

BRIEF_LIST_KEYS: dict[str, tuple[str, ...]] = {
    "arc": ("arc", "beats", "structure"),
    "must_include": ("must_include", "must_have"),
    "avoid": ("avoid", "exclude"),
}

DURATION_KEYS = ("target_duration_seconds", "target_duration", "duration_seconds")


def _brief_text(document: dict, names: tuple[str, ...]) -> str:
    for name in names:
        value = document.get(name)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _brief_list(document: dict, names: tuple[str, ...]) -> list[str]:
    """A brief's list field, however the creator happened to write it.

    A YAML list is the documented spelling; a block of lines is what somebody
    types when they are in a hurry, and refusing that would be pedantry about
    punctuation rather than about meaning.
    """
    for name in names:
        value = document.get(name)
        if value is None:
            continue
        if isinstance(value, str):
            items = [line.strip().lstrip("-*• ").strip() for line in value.splitlines()]
        elif isinstance(value, (list, tuple)):
            items = [str(item).strip() for item in value]
        else:
            items = [str(value).strip()]
        items = [item for item in items if item]
        if items:
            return items
    return []


def _brief_seconds(document: dict) -> float:
    text = _brief_text(document, DURATION_KEYS)
    if not text:
        return 0.0
    try:
        return float(text)
    except ValueError:
        raise RuntimeError(
            f"project.yaml gives a target duration of {text!r}, which is not a "
            "number of seconds. Write e.g. target_duration_seconds: 180."
        ) from None


def _project_yaml(project_dir: Path) -> dict:
    """`project.yaml` as a mapping, or empty when there is nothing to parse.

    Free prose in `project.yaml` loads as a plain string rather than a mapping.
    That is not an error — it has no fields to find and still reaches the model
    through `Brief.raw` — but YAML that does not parse at all is, because the
    creator wrote keys believing something would read them.
    """
    import yaml

    path = project_dir / "project.yaml"
    if not path.is_file():
        return {}
    try:
        document = yaml.safe_load(_read_one(path))
    except yaml.YAMLError as error:
        raise RuntimeError(
            f"cannot parse {path}: {error}. The arc, must_include and avoid lists "
            "in it are read by the pipeline, so a file it cannot parse is a brief "
            "nothing will honour."
        ) from error
    return document if isinstance(document, dict) else {}


def load_brief(project_dir: Path) -> Brief:
    """The creator's brief as fields, with the whole text kept alongside.

    Only `project.yaml` is parsed. `notes.md` and `description/` are prose a
    person wrote for a person — there are no keys in them to find — so they reach
    the model through `raw`, exactly as they always did.
    """
    document = _project_yaml(project_dir)
    fields = {
        name: _brief_text(document, names) for name, names in BRIEF_TEXT_KEYS.items()
    }
    fields.update(
        {name: _brief_list(document, names) for name, names in BRIEF_LIST_KEYS.items()}
    )
    return Brief(
        **fields,
        target_duration_seconds=_brief_seconds(document),
        raw=read_brief(project_dir),
    )


def brief_prompt(brief: Brief) -> str:
    """The brief as a model should read it: what was parsed, then what was written.

    The parsed fields are repeated above the raw text on purpose. They are the
    part the pipeline can hold a plan to — an arc it can name sections after, a
    must-include list it can report a miss against — and a model cannot tell
    which half of a pasted YAML file that is. The raw text stays underneath so a
    key this schema never anticipated is still in front of the model.
    """
    if brief.is_empty():
        return ""
    lines: list[str] = []
    if brief.subject:
        lines.append(f"Subject: {brief.subject}")
    if brief.subject_description:
        lines.append(f"About the subject: {brief.subject_description}")
    if brief.summary:
        lines.append(f"This video: {brief.summary}")
    if brief.style:
        lines.append(f"Style: {brief.style}")
    if brief.target_duration_seconds:
        lines.append(f"Target duration: {brief.target_duration_seconds:.0f}s")
    for label, items in (
        ("The arc the creator shot, in order", brief.arc),
        ("Moments that must survive the edit", brief.must_include),
        ("Never let this reach the video", brief.avoid),
    ):
        if items:
            lines.append(label + ":\n" + "\n".join(f"  - {item}" for item in items))
    if not lines:
        return brief.raw
    return "\n".join(lines) + "\n\nThe brief as written:\n" + brief.raw


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
