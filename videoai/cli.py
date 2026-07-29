"""VideoAI command line interface."""
from __future__ import annotations

from pathlib import Path

import typer

import videoai.stages  # noqa: F401  (imports register every stage)
from videoai.config import load_config
from videoai.core.project import BRIEF_SUFFIXES, list_camera_clips, resolve_clip_dir
from videoai.core.registry import StageContext
from videoai.core.runner import ordered_stages, run_pipeline, stale_downstream
from videoai.core.store import ArtifactStore, hash_parts

app = typer.Typer(add_completion=False, help="Automated video pipeline.")


def _media_fingerprint(project_dir: Path) -> str:
    """Fingerprint every clip ingest will read, including per-camera subfolders."""
    parts: list[str] = []
    clip_dir = resolve_clip_dir(project_dir)
    cameras = list_camera_clips(clip_dir)
    for camera in sorted(cameras):
        for path in cameras[camera]:
            if path.is_file():
                stat = path.stat()
                key = path.relative_to(project_dir) if path.is_relative_to(project_dir) else path
                parts.append(f"{key}:{stat.st_size}:{int(stat.st_mtime)}")
    return hash_parts(*parts)


def _brief_fingerprint(project_dir: Path) -> str:
    """Fingerprint the creator's brief: project.yaml, notes.md, and description/*.
    Kept separate from the media fingerprint so an edit here only invalidates the
    stages that actually read the brief (analyze, plan), not the whole pipeline."""
    parts: list[str] = []

    for name in ("project.yaml", "notes.md"):
        path = project_dir / name
        if path.is_file():
            stat = path.stat()
            parts.append(f"{name}:{stat.st_size}:{int(stat.st_mtime)}")

    description = project_dir / "description"
    if description.is_dir():
        for path in sorted(description.iterdir()):
            if path.is_file() and not path.name.startswith(".") and path.suffix.lower() in BRIEF_SUFFIXES:
                stat = path.stat()
                parts.append(f"description/{path.name}:{stat.st_size}:{int(stat.st_mtime)}")

    return hash_parts(*parts)


@app.command()
def run(
    project: Path = typer.Argument(..., help="Project directory holding input/, video/, or a flat folder of clips"),
    config_path: Path = typer.Option(Path("config.yaml"), "--config", help="Config file"),
    stage_id: str | None = typer.Option(None, "--stage", help="Run a single stage by id"),
    force: bool = typer.Option(False, "--force", help="Ignore the cache and re-run"),
) -> None:
    """Run the pipeline over a project folder."""
    clip_dir = resolve_clip_dir(project)
    cameras = list_camera_clips(clip_dir)
    if not any(sources for sources in cameras.values()):
        raise typer.BadParameter(
            f"no video files found in {clip_dir} "
            "(clips may live in input/, in video/, in per-camera subfolders of either, "
            "or directly in the project folder)"
        )

    work_dir = project / "work"
    output_dir = project / "output"
    work_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    ctx = StageContext(
        project_dir=project,
        input_dir=project,
        work_dir=work_dir,
        output_dir=output_dir,
        config=load_config(config_path),
        store=ArtifactStore(work_dir),
    )

    media_fingerprint = _media_fingerprint(project)
    brief_fingerprint = _brief_fingerprint(project)

    executed = run_pipeline(
        ctx,
        only=stage_id,
        force=force,
        media_fingerprint=media_fingerprint,
        brief_fingerprint=brief_fingerprint,
    )
    if executed:
        typer.echo("Executed: " + ", ".join(executed))
    else:
        typer.echo("Nothing to do — every stage is up to date.")

    if stage_id is not None:
        stale = stale_downstream(ctx, stage_id, media_fingerprint, brief_fingerprint)
        if stale:
            typer.echo(
                "Note: " + ", ".join(stale) + " now depend on stale input from this "
                "single-stage run — re-run without --stage to bring them up to date."
            )


@app.command()
def stages() -> None:
    """List pipeline stages in execution order."""
    for spec in ordered_stages():
        requires = ", ".join(spec.requires) or "-"
        typer.echo(f"{spec.id:<14} produces={spec.produces:<16} requires={requires}")


@app.command()
def config(path: Path = typer.Option(Path("config.yaml"), help="Config file path")) -> None:
    """Print the effective configuration."""
    typer.echo(load_config(path).model_dump_json(indent=2))


if __name__ == "__main__":
    app()
