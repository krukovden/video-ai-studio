"""VideoAI command line interface."""
from __future__ import annotations

from pathlib import Path

import typer

import videoai.stages  # noqa: F401  (imports register every stage)
from videoai.config import load_config
from videoai.core.project import BRIEF_SUFFIXES, list_camera_clips, resolve_clip_dir
from videoai.core.registry import StageContext
from videoai.core.runner import ordered_stages, run_pipeline
from videoai.core.store import ArtifactStore, hash_parts

app = typer.Typer(add_completion=False, help="Automated video pipeline.")


def _source_fingerprint(project_dir: Path) -> str:
    """Fingerprint everything that feeds the pipeline: the clips ingest will read,
    plus the brief (project.yaml, notes.md, description/*) that analysis and plan
    prompt on. A brief edit must invalidate the cache just like a new clip does."""
    parts: list[str] = []

    clip_dir = resolve_clip_dir(project_dir)
    cameras = list_camera_clips(clip_dir)
    for camera in sorted(cameras):
        for path in cameras[camera]:
            if path.is_file():
                stat = path.stat()
                key = path.relative_to(project_dir) if path.is_relative_to(project_dir) else path
                parts.append(f"{key}:{stat.st_size}:{int(stat.st_mtime)}")

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

    executed = run_pipeline(
        ctx, only=stage_id, force=force, extra_fingerprint=_source_fingerprint(project)
    )
    if executed:
        typer.echo("Executed: " + ", ".join(executed))
    else:
        typer.echo("Nothing to do — every stage is up to date.")


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
