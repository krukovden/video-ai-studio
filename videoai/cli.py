"""VideoAI command line interface."""
from __future__ import annotations

from pathlib import Path

import typer

import videoai.stages  # noqa: F401  (imports register every stage)
from videoai.config import load_config
from videoai.core.ffmpeg import VIDEO_SUFFIXES
from videoai.core.registry import StageContext
from videoai.core.runner import ordered_stages, run_pipeline
from videoai.core.store import ArtifactStore, hash_parts

app = typer.Typer(add_completion=False, help="Automated video pipeline.")


def _source_fingerprint(input_dir: Path) -> str:
    parts: list[str] = []
    for path in sorted(input_dir.rglob("*")):
        if path.is_file() and (path.suffix.lower() in VIDEO_SUFFIXES or path.name in {"project.yaml", "notes.md"}):
            stat = path.stat()
            parts.append(f"{path.name}:{stat.st_size}:{int(stat.st_mtime)}")
    return hash_parts(*parts)


@app.command()
def run(
    project: Path = typer.Argument(..., help="Project directory containing input/"),
    config_path: Path = typer.Option(Path("config.yaml"), "--config", help="Config file"),
    stage_id: str | None = typer.Option(None, "--stage", help="Run a single stage by id"),
    force: bool = typer.Option(False, "--force", help="Ignore the cache and re-run"),
) -> None:
    """Run the pipeline over a project folder."""
    input_dir = project / "input"
    if not input_dir.is_dir():
        raise typer.BadParameter(f"no input directory: {input_dir}")

    work_dir = project / "work"
    output_dir = project / "output"
    work_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    ctx = StageContext(
        project_dir=project,
        input_dir=input_dir,
        work_dir=work_dir,
        output_dir=output_dir,
        config=load_config(config_path),
        store=ArtifactStore(work_dir),
    )

    executed = run_pipeline(
        ctx, only=stage_id, force=force, extra_fingerprint=_source_fingerprint(input_dir)
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
