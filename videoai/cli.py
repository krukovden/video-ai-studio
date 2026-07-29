"""VideoAI command line interface."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import typer

import videoai.stages  # noqa: F401  (imports register every stage)
from videoai.config import load_config
from videoai.core.models import Approval, FinalResult
from videoai.core.project import BRIEF_SUFFIXES, list_camera_clips, resolve_clip_dir
from videoai.core.registry import StageContext
from videoai.core.runner import StageFailure, ordered_stages, run_pipeline, stale_downstream
from videoai.core.store import ArtifactStore, hash_file, hash_parts

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
    debug: bool = typer.Option(False, "--debug", help="Re-raise stage failures with their traceback"),
    auto_fix: int = typer.Option(
        0, "--auto-fix", min=0,
        help="When the visual check rejects segments, re-plan without them and check "
             "again, at most this many times",
    ),
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

    executed: list[str] = []
    rounds = 0
    while True:
        try:
            executed = run_pipeline(
                ctx,
                only=stage_id,
                force=force,
                media_fingerprint=media_fingerprint,
                brief_fingerprint=brief_fingerprint,
            )
            break
        except StageFailure as failure:
            # Only a full run can self-correct: the fix is a fresh plan, and a
            # `--stage visual_check` run would re-check the same timeline forever.
            # `rounds` is capped by `--auto-fix` so the loop always terminates.
            if failure.stage_id == "visual_check" and stage_id is None and rounds < auto_fix:
                rounds += 1
                typer.echo(f"Auto-fix round {rounds} of {auto_fix}: {failure.cause}")
                typer.echo("Re-planning without the rejected segments.")
                continue
            typer.echo(f"Stage '{failure.stage_id}' failed: {failure.cause}", err=True)
            if rounds:
                typer.echo(
                    f"Auto-fix gave up after {rounds} re-planning round(s); every "
                    "rejected segment is listed above and in work/05c-rejected.json.",
                    err=True,
                )
            typer.echo(
                "Artifacts from earlier stages are kept, so fix the cause and re-run just "
                f"this stage:\n  videoai run {project} --config {config_path} "
                f"--stage {failure.stage_id}",
                err=True,
            )
            if debug:
                raise
            typer.echo("Run again with --debug for the full traceback.", err=True)
            raise typer.Exit(1) from failure
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
def approve(
    project: Path = typer.Argument(..., help="Project whose current draft was reviewed"),
    config_path: Path = typer.Option(Path("config.yaml"), "--config", help="Config file"),
) -> None:
    """Approve the current timeline for delivery rendering."""
    store = ArtifactStore(project / "work")
    timeline_hash = store.content_hash("05-timeline")
    if timeline_hash is None or not store.exists("06-draft"):
        raise typer.BadParameter(
            "the project has no current timeline and draft; run the pipeline and review "
            "output/draft.mp4 first"
        )
    from videoai.core.models import DraftResult

    draft = store.read("06-draft", DraftResult)
    if draft.timeline_hash != timeline_hash:
        raise typer.BadParameter(
            "draft.mp4 is stale: it was not rendered from the current timeline; "
            "run the pipeline through render_draft and review the new file"
        )
    draft_hash = hash_file(Path(draft.path)) if Path(draft.path).is_file() else ""
    if not draft_hash:
        raise typer.BadParameter(f"draft file is missing: {draft.path}")
    config_hash = hash_parts(load_config(config_path).model_dump_json())
    store.write(
        "06-approval",
        Approval(
            timeline_hash=timeline_hash,
            draft_hash=draft_hash,
            config_hash=config_hash,
            approved_at=datetime.now(timezone.utc).isoformat(),
        ),
        fingerprint="manual-approval",
    )
    typer.echo(
        "Approved the current timeline, draft, and effective config. Any change "
        "to one of them invalidates this approval."
    )


@app.command()
def produce(
    project: Path = typer.Argument(..., help="Project to take through the production contract"),
    config_path: Path = typer.Option(Path("config.yaml"), "--config", help="Config file"),
) -> None:
    """Build a review draft, then a contract-validated final after approval."""
    clip_dir = resolve_clip_dir(project)
    cameras = list_camera_clips(clip_dir)
    if not any(cameras.values()):
        raise typer.BadParameter(f"no video files found in {clip_dir}")
    work_dir = project / "work"
    output_dir = project / "output"
    work_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    loaded = load_config(config_path)
    if not loaded.polish.strict_contract:
        raise typer.BadParameter(
            "production requires polish.strict_contract: true in the selected config"
        )
    ctx = StageContext(
        project_dir=project,
        input_dir=project,
        work_dir=work_dir,
        output_dir=output_dir,
        config=loaded,
        store=ArtifactStore(work_dir),
    )
    media_fingerprint = _media_fingerprint(project)
    brief_fingerprint = _brief_fingerprint(project)
    try:
        executed = run_pipeline(
            ctx,
            stop_after="render_draft",
            media_fingerprint=media_fingerprint,
            brief_fingerprint=brief_fingerprint,
        )
    except StageFailure as failure:
        raise typer.BadParameter(
            f"production stopped at {failure.stage_id}: {failure.cause}"
        ) from failure
    if executed:
        typer.echo("Prepared review draft: " + ", ".join(executed))

    from videoai.core.models import DraftResult
    from videoai.stages.s08_polish import _approval_is_current

    draft = ctx.store.read("06-draft", DraftResult)
    try:
        _approval_is_current(ctx, draft)
    except RuntimeError as error:
        typer.echo(str(error))
        typer.echo(f"Review: {draft.path}")
        typer.echo(
            f"Then approve: videoai approve {project} --config {config_path}"
        )
        return

    try:
        executed = run_pipeline(
            ctx,
            media_fingerprint=media_fingerprint,
            brief_fingerprint=brief_fingerprint,
        )
    except StageFailure as failure:
        raise typer.BadParameter(
            f"production stopped at {failure.stage_id}: {failure.cause}"
        ) from failure
    result = ctx.store.read("08-final", FinalResult)
    if not result.fully_decoded or not result.production_report:
        raise typer.BadParameter("final artifact did not pass the production contract")
    typer.echo("Production passed: " + result.path)
    typer.echo("Report: " + result.production_report)


@app.command()
def config(path: Path = typer.Option(Path("config.yaml"), help="Config file path")) -> None:
    """Print the effective configuration."""
    typer.echo(load_config(path).model_dump_json(indent=2))


if __name__ == "__main__":
    app()
