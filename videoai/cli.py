"""VideoAI command line interface."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import typer

import videoai.stages  # noqa: F401  (imports register every stage)
from videoai.config import load_config
from videoai.core.models import Approval, DraftResult, FinalResult
from videoai.core.project import (
    BRIEF_SUFFIXES,
    list_camera_clips,
    resolve_clip_dir,
    resolve_media_path,
    snapshot_output,
)
from videoai.core.registry import StageContext
from videoai.core.runner import StageFailure, ordered_stages, run_pipeline, stale_downstream
from videoai.core.store import ArtifactStore, hash_file, hash_parts
from videoai.stages.s08_polish import approval_is_current

app = typer.Typer(add_completion=False, help="Automated video pipeline.")


def _report_stage_failure(
    failure: StageFailure,
    project: Path,
    config_path: Path,
    debug: bool,
    note: str = "",
) -> None:
    """Name the stage, hand over the exact re-run command, and offer a traceback.

    Every command that runs the pipeline reports a failure the same way: a stage
    id, a cause, the one command that resumes from there, and how to get the
    traceback. A creator should never have to know which command they used to
    learn what to do next.
    """
    typer.echo(f"Stage '{failure.stage_id}' failed: {failure.cause}", err=True)
    if note:
        typer.echo(note, err=True)
    typer.echo(
        "Artifacts from earlier stages are kept, so fix the cause and re-run just "
        f"this stage:\n  videoai run {project} --config {config_path} "
        f"--stage {failure.stage_id}",
        err=True,
    )
    if debug:
        raise failure
    typer.echo("Run again with --debug for the full traceback.", err=True)
    raise typer.Exit(1) from failure


def _snapshot_if_executed(ctx: StageContext, project: Path, executed: bool) -> None:
    """Archive output/ into a timestamped subfolder after a run that changed
    something. A "nothing to do" run (executed is falsy) must leave no trace,
    which is why the caller decides whether anything ran, not this function."""
    if not executed or not ctx.config.output_snapshots:
        return
    snapshot_dir = snapshot_output(ctx.output_dir)
    if snapshot_dir is None:
        return
    count = sum(1 for path in snapshot_dir.iterdir() if path.is_file())
    typer.echo(f"Snapshot: {snapshot_dir.relative_to(project)}/ ({count} files)")


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
            _report_stage_failure(
                failure, project, config_path, debug,
                note=(
                    f"Auto-fix gave up after {rounds} re-planning round(s); every "
                    "rejected segment is listed above and in work/05c-rejected.json."
                    if rounds
                    else ""
                ),
            )
    if executed:
        typer.echo("Executed: " + ", ".join(executed))
    else:
        typer.echo("Nothing to do — every stage is up to date.")
    _snapshot_if_executed(ctx, project, bool(executed))

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


@app.command("seed-effects")
def seed_effects(
    directory: Path = typer.Option(
        None, "--dir", help="Where to write the library (default: the repo's assets/effects)"
    ),
) -> None:
    """Redraw the built-in cartoon sprite library and its manifest.

    The library is committed, so this is only needed after editing a drawing in
    `videoai/logic/effect_seeds.py`. It makes no network calls.
    """
    from videoai.logic.effect_seeds import seed_library
    from videoai.logic.effects import default_library_dir, load_library

    target = directory or default_library_dir()
    manifest = seed_library(target)
    library = load_library(target)
    typer.echo(f"Wrote {manifest}")
    for sprite in library.sprites:
        image = library.path_of(sprite)
        typer.echo(f"  {sprite.name:<18} {image.name:<22} {image.stat().st_size:>7} bytes")


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
    debug: bool = typer.Option(False, "--debug", help="Re-raise stage failures with their traceback"),
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
        _report_stage_failure(failure, project, config_path, debug)
    if executed:
        typer.echo("Prepared review draft: " + ", ".join(executed))
    draft_executed = bool(executed)

    draft = ctx.store.read("06-draft", DraftResult)
    try:
        approval_is_current(ctx, draft)
    except RuntimeError as error:
        typer.echo(str(error))
        typer.echo(f"Review: {draft.path}")
        typer.echo(
            f"Then approve: videoai approve {project} --config {config_path}"
        )
        _snapshot_if_executed(ctx, project, draft_executed)
        return

    try:
        executed = run_pipeline(
            ctx,
            media_fingerprint=media_fingerprint,
            brief_fingerprint=brief_fingerprint,
        )
    except StageFailure as failure:
        _report_stage_failure(failure, project, config_path, debug)
    result = ctx.store.read("08-final", FinalResult)
    if not result.fully_decoded or not result.production_report:
        raise typer.BadParameter("final artifact did not pass the production contract")
    typer.echo("Production passed: " + result.path)
    typer.echo("Report: " + result.production_report)
    _snapshot_if_executed(ctx, project, draft_executed or bool(executed))


@app.command()
def config(path: Path = typer.Option(Path("config.yaml"), help="Config file path")) -> None:
    """Print the effective configuration."""
    typer.echo(load_config(path).model_dump_json(indent=2))


if __name__ == "__main__":
    app()


@app.command("preview-effects")
def preview_effects(
    project: Path = typer.Argument(..., help="Project whose planned effects to draw"),
    config_path: Path = typer.Option(Path("config.yaml"), "--config", help="Config file"),
) -> None:
    """Draw every planned accent on the frame it will land on, for approval.

    An effect plan read as text cannot be judged: a name and a timestamp say
    nothing about whether the accent covers a face, points at what is happening,
    or sits in an empty corner. This writes a page showing each one composited
    exactly as the delivery would, next to where the picture is actually moving.
    """
    import base64  # noqa: F401  (used via the preview module)

    import cv2

    from videoai.core.models import EffectPlan, Manifest, StoryPlan, Timeline
    from videoai.logic.effects import load_library
    from videoai.logic.effects_preview import (
        EffectProposal,
        compose_preview,
        render_preview_html,
    )
    from videoai.logic.motion import busiest_cell
    from videoai.stages.s08_polish import _sprite_base_image

    work_dir = project / "work"
    store = ArtifactStore(work_dir)
    for name in ("05-timeline", "05d-effects"):
        if not store.exists(name):
            raise typer.BadParameter(
                f"{name} is missing; run the pipeline through the effects stage first"
            )
    timeline = store.read("05-timeline", Timeline)
    plan = store.read("05d-effects", EffectPlan)
    manifest = store.read("01-manifest", Manifest)
    story = store.read("05a-storyplan", StoryPlan) if store.exists("05a-storyplan") else StoryPlan()
    if not plan.events:
        typer.echo("No effects were planned, so there is nothing to approve.")
        return

    library = load_library(Path(plan.library) if plan.library else None)
    frame = (timeline.width, timeline.height)
    preview_dir = work_dir / "effects-preview"
    preview_dir.mkdir(parents=True, exist_ok=True)

    # Each event sits on the edit's clock; the source frame behind it is found by
    # walking the timeline, so the preview shows the same footage the delivery
    # will cut — not the draft, which is a different encode.
    proposals: list[EffectProposal] = []
    for index, event in enumerate(plan.events):
        clip, offset = _clip_at(timeline, event.at_seconds)
        if clip is None:
            continue
        source = resolve_media_path(project, manifest.by_id(clip.src).proxy_path
                                    or manifest.by_id(clip.src).path)
        capture = cv2.VideoCapture(str(source))
        capture.set(cv2.CAP_PROP_POS_MSEC, offset * 1000.0)
        ok, picture = capture.read()
        capture.release()
        if not ok:
            continue
        picture = cv2.resize(picture, frame, interpolation=cv2.INTER_AREA)
        sprite = _sprite_base_image(library, event, frame, preview_dir, index)
        drawn = compose_preview(
            picture, sprite, event, library, busiest_cell(source, offset)
        )
        # JPEG, not PNG: these are photographs with one flat overlay, and seven
        # full-resolution PNGs made a 40MB page nobody wants to open.
        preview_width = 1280
        if drawn.shape[1] > preview_width:
            scale = preview_width / drawn.shape[1]
            drawn = cv2.resize(
                drawn,
                (preview_width, max(1, round(drawn.shape[0] * scale))),
                interpolation=cv2.INTER_AREA,
            )
        ok, encoded = cv2.imencode(".jpg", drawn, [cv2.IMWRITE_JPEG_QUALITY, 88])
        if not ok:
            continue
        proposals.append(
            EffectProposal(
                index=index,
                event=event,
                measured_cell=busiest_cell(source, offset),
                image_jpeg=encoded.tobytes(),
            )
        )

    output_dir = project / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "effects-preview.html"
    target.write_text(
        render_preview_html(proposals, story.title or project.name), encoding="utf-8"
    )
    agree = sum(1 for item in proposals if item.agrees_with_motion)
    typer.echo(f"Drew {len(proposals)} accent(s): {target}")
    typer.echo(
        f"{agree} of {len(proposals)} sit where the picture is actually moving. "
        "Open the page, then say which to keep, move or drop."
    )


def _clip_at(timeline, at_seconds: float):
    """The timeline segment playing at `at_seconds`, and the offset inside its source."""
    clock = 0.0
    for clip in timeline.clips:
        if clock <= at_seconds < clock + clip.dur:
            return clip, clip.offset + (at_seconds - clock)
        clock += clip.dur
    return None, 0.0
