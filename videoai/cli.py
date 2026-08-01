"""VideoAI command line interface."""
from __future__ import annotations

import time
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
from videoai.core.store import ArtifactStore, hash_file, hash_file_ends, hash_parts
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
    """Fingerprint every clip ingest will read, including per-camera subfolders.

    Addressed by content, exactly like `store.source_key`, and for the same
    reason: this value is mixed into EVERY stage's fingerprint, so keying it on
    mtime meant an rsync, a Time Machine restore or a cloud-sync round-trip
    re-ran the whole pipeline — including re-uploading the entire shoot to a
    metered video model — over footage that had not changed by a byte.
    """
    parts: list[str] = []
    clip_dir = resolve_clip_dir(project_dir)
    cameras = list_camera_clips(clip_dir)
    for camera in sorted(cameras):
        for path in cameras[camera]:
            if path.is_file():
                key = path.relative_to(project_dir) if path.is_relative_to(project_dir) else path.name
                parts.append(f"{key}:{hash_file_ends(path)}")
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
    open_editor: bool = typer.Option(
        False, "--edit",
        help="When the draft needs approving, open the edit page instead of just naming it",
    ),
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
        # The one moment a creator is deciding anything, and the only place the
        # edit page has ever been reachable from the workflow they actually run.
        if open_editor:
            saved = _serve_edit_page(project, config_path)
            if saved:
                typer.echo(
                    "Your changes are saved. Run produce again to rebuild the draft "
                    "from them before approving."
                )
                _snapshot_if_executed(ctx, project, draft_executed)
                return
        else:
            typer.echo(
                "Change it — reorder or switch off shots, and swap, move, resize or "
                f"turn any accent:\n  videoai edit {project} --config {config_path}"
            )
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


def _grab_jpeg(
    source: Path, at_seconds: float, frame: tuple[int, int], width: int
) -> bytes | None:
    """One frame of a source file, in the delivery's shape, as JPEG bytes.

    JPEG rather than PNG, and downscaled: these are photographs, and a page of
    full-resolution stills was 40MB and nobody opened it. Returns None when the
    file will not seek or decode, which is a shot the page simply does not draw
    rather than a run that fails.
    """
    import cv2

    capture = cv2.VideoCapture(str(source))
    capture.set(cv2.CAP_PROP_POS_MSEC, at_seconds * 1000.0)
    ok, picture = capture.read()
    capture.release()
    if not ok:
        return None
    picture = cv2.resize(picture, frame, interpolation=cv2.INTER_AREA)
    if picture.shape[1] > width:
        scale = width / picture.shape[1]
        picture = cv2.resize(
            picture, (width, max(1, round(picture.shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        )
    ok, encoded = cv2.imencode(".jpg", picture, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return encoded.tobytes() if ok else None


def _shown_order(store, timeline):
    """Every shot the creator may arrange, in the order they last left it.

    Deliberately built from the PROPOSAL and not from `05-timeline`: assembly has
    already removed the shots that were switched off, and a shot that vanishes
    from the page cannot be switched back on — there would be nothing left to
    click. So the running order is reapplied here with every shot enabled, which
    reuses assembly's own placement rule (including where a shot the creator has
    never seen belongs) and then marks the disabled ones instead of dropping them.
    """
    from videoai.core.models import Timeline
    from videoai.logic.decisions import read_overrides
    from videoai.logic.timeline import apply_clip_overrides

    overrides = read_overrides(store)
    proposal = store.read("05-proposal", Timeline) if store.exists("05-proposal") else timeline
    arranged, _ = apply_clip_overrides(
        proposal,
        overrides.model_copy(update={
            "clips": [
                decided.model_copy(update={"enabled": True})
                for decided in overrides.clips
            ]
        }),
    )
    switched_off = {decided.ref for decided in overrides.clips if not decided.enabled}
    return [(clip, clip.ref not in switched_off) for clip in arranged.clips]


def _accent_shot(timeline, event):
    """The shot an accent is drawn over, and where in its source that moment is.

    Found by ref wherever there is one: an accent belongs to a moment in a shot,
    so looking it up by second would go hunting on whatever footage the last
    reorder happened to move underneath it. Only an accent from before refs
    existed falls back to the clock.
    """
    if event.clip_ref:
        try:
            clip = timeline.clips[timeline.index_of(event.clip_ref)]
        except KeyError:
            return None, 0.0
        return clip, clip.offset + min(max(0.0, event.at_in_clip), clip.dur)
    return _clip_at(timeline, event.at_seconds)


def _build_preview(
    project: Path, config_path: Path, save_url: str = "", token: str = ""
) -> str | None:
    """The edit page as HTML, or None when there is no edit to show yet.

    Shared by the command that writes it to a file and the one that serves it, so
    what a creator changes is the same page either way.
    """
    import cv2

    from videoai.core.models import EffectPlan, Manifest, StoryPlan, Timeline
    from videoai.logic.decisions import (
        apply_effect_overrides,
        read_overrides,
        reproject_events,
    )
    from videoai.logic.effects import load_library
    from videoai.logic.effects_preview import (
        EffectProposal,
        ShotCard,
        proposal_geometry,
        render_preview_html,
    )
    from videoai.logic.motion import busiest_cell, cell_centre
    from videoai.stages.s08_polish import _sprite_base_image

    work_dir = project / "work"
    store = ArtifactStore(work_dir)
    if not store.exists("05-timeline"):
        raise typer.BadParameter(
            "05-timeline is missing; run the pipeline through the assemble stage first"
        )
    timeline = store.read("05-timeline", Timeline)
    # The page has to show the video the creator will get, which means their own
    # earlier decisions — not the proposal the model made against whatever order
    # it happened to be shown. A second visit that silently reverted the first
    # visit's work would be the same silent loss this whole path exists to stop.
    # Effects are optional: a calm review with none is still an edit to arrange.
    planned = store.read("05d-effects", EffectPlan) if store.exists("05d-effects") else EffectPlan()
    plan, _ = apply_effect_overrides(planned, read_overrides(store))
    plan, _ = reproject_events(plan, timeline)
    manifest = store.read("01-manifest", Manifest)
    story = store.read("05a-storyplan", StoryPlan) if store.exists("05a-storyplan") else StoryPlan()

    library = load_library(Path(plan.library) if plan.library else None)
    frame = (timeline.width, timeline.height)
    preview_dir = work_dir / "effects-preview"
    preview_dir.mkdir(parents=True, exist_ok=True)

    shots: list[ShotCard] = []
    for clip, enabled in _shown_order(store, timeline):
        info = manifest.by_id(clip.src)
        thumb = _grab_jpeg(
            resolve_media_path(project, info.proxy_path or info.path),
            clip.offset + clip.dur / 2.0, frame, 320,
        )
        if thumb is None:
            continue
        shots.append(ShotCard(
            ref=clip.ref or clip.src, beat=clip.beat, quote=clip.quote,
            duration=clip.dur, enabled=enabled, thumb_jpeg=thumb,
        ))
    if not shots:
        typer.echo("This project has no edit to show yet.")
        return None

    proposals = []
    for index, event in enumerate(plan.events):
        clip, offset = _accent_shot(timeline, event)
        if clip is None:
            continue
        info = manifest.by_id(clip.src)
        source = resolve_media_path(project, info.proxy_path or info.path)
        frame_bytes = _grab_jpeg(source, offset, frame, 1280)
        if frame_bytes is None:
            continue

        sprite = _sprite_base_image(library, event, frame, preview_dir, index)
        ok, sprite_bytes = cv2.imencode(".png", sprite)
        if not ok:
            continue
        width_frac, height_frac, x_frac, y_frac = proposal_geometry(
            event, library, (sprite.shape[1], sprite.shape[0]), frame
        )

        cell = busiest_cell(source, offset)
        motion_xy = None
        if cell:
            cx, cy = cell_centre(cell, frame[0], frame[1])
            motion_xy = (cx / frame[0], cy / frame[1])

        proposals.append(
            EffectProposal(
                index=index, event=event, clip_ref=clip.ref or clip.src,
                motion_xy=motion_xy, motion_cell=cell,
                frame_jpeg=frame_bytes, sprite_png=sprite_bytes.tobytes(),
                sprite_width=width_frac, sprite_height=height_frac,
                start_x=x_frac, start_y=y_frac, frame_size=frame,
            )
        )

    agree = sum(
        1 for item in proposals
        if item.motion_cell and item.motion_cell == item.event.screen_position
    )
    typer.echo(
        f"{len(shots)} shot(s) and {len(proposals)} accent(s); "
        f"{agree} sit where the picture is actually moving."
    )
    return render_preview_html(
        shots, proposals, story.title or project.name,
        _sprite_choices(library, frame),
        # Scoped to this project's own folder, so two projects cannot inherit
        # each other's half-finished work out of one browser's storage.
        project_key=hash_parts(str(project.resolve())),
        save_url=save_url, token=token,
    )


def _serve_edit_page(project: Path, config_path: Path) -> bool:
    """Serve the edit page until Ctrl-C, and say whether anything was saved.

    The page cannot write to disk, so it posts to a small server that lives only
    while the editing is open: press Save and the decisions are in the edit
    before the button stops looking pressed.
    """
    import webbrowser

    from videoai.logic.approval_server import ApprovalSession, serve

    # The session is created first because the page has to be built around the
    # token it mints. No stage fingerprint: the decisions land in the creator's
    # own artifact, which no stage writes, so there is nothing to call stale.
    session = ApprovalSession(project_dir=project, page_html="")
    page = _build_preview(project, config_path, save_url="/save", token=session.token)
    if page is None:
        return False
    session.page_html = page
    session.on_save = lambda result: typer.echo("  Saved: " + result.summary())

    server, url = serve(session)
    typer.echo(f"Edit page: {url}")
    typer.echo("Rearrange it, press Save, then stop this with Ctrl-C.")
    webbrowser.open(url)
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        typer.echo("")
    finally:
        server.shutdown()
    if not session.saves:
        typer.echo("Nothing was saved, so the edit is unchanged.")
    return bool(session.saves)


@app.command("edit")
def edit(
    project: Path = typer.Argument(..., help="Project whose edit to arrange"),
    config_path: Path = typer.Option(Path("config.yaml"), "--config", help="Config file"),
    write: bool = typer.Option(
        False, "--write",
        help="Write the page to output/edit.html instead of serving it; edits then "
             "stay in the browser until you copy them back",
    ),
) -> None:
    """Arrange the edit: the running order, and every accent on its own frame.

    Drag a shot to move it or untick it to take it out; swap, drag, resize or
    turn any badge. Press Save and it is on disk. `--write` produces the same
    page as a file, for looking at later or on another machine.
    """
    if not write:
        if _serve_edit_page(project, config_path):
            typer.echo("Render the changes: videoai produce " + str(project))
        return
    page = _build_preview(project, config_path)
    if page is None:
        return
    output_dir = project / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "edit.html"
    target.write_text(page, encoding="utf-8")
    typer.echo(f"Wrote {target}")
    typer.echo("Edits there stay in the browser — run 'videoai edit' to save them.")


@app.command("preview-effects", hidden=True)
def preview_effects(
    project: Path = typer.Argument(...),
    config_path: Path = typer.Option(Path("config.yaml"), "--config"),
) -> None:
    """The old name for `videoai edit --write`."""
    edit(project, config_path, write=True)


@app.command("approve-effects", hidden=True)
def approve_effects(
    project: Path = typer.Argument(...),
    config_path: Path = typer.Option(Path("config.yaml"), "--config"),
) -> None:
    """The old name for `videoai edit`."""
    edit(project, config_path, write=False)


@app.command("apply-effects")
def apply_effects(
    project: Path = typer.Argument(..., help="Project to apply the decisions to"),
    decisions: Path = typer.Option(
        ..., "--decisions", help="effects-decisions.json downloaded from the edit page"
    ),
    config_path: Path = typer.Option(Path("config.yaml"), "--config", help="Config file"),
) -> None:
    """Write the creator's decisions from the edit page into work/05e-overrides.json.

    Dropped accents are recorded as refused rather than deleted: a decision that
    vanishes is one the next re-plan will happily make again, and the record of
    what was looked at and turned down is worth keeping.
    """
    import json as _json

    from videoai.logic.approval import apply_decisions

    store = ArtifactStore(project / "work")
    if not store.exists("05-timeline"):
        raise typer.BadParameter("this project has no edit to apply decisions to")
    result = apply_decisions(
        store, _json.loads(decisions.read_text(encoding="utf-8"))
    )
    typer.echo("Applied: " + result.summary() + ".")
    typer.echo("Re-run the delivery to render them: videoai produce " + str(project))


def _sprite_choices(library, frame: tuple[int, int]) -> list[dict]:
    """Every sprite the library offers, small enough to embed in a page.

    Each carries its own picture and aspect so the page can swap the drawing AND
    resize it the way the compositor would. Downscaled hard: the preview shows
    them a couple of hundred pixels wide, and fifty-seven full-resolution PNGs
    would be a page nobody waits for.
    """
    import base64

    import cv2

    from videoai.logic.effects import SPRITE_SCALES

    choices: list[dict] = []
    for sprite in library.sprites:
        image = cv2.imread(str(library.path_of(sprite)), cv2.IMREAD_UNCHANGED)
        if image is None:
            continue
        preview_height = 192
        scale = preview_height / image.shape[0]
        small = cv2.resize(
            image,
            (max(1, round(image.shape[1] * scale)), preview_height),
            interpolation=cv2.INTER_AREA,
        )
        ok, encoded = cv2.imencode(".png", small)
        if not ok:
            continue
        choices.append({
            "name": sprite.name,
            "aspect": round(image.shape[1] / image.shape[0], 4),
            # What fraction of the frame's height a medium accent occupies; the
            # page multiplies by the aspect to get its width.
            "height_frac": SPRITE_SCALES["medium"],
            "takes_text": sprite.takes_text,
            "data": base64.b64encode(encoded.tobytes()).decode("ascii"),
        })
    return choices


def _clip_at(timeline, at_seconds: float):
    """The timeline segment playing at `at_seconds`, and the offset inside its source."""
    clock = 0.0
    for clip in timeline.clips:
        if clock <= at_seconds < clock + clip.dur:
            return clip, clip.offset + (at_seconds - clock)
        clock += clip.dur
    return None, 0.0


@app.command("fetch-badges")
def fetch_badges(
    library: Path = typer.Option(
        None, "--library",
        help="Where the sprite library lives (default: the packaged assets/effects)",
    ),
) -> None:
    """Download the badge set and add it to the sprite library.

    Run once. The drawings are fetched from Twemoji, rasterised to a size the
    compositor never has to upscale, and written into the library's manifest with
    the credit their licence requires attached to each one.
    """
    import yaml

    from videoai.logic.badges import BADGES, manifest_entry, rasterise_badge
    from videoai.logic.effects import default_library_dir, manifest_path

    directory = library or default_library_dir()
    directory.mkdir(parents=True, exist_ok=True)
    path = manifest_path(directory)
    document = yaml.safe_load(path.read_text(encoding="utf-8")) if path.is_file() else {}
    sprites = list(document.get("sprites") or [])
    known = {entry.get("name") for entry in sprites}

    added, refreshed = 0, 0
    for badge in BADGES:
        target = directory / f"{badge.name}.png"
        try:
            target.write_bytes(rasterise_badge(badge.code))
        except Exception as error:  # noqa: BLE001 - reported, not swallowed
            typer.echo(f"  {badge.name}: could not fetch ({error})", err=True)
            continue
        entry = manifest_entry(badge)
        if badge.name in known:
            sprites = [entry if item.get("name") == badge.name else item for item in sprites]
            refreshed += 1
        else:
            sprites.append(entry)
            added += 1

    document["sprites"] = sprites
    path.write_text(yaml.safe_dump(document, sort_keys=False, allow_unicode=True),
                    encoding="utf-8")
    typer.echo(f"Badges: {added} added, {refreshed} refreshed, in {directory}")
    typer.echo(
        "Their licence needs a credit, which is attached to each sprite and is "
        "written into output/metadata.md only when one of them is actually used."
    )


@app.command("doctor")
def doctor(
    project_dir: Path = typer.Option(Path("."), "--dir", help="Project or repo directory to check"),
    fix: bool = typer.Option(False, "--fix", help="Automatically create missing .env from .env.example"),
) -> None:
    """Validate system configuration, environment variables, and dependencies."""
    import os
    import platform
    import shutil
    import subprocess
    import sys

    typer.echo("🔍 VideoAI System & Environment Health Check")
    typer.echo("=" * 45)

    all_ok = True

    # 1. OS & Architecture
    is_macos = sys.platform == "darwin"
    arch = platform.machine()
    is_arm = arch in ("arm64", "aarch64")
    if is_macos and is_arm:
        typer.echo("  [✓] Operating System: macOS Apple Silicon (" + arch + ")")
    else:
        typer.echo(f"  [!] Operating System: {sys.platform} ({arch}) - Apple Silicon Mac recommended for local MLX GPU ASR")

    # 2. Python version
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    if sys.version_info >= (3, 13):
        typer.echo(f"  [✓] Python: {py_ver}")
    else:
        typer.echo(f"  [✗] Python: {py_ver} (Python >= 3.13 is required)")
        all_ok = False

    # 3. Executable CLI checks
    for tool, required in [("ffmpeg", True), ("uv", True), ("git", True), ("claude", False), ("codex", False)]:
        path = shutil.which(tool)
        if path:
            ver = ""
            try:
                out = subprocess.check_output([tool, "--version"], stderr=subprocess.STDOUT, text=True)
                ver = out.splitlines()[0] if out else ""
            except Exception:
                pass
            typer.echo(f"  [✓] CLI tool {tool}: found ({ver[:40]})")
        elif required:
            typer.echo(f"  [✗] CLI tool {tool}: missing (Install via: brew install {tool})")
            all_ok = False
        else:
            typer.echo(f"  [-] CLI tool {tool}: not found (Optional subscription CLI)")

    # 4. Cairo / CairoSVG check
    try:
        import cairosvg  # noqa: F401
        typer.echo("  [✓] Library cairosvg: available")
    except ImportError:
        typer.echo("  [!] Library cairosvg: missing (Install via: brew install cairo && uv sync)")

    # 5. Check .env file and keys
    env_file = project_dir / ".env"
    env_example = project_dir / ".env.example"
    if env_file.is_file():
        typer.echo(f"  [✓] Environment file: {env_file.name} exists")
        from dotenv import load_dotenv
        load_dotenv(env_file)
        gemini_key = os.getenv("GEMINI_API_KEY")
        if gemini_key:
            typer.echo("  [✓] Secret GEMINI_API_KEY: configured")
        else:
            typer.echo("  [-] Secret GEMINI_API_KEY: not set (Optional for Gemini multimodal provider)")
    else:
        typer.echo(f"  [!] Environment file: .env missing in {project_dir.resolve()}")
        if env_example.is_file():
            if fix:
                shutil.copy(env_example, env_file)
                typer.echo(f"      -> Auto-created .env from {env_example.name}")
            else:
                typer.echo(f"      -> Fix by running: cp .env.example .env (or run 'videoai doctor --fix')")
        all_ok = False

    typer.echo("=" * 45)
    if all_ok:
        typer.echo("🎉 Environment is correctly configured and ready!")
    else:
        typer.echo("⚠️  Some required items are missing or misconfigured. Please follow the instructions above.")


# At the very bottom, and not in the middle of the file: everything below the
# `app()` call would otherwise never be defined when the module is run with
# `python -m videoai.cli`, so half the commands did not exist that way.
if __name__ == "__main__":
    app()
