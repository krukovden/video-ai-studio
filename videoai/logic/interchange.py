"""Turn the pipeline's own `Timeline` + `Manifest` into an in-memory
OpenTimelineIO timeline, ready to be written out as `.otio` and `.edl` for a
human editor to open by hand in DaVinci Resolve (free) or any other
OTIO/EDL-capable NLE.

Free DaVinci Resolve has no scripting API — that is Studio-only — so nothing
here drives Resolve itself. This module only builds an interchange document;
the creator imports it themselves through Resolve's own File -> Import
Timeline dialog.
"""
from __future__ import annotations

from pathlib import Path

import opentimelineio as otio

from videoai.core.models import Manifest, Timeline, TimelineClip


def _clip_metadata(clip: TimelineClip) -> dict:
    """Carried as a generic metadata dict on both the `Clip` and its `Marker` so
    beat/quote/reason/is_insert survive any OTIO round-trip, not just the
    human-readable marker comment."""
    return {
        "videoai": {
            "beat": clip.beat,
            "quote": clip.quote,
            "reason": clip.reason,
            "is_insert": clip.is_insert,
        }
    }


def _marker_comment(clip: TimelineClip) -> str:
    """Human-readable text so opening the project in Resolve shows, at a glance,
    why each cut is there — the whole point of the handoff, not just data that
    happens to survive it."""
    lines = []
    if clip.beat:
        lines.append(f"beat: {clip.beat}")
    if clip.is_insert:
        lines.append("visual insert (no speech)")
    if clip.quote:
        lines.append(f'quote: "{clip.quote}"')
    if clip.reason:
        lines.append(f"reason: {clip.reason}")
    return "\n".join(lines)


def _resolve_source_path(raw_path: str, project_dir: Path) -> Path:
    """Manifest paths are written by `ingest` relative to wherever `videoai run`
    happened to be invoked from (its process cwd) — e.g.
    `assets/.../video/IMG_5665.MOV` when run from the repo root, or just
    `video/IMG_5665.MOV` when run from inside the project folder. `project_dir`
    is itself just as cwd-relative, so naively joining it onto an already
    cwd-relative path double-prefixes it (`project/project/video/...`) instead
    of fixing it. Resolving against the current cwd first reproduces exactly
    what `ingest` saw; only if that path is absent does resolving against
    `project_dir` get tried, for the case where this stage runs from a
    different cwd than ingest did (e.g. invoked from inside the project
    folder while the manifest was written from the repo root)."""
    path = Path(raw_path)
    if path.is_absolute():
        return path
    cwd_relative = path.resolve()
    if cwd_relative.exists():
        return cwd_relative
    project_relative = (project_dir / path).resolve()
    if project_relative.exists():
        return project_relative
    return cwd_relative


def build_otio_timeline(
    timeline: Timeline, manifest: Manifest, project_dir: Path
) -> tuple[otio.schema.Timeline, list[str]]:
    """Build the OTIO object graph. Returns the timeline plus the absolute source
    paths that do not currently exist on disk (still referenced, not skipped —
    see `ExportResult.missing_media`).

    Media references always resolve to `ClipInfo.path`, the original
    full-quality source, never `ClipInfo.proxy_path`. The 540p proxy exists only
    to make the earlier analysis stages (quality, sync, transcribe, analyze)
    fast; the creator finishes the video by hand at full quality, and handing
    off the proxy as "the" media would silently cap the final export at 540p.
    """
    otio_timeline = otio.schema.Timeline(name="VideoAI edit")
    track = otio.schema.Track(name="V1", kind=otio.schema.TrackKind.Video)
    otio_timeline.tracks.append(track)

    # Per task spec: rational time values use the *timeline's own* fps, not each
    # clip's individually reported fps — this project's sources already agree at
    # a single fps, and one shared rate keeps every clip's in/out points and
    # timeline position commensurable in the same track.
    rate = timeline.fps

    missing: list[str] = []
    position = 0.0
    for clip in timeline.clips:
        source = manifest.by_id(clip.src)
        source_path = _resolve_source_path(source.path, project_dir)
        if not source_path.exists():
            missing.append(str(source_path))

        # A hand-edited `05-timeline.json` could introduce a gap between clips;
        # `build_timeline()` itself never does (each clip's `start` is the exact
        # running sum of prior durations), but deriving position from `start`
        # rather than trusting sequential append keeps that an enforced
        # invariant of the export, not an assumption about the file's origin.
        gap = clip.start - position
        if gap > 0:
            track.append(
                otio.schema.Gap(
                    source_range=otio.opentime.TimeRange(
                        start_time=otio.opentime.RationalTime(0, rate),
                        duration=otio.opentime.RationalTime.from_seconds(gap, rate),
                    )
                )
            )

        media_reference = otio.schema.ExternalReference(target_url=str(source_path))
        source_range = otio.opentime.TimeRange(
            start_time=otio.opentime.RationalTime.from_seconds(clip.offset, rate),
            duration=otio.opentime.RationalTime.from_seconds(clip.dur, rate),
        )
        otio_clip = otio.schema.Clip(
            name=f"{clip.src} ({clip.beat})" if clip.beat else clip.src,
            media_reference=media_reference,
            source_range=source_range,
            metadata=_clip_metadata(clip),
        )

        comment = _marker_comment(clip)
        if comment:
            otio_clip.markers.append(
                otio.schema.Marker(
                    name="insert" if clip.is_insert else (clip.beat or "cut"),
                    marked_range=otio.opentime.TimeRange(
                        start_time=otio.opentime.RationalTime(0, rate),
                        duration=source_range.duration,
                    ),
                    color=otio.schema.MarkerColor.PURPLE
                    if clip.is_insert
                    else otio.schema.MarkerColor.GREEN,
                    comment=comment,
                    metadata=_clip_metadata(clip),
                )
            )

        track.append(otio_clip)
        position = clip.start + clip.dur

    return otio_timeline, missing
