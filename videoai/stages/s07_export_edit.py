"""s07 export_edit: hand the cut off to a human editor.

Free DaVinci Resolve has no scripting API (that is Studio-only), so this stage
never drives Resolve. It writes two interchange files into `output/` —
`edit.otio` (OpenTimelineIO, primary) and `edit.edl` (CMX3600, fallback) — that
the creator imports themselves via Resolve's own File -> Import Timeline
dialog. See `docs/EDITOR-HANDOFF.md` for the format research behind this
choice.
"""
from __future__ import annotations

import opentimelineio as otio

from videoai.core.models import ExportResult, Manifest, Timeline
from videoai.core.registry import StageContext, stage
from videoai.logic.interchange import build_otio_timeline

# The cmx_3600 (EDL) adapter ships as a separate plugin package, not inside
# core `opentimelineio` — naming it here lets `_require_cmx_3600_adapter`
# fail with the exact `uv add` command instead of a bare adapter-not-found
# error surfacing from deep inside `otio.adapters.write_to_file`.
CMX_3600_ADAPTER_PACKAGE = "otio-cmx3600-adapter"


def _require_cmx_3600_adapter() -> None:
    active = otio.plugins.ActiveManifest()
    if not any(adapter.name == "cmx_3600" for adapter in active.adapters):
        raise RuntimeError(
            "OpenTimelineIO's 'cmx_3600' adapter plugin is not installed; "
            f"add it with: uv add {CMX_3600_ADAPTER_PACKAGE}"
        )


@stage(
    id="export_edit",
    produces="07-export",
    requires=("01-manifest", "05-timeline"),
    model=ExportResult,
)
def export_edit(ctx: StageContext) -> ExportResult:
    _require_cmx_3600_adapter()

    manifest = ctx.store.read("01-manifest", Manifest)
    timeline = ctx.store.read("05-timeline", Timeline)
    if not timeline.clips:
        raise RuntimeError("cannot export an empty timeline")

    otio_timeline, missing_media = build_otio_timeline(timeline, manifest, ctx.project_dir)

    ctx.output_dir.mkdir(parents=True, exist_ok=True)
    otio_path = ctx.output_dir / "edit.otio"
    edl_path = ctx.output_dir / "edit.edl"

    otio.adapters.write_to_file(otio_timeline, str(otio_path))
    otio.adapters.write_to_file(otio_timeline, str(edl_path), adapter_name="cmx_3600")

    return ExportResult(
        otio_path=str(otio_path),
        edl_path=str(edl_path),
        clip_count=len(timeline.clips),
        missing_media=missing_media,
    )
