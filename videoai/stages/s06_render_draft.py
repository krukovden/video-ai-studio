"""s06 draft render: cut the approved segments and concatenate them.

Segments are re-encoded from the proxy so cuts land exactly where the timeline
says, and each boundary gets a short audio fade to avoid clicks.
"""
from __future__ import annotations

from pathlib import Path

from videoai.core.ffmpeg import probe, run_ffmpeg
from videoai.core.models import DraftResult, Manifest, Timeline
from videoai.core.registry import StageContext, stage


def _render_segment(source: Path, offset: float, duration: float, fade: float, crf: int, dst: Path) -> None:
    fade_out_start = max(0.0, duration - fade)
    run_ffmpeg([
        "-ss", f"{offset:.3f}", "-i", str(source), "-t", f"{duration:.3f}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
        "-c:a", "aac", "-b:a", "160k",
        "-af", f"afade=t=in:st=0:d={fade},afade=t=out:st={fade_out_start:.3f}:d={fade}",
        "-avoid_negative_ts", "make_zero",
        str(dst),
    ])


@stage(
    id="render_draft",
    produces="06-draft",
    requires=("01-manifest", "05-timeline"),
    model=DraftResult,
)
def render_draft(ctx: StageContext) -> DraftResult:
    manifest = ctx.store.read("01-manifest", Manifest)
    timeline = ctx.store.read("05-timeline", Timeline)
    if not timeline.clips:
        raise RuntimeError("cannot render an empty timeline")

    segments_dir = ctx.work_dir / "segments"
    segments_dir.mkdir(parents=True, exist_ok=True)
    fade = ctx.config.render.audio_fade_seconds
    crf = ctx.config.render.draft_crf

    segment_paths: list[Path] = []
    for index, clip in enumerate(timeline.clips):
        source_info = manifest.by_id(clip.src)
        source = Path(source_info.proxy_path or source_info.path)
        target = segments_dir / f"seg-{index:03d}.mp4"
        _render_segment(source, clip.offset, clip.dur, fade, crf, target)
        segment_paths.append(target)

    list_file = segments_dir / "concat.txt"
    list_file.write_text(
        "\n".join(f"file '{path.name}'" for path in segment_paths) + "\n",
        encoding="utf-8",
    )

    ctx.output_dir.mkdir(parents=True, exist_ok=True)
    output = ctx.output_dir / "draft.mp4"
    run_ffmpeg([
        "-f", "concat", "-safe", "0", "-i", str(list_file),
        "-c", "copy", str(output),
    ])

    return DraftResult(
        path=str(output),
        duration=probe(output).duration,
        segment_count=len(segment_paths),
    )
