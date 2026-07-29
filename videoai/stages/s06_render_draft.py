"""s06 draft render: cut the approved segments and concatenate them.

Segments are re-encoded from the proxy so cuts land exactly where the timeline
says, and each boundary gets a short audio fade to avoid clicks. The concat step
that follows uses `-c copy`, which is only valid when every segment has an
identical stream layout — so a source with no audio track still gets one
synthesised, rather than producing a video-only segment that breaks that
assumption.
"""
from __future__ import annotations

from pathlib import Path

from videoai.core.ffmpeg import probe, run_ffmpeg
from videoai.core.models import DraftResult, Manifest, Timeline
from videoai.core.registry import StageContext, stage

# Silent audio synthesised for sources with no audio track, so every rendered
# segment carries an audio stream with the same codec, sample rate and channel
# layout — required for the concat demuxer's `-c copy` step to stay valid.
SILENT_AUDIO_SOURCE = "anullsrc=channel_layout=mono:sample_rate=44100"


def _render_segment(
    source: Path, offset: float, duration: float, fade: float, crf: int, has_audio: bool, dst: Path
) -> None:
    video_args = ["-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf)]
    audio_args = ["-c:a", "aac", "-b:a", "160k"]
    if has_audio:
        fade_out_start = max(0.0, duration - fade)
        run_ffmpeg([
            "-ss", f"{offset:.3f}", "-i", str(source), "-t", f"{duration:.3f}",
            *video_args,
            *audio_args,
            "-af", f"afade=t=in:st=0:d={fade},afade=t=out:st={fade_out_start:.3f}:d={fade}",
            "-avoid_negative_ts", "make_zero",
            str(dst),
        ])
    else:
        # No audio to fade; synthesise a silent track instead of leaving this
        # segment video-only.
        run_ffmpeg([
            "-ss", f"{offset:.3f}", "-i", str(source),
            "-f", "lavfi", "-i", SILENT_AUDIO_SOURCE,
            "-t", f"{duration:.3f}",
            "-map", "0:v:0", "-map", "1:a:0",
            *video_args,
            *audio_args,
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
        _render_segment(source, clip.offset, clip.dur, fade, crf, source_info.has_audio, target)
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
