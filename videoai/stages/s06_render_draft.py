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

# The draft's fixed audio target. Mono at 44100 Hz is right for a review draft:
# it's smaller, and the draft exists to be watched for edit decisions, not to be
# the master. Both branches of `_render_segment` force these values explicitly —
# without an explicit `-ar`/`-ac`, `aac` just inherits whatever the source proxy
# has (this project's real footage is AAC stereo at 48 kHz from an iPhone), so an
# audio-bearing segment and a synthesised-silent segment would drift apart and
# violate the homogeneity the concat demuxer's `-c copy` step depends on.
DRAFT_AUDIO_CODEC = "aac"
DRAFT_AUDIO_BITRATE = "160k"
DRAFT_AUDIO_SAMPLE_RATE = 44100
DRAFT_AUDIO_CHANNELS = 1
DRAFT_AUDIO_CHANNEL_LAYOUT = "mono"

SILENT_AUDIO_SOURCE = (
    f"anullsrc=channel_layout={DRAFT_AUDIO_CHANNEL_LAYOUT}:sample_rate={DRAFT_AUDIO_SAMPLE_RATE}"
)


def _render_segment(
    source: Path, offset: float, duration: float, fade: float, crf: int, has_audio: bool, dst: Path
) -> None:
    video_args = ["-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf)]
    audio_args = [
        "-c:a", DRAFT_AUDIO_CODEC, "-b:a", DRAFT_AUDIO_BITRATE,
        "-ar", str(DRAFT_AUDIO_SAMPLE_RATE), "-ac", str(DRAFT_AUDIO_CHANNELS),
    ]
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
    config_keys=("render.audio_fade_seconds", "render.draft_crf"),
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
