"""s08 polish: turn the reviewed cut into something a viewer would watch.

The draft is deliberately bare — it exists to check edit decisions, and it stays
exactly as it was. This stage builds a second file beside it with the things a
finished video has and an edit review does not: a title card, a lower third when
the story moves to a new section, a music bed ducked under the child's voice, and
a soft dissolve where one section becomes the next.

Assembly is a single ffmpeg invocation: the cross-dissolves, the title card, the
overlays and the music mix all happen in one filter graph, so the picture is
encoded once more rather than once per element. The one thing that cannot be done
inside that graph is slicing the draft — see `_cut_runs`.
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import yaml

from videoai.core.ffmpeg import probe, run_ffmpeg
from videoai.core.models import DraftResult, FinalResult, StoryPlan, Timeline
from videoai.core.registry import StageContext, stage
from videoai.core.text import (
    drawtext_available,
    drawtext_filter,
    render_text_image,
    resolve_font,
    wrap_text,
)
from videoai.logic.music import attribution_line, list_tracks, select_track
from videoai.stages.s06_render_draft import (
    DRAFT_AUDIO_BITRATE,
    DRAFT_AUDIO_CHANNEL_LAYOUT,
    DRAFT_AUDIO_CHANNELS,
    DRAFT_AUDIO_CODEC,
    DRAFT_AUDIO_SAMPLE_RATE,
)

FINAL_CRF = 20
# The run cuts are an intermediate that is immediately re-encoded into the final,
# so they are kept visually transparent rather than small.
RUN_CRF = 16
INTRO_BACKGROUND = "0x111a2b"
INTRO_ACCENT = "0xe8c46a"
INTRO_FADE_SECONDS = 0.4
INTRO_TEXT_FADE_SECONDS = 0.6
TITLE_FADE_SECONDS = 0.35
MUSIC_FADE_SECONDS = 2.0

# `sidechaincompress` attenuates the bed by `overshoot * (1 - 1/ratio)` dB, where
# the overshoot is how far the key signal sits above the threshold. The threshold
# below is far under speech but above this footage's room tone, and speech in the
# rendered drafts sits roughly this far above it — so that is the overshoot the
# configured duck is solved for.
SIDECHAIN_THRESHOLD = 0.015
SPEECH_OVERSHOOT_DB = 16.0
SIDECHAIN_ATTACK_MS = 20
SIDECHAIN_RELEASE_MS = 300

# A cut may only become a dissolve when both sides have material to dissolve
# through; a run shorter than this is left as a hard cut.
MIN_RUN_FACTOR = 3.0

SILENCE_SOURCE = (
    f"anullsrc=channel_layout={DRAFT_AUDIO_CHANNEL_LAYOUT}:sample_rate={DRAFT_AUDIO_SAMPLE_RATE}"
)


@dataclass(frozen=True)
class _TextOverlay:
    """One RGBA strip faded in over the picture between `start` and `start+dur`."""

    text: str
    start: float
    duration: float
    width: int
    height: int
    y_expression: str
    plate_alpha: float
    fade_in: float
    fade_out: float
    max_lines: int = 2


def _project_style(project_dir: Path) -> str:
    """The `style` the creator declared in project.yaml, or "" when there is none.

    Read directly rather than through the store because it is part of the brief,
    not a pipeline artifact — the same way `plan` reads the brief itself.
    """
    path = project_dir / "project.yaml"
    if not path.is_file():
        return ""
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError:
        return ""
    style = raw.get("style") if isinstance(raw, dict) else None
    return style if isinstance(style, str) else ""


def _draft_path(ctx: StageContext, draft: DraftResult) -> Path:
    """The draft on disk. The recorded path is relative when the CLI was invoked
    with a relative project directory, so fall back to the output folder."""
    recorded = Path(draft.path)
    if recorded.is_file():
        return recorded
    beside = ctx.output_dir / recorded.name
    if beside.is_file():
        return beside
    raise RuntimeError(f"draft not found at {recorded} or {beside}")


def segment_starts(ctx: StageContext, timeline: Timeline, draft_duration: float) -> list[float]:
    """Where each timeline clip begins inside the rendered draft.

    Rendered segments are whole numbers of frames, so each is a frame or so longer
    than the timeline asked for and the error accumulates across twenty-odd cuts —
    enough to put a dissolve on the wrong side of a cut. The segment files carry
    the truth and are measured when the render stage's work directory still holds
    them; otherwise the timeline's own starts are stretched onto the draft's
    measured length, which spreads the same drift rather than ignoring it.
    """
    segments = sorted((ctx.work_dir / "segments").glob("seg-*.mp4"))
    starts = [0.0]
    if len(segments) == len(timeline.clips):
        for path in segments[:-1]:
            starts.append(starts[-1] + probe(path).duration)
        return starts
    planned = timeline.duration
    scale = draft_duration / planned if planned > 0 else 1.0
    running = 0.0
    for clip in timeline.clips[:-1]:
        running += clip.dur
        starts.append(running * scale)
    return starts


def section_changes(timeline: Timeline) -> list[int]:
    """Clip indices where the story moves to a new beat.

    Index 0 is never one: it has no previous clip to differ from, and the title
    card has just named the video anyway.
    """
    return [
        index
        for index in range(1, len(timeline.clips))
        if timeline.clips[index].beat != timeline.clips[index - 1].beat
    ]


def duck_ratio(duck_db: float) -> float:
    """The compressor ratio that attenuates the bed by `duck_db` under speech."""
    reduction = min(abs(duck_db), SPEECH_OVERSHOOT_DB - 1.0)
    if reduction <= 0.0:
        return 1.0
    return max(1.0, min(20.0, 1.0 / (1.0 - reduction / SPEECH_OVERSHOOT_DB)))


def write_attribution(output_dir: Path, line: str) -> bool:
    """Append the credit to output/metadata.md. True when it was actually added.

    Re-rendering the final must not stack up copies of the same credit, so a line
    already present is left alone.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "metadata.md"
    existing = path.read_text(encoding="utf-8") if path.is_file() else ""
    if line in existing:
        return False
    if existing:
        separator = "" if existing.endswith("\n") else "\n"
        path.write_text(f"{existing}{separator}{line}\n", encoding="utf-8")
    else:
        path.write_text(f"# Credits\n\n{line}\n", encoding="utf-8")
    return True


def _cut_runs(
    source: Path, bounds: list[tuple[float, float]], work_dir: Path, fps: float
) -> list[Path]:
    """One normalised file per stretch of the draft between section dissolves.

    The draft cannot be sliced inside the assembly graph. It is a concat of
    twenty-odd separately encoded segments, each rounded up to a whole frame, so
    it carries a slightly variable frame rate and a video stream that starts a
    frame late — and against a file like that neither the input `-t` option nor
    the `trim` filter cuts where it is asked to (measured: `-ss 55.158 -t 58.837`
    reads 127 seconds, and `trim=end=58.837` keeps 29). Output-side `-t` is exact,
    so each run is cut and re-encoded to constant frame rate here, and the
    assembly graph then only ever sees files that behave.

    A single run needs no cut: with nothing to slice, the draft goes straight in.
    """
    if len(bounds) < 2:
        return [source]
    paths: list[Path] = []
    for index, (start, end) in enumerate(bounds):
        target = work_dir / f"run-{index:02d}.mp4"
        args = ["-ss", f"{start:.3f}", "-i", str(source)]
        if index < len(bounds) - 1:
            args += ["-t", f"{end - start:.3f}"]
        run_ffmpeg([
            *args,
            "-vf", f"fps={fps},format=yuv420p,setsar=1",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", str(RUN_CRF),
            "-c:a", DRAFT_AUDIO_CODEC, "-b:a", DRAFT_AUDIO_BITRATE,
            "-ar", str(DRAFT_AUDIO_SAMPLE_RATE), "-ac", str(DRAFT_AUDIO_CHANNELS),
            str(target),
        ])
        paths.append(target)
    return paths


def _overlay_inputs(
    overlay: _TextOverlay, index: int, work_dir: Path, fps: float, use_drawtext: bool
) -> tuple[list[str], str]:
    """Input arguments and a filter chain producing one faded RGBA overlay stream.

    Both text routes end in the same kind of stream, so everything after this —
    the alpha fades, the delay, the overlay itself — is shared.
    """
    span = overlay.duration
    if use_drawtext:
        font_size = max(12, int(overlay.height * 0.34))
        text_file = work_dir / f"text-{index:02d}.txt"
        text_file.parent.mkdir(parents=True, exist_ok=True)
        text_file.write_text(
            wrap_text(
                overlay.text, int(overlay.width * 0.92), font_size, overlay.max_lines
            ) + "\n",
            encoding="utf-8",
        )
        args = [
            "-f", "lavfi", "-t", f"{span:.3f}",
            "-i", f"color=c=black@0:s={overlay.width}x{overlay.height}:r={fps}",
        ]
        source = "format=rgba," + drawtext_filter(
            text_file, font_size, overlay.plate_alpha, "(h-text_h)/2"
        )
    else:
        image = work_dir / f"text-{index:02d}.png"
        render_text_image(
            image, overlay.text, overlay.width, overlay.height,
            plate_alpha=overlay.plate_alpha, max_lines=overlay.max_lines,
        )
        args = [
            "-loop", "1", "-framerate", f"{fps}", "-t", f"{span:.3f}",
            "-i", str(image),
        ]
        source = "format=rgba"

    chain = source
    if overlay.fade_in > 0:
        chain += f",fade=t=in:st=0:d={overlay.fade_in:.3f}:alpha=1"
    if overlay.fade_out > 0:
        chain += f",fade=t=out:st={span - overlay.fade_out:.3f}:d={overlay.fade_out:.3f}:alpha=1"
    if overlay.start > 0:
        # Transparent padding rather than `enable=`: it gives the strip its own
        # timestamps, so `overlay` never waits on a stream that has not started.
        chain += f",tpad=start_duration={overlay.start:.3f}:start_mode=add:color=0x00000000"
    return args, chain


@stage(
    id="polish",
    produces="08-final",
    # "01-manifest" is declared rather than read: it is what pins this stage to
    # the set of clips the draft was cut from, so replacing a source clip
    # invalidates the final as well as everything between.
    requires=("01-manifest", "05-timeline", "05a-storyplan", "06-draft"),
    model=FinalResult,
    config_keys=(
        "polish.enabled",
        "polish.intro_seconds",
        "polish.title_seconds",
        "polish.music_gain_db",
        "polish.music_duck_db",
        "polish.transition_frames",
        "polish.music_track",
        "polish.music_dir",
    ),
)
def polish(ctx: StageContext) -> FinalResult:
    settings = ctx.config.polish
    draft = ctx.store.read("06-draft", DraftResult)
    source = _draft_path(ctx, draft)
    ctx.output_dir.mkdir(parents=True, exist_ok=True)
    output = ctx.output_dir / "final.mp4"

    if not settings.enabled:
        # The draft is copied rather than re-encoded: "polish off" must cost the
        # picture nothing, and a re-encode is not nothing.
        shutil.copyfile(source, output)
        return FinalResult(
            path=str(output),
            duration=probe(output).duration,
            notes=["polish.enabled is false: final.mp4 is a copy of the draft"],
        )

    info = probe(source)
    if not info.has_audio:
        raise RuntimeError(
            f"draft {source} has no audio stream; the render stage guarantees one"
        )

    timeline = ctx.store.read("05-timeline", Timeline)
    story = (
        ctx.store.read("05a-storyplan", StoryPlan)
        if ctx.store.exists("05a-storyplan")
        else StoryPlan()
    )

    # The timeline's rate, not the draft's: `probe` reads `r_frame_rate`, which a
    # concatenated file reports as the largest rate any of its parts could carry
    # (120 for a 30 fps draft), and that number would end up as the final's own
    # frame rate and shrink every transition to a quarter of its length.
    fps = timeline.fps if timeline.fps > 0 else info.fps
    width, height = info.width, info.height
    work_dir = ctx.work_dir / "polish"
    work_dir.mkdir(parents=True, exist_ok=True)
    notes: list[str] = []

    use_drawtext = drawtext_available()
    if not use_drawtext:
        notes.append(
            "this ffmpeg build has no drawtext filter (no libfreetype); titles were "
            "rasterised and overlaid as images"
        )
    elif resolve_font() is None:
        notes.append("no preferred macOS font found; drawtext used its default face")

    # --- where the dissolves go -------------------------------------------------
    transition = max(0.0, settings.transition_frames) / fps
    starts = segment_starts(ctx, timeline, info.duration)
    changes = section_changes(timeline)

    splits: list[float] = []
    if transition > 0:
        minimum = max(transition * MIN_RUN_FACTOR, 2.0 / fps)
        previous = 0.0
        for index in changes:
            at = starts[index]
            if at - previous >= minimum and info.duration - at >= minimum:
                splits.append(at)
                previous = at
    run_bounds = list(zip([0.0, *splits], [*splits, info.duration]))
    run_sources = _cut_runs(source, run_bounds, work_dir, fps)
    run_lengths = [probe(path).duration for path in run_sources] if len(run_sources) > 1 \
        else [info.duration]
    body_duration = sum(run_lengths) - len(splits) * transition

    # --- the intro card ---------------------------------------------------------
    intro_duration = max(0.0, settings.intro_seconds)
    intro_fade = 0.0
    if intro_duration > 0:
        intro_fade = min(transition if transition > 0 else 1.0 / fps,
                         intro_duration / 2, run_lengths[0])
    intro_offset = intro_duration - intro_fade if intro_duration > 0 else 0.0
    total = intro_offset + body_duration

    intro_title = (story.title or "").strip()

    # --- the lower thirds -------------------------------------------------------
    strip_height = max(48, int(height * 0.16))
    intro_overlays: list[_TextOverlay] = []
    if intro_duration > 0 and intro_title:
        intro_overlays.append(_TextOverlay(
            text=intro_title,
            start=0.0,
            duration=intro_duration,
            width=int(width * 0.86) // 2 * 2,
            height=int(height * 0.36) // 2 * 2,
            y_expression="(H-h)/2-(H*0.04)",
            plate_alpha=0.0,
            fade_in=min(INTRO_TEXT_FADE_SECONDS, intro_duration / 3),
            fade_out=0.0,
            max_lines=3,
        ))

    title_overlays: list[_TextOverlay] = []
    title_duration = max(0.0, settings.title_seconds)
    if title_duration > 0:
        for index in changes:
            beat = timeline.clips[index].beat.strip()
            if not beat:
                continue
            at = starts[index]
            run = sum(1 for split in splits if split <= at + 1e-6)
            # A cut that became a dissolve reads at the middle of that dissolve;
            # one that stayed a hard cut reads where it always was. Both are
            # measured on the assembled body, which is shorter than the draft by
            # one transition per dissolve.
            body_at = sum(run_lengths[:run]) - run * transition
            if any(abs(split - at) < 1e-6 for split in splits):
                body_at += transition / 2
            else:
                body_at += at - (splits[run - 1] if run else 0.0)
            start = body_at + intro_offset
            if start < 0 or start + title_duration > total:
                continue
            title_overlays.append(_TextOverlay(
                text=beat,
                start=start,
                duration=title_duration,
                width=width,
                height=strip_height,
                y_expression=f"H-h-{max(8, int(height * 0.06))}",
                plate_alpha=0.55,
                fade_in=min(TITLE_FADE_SECONDS, title_duration / 3),
                fade_out=min(TITLE_FADE_SECONDS, title_duration / 3),
            ))

    # --- the music bed ----------------------------------------------------------
    music_dir = Path(settings.music_dir)
    if not music_dir.is_absolute():
        candidate = ctx.project_dir / music_dir
        music_dir = candidate if candidate.is_dir() else music_dir
    tracks = list_tracks(music_dir)
    track: Path | None = None
    if settings.music_track:
        track = music_dir / settings.music_track
        if not track.is_file():
            raise RuntimeError(
                f"polish.music_track {settings.music_track!r} is not in {music_dir}; "
                f"available: {', '.join(path.name for path in tracks) or 'nothing'}"
            )
    elif tracks:
        track = select_track(tracks, _project_style(ctx.project_dir), ctx.project_dir.name)
    else:
        notes.append(f"no music: {music_dir} holds no playable tracks")

    attribution = attribution_line(track) if track is not None else ""
    if attribution:
        write_attribution(ctx.output_dir, attribution)

    # --- the single ffmpeg invocation ------------------------------------------
    args: list[str] = []
    chains: list[str] = []
    next_input = 0
    overlay_order = 0

    intro_video_label = ""
    intro_audio_label = ""
    if intro_duration > 0:
        video_index, audio_index = next_input, next_input + 1
        next_input += 2
        args += [
            "-f", "lavfi", "-t", f"{intro_duration:.3f}",
            "-i", f"color=c={INTRO_BACKGROUND}:s={width}x{height}:r={fps}",
            "-f", "lavfi", "-t", f"{intro_duration:.3f}", "-i", SILENCE_SOURCE,
        ]
        bar_width = max(24, int(width * 0.16))
        bar_height = max(2, int(height / 180))
        # `iw`/`ih`, not `w`/`h`: inside drawbox those name the box being drawn.
        chains.append(
            f"[{video_index}:v]drawbox=x=(iw-{bar_width})/2:y=ih*0.66:"
            f"w={bar_width}:h={bar_height}:color={INTRO_ACCENT}@0.9:t=fill[card0]"
        )
        intro_video_label = "[card0]"
        # The card's text is composited onto the card, not onto the finished
        # picture: the dissolve into the first segment then carries the title away
        # with the background it sits on.
        for overlay in intro_overlays:
            overlay_args, chain = _overlay_inputs(
                overlay, overlay_order, work_dir, fps, use_drawtext
            )
            args += overlay_args
            chains.append(f"[{next_input}:v]{chain}[ct{overlay_order}]")
            chains.append(
                f"{intro_video_label}[ct{overlay_order}]overlay=x=(W-w)/2:"
                f"y={overlay.y_expression}:eof_action=pass[card{overlay_order + 1}]"
            )
            intro_video_label = f"[card{overlay_order + 1}]"
            next_input += 1
            overlay_order += 1
        chains.append(
            f"{intro_video_label}fps={fps},format=yuv420p,setsar=1[introv]"
        )
        intro_video_label = "[introv]"
        intro_audio_label = f"[{audio_index}:a]"

    run_video_labels: list[str] = []
    run_audio_labels: list[str] = []
    for position, run_source in enumerate(run_sources):
        args += ["-i", str(run_source)]
        chains.append(f"[{next_input}:v]fps={fps},format=yuv420p,setsar=1[rv{position}]")
        chains.append(
            f"[{next_input}:a]aformat=sample_rates={DRAFT_AUDIO_SAMPLE_RATE}:"
            f"channel_layouts={DRAFT_AUDIO_CHANNEL_LAYOUT}[ra{position}]"
        )
        run_video_labels.append(f"[rv{position}]")
        run_audio_labels.append(f"[ra{position}]")
        next_input += 1

    overlay_streams: list[tuple[_TextOverlay, str]] = []
    for overlay in title_overlays:
        overlay_args, chain = _overlay_inputs(
            overlay, overlay_order, work_dir, fps, use_drawtext
        )
        args += overlay_args
        label = f"[ov{overlay_order}]"
        chains.append(f"[{next_input}:v]{chain}{label}")
        overlay_streams.append((overlay, label))
        next_input += 1
        overlay_order += 1

    music_index: int | None = None
    if track is not None:
        args += ["-stream_loop", "-1", "-i", str(track)]
        music_index = next_input
        next_input += 1

    # Video: dissolve the runs together, dissolve the card in, then the overlays.
    body_video = run_video_labels[0]
    accumulated = run_lengths[0]
    for position in range(1, len(run_video_labels)):
        offset = accumulated - transition
        chains.append(
            f"{body_video}{run_video_labels[position]}xfade=transition=fade:"
            f"duration={transition:.3f}:offset={offset:.3f}[bv{position}]"
        )
        body_video = f"[bv{position}]"
        accumulated += run_lengths[position] - transition

    video_label = body_video
    if intro_video_label:
        chains.append(
            f"{intro_video_label}{video_label}xfade=transition=fade:"
            f"duration={intro_fade:.3f}:offset={intro_offset:.3f}[vintro]"
        )
        video_label = "[vintro]"

    for order, (overlay, label) in enumerate(overlay_streams):
        chains.append(
            f"{video_label}{label}overlay=x=(W-w)/2:y={overlay.y_expression}:"
            f"eof_action=pass[vt{order}]"
        )
        video_label = f"[vt{order}]"

    if intro_duration > 0:
        chains.append(f"{video_label}fade=t=in:st=0:d={INTRO_FADE_SECONDS}[vout]")
        video_label = "[vout]"

    # Audio: join the runs, prepend the card's silence, then duck the bed under it.
    body_audio = run_audio_labels[0]
    for position in range(1, len(run_audio_labels)):
        chains.append(
            f"{body_audio}{run_audio_labels[position]}acrossfade="
            f"d={transition:.3f}:c1=tri:c2=tri[ba{position}]"
        )
        body_audio = f"[ba{position}]"

    audio_label = body_audio
    if intro_audio_label:
        chains.append(
            f"{intro_audio_label}{audio_label}acrossfade="
            f"d={intro_fade:.3f}:c1=tri:c2=tri[aintro]"
        )
        audio_label = "[aintro]"

    if music_index is not None:
        fade = min(MUSIC_FADE_SECONDS, total / 4)
        chains.append(f"{audio_label}asplit=2[speechmix][speechkey]")
        chains.append(
            f"[{music_index}:a]atrim=0:{total:.3f},asetpts=N/SR/TB,"
            f"aformat=sample_rates={DRAFT_AUDIO_SAMPLE_RATE}:"
            f"channel_layouts={DRAFT_AUDIO_CHANNEL_LAYOUT},"
            f"volume={settings.music_gain_db}dB,"
            f"afade=t=in:st=0:d={fade:.3f},"
            f"afade=t=out:st={max(0.0, total - fade):.3f}:d={fade:.3f}[bed]"
        )
        chains.append(
            f"[bed][speechkey]sidechaincompress=threshold={SIDECHAIN_THRESHOLD}:"
            f"ratio={duck_ratio(settings.music_duck_db):.3f}:"
            f"attack={SIDECHAIN_ATTACK_MS}:release={SIDECHAIN_RELEASE_MS}:"
            "detection=rms[ducked]"
        )
        chains.append(
            "[speechmix][ducked]amix=inputs=2:normalize=0:duration=first[aout]"
        )
        audio_label = "[aout]"

    run_ffmpeg([
        *args,
        "-filter_complex", ";".join(chains),
        "-map", video_label, "-map", audio_label,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", str(FINAL_CRF),
        "-pix_fmt", "yuv420p",
        "-c:a", DRAFT_AUDIO_CODEC, "-b:a", DRAFT_AUDIO_BITRATE,
        "-ar", str(DRAFT_AUDIO_SAMPLE_RATE), "-ac", str(DRAFT_AUDIO_CHANNELS),
        "-movflags", "+faststart",
        str(output),
    ])

    return FinalResult(
        path=str(output),
        duration=probe(output).duration,
        intro=intro_duration > 0,
        intro_title=intro_title,
        title_count=len(title_overlays),
        transition_count=len(splits),
        music_track=track.name if track is not None else None,
        music_attribution=attribution,
        notes=notes,
    )
