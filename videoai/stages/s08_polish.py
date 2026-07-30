"""s08 polish: turn the reviewed cut into something a viewer would watch.

The draft is deliberately bare — it exists to check edit decisions, and it stays
exactly as it was: cut from the 540p proxies, fast, disposable. This stage builds
a second file beside it with the things a finished video has and an edit review
does not: a title card, a lower third when the story moves to a new section, a
music bed ducked under the child's voice, and a soft dissolve where one section
becomes the next.

It does not build any of that on top of the draft. The draft is a sixteenth of
the source's pixels, and a deliverable made from it would be a 540p video with
titles on it. The timeline's in and out points are exact and the manifest still
names the original files, so this stage cuts its own segments straight from those
originals at delivery resolution — nothing here is ever upscaled.

Assembly is a single ffmpeg invocation: the cross-dissolves, the title card, the
overlays and the music mix all happen in one filter graph, so the picture is
encoded once more rather than once per element. What cannot happen inside that
graph is the cutting itself — see `_cut_segment`.
"""
from __future__ import annotations

import shutil
import time
import json
import math
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import yaml

from videoai.core.ffmpeg import (
    ProbeResult,
    filter_available,
    h264_encode_args,
    hardware_decode_args,
    probe,
    run_ffmpeg,
    videotoolbox_available,
    _run_ffmpeg_to,
)
from videoai.core.models import (
    Approval,
    DraftResult,
    FinalResult,
    Manifest,
    StoryPlan,
    Timeline,
    TimelineClip,
    Transcript,
    Word,
)
from videoai.core.registry import StageContext, stage
from videoai.core.text import (
    drawtext_available,
    drawtext_filter,
    render_text_image,
    resolve_font,
    wrap_text,
)
from videoai.core.store import hash_file, hash_parts
from videoai.logic.music import attribution_line, list_tracks, select_track
from videoai.logic.contract import has_closing_beat, validate_production_report
from videoai.stages.s06_render_draft import (
    DRAFT_AUDIO_BITRATE,
    DRAFT_AUDIO_CHANNEL_LAYOUT,
    DRAFT_AUDIO_CHANNELS,
    DRAFT_AUDIO_CODEC,
    DRAFT_AUDIO_SAMPLE_RATE,
    SILENT_AUDIO_SOURCE,
    _audio_filter_chain,
)

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


@dataclass(frozen=True)
class _Caption:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class _CartoonEffect:
    """One deterministic, locally rendered animation on the delivery clock."""

    kind: str
    start: float
    duration: float
    seed: int = 0


_MAJOR_ACTIONS = (
    ("box", "OPENING THE BOX!", ("box", "unbox", "open")),
    ("injection", "INJECTION TIME!", ("inject", "shot", "syringe", "fill")),
    ("popping", "POPPING TIME!", ("pop",)),
)


def _action_for_beat(beat: str) -> tuple[str, str] | None:
    lowered = beat.strip().lower()
    for key, label, words in _MAJOR_ACTIONS:
        if any(word in lowered for word in words):
            return key, label
    return None


def build_major_action_titles(
    timeline: Timeline,
    starts: list[float],
    durations: list[float],
    intro_offset: float,
    title_seconds: float,
    width: int,
    height: int,
) -> list[_TextOverlay]:
    """Title only the first occurrence of a major viewer-facing action."""
    seen: set[str] = set()
    overlays: list[_TextOverlay] = []
    for index, clip in enumerate(timeline.clips):
        action = _action_for_beat(clip.beat)
        if action is None or action[0] in seen:
            continue
        seen.add(action[0])
        overlays.append(
            _TextOverlay(
                text=action[1],
                start=intro_offset + starts[index],
                duration=min(title_seconds, max(0.2, durations[index])),
                width=width,
                height=max(72, int(height * 0.14)),
                y_expression="",
                plate_alpha=0.62,
                fade_in=0,
                fade_out=0,
            )
        )
    return overlays


def lower_third_y(frame_height: int, overlay_height: int) -> int:
    """Place an overlay inside the bottom title-safe area."""
    return max(0, frame_height - overlay_height - int(frame_height * 0.06))


def build_cartoon_effects(
    timeline: Timeline,
    starts: list[float],
    durations: list[float],
    intro_offset: float,
    title_seconds: float,
) -> list[_CartoonEffect]:
    """Place a small number of story-aware effects without another model call."""
    effects: list[_CartoonEffect] = []
    action_indices: dict[str, list[int]] = {}
    for index, clip in enumerate(timeline.clips):
        action = _action_for_beat(clip.beat)
        if action is not None:
            action_indices.setdefault(action[0], []).append(index)

    def after_title(index: int) -> float:
        room = max(0.0, durations[index] - 0.8)
        return intro_offset + starts[index] + min(title_seconds + 0.15, room)

    if action_indices.get("box"):
        index = action_indices["box"][0]
        effects.append(_CartoonEffect("box_fly", after_title(index), 1.05))

    if action_indices.get("injection"):
        index = action_indices["injection"][0]
        close_up = next(
            (
                candidate
                for candidate in action_indices["injection"]
                if timeline.clips[candidate].is_insert
            ),
            None,
        )
        if close_up is not None:
            close_up_start = intro_offset + starts[close_up]
            effects.append(
                _CartoonEffect("injection_burst", close_up_start + 0.12, 0.9)
            )
            effects.append(
                _CartoonEffect(
                    "toy_reaction",
                    close_up_start + 1.12,
                    min(3.2, max(1.0, durations[close_up] - 1.35)),
                )
            )
        else:
            effects.append(_CartoonEffect("injection_burst", after_title(index), 0.9))

    pop_indices = action_indices.get("popping", [])
    pop_times: list[float] = []
    for index in pop_indices:
        first = intro_offset + starts[index] + min(0.45, durations[index] / 3)
        pop_times.append(first)
        extra = 5.5
        while extra < durations[index] - 0.5:
            pop_times.append(intro_offset + starts[index] + extra)
            extra += 6.0
    # Leave the title readable, then use no more than five accents across the
    # whole popping sequence.
    first_pop = pop_indices[0] if pop_indices else None
    minimum = after_title(first_pop) if first_pop is not None else 0.0
    selected: list[float] = []
    for at in sorted(pop_times):
        at = max(at, minimum)
        if selected and at - selected[-1] < 1.2:
            continue
        selected.append(at)
        if len(selected) == 5:
            break
    effects.extend(
        _CartoonEffect("pop", at, 0.62, seed=index)
        for index, at in enumerate(selected)
    )
    return effects


def _ass_time(seconds: float) -> str:
    centiseconds = max(0, round(seconds * 100))
    hours, rest = divmod(centiseconds, 360000)
    minutes, rest = divmod(rest, 6000)
    whole, fraction = divmod(rest, 100)
    return f"{hours}:{minutes:02d}:{whole:02d}.{fraction:02d}"


def _ass_text(text: str) -> str:
    return text.replace("\\", r"\\").replace("{", r"\{").replace("}", r"\}").replace("\n", r"\N")


def _caption_groups(words: list[Word], maximum: int) -> list[list[Word]]:
    groups: list[list[Word]] = []
    current: list[Word] = []
    maximum = max(1, maximum)
    for word in words:
        if current and (
            len(current) >= maximum
            or word.start - current[-1].end > 0.65
            or word.end - current[0].start > 2.4
        ):
            groups.append(current)
            current = []
        current.append(word)
    if current:
        groups.append(current)
    return groups


def build_captions(
    timeline: Timeline,
    transcript: Transcript,
    measured_starts: list[float],
    split_indices: list[int],
    transition: float,
    intro_offset: float,
    words_per_caption: int,
) -> list[_Caption]:
    """Map source word timestamps onto the assembled delivery timeline."""
    known = {clip.clip_id for clip in transcript.clips}
    captions: list[_Caption] = []
    for index, clip in enumerate(timeline.clips):
        if clip.is_insert or clip.src not in known:
            continue
        words = [
            word
            for word in transcript.by_id(clip.src).words
            if word.start >= clip.offset - 0.01
            and word.end <= clip.offset + clip.dur + 0.01
        ]
        if not words:
            continue
        prior_dissolves = sum(1 for split in split_indices if split <= index)
        base = measured_starts[index] - prior_dissolves * transition + intro_offset
        for group in _caption_groups(words, words_per_caption):
            start = base + group[0].start - clip.offset
            end = base + group[-1].end - clip.offset
            captions.append(
                _Caption(
                    start=max(0.0, start - 0.04),
                    end=max(start + 0.2, end + 0.08),
                    text=" ".join(word.text for word in group),
                )
            )
    return captions


def write_ass_captions(path: Path, captions: list[_Caption], width: int, height: int) -> None:
    """Write readable, YouTube-safe burned-in captions without another dependency."""
    font_size = max(28, round(height * 0.052))
    margin_v = max(36, round(height * 0.08))
    lines = [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {width}",
        f"PlayResY: {height}",
        "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, "
        "ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, "
        "MarginR, MarginV, Encoding",
        f"Style: Default,Arial Rounded MT Bold,{font_size},&H00FFFFFF,&H0000FFFF,"
        f"&H00101010,&H80000000,-1,0,0,0,100,100,0,0,1,4,1,2,60,60,{margin_v},1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    lines.extend(
        f"Dialogue: 0,{_ass_time(caption.start)},{_ass_time(caption.end)},"
        f"Default,,0,0,0,,{_ass_text(caption.text)}"
        for caption in captions
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_srt_captions(path: Path, captions: list[_Caption]) -> None:
    def timestamp(seconds: float) -> str:
        milliseconds = max(0, round(seconds * 1000))
        hours, rest = divmod(milliseconds, 3_600_000)
        minutes, rest = divmod(rest, 60_000)
        whole, fraction = divmod(rest, 1000)
        return f"{hours:02d}:{minutes:02d}:{whole:02d},{fraction:03d}"

    blocks = [
        f"{index}\n{timestamp(caption.start)} --> {timestamp(caption.end)}\n"
        f"{caption.text}\n"
        for index, caption in enumerate(captions, start=1)
    ]
    path.write_text("\n".join(blocks), encoding="utf-8")


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


def cumulative_starts(durations: list[float]) -> list[float]:
    """Where each segment begins in the concatenated body.

    Measured, not planned: a rendered segment is a whole number of frames, so it
    is a frame or so longer than the timeline asked for and the error accumulates
    across twenty-odd cuts — enough to put a lower third on the wrong side of a
    dissolve.
    """
    starts = [0.0]
    for duration in durations[:-1]:
        starts.append(starts[-1] + duration)
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


def _source_path(project_dir: Path, raw_path: str, clip_id: str) -> Path:
    """The original file behind a timeline clip, or a refusal naming it.

    The manifest stores the path ingest was handed, which is relative whenever the
    CLI was invoked with a relative project directory — so it is tried as recorded
    and then under the project folder, the same fallback the draft path uses.

    A miss is fatal on purpose. The proxy is sitting right there and using it
    would produce a final that renders, plays, and is quietly a sixteenth of the
    resolution the creator shot: exactly the defect this stage was rewritten to
    remove. Better to stop and name the file.
    """
    recorded = Path(raw_path)
    candidates = [recorded] if recorded.is_absolute() else [recorded, project_dir / recorded]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    looked = ", ".join(str(candidate) for candidate in candidates)
    raise RuntimeError(
        f"polish cuts the final from the original footage, but the source for "
        f"{clip_id} is missing: {raw_path} (looked in {looked}). Restore the file "
        f"or re-run ingest; the final is not rendered from the proxy."
    )


def delivery_frame(source: ProbeResult, output_height: int) -> tuple[int, int]:
    """The finished frame: `output_height` tall, at the source's display aspect.

    Display, not coded: a portrait iPhone clip is stored as landscape frames plus
    a quarter turn, so its recorded width and height describe a sideways picture
    and would hand the final a landscape frame to letterbox the upright footage
    into. Both dimensions are even, which yuv420p requires.
    """
    height = max(2, output_height // 2 * 2)
    ratio = source.display_width / max(1, source.display_height)
    width = max(2, round(ratio * height / 2) * 2)
    return width, height


def _cut_segment(
    source: Path, clip: TimelineClip, frame: tuple[int, int], fps: float, fade: float,
    has_audio: bool, crf: int, hardware: bool, dst: Path,
) -> None:
    """One timeline clip, cut from the original at the delivery frame.

    Rotation needs no filter of its own. ffmpeg autorotates on decode whenever the
    output goes through a filter chain, so a turned source arrives here already
    upright and `force_original_aspect_ratio` measures the picture a viewer would
    see (measured: a 640x480 clip carrying a 90-degree display matrix lands as
    180x240 through `scale=-2:240`, and as a sideways 320x240 only under
    `-noautorotate`).

    Every segment is normalised to the same frame, rate, pixel format and audio
    layout, which is what lets the runs below be concatenated with `-c copy` — and
    what keeps the final's audio identical to the draft's.
    """
    width, height = frame
    video_filter = (
        f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,"
        f"fps={fps},format=yuv420p,setsar=1"
    )
    video_args = h264_encode_args(crf, hardware)
    audio_args = [
        "-c:a", DRAFT_AUDIO_CODEC, "-b:a", DRAFT_AUDIO_BITRATE,
        "-ar", str(DRAFT_AUDIO_SAMPLE_RATE), "-ac", str(DRAFT_AUDIO_CHANNELS),
    ]
    # Input-side `-ss` so the decoder seeks rather than decoding from zero, which
    # on a three-minute 4K HEVC clip is the difference between seconds and minutes.
    seek = [*hardware_decode_args(), "-ss", f"{clip.offset:.3f}", "-i", str(source)]
    if has_audio:
        run_ffmpeg([
            *seek, "-t", f"{clip.dur:.3f}",
            "-filter:v", video_filter,
            *video_args, *audio_args,
            "-af", _audio_filter_chain(fade, clip.dur, clip.gain_db),
            "-avoid_negative_ts", "make_zero",
            str(dst),
        ])
    else:
        # A silent original still has to arrive with an audio track: the concat
        # below copies streams, and a video-only run would break its layout.
        run_ffmpeg([
            *seek,
            "-f", "lavfi", "-i", SILENT_AUDIO_SOURCE,
            "-t", f"{clip.dur:.3f}",
            "-map", "0:v:0", "-map", "1:a:0",
            "-filter:v", video_filter,
            *video_args, *audio_args,
            "-avoid_negative_ts", "make_zero",
            str(dst),
        ])


def _build_runs(
    segments: list[Path], splits: list[int], work_dir: Path
) -> list[Path]:
    """One file per stretch of segments between section dissolves.

    `splits` are segment indices where a run ends and the next begins. Segments
    are homogeneous by construction, so a run is a stream copy rather than a
    re-encode — the delivery picture is encoded exactly twice on its way out (once
    into the segments, once out of the assembly graph), never three times.
    """
    bounds = list(zip([0, *splits], [*splits, len(segments)]))
    paths: list[Path] = []
    for index, (first, last) in enumerate(bounds):
        members = segments[first:last]
        if len(members) == 1:
            paths.append(members[0])
            continue
        list_file = work_dir / f"run-{index:02d}.txt"
        list_file.write_text(
            "\n".join(f"file '{path.name}'" for path in members) + "\n", encoding="utf-8"
        )
        target = work_dir / f"run-{index:02d}.mp4"
        run_ffmpeg([
            "-f", "concat", "-safe", "0", "-i", str(list_file),
            "-c", "copy", str(target),
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


def _render_card(
    path: Path,
    title: str,
    subtitle: str,
    frame: tuple[int, int],
) -> None:
    """A bright but repeatable kids-channel card rendered entirely locally."""
    import cv2
    import numpy as np

    width, height = frame
    image = np.zeros((height, width, 3), dtype=np.uint8)
    top = np.array((58, 35, 122), dtype=np.float32)
    bottom = np.array((18, 102, 156), dtype=np.float32)
    for row in range(height):
        mix = row / max(1, height - 1)
        image[row, :, :] = top * (1 - mix) + bottom * mix
    accent = (74, 220, 255)
    cv2.circle(image, (int(width * 0.12), int(height * 0.18)), int(height * 0.12),
               accent, -1, cv2.LINE_AA)
    cv2.circle(image, (int(width * 0.9), int(height * 0.78)), int(height * 0.18),
               (247, 105, 92), -1, cv2.LINE_AA)
    cv2.rectangle(
        image,
        (int(width * 0.2), int(height * 0.69)),
        (int(width * 0.8), int(height * 0.705)),
        accent,
        -1,
    )
    rgba = path.with_suffix(".rgba.png")
    render_text_image(
        rgba, title, int(width * 0.82), int(height * 0.42),
        plate_alpha=0.0, max_lines=3,
    )
    overlay = cv2.imread(str(rgba), cv2.IMREAD_UNCHANGED)
    x = (width - overlay.shape[1]) // 2
    y = int(height * 0.20)
    alpha = overlay[:, :, 3:4].astype(np.float32) / 255.0
    image[y:y + overlay.shape[0], x:x + overlay.shape[1]] = (
        overlay[:, :, :3] * alpha
        + image[y:y + overlay.shape[0], x:x + overlay.shape[1]] * (1 - alpha)
    ).astype(np.uint8)
    if subtitle.strip():
        sub = path.with_suffix(".subtitle.png")
        render_text_image(
            sub, subtitle, int(width * 0.72), int(height * 0.13),
            plate_alpha=0.0, max_lines=2,
        )
        layer = cv2.imread(str(sub), cv2.IMREAD_UNCHANGED)
        sx = (width - layer.shape[1]) // 2
        sy = int(height * 0.74)
        alpha = layer[:, :, 3:4].astype(np.float32) / 255.0
        image[sy:sy + layer.shape[0], sx:sx + layer.shape[1]] = (
            layer[:, :, :3] * alpha
            + image[sy:sy + layer.shape[0], sx:sx + layer.shape[1]] * (1 - alpha)
        ).astype(np.uint8)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"could not render card image: {path}")


def _card_segment(
    image: Path,
    duration: float,
    frame: tuple[int, int],
    fps: float,
    dst: Path,
) -> None:
    width, height = frame
    fade = min(0.4, duration / 4)
    _run_ffmpeg_to(
        [
            "-loop", "1", "-framerate", f"{fps}", "-t", f"{duration:.3f}",
            "-i", str(image),
            "-f", "lavfi", "-t", f"{duration:.3f}", "-i", SILENCE_SOURCE,
            "-map", "0:v:0", "-map", "1:a:0",
            "-vf",
            f"scale={width}:{height},format=yuv420p,"
            f"fade=t=in:st=0:d={fade:.3f},"
            f"fade=t=out:st={max(0.0, duration - fade):.3f}:d={fade:.3f}",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "0",
            "-c:a", DRAFT_AUDIO_CODEC, "-b:a", DRAFT_AUDIO_BITRATE,
            "-ar", str(DRAFT_AUDIO_SAMPLE_RATE), "-ac", str(DRAFT_AUDIO_CHANNELS),
        ],
        dst,
    )


def _cut_delivery_segment(
    source: Path,
    clip: TimelineClip,
    frame: tuple[int, int],
    fps: float,
    audio_fade: float,
    has_audio: bool,
    transition: float,
    fade_in: bool,
    fade_out: bool,
    dst: Path,
) -> None:
    width, height = frame
    filters = [
        f"scale={width}:{height}:force_original_aspect_ratio=decrease",
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2",
        f"fps={fps}",
        "format=yuv420p",
        "setsar=1",
    ]
    duration = clip.dur
    local_transition = min(transition, duration / 3)
    if fade_in and local_transition > 0:
        filters.append(f"fade=t=in:st=0:d={local_transition:.3f}")
    if fade_out and local_transition > 0:
        filters.append(
            f"fade=t=out:st={max(0.0, duration - local_transition):.3f}:"
            f"d={local_transition:.3f}"
        )
    seek = ["-ss", f"{clip.offset:.3f}", "-i", str(source)]
    common = [
        "-t", f"{duration:.3f}", "-filter:v", ",".join(filters),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "0",
        "-c:a", DRAFT_AUDIO_CODEC, "-b:a", DRAFT_AUDIO_BITRATE,
        "-ar", str(DRAFT_AUDIO_SAMPLE_RATE), "-ac", str(DRAFT_AUDIO_CHANNELS),
        "-avoid_negative_ts", "make_zero",
    ]
    if has_audio:
        _run_ffmpeg_to(
            [
                *seek, *common,
                "-af", _audio_filter_chain(audio_fade, duration, clip.gain_db),
            ],
            dst,
        )
    else:
        _run_ffmpeg_to(
            [
                *seek,
                "-f", "lavfi", "-i", SILENT_AUDIO_SOURCE,
                "-map", "0:v:0", "-map", "1:a:0",
                *common,
            ],
            dst,
        )


def _concat_copy(parts: list[Path], work_dir: Path, dst: Path) -> None:
    listing = work_dir / f"{dst.stem}-concat.txt"
    listing.write_text(
        "\n".join(f"file '{part.name}'" for part in parts) + "\n",
        encoding="utf-8",
    )
    _run_ffmpeg_to(
        ["-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy"],
        dst,
    )


def _draw_centered_text(
    canvas, text: str, center: tuple[int, int], scale: float, color: tuple[int, int, int, int],
    thickness: int,
) -> None:
    import cv2

    size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, scale, thickness)
    origin = (center[0] - size[0] // 2, center[1] + size[1] // 2)
    cv2.putText(
        canvas, text, origin, cv2.FONT_HERSHEY_DUPLEX, scale, color,
        thickness, cv2.LINE_AA,
    )


def _draw_cartoon_effect(canvas, effect: _CartoonEffect, at: float) -> None:
    """Draw one frame of a bright comic effect into an RGBA canvas."""
    import cv2
    import numpy as np

    progress = min(1.0, max(0.0, (at - effect.start) / effect.duration))
    width = canvas.shape[1]
    height = canvas.shape[0]
    fade = min(1.0, progress / 0.12, (1.0 - progress) / 0.18)
    alpha = max(0, min(255, round(255 * fade)))
    if alpha <= 0:
        return

    if effect.kind == "box_fly":
        travel = 1.0 - (1.0 - progress) ** 3
        box_w = int(width * 0.13)
        box_h = int(height * 0.13)
        x = int(-box_w + travel * width * 0.84)
        y = int(height * 0.47 + math.sin(progress * math.pi * 2) * height * 0.035)
        for offset in (0, int(width * 0.055), int(width * 0.11)):
            cv2.line(
                canvas,
                (max(0, x - offset - int(width * 0.13)), y + box_h // 2),
                (max(0, x - offset - int(width * 0.025)), y + box_h // 2),
                (255, 220, 70, alpha),
                max(3, height // 120),
                cv2.LINE_AA,
            )
        cv2.rectangle(
            canvas, (x, y), (x + box_w, y + box_h),
            (225, 157, 62, alpha), -1, cv2.LINE_AA,
        )
        cv2.line(
            canvas, (x + box_w // 2, y), (x + box_w // 2, y + box_h),
            (120, 72, 24, alpha), max(2, height // 180), cv2.LINE_AA,
        )
        cv2.line(
            canvas, (x, y + box_h // 2), (x + box_w, y + box_h // 2),
            (120, 72, 24, alpha), max(2, height // 180), cv2.LINE_AA,
        )
        _draw_centered_text(
            canvas, "WHOOSH!", (int(width * 0.72), int(height * 0.40)),
            max(0.8, height / 720), (255, 244, 72, alpha),
            max(2, height // 240),
        )
        return

    if effect.kind == "injection_burst":
        # Keep the burst around the toy/work surface rather than across the
        # child's face or torso. The preferred event is a toy-only close-up.
        center = (int(width * 0.52), int(height * 0.67))
        radius = int(height * (0.06 + 0.18 * math.sin(progress * math.pi)))
        for ray in range(12):
            angle = ray * math.pi / 6
            inner = radius * 0.58
            outer = radius * (1.0 if ray % 2 else 1.22)
            cv2.line(
                canvas,
                (
                    center[0] + int(math.cos(angle) * inner),
                    center[1] + int(math.sin(angle) * inner),
                ),
                (
                    center[0] + int(math.cos(angle) * outer),
                    center[1] + int(math.sin(angle) * outer),
                ),
                (255, 226, 55, alpha),
                max(3, height // 135),
                cv2.LINE_AA,
            )
        # A simple comic syringe entering the burst diagonally.
        start = (center[0] + int(width * 0.16), center[1] - int(height * 0.18))
        end = (center[0] + int(width * 0.025), center[1] - int(height * 0.02))
        cv2.line(canvas, start, end, (85, 235, 255, alpha), max(10, height // 65), cv2.LINE_AA)
        cv2.line(canvas, end, center, (245, 245, 255, alpha), max(2, height // 240), cv2.LINE_AA)
        _draw_centered_text(
            canvas, "ZAP!", (center[0], center[1]),
            max(0.85, height / 650), (255, 255, 255, alpha),
            max(2, height // 220),
        )
        return

    if effect.kind == "toy_reaction":
        # The toy's face is partly covered by translucent bubbles, so fixed
        # synthetic eyes/mouth cannot land reliably. Animate the speech bubble
        # instead and point it at the real face.
        bob = int(math.sin(progress * math.pi * 6) * height * 0.004)
        bubble = (int(width * 0.68), int(height * 0.27) + bob)
        pop_scale = min(1.0, max(0.72, progress / 0.16))
        axes_bubble = (
            int(width * 0.22 * pop_scale),
            int(height * 0.105 * pop_scale),
        )
        cv2.ellipse(canvas, bubble, axes_bubble, 0, 0, 360, (255, 255, 255, alpha), -1, cv2.LINE_AA)
        cv2.ellipse(
            canvas, bubble, axes_bubble, 0, 0, 360,
            (35, 25, 45, alpha), max(3, height // 220), cv2.LINE_AA,
        )
        cv2.fillConvexPoly(
            canvas,
            np.array(
                [
                    (int(width * 0.52), int(height * 0.33) + bob),
                    (int(width * 0.43), int(height * 0.45)),
                    (int(width * 0.57), int(height * 0.37) + bob),
                ],
                dtype="int32",
            ),
            (255, 255, 255, alpha),
            cv2.LINE_AA,
        )
        _draw_centered_text(
            canvas, "OH NO! NO!", bubble, max(0.75, height / 760),
            (35, 25, 45, alpha), max(2, height // 260),
        )
        return

    if effect.kind == "pop":
        positions = (
            (0.54, 0.58), (0.43, 0.64), (0.61, 0.52), (0.48, 0.49), (0.58, 0.67),
        )
        px, py = positions[effect.seed % len(positions)]
        center = (int(width * px), int(height * py))
        radius = int(height * (0.025 + progress * 0.13))
        colors = ((255, 225, 40, alpha), (80, 235, 255, alpha), (255, 85, 185, alpha))
        for ring, color in enumerate(colors):
            cv2.circle(
                canvas, center, max(4, radius - ring * int(height * 0.022)),
                color, max(3, height // 150), cv2.LINE_AA,
            )
        for ray in range(8):
            angle = ray * math.pi / 4
            inner = radius * 1.05
            outer = radius * 1.38
            cv2.line(
                canvas,
                (center[0] + int(math.cos(angle) * inner), center[1] + int(math.sin(angle) * inner)),
                (center[0] + int(math.cos(angle) * outer), center[1] + int(math.sin(angle) * outer)),
                colors[ray % len(colors)], max(3, height // 160), cv2.LINE_AA,
            )
        if progress < 0.72:
            _draw_centered_text(
                canvas, "POP!", center, max(0.8, height / 700),
                (255, 255, 255, alpha), max(2, height // 220),
            )


def _render_graphics_track(
    path: Path,
    frame: tuple[int, int],
    fps: float,
    duration: float,
    captions: list[_Caption],
    titles: list[_TextOverlay],
    effects: list[_CartoonEffect],
    work_dir: Path,
) -> None:
    """One finite alpha video containing every title and caption.

    The previous implementation opened one infinite-loop image input per title
    inside the final graph. On a three-minute video ffmpeg could deadlock after
    encoding the whole picture. This renderer has one stdin and an exact frame
    count, so it must terminate.
    """
    import cv2
    import numpy as np

    width, height = frame
    assets: list[tuple[float, float, int, np.ndarray]] = []
    for index, title in enumerate(titles):
        image_path = work_dir / f"section-title-{index:02d}.png"
        render_text_image(
            image_path, title.text, int(width * 0.74), max(72, int(height * 0.14)),
            plate_alpha=0.62, max_lines=2,
        )
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        # Major-action labels are lower thirds. Keeping them below the child's
        # face matters more than reserving a separate caption lane because
        # captions are delivered as a viewer-controlled sidecar by default.
        # The six-percent bottom margin keeps the complete plate inside the
        # title-safe area on 16:9 delivery frames.
        title_y = lower_third_y(height, image.shape[0])
        assets.append((title.start, title.start + title.duration, title_y, image))
    caption_y = int(height * 0.80)
    for index, caption in enumerate(captions):
        image_path = work_dir / f"caption-{index:03d}.png"
        render_text_image(
            image_path, caption.text, int(width * 0.78), max(80, int(height * 0.12)),
            plate_alpha=0.72, max_lines=2,
        )
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        assets.append((caption.start, caption.end, caption_y, image))

    command = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgba",
        "-s", f"{width}x{height}", "-r", f"{fps}", "-i", "-",
        "-an", "-c:v", "qtrle", "-pix_fmt", "argb", str(path),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert process.stdin is not None
    frame_count = max(1, round(duration * fps))
    try:
        for number in range(frame_count):
            at = number / fps
            canvas = np.zeros((height, width, 4), dtype=np.uint8)
            for effect in effects:
                if effect.start <= at < effect.start + effect.duration:
                    _draw_cartoon_effect(canvas, effect, at)
            for start, end, y, image in assets:
                if not (start <= at < end):
                    continue
                x = (width - image.shape[1]) // 2
                bottom = min(height, y + image.shape[0])
                right = min(width, x + image.shape[1])
                canvas[y:bottom, x:right] = image[:bottom - y, :right - x]
            process.stdin.write(canvas.tobytes())
        process.stdin.close()
        stderr = process.stderr.read().decode("utf-8", errors="replace")
        returncode = process.wait()
    except BaseException:
        process.kill()
        process.wait()
        path.unlink(missing_ok=True)
        raise
    if returncode != 0:
        path.unlink(missing_ok=True)
        raise RuntimeError(f"graphics-track render failed: {stderr[-1000:]}")


def _render_music_bed(
    track: Path,
    duration: float,
    gain_db: float,
    dst: Path,
) -> None:
    fade = min(MUSIC_FADE_SECONDS, duration / 4)
    _run_ffmpeg_to(
        [
            "-stream_loop", "-1", "-i", str(track),
            "-t", f"{duration:.3f}",
            "-af",
            f"aformat=sample_rates={DRAFT_AUDIO_SAMPLE_RATE}:"
            f"channel_layouts={DRAFT_AUDIO_CHANNEL_LAYOUT},"
            f"volume={gain_db}dB,"
            f"afade=t=in:st=0:d={fade:.3f},"
            f"afade=t=out:st={max(0.0, duration - fade):.3f}:d={fade:.3f}",
            "-c:a", "pcm_s16le",
        ],
        dst,
    )


def _mix_delivery_audio(
    picture: Path,
    bed: Path,
    duration: float,
    duck_db: float,
    dst: Path,
) -> None:
    _run_ffmpeg_to(
        [
            "-i", str(picture), "-i", str(bed),
            "-filter_complex",
            f"[0:a]asplit=2[speech][key];"
            f"[1:a][key]sidechaincompress=threshold={SIDECHAIN_THRESHOLD}:"
            f"ratio={duck_ratio(duck_db):.3f}:attack={SIDECHAIN_ATTACK_MS}:"
            f"release={SIDECHAIN_RELEASE_MS}:detection=rms[ducked];"
            f"[speech][ducked]amix=inputs=2:normalize=0:duration=first[a]",
            "-map", "[a]", "-t", f"{duration:.3f}", "-c:a", "pcm_s16le",
        ],
        dst,
    )


def _full_decode(path: Path) -> None:
    result = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "null", "-"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"final failed full decode: {result.stderr.strip()[-1000:]}")


def _approval_is_current(ctx: StageContext, draft: DraftResult) -> None:
    from videoai.core.models import Approval

    if not ctx.store.exists("06-approval"):
        raise RuntimeError(
            "delivery requires review: watch output/draft.mp4, then run "
            f"'videoai approve {ctx.project_dir}'"
        )
    approval = ctx.store.read("06-approval", Approval)
    timeline_hash = ctx.store.content_hash("05-timeline") or ""
    draft_path = Path(draft.path)
    current_draft_hash = hash_file(draft_path) if draft_path.is_file() else ""
    config_hash = hash_parts(ctx.config.model_dump_json())
    mismatches: list[str] = []
    if approval.timeline_hash != timeline_hash or draft.timeline_hash != timeline_hash:
        mismatches.append("timeline")
    if approval.draft_hash != current_draft_hash:
        mismatches.append("draft")
    if approval.config_hash != config_hash:
        mismatches.append("config")
    if mismatches:
        raise RuntimeError(
            "approval is stale for: " + ", ".join(mismatches)
            + f"; rebuild/review the draft and run 'videoai approve {ctx.project_dir} "
              "--config <the same config>' again"
        )


def _polish_multiphase(ctx: StageContext) -> FinalResult:
    started = time.monotonic()
    settings = ctx.config.polish
    draft = ctx.store.read("06-draft", DraftResult)
    ctx.output_dir.mkdir(parents=True, exist_ok=True)
    output = ctx.output_dir / "final.mp4"

    if not settings.enabled:
        source = _draft_path(ctx, draft)
        shutil.copyfile(source, output)
        copied = probe(output)
        return FinalResult(
            path=str(output), duration=copied.duration,
            width=copied.width, height=copied.height,
            render_seconds=time.monotonic() - started,
            notes=["polish.enabled is false: final.mp4 is a copy of the draft"],
        )
    if settings.require_approval:
        _approval_is_current(ctx, draft)

    manifest = ctx.store.read("01-manifest", Manifest)
    timeline = ctx.store.read("05-timeline", Timeline)
    transcript = (
        ctx.store.read("03-transcript", Transcript)
        if ctx.store.exists("03-transcript")
        else Transcript(provider="", clips=[])
    )
    story = (
        ctx.store.read("05a-storyplan", StoryPlan)
        if ctx.store.exists("05a-storyplan")
        else StoryPlan()
    )
    if not timeline.clips:
        raise RuntimeError("cannot polish an empty timeline")

    work_dir = ctx.work_dir / "delivery"
    work_dir.mkdir(parents=True, exist_ok=True)
    sources = [
        _source_path(ctx.project_dir, manifest.by_id(clip.src).path, clip.src)
        for clip in timeline.clips
    ]
    frame = delivery_frame(probe(sources[0]), settings.output_height)
    width, height = frame
    fps = timeline.fps or manifest.by_id(timeline.clips[0].src).fps
    transition = max(0.0, settings.transition_frames / fps)

    segments: list[Path] = []
    changes = set(section_changes(timeline))
    for index, (clip, source) in enumerate(zip(timeline.clips, sources)):
        target = work_dir / f"segment-{index:03d}.mp4"
        _cut_delivery_segment(
            source, clip, frame, fps, ctx.config.render.audio_fade_seconds,
            manifest.by_id(clip.src).has_audio,
            transition,
            fade_in=index == 0 or index in changes,
            fade_out=index == len(timeline.clips) - 1 or index + 1 in changes,
            dst=target,
        )
        segments.append(target)
    segment_durations = [probe(path).duration for path in segments]
    starts = cumulative_starts(segment_durations)

    intro_duration = max(0.0, settings.intro_seconds)
    outro_duration = max(0.0, settings.outro_seconds)
    intro_path = work_dir / "intro.mp4"
    outro_path = work_dir / "outro.mp4"
    intro_image = work_dir / "intro.png"
    outro_image = work_dir / "outro.png"
    if intro_duration <= 0 or not (story.title or "").strip():
        raise RuntimeError("production contract requires a non-empty intro")
    if outro_duration <= 0 or not settings.outro_text.strip():
        raise RuntimeError("production contract requires a non-empty outro")
    _render_card(intro_image, story.title.strip(), "Toy review", frame)
    _render_card(outro_image, settings.outro_text.strip(), "See you next time!", frame)
    _card_segment(intro_image, intro_duration, frame, fps, intro_path)
    _card_segment(outro_image, outro_duration, frame, fps, outro_path)

    picture = work_dir / "picture-master.mp4"
    _concat_copy([intro_path, *segments, outro_path], work_dir, picture)
    measured_picture = probe(picture)
    total = measured_picture.duration

    captions = build_captions(
        timeline, transcript, starts, [], 0.0, probe(intro_path).duration,
        settings.caption_words,
    )
    if settings.captions_enabled and not captions:
        raise RuntimeError("production contract requires captions, but none were generated")
    srt_path = ctx.output_dir / "final.srt"
    write_srt_captions(srt_path, captions)

    intro_offset = probe(intro_path).duration
    title_overlays = build_major_action_titles(
        timeline, starts, segment_durations, intro_offset,
        settings.title_seconds, width, height,
    )
    if not title_overlays:
        raise RuntimeError("production contract requires section titles")
    cartoon_effects = build_cartoon_effects(
        timeline, starts, segment_durations, intro_offset, settings.title_seconds,
    )

    graphics = work_dir / "graphics.mov"
    _render_graphics_track(
        graphics, frame, fps, total,
        captions if settings.burn_captions else [],
        title_overlays, cartoon_effects, work_dir,
    )

    music_dir = Path(settings.music_dir)
    if not music_dir.is_absolute():
        project_candidate = ctx.project_dir / music_dir
        music_dir = project_candidate if project_candidate.is_dir() else music_dir
    tracks = list_tracks(music_dir)
    if settings.music_track:
        track = music_dir / settings.music_track
        if not track.is_file():
            raise RuntimeError(f"required music track is missing: {track}")
    else:
        track = select_track(
            tracks, _project_style(ctx.project_dir), ctx.project_dir.name
        ) if tracks else None
    if track is None:
        raise RuntimeError(f"production contract requires music; none found in {music_dir}")
    bed = work_dir / "music-bed.wav"
    mixed = work_dir / "mixed-audio.wav"
    _render_music_bed(track, total, settings.music_gain_db, bed)
    _mix_delivery_audio(picture, bed, total, settings.music_duck_db, mixed)

    hardware = settings.hardware_encode and videotoolbox_available()
    _run_ffmpeg_to(
        [
            "-i", str(picture), "-i", str(graphics), "-i", str(mixed),
            "-filter_complex", "[0:v][1:v]overlay=eof_action=pass:shortest=1[v]",
            "-map", "[v]", "-map", "2:a:0", "-t", f"{total:.3f}",
            *h264_encode_args(settings.output_crf, hardware),
            "-pix_fmt", "yuv420p",
            "-c:a", DRAFT_AUDIO_CODEC, "-b:a", DRAFT_AUDIO_BITRATE,
            "-ar", str(DRAFT_AUDIO_SAMPLE_RATE), "-ac", str(DRAFT_AUDIO_CHANNELS),
            "-movflags", "+faststart",
        ],
        output,
    )
    _full_decode(output)
    measured = probe(output)
    attribution = attribution_line(track)
    if attribution:
        write_attribution(ctx.output_dir, attribution)

    report_path = ctx.output_dir / "production-report.json"
    report = {
        "contract_version": 1,
        "status": "passed",
        "output": str(output),
        "duration": measured.duration,
        "width": measured.width,
        "height": measured.height,
        "features": {
            "intro": True,
            "outro": True,
            "section_titles": len(title_overlays),
            "captions": len(captions),
            "soft_captions": len(captions),
            "burned_in_captions": settings.burn_captions,
            "music": track.name,
            "music_ducking": True,
            "transitions": len(changes),
            "cartoon_animations": len(cartoon_effects),
            "toy_reaction": any(effect.kind == "toy_reaction" for effect in cartoon_effects),
            "closing_beat": has_closing_beat(timeline),
            "full_decode": True,
        },
        "quality": {
            "source": "originals",
            "lossless_intermediates": True,
            "lossy_video_generations": 1,
        },
    }
    try:
        validate_production_report(report, ctx.project_dir)
    except RuntimeError:
        output.unlink(missing_ok=True)
        raise
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    elapsed = time.monotonic() - started
    return FinalResult(
        path=str(output),
        duration=measured.duration,
        width=measured.width,
        height=measured.height,
        render_seconds=elapsed,
        intro=True,
        intro_title=story.title.strip(),
        title_count=len(title_overlays),
        transition_count=len(changes),
        caption_count=len(captions),
        burned_in_captions=settings.burn_captions,
        cartoon_effect_count=len(cartoon_effects),
        toy_reaction=any(effect.kind == "toy_reaction" for effect in cartoon_effects),
        outro=True,
        music_track=track.name,
        music_attribution=attribution,
        music_ducking=True,
        fully_decoded=True,
        production_report=str(report_path),
        notes=[
            "delivery used finite multi-pass rendering",
            "source segments and picture master are lossless x264",
            "final.mp4 is the only lossy video generation",
            "major-action titles only; cartoon effects rendered locally",
            (
                "captions burned into picture"
                if settings.burn_captions
                else "captions delivered as viewer-controlled final.srt"
            ),
        ],
    )


@stage(
    id="polish",
    produces="08-final",
    # "01-manifest" names the original files this stage cuts from. "06-draft" is
    # required for its ordering rather than its pixels: the draft is the review
    # the final is only worth building after, and it is what `polish.enabled:
    # false` copies through.
    requires=("01-manifest", "03-transcript", "05-timeline", "05a-storyplan", "06-draft"),
    version="4",
    model=FinalResult,
    config_keys=(
        "polish.enabled",
        "polish.strict_contract",
        "polish.require_approval",
        "polish.intro_seconds",
        "polish.outro_seconds",
        "polish.outro_text",
        "polish.title_seconds",
        "polish.captions_enabled",
        "polish.burn_captions",
        "polish.caption_words",
        "polish.music_gain_db",
        "polish.music_duck_db",
        "polish.transition_frames",
        "polish.music_track",
        "polish.music_dir",
        "polish.output_height",
        "polish.output_crf",
        "polish.lossless_intermediates",
        "polish.hardware_encode",
        # Read, not inherited: the final does its own cutting now, and the fade at
        # each cut has to be the one the draft was reviewed with.
        "render.audio_fade_seconds",
    ),
)
def polish(ctx: StageContext) -> FinalResult:
    if ctx.config.polish.strict_contract:
        return _polish_multiphase(ctx)

    # Legacy monolithic implementation retained temporarily below as a readable
    # reference while the finite multi-pass renderer is validated against the
    # existing test suite. It is intentionally unreachable.
    started = time.monotonic()
    settings = ctx.config.polish
    draft = ctx.store.read("06-draft", DraftResult)
    ctx.output_dir.mkdir(parents=True, exist_ok=True)
    output = ctx.output_dir / "final.mp4"

    if not settings.enabled:
        # The draft is copied rather than re-encoded: "polish off" must cost the
        # picture nothing, and a re-encode is not nothing. This is the one path
        # that ships proxy resolution, and it says so.
        source = _draft_path(ctx, draft)
        shutil.copyfile(source, output)
        copied = probe(output)
        return FinalResult(
            path=str(output),
            duration=copied.duration,
            width=copied.width,
            height=copied.height,
            render_seconds=time.monotonic() - started,
            notes=["polish.enabled is false: final.mp4 is a copy of the draft"],
        )

    manifest = ctx.store.read("01-manifest", Manifest)
    timeline = ctx.store.read("05-timeline", Timeline)
    transcript = (
        ctx.store.read("03-transcript", Transcript)
        if ctx.store.exists("03-transcript")
        else Transcript(provider="", clips=[])
    )
    if not timeline.clips:
        raise RuntimeError("cannot polish an empty timeline")
    if settings.require_approval:
        if not ctx.store.exists("06-approval"):
            raise RuntimeError(
                "delivery requires review: watch output/draft.mp4, then run "
                f"'videoai approve {ctx.project_dir}'"
            )
        approval = ctx.store.read("06-approval", Approval)
        current_timeline_hash = ctx.store.content_hash("05-timeline")
        if approval.timeline_hash != current_timeline_hash:
            raise RuntimeError(
                "the timeline changed after approval; review the new output/draft.mp4 "
                f"and run 'videoai approve {ctx.project_dir}' again"
            )
    story = (
        ctx.store.read("05a-storyplan", StoryPlan)
        if ctx.store.exists("05a-storyplan")
        else StoryPlan()
    )

    fps = timeline.fps if timeline.fps > 0 else manifest.by_id(timeline.clips[0].src).fps
    work_dir = ctx.work_dir / "polish"
    work_dir.mkdir(parents=True, exist_ok=True)
    notes: list[str] = []

    # --- the cut, at delivery quality, from the originals -----------------------
    sources = [
        _source_path(ctx.project_dir, manifest.by_id(clip.src).path, clip.src)
        for clip in timeline.clips
    ]
    frame = delivery_frame(probe(sources[0]), settings.output_height)
    width, height = frame
    hardware = settings.hardware_encode and videotoolbox_available()
    if settings.hardware_encode and not hardware:
        notes.append(
            "polish.hardware_encode is on but this ffmpeg build has no VideoToolbox "
            "encoder; the final was encoded with libx264"
        )
    # Lossless x264 intermediates make the final composition the only lossy
    # encode. VideoToolbox has no equivalent CRF 0 mode, so segment cutting is
    # intentionally software-only in this mode; the final encode may still use
    # the media engine.
    segment_crf = 0 if settings.lossless_intermediates else max(0, settings.output_crf - 4)
    segment_hardware = hardware and not settings.lossless_intermediates
    fade = ctx.config.render.audio_fade_seconds

    segments: list[Path] = []
    for index, (clip, clip_source) in enumerate(zip(timeline.clips, sources)):
        target = work_dir / f"seg-{index:03d}.mp4"
        _cut_segment(
            clip_source, clip, frame, fps, fade,
            manifest.by_id(clip.src).has_audio, segment_crf, segment_hardware, target,
        )
        segments.append(target)
    segment_durations = [probe(path).duration for path in segments]

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
    starts = cumulative_starts(segment_durations)
    body_total = sum(segment_durations)
    changes = section_changes(timeline)

    # A dissolve lands on a segment boundary, so it is recorded as the index of
    # the segment that opens the next run and as the time that index sits at.
    split_indices: list[int] = []
    splits: list[float] = []
    if transition > 0:
        minimum = max(transition * MIN_RUN_FACTOR, 2.0 / fps)
        previous = 0.0
        for index in changes:
            at = starts[index]
            if at - previous >= minimum and body_total - at >= minimum:
                split_indices.append(index)
                splits.append(at)
                previous = at
    run_sources = _build_runs(segments, split_indices, work_dir)
    run_lengths = [probe(path).duration for path in run_sources]
    body_duration = sum(run_lengths) - len(splits) * transition

    # --- the intro card ---------------------------------------------------------
    intro_duration = max(0.0, settings.intro_seconds)
    intro_fade = 0.0
    if intro_duration > 0:
        intro_fade = min(transition if transition > 0 else 1.0 / fps,
                         intro_duration / 2, run_lengths[0])
    intro_offset = intro_duration - intro_fade if intro_duration > 0 else 0.0
    total = intro_offset + body_duration

    # --- captions ---------------------------------------------------------------
    captions: list[_Caption] = []
    caption_file: Path | None = None
    if settings.captions_enabled:
        if filter_available("subtitles"):
            captions = build_captions(
                timeline, transcript, starts, split_indices, transition,
                intro_offset, settings.caption_words,
            )
            if captions:
                caption_file = work_dir / "captions.ass"
                write_ass_captions(caption_file, captions, width, height)
        else:
            notes.append(
                "captions requested but this ffmpeg build has no subtitles/libass filter"
            )

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
            # Measured on the assembled body, which is shorter than the cut by one
            # transition per dissolve. A cut that became a dissolve reads at the
            # middle of it; one that stayed a hard cut reads where it always was.
            run = sum(1 for split in split_indices if split <= index)
            body_at = starts[index] - run * transition
            if index in split_indices:
                body_at += transition / 2
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

    if caption_file is not None:
        escaped_caption_path = (
            str(caption_file).replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
        )
        chains.append(
            f"{video_label}subtitles=filename='{escaped_caption_path}'[vcaptions]"
        )
        video_label = "[vcaptions]"

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
        *h264_encode_args(settings.output_crf, hardware),
        "-pix_fmt", "yuv420p",
        "-c:a", DRAFT_AUDIO_CODEC, "-b:a", DRAFT_AUDIO_BITRATE,
        "-ar", str(DRAFT_AUDIO_SAMPLE_RATE), "-ac", str(DRAFT_AUDIO_CHANNELS),
        "-movflags", "+faststart",
        str(output),
    ])

    measured = probe(output)
    elapsed = time.monotonic() - started
    encoder = "h264_videotoolbox" if hardware else "libx264"
    notes.append(
        f"cut {len(segments)} segments from the original sources at {width}x{height} "
        f"(crf {settings.output_crf}, {encoder}) in {elapsed:.1f}s"
    )
    if settings.lossless_intermediates:
        notes.append(
            "delivery segments used lossless x264 intermediates; final.mp4 is the "
            "only lossy video generation"
        )

    return FinalResult(
        path=str(output),
        duration=measured.duration,
        width=measured.width,
        height=measured.height,
        render_seconds=elapsed,
        intro=intro_duration > 0,
        intro_title=intro_title,
        title_count=len(title_overlays),
        transition_count=len(splits),
        caption_count=len(captions),
        music_track=track.name if track is not None else None,
        music_attribution=attribution,
        notes=notes,
    )
