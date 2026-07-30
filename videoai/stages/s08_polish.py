"""s08 polish: turn the reviewed cut into the delivered video.

The draft is deliberately bare — it exists to check edit decisions, and it stays
exactly as it was: cut from the 540p proxies, fast, disposable. This stage builds
a second file beside it with the things a finished video has and an edit review
does not: a title card, a lower third when the story moves to a new section, a
music bed ducked under the child's voice, and a fade where one section becomes
the next.

It does not build any of that on top of the draft. The draft is a sixteenth of
the source's pixels, and a deliverable made from it would be a 540p video with
titles on it. The timeline's in and out points are exact and the manifest still
names the original files, so this stage cuts its own segments straight from those
originals at delivery resolution — nothing here is ever upscaled.

Assembly is a sequence of finite, independently inspectable passes rather than
one large filter graph: segments, then the picture master, then a fixed-length
alpha graphics track, then the ducked music mix, then one composite encode. An
earlier single-graph version opened an unbounded image loop per overlay and could
deadlock after encoding the whole picture; every input here has an explicit
duration or frame count, so each pass must terminate. The picture is encoded
losslessly on the way in and lossily exactly once on the way out.

Nothing this stage produces is called `final.mp4` until `production-contract.yaml`
has been measured against the file that was actually written.
"""
from __future__ import annotations

import shutil
import time
import json
import math
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import yaml

from videoai.core.ffmpeg import (
    ProbeResult,
    h264_encode_args,
    probe,
    videotoolbox_available,
    _run_ffmpeg_to,
)
from videoai.core.models import (
    DraftResult,
    EffectEvent,
    EffectPlan,
    FinalResult,
    Manifest,
    StoryPlan,
    Timeline,
    TimelineClip,
    Transcript,
    Word,
)
from videoai.core.registry import StageContext, stage
from videoai.core.text import render_text_image
from videoai.core.store import hash_file, hash_parts
from videoai.logic.effects import (
    EffectLibrary,
    SpriteTransform,
    animation_transform,
    bubble_max_width,
    library_fingerprint,
    load_library,
    place_sprite,
    render_speech_bubble,
    sprite_target_height,
)
from videoai.logic.music import attribution_line, list_tracks, select_track
from videoai.logic.contract import (
    contract_hash,
    estimated_delivery_bytes,
    fallback_output_name,
    free_bytes,
    has_closing_beat,
    load_contract,
    preflight,
    validate_production_report,
)
from videoai.stages.s06_render_draft import (
    DRAFT_AUDIO_BITRATE,
    DRAFT_AUDIO_CHANNEL_LAYOUT,
    DRAFT_AUDIO_CHANNELS,
    DRAFT_AUDIO_CODEC,
    DRAFT_AUDIO_SAMPLE_RATE,
    SILENT_AUDIO_SOURCE,
    _audio_filter_chain,
)

MUSIC_FADE_SECONDS = 2.0

# `sidechaincompress` attenuates the bed by how far the key signal sits above the
# threshold, so how much duck a given setting produces depends on how loudly the
# video happens to have been recorded. A fixed threshold therefore encodes an
# assumption about the creator's microphone: the constant this stage used to carry
# assumed speech around -20 dBFS, and on the real footage (-27 dBFS) it delivered
# 2.3 dB of duck while the report simply claimed ducking was applied.
#
# So nothing here is assumed. The threshold is placed under the MEASURED speech
# level so that speech really is `SPEECH_OVERSHOOT_DB` above it, and `level_sc` —
# the gain ffmpeg applies to the key before detection — is then bisected until the
# bed measurably drops by the amount `polish.music_duck_db` asked for. Each step is
# one short audio pass, and the step count is fixed, so the search terminates and
# the same input always produces the same mix.
SPEECH_OVERSHOOT_DB = 16.0
SIDECHAIN_ATTACK_MS = 20
SIDECHAIN_RELEASE_MS = 300
# ffmpeg's own limits for `threshold` and `level_sc`.
MIN_SIDECHAIN_THRESHOLD = 0.000977
SIDECHAIN_KEY_GAIN_LIMITS = (0.015625, 64.0)
DUCK_SEARCH_STEPS = 6
DUCK_TOLERANCE_DB = 0.75

SILENCE_SOURCE = (
    f"anullsrc=channel_layout={DRAFT_AUDIO_CHANNEL_LAYOUT}:sample_rate={DRAFT_AUDIO_SAMPLE_RATE}"
)


@dataclass(frozen=True)
class _TextOverlay:
    """One RGBA strip shown over the picture between `start` and `start+duration`."""

    text: str
    start: float
    duration: float


@dataclass(frozen=True)
class _Caption:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class _EffectOverlay:
    """One cartoon accent, resolved onto the delivery clock and ready to draw.

    `image` is the sprite at its resting size — already stretched around its text
    if it is a nine-patch, already scaled if it is not — so the frame loop only has
    to apply the animation curve.
    """

    name: str
    start: float
    duration: float
    cell: str
    anchor: str
    animation: str
    image: object
    # Where the creator dragged it, as a fraction of the frame. When set it wins
    # over `cell`: nine cells are what a model aims with, and a person who can
    # see the shot can say "just above his hand".
    point: tuple[float, float] | None = None


@dataclass(frozen=True)
class _Duck:
    """A measured duck, and the settings that produced the file it was measured on."""

    attenuation_db: float
    threshold: float
    key_gain: float


def lower_third_y(frame_height: int, overlay_height: int) -> int:
    """Place an overlay inside the bottom title-safe area.

    The six-percent bottom margin keeps the whole plate inside the title-safe
    area of a 16:9 delivery frame, so nothing is clipped by an overscanning TV.
    """
    return max(0, frame_height - overlay_height - int(frame_height * 0.06))


def section_title_height(frame_height: int) -> int:
    """The height of a section-title strip on a frame this tall."""
    return max(72, int(frame_height * 0.14))


def caption_lane_y(frame_height: int, caption_height: int, title_height: int) -> int:
    """The burned-caption lane: immediately above the section-title lower third.

    Burned captions and section titles both want the bottom of the frame, and a
    section title can begin mid-sentence. The titles own the lower third, so the
    caption lane is placed above them with a small gap instead of on top of them.
    """
    gap = max(8, int(frame_height * 0.015))
    return max(0, lower_third_y(frame_height, title_height) - caption_height - gap)


def build_section_titles(
    timeline: Timeline,
    starts: list[float],
    durations: list[float],
    intro_offset: float,
    title_seconds: float,
) -> list[_TextOverlay]:
    """A lower third naming the beat wherever the story moves to a new section.

    Generic by design: the beat text is whatever the planner named the section,
    so nothing here has to know what the video is about.
    """
    return [
        _TextOverlay(
            text=timeline.clips[index].beat.strip(),
            start=intro_offset + starts[index],
            duration=min(title_seconds, max(0.2, durations[index])),
        )
        for index in section_changes(timeline)
        if timeline.clips[index].beat.strip()
    ]


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


def clamp_caption_ends(captions: list[_Caption]) -> list[_Caption]:
    """Stop each cue at the moment the next one starts.

    The mapped cues come from word timings padded outwards at both edges, so a
    cue's end can land after the next cue's start. YouTube, VLC and QuickTime all
    react to overlapping SRT cues differently — stacking them, dropping one, or
    flickering between the two — so the overlap is removed here rather than left
    to the player. A cue is never shortened below 200 ms, and a cue that would be
    is dropped instead of being shown for a single frame.
    """
    clamped: list[_Caption] = []
    for index, caption in enumerate(captions):
        end = caption.end
        if index + 1 < len(captions):
            end = min(end, captions[index + 1].start)
        if end - caption.start < 0.2:
            continue
        clamped.append(_Caption(start=caption.start, end=end, text=caption.text))
    return clamped


def speech_windows(
    captions: list[_Caption], pad: float = 0.1, merge_gap: float = 0.4, limit: int = 40
) -> list[tuple[float, float]]:
    """Where speech is on the delivery clock, as few windows as possible.

    Derived from the mapped caption cues because those are already the spoken
    words in delivery time. Neighbouring windows are merged so the measurement
    filter stays a short expression rather than one term per word group.
    """
    spans = sorted(
        (max(0.0, caption.start - pad), caption.end + pad) for caption in captions
    )
    merged: list[list[float]] = []
    for start, end in spans:
        if merged and start - merged[-1][1] <= merge_gap:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged[:limit]]


def _windowed_rms_db(path: Path, windows: list[tuple[float, float]]) -> float:
    """The overall RMS level of `path`, counting only `windows`.

    `-inf` comes back as a very low number rather than a float infinity so callers
    can subtract two of these without special cases.
    """
    chain = ""
    if windows:
        # Commas inside `between(t,a,b)` are filter-graph separators unless they
        # are escaped, and this expression is passed to ffmpeg directly (no shell).
        terms = "+".join(
            f"between(t\\,{start:.3f}\\,{end:.3f})" for start, end in windows
        )
        chain = f"aselect=expr={terms},asetpts=N/SR/TB,"
    result = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-nostats", "-i", str(path),
            "-af", f"{chain}astats=metadata=0", "-f", "null", "-",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"could not measure {path}: {result.stderr.strip()[-500:]}")
    levels = re.findall(r"RMS level dB:\s*(-?[\d.]+|-inf|nan)", result.stderr)
    if not levels:
        raise RuntimeError(f"astats reported no RMS level for {path}")
    value = levels[-1]
    return -120.0 if value in ("-inf", "nan") else float(value)


def sidechain_threshold(speech_rms_db: float) -> float:
    """The key threshold that puts a speech level of `speech_rms_db` above itself by
    `SPEECH_OVERSHOOT_DB`, inside the range ffmpeg accepts."""
    wanted = 10 ** ((speech_rms_db - SPEECH_OVERSHOOT_DB) / 20)
    return max(MIN_SIDECHAIN_THRESHOLD, min(1.0, wanted))


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


def _load_rgba(path: Path):
    """An RGBA image as a numpy array in R, G, B, A order.

    OpenCV reads and writes BGR(A), and the graphics track's raw pipe to ffmpeg is
    declared `rgba`, so the channel order has to be resolved in exactly one place.
    """
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"could not read text image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)


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
    def stamp(strip: Path, text: str, strip_width: int, strip_height: int,
              max_lines: int, top_fraction: float) -> None:
        render_text_image(
            strip, text, strip_width, strip_height, plate_alpha=0.0,
            max_lines=max_lines,
        )
        # The strip is RGBA; the card is the BGR image cv2 will write. Reversing
        # the colour channels here is what keeps a coloured title from coming out
        # of the card with its red and blue swapped.
        layer = _load_rgba(strip)[:, :, ::-1]
        colour, alpha = layer[:, :, 1:], layer[:, :, :1].astype(np.float32) / 255.0
        x = (width - layer.shape[1]) // 2
        y = int(height * top_fraction)
        region = image[y:y + layer.shape[0], x:x + layer.shape[1]]
        region[:] = (colour * alpha + region * (1 - alpha)).astype(np.uint8)

    stamp(path.with_suffix(".rgba.png"), title, int(width * 0.82),
          int(height * 0.42), 3, 0.20)
    if subtitle.strip():
        stamp(path.with_suffix(".subtitle.png"), subtitle, int(width * 0.72),
              int(height * 0.13), 2, 0.74)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"could not render card image: {path}")


def _is_lossless_x264_encode(args: list[str]) -> bool:
    """True only for `-c:v libx264 -crf 0`: a mathematically lossless generation.

    Anything else — a different crf, no crf at all, or VideoToolbox's constant-
    quality mode — throws pixels away. This is checked against the actual
    argument list handed to ffmpeg rather than assumed, so a future change that
    quietly makes an intermediate lossy is counted, not hidden.
    """
    if "-c:v" not in args or args[args.index("-c:v") + 1] != "libx264":
        return False
    if "-crf" not in args:
        return False
    return args[args.index("-crf") + 1] == "0"


def _card_segment(
    image: Path,
    duration: float,
    frame: tuple[int, int],
    fps: float,
    dst: Path,
    lossless_sink: list[bool] | None = None,
) -> None:
    width, height = frame
    fade = min(0.4, duration / 4)
    args = [
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
    ]
    if lossless_sink is not None:
        lossless_sink.append(_is_lossless_x264_encode(args))
    _run_ffmpeg_to(args, dst)


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
    lossless_sink: list[bool] | None = None,
) -> int:
    """Cut one segment; return how many fade filters were actually emitted.

    The count is what the production report records as a transition. A boundary
    the planner marked is not a transition until a filter for it exists: with
    `transition_frames: 0`, or on a segment too short to fade through, the cut
    stays hard and nothing should claim otherwise.
    """
    width, height = frame
    filters = [
        f"scale={width}:{height}:force_original_aspect_ratio=decrease",
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2",
        f"fps={fps}",
        "format=yuv420p",
        "setsar=1",
    ]
    duration = clip.dur
    # A segment cannot fade for longer than a third of itself at each edge, or the
    # picture would never be at full strength between the two fades.
    local_transition = min(transition, duration / 3)
    emitted = 0
    if fade_in and local_transition > 0:
        filters.append(f"fade=t=in:st=0:d={local_transition:.3f}")
        emitted += 1
    if fade_out and local_transition > 0:
        filters.append(
            f"fade=t=out:st={max(0.0, duration - local_transition):.3f}:"
            f"d={local_transition:.3f}"
        )
        emitted += 1
    seek = ["-ss", f"{clip.offset:.3f}", "-i", str(source)]
    common = [
        "-t", f"{duration:.3f}", "-filter:v", ",".join(filters),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "0",
        "-c:a", DRAFT_AUDIO_CODEC, "-b:a", DRAFT_AUDIO_BITRATE,
        "-ar", str(DRAFT_AUDIO_SAMPLE_RATE), "-ac", str(DRAFT_AUDIO_CHANNELS),
        "-avoid_negative_ts", "make_zero",
    ]
    if lossless_sink is not None:
        lossless_sink.append(_is_lossless_x264_encode(common))
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
    return emitted


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


def _blend_at(canvas, overlay, x: int, y: int) -> None:
    """Composite an RGBA overlay onto an RGBA canvas at `(x, y)`, clipping edges.

    Alpha compositing rather than assignment: a strip is only partly opaque (a
    semi-transparent plate with hard glyphs on it), and two strips can overlap in
    time, so writing the pixels straight in would punch a rectangular hole in
    whatever was already there and hand ffmpeg the plate's alpha as if it were the
    text's.

    Negative positions are clipped rather than refused: a cartoon accent is placed
    by an anchor point and scaled per frame, so a large sprite in a corner cell can
    legitimately want to start off the edge of the frame.
    """
    import numpy as np

    height, width = canvas.shape[:2]
    left = max(0, x)
    top = max(0, y)
    right = min(width, x + overlay.shape[1])
    bottom = min(height, y + overlay.shape[0])
    if bottom <= top or right <= left:
        return
    source = overlay[top - y : bottom - y, left - x : right - x].astype(np.float32)
    target = canvas[top:bottom, left:right].astype(np.float32)
    source_alpha = source[:, :, 3:4] / 255.0
    target_alpha = target[:, :, 3:4] / 255.0
    out_alpha = source_alpha + target_alpha * (1.0 - source_alpha)
    safe_alpha = np.where(out_alpha > 0, out_alpha, 1.0)
    colour = (
        source[:, :, :3] * source_alpha
        + target[:, :, :3] * target_alpha * (1.0 - source_alpha)
    ) / safe_alpha
    canvas[top:bottom, left:right, :3] = np.clip(colour, 0, 255).astype(np.uint8)
    canvas[top:bottom, left:right, 3:4] = np.clip(out_alpha * 255.0, 0, 255).astype(np.uint8)


def _blend(canvas, overlay, y: int) -> None:
    """Composite a horizontally centred RGBA strip. Titles and captions are lanes:
    they span most of the frame's width and are always centred in it."""
    _blend_at(canvas, overlay, (canvas.shape[1] - overlay.shape[1]) // 2, y)


def _sprite_base_image(
    library: EffectLibrary,
    event: EffectEvent,
    frame: tuple[int, int],
    work_dir: Path,
    index: int,
):
    """The sprite for one event at its final resting size, as an RGBA array.

    A nine-patch is stretched around its text first, so its size comes from the
    words; every other sprite is scaled to a fraction of the frame's height. The
    per-frame animation then works from this image, so the overshoot of a pop-in is
    an overshoot of the size the sprite is meant to end at.
    """
    import cv2

    sprite = library.get(event.effect_name)
    source = library.path_of(sprite)
    if sprite.nine_patch is not None:
        target = work_dir / f"effect-{index:02d}-{sprite.name}.png"
        render_speech_bubble(
            source,
            sprite.nine_patch,
            event.text,
            bubble_max_width(event.scale, frame[0]),
            target,
        )
        return _load_rgba(target)
    image = _load_rgba(source)
    height = sprite_target_height(event.scale, frame[1])
    width = max(1, round(image.shape[1] * height / max(1, image.shape[0])))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def _transform_sprite(image, transform: SpriteTransform):
    """One animation frame of a sprite: scaled, rotated, and alpha-scaled.

    Rotation happens on a padded canvas so a shake never clips the sprite's own
    corners, and the alpha is scaled rather than the colour so a fading accent goes
    transparent instead of going black.
    """
    import cv2
    import numpy as np

    scale = max(0.01, transform.scale)
    width = max(1, round(image.shape[1] * scale))
    height = max(1, round(image.shape[0] * scale))
    out = cv2.resize(
        image,
        (width, height),
        interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
    )
    if abs(transform.rotation) > 0.01:
        pad = max(2, round(max(width, height) * 0.12))
        out = cv2.copyMakeBorder(
            out, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0, 0)
        )
        centre = (out.shape[1] / 2, out.shape[0] / 2)
        matrix = cv2.getRotationMatrix2D(centre, transform.rotation, 1.0)
        out = cv2.warpAffine(
            out,
            matrix,
            (out.shape[1], out.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )
    alpha = max(0.0, min(1.0, transform.alpha))
    if alpha < 1.0:
        out = out.copy()
        out[:, :, 3] = np.clip(out[:, :, 3].astype(np.float32) * alpha, 0, 255).astype(
            np.uint8
        )
    return out


def build_effect_overlays(
    events: list[EffectEvent],
    library: EffectLibrary,
    timeline: Timeline,
    measured_starts: list[float],
    intro_offset: float,
    frame: tuple[int, int],
    work_dir: Path,
) -> list[_EffectOverlay]:
    """Map planned events onto the delivery clock and prepare their sprites.

    The planned time is on the timeline's clock; a rendered segment is a whole
    number of frames, so by the twentieth cut the delivery has drifted from the
    plan by more than a frame. The event is therefore placed at the same offset
    inside the same segment, measured — the identical correction captions get.
    """
    planned = cumulative_starts([clip.dur for clip in timeline.clips])
    overlays: list[_EffectOverlay] = []
    for index, event in enumerate(events):
        if not event.keep:
            # Turned down on the approval page. Dropped here rather than earlier
            # so the plan keeps the record of what was considered and refused.
            continue
        segment = 0
        for candidate, start in enumerate(planned):
            if event.at_seconds + 1e-6 >= start:
                segment = candidate
        inside = max(0.0, event.at_seconds - planned[segment])
        image = _sprite_base_image(library, event, frame, work_dir, index)
        sprite = library.get(event.effect_name)
        duration = event.seconds if event.seconds > 0 else sprite.default_seconds
        overlays.append(
            _EffectOverlay(
                name=event.effect_name,
                start=intro_offset + measured_starts[segment] + inside,
                duration=max(0.1, duration),
                cell=event.screen_position,
                anchor=sprite.anchor,
                animation=sprite.animation,
                image=image,
                point=(event.x, event.y) if event.x is not None and event.y is not None else None,
            )
        )
    return overlays


def _render_graphics_track(
    path: Path,
    frame: tuple[int, int],
    fps: float,
    duration: float,
    captions: list[_Caption],
    titles: list[_TextOverlay],
    work_dir: Path,
    effects: list[_EffectOverlay] = (),
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
    title_height = section_title_height(height)
    title_y = lower_third_y(height, title_height)
    for index, title in enumerate(titles):
        image_path = work_dir / f"section-title-{index:02d}.png"
        render_text_image(
            image_path, title.text, int(width * 0.74), title_height,
            plate_alpha=0.62, max_lines=2,
        )
        image = _load_rgba(image_path)
        # A section title is a lower third: below the presenter, inside the
        # bottom title-safe area.
        assets.append((title.start, title.start + title.duration, title_y, image))
    caption_height = max(80, int(height * 0.12))
    caption_y = caption_lane_y(height, caption_height, title_height)
    for index, caption in enumerate(captions):
        image_path = work_dir / f"caption-{index:03d}.png"
        render_text_image(
            image_path, caption.text, int(width * 0.78), caption_height,
            plate_alpha=0.72, max_lines=2,
        )
        image = _load_rgba(image_path)
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
            for start, end, y, image in assets:
                if not (start <= at < end):
                    continue
                _blend(canvas, image, y)
            # Accents go on last: they are the top layer, and a starburst behind a
            # lower third would be a starburst nobody sees.
            for effect in effects:
                if not (effect.start <= at < effect.start + effect.duration):
                    continue
                progress = (at - effect.start) / effect.duration
                transform = animation_transform(effect.animation, progress)
                sprite = _transform_sprite(effect.image, transform)
                if effect.point is not None:
                    # A dragged point is the sprite's centre, which is what a
                    # person aims at; clamped so a drag to the edge cannot push
                    # the accent off the frame.
                    x = round(effect.point[0] * frame[0] - sprite.shape[1] / 2)
                    y = round(effect.point[1] * frame[1] - sprite.shape[0] / 2)
                    x = min(max(x, 0), max(0, frame[0] - sprite.shape[1]))
                    y = min(max(y, 0), max(0, frame[1] - sprite.shape[0]))
                else:
                    x, y = place_sprite(
                        (sprite.shape[1], sprite.shape[0]), effect.cell, effect.anchor, frame
                    )
                _blend_at(
                    canvas,
                    sprite,
                    x + round(transform.dx * sprite.shape[0]),
                    y + round(transform.dy * sprite.shape[0]),
                )
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


def _extract_speech_track(picture: Path, dst: Path) -> None:
    """The picture master's own audio, on its own.

    The duck is keyed on it and the search below measures it repeatedly, and the
    picture master is a multi-gigabyte lossless file — decoding its video once per
    measurement would cost minutes for an answer about its audio.
    """
    _run_ffmpeg_to(["-i", str(picture), "-vn", "-c:a", "pcm_s16le"], dst)


def _duck_bed(
    speech: Path,
    bed: Path,
    duration: float,
    duck_db: float,
    threshold: float,
    key_gain: float,
    dst: Path,
) -> None:
    """The music bed alone, pushed down under the delivered speech.

    Written out as its own file rather than ducked inside the mix graph so the
    attenuation can be measured against the unducked bed afterwards. The mix that
    goes into the final uses exactly this file, so the measurement is of the
    delivered audio and not of a re-created approximation of it.
    """
    _run_ffmpeg_to(
        [
            "-i", str(bed), "-i", str(speech),
            "-filter_complex",
            f"[0:a][1:a]sidechaincompress=threshold={threshold:.6f}:"
            f"ratio={duck_ratio(duck_db):.3f}:attack={SIDECHAIN_ATTACK_MS}:"
            f"release={SIDECHAIN_RELEASE_MS}:detection=rms:"
            f"level_sc={key_gain:.4f}[ducked]",
            "-map", "[ducked]", "-t", f"{duration:.3f}", "-c:a", "pcm_s16le",
        ],
        dst,
    )


def duck_bed_under_speech(
    speech: Path,
    bed: Path,
    duration: float,
    duck_db: float,
    windows: list[tuple[float, float]],
    dst: Path,
) -> _Duck:
    """Duck `bed` under `speech` until it measurably drops by `duck_db`.

    Returns the duck that `dst` actually holds, so the figure the production report
    carries is a measurement of the file that goes into the delivered mix. The two
    files compared are the same music at the same gain and differ only by the
    sidechain compressor, so measuring both over the same speech windows measures
    the duck itself rather than the music.

    `level_sc` is bisected geometrically because it is a gain, and the search is a
    fixed number of steps: a settled answer matters less than a repeatable one, and
    a mix that differed between two runs of the same project would be worse than a
    mix that is half a decibel off.
    """
    requested = abs(duck_db)
    threshold = sidechain_threshold(_windowed_rms_db(speech, windows))
    unducked = _windowed_rms_db(bed, windows)

    def render(key_gain: float) -> _Duck:
        _duck_bed(speech, bed, duration, duck_db, threshold, key_gain, dst)
        return _Duck(unducked - _windowed_rms_db(dst, windows), threshold, key_gain)

    low, high = SIDECHAIN_KEY_GAIN_LIMITS
    if requested <= 0:
        return render(low)  # the quietest key there is: as close to no duck as possible
    best = None
    for _ in range(DUCK_SEARCH_STEPS):
        attempt = render(math.sqrt(low * high))
        if best is None or abs(attempt.attenuation_db - requested) < abs(
            best.attenuation_db - requested
        ):
            best = attempt
        if abs(attempt.attenuation_db - requested) <= DUCK_TOLERANCE_DB:
            return attempt
        if attempt.attenuation_db < requested:
            low = attempt.key_gain
        else:
            high = attempt.key_gain
    # The last step is not necessarily the closest one, and `dst` must hold the mix
    # whose measurement is reported, so the winner is rendered again.
    return render(best.key_gain)


def _mix_delivery_audio(
    speech: Path,
    ducked_bed: Path,
    duration: float,
    dst: Path,
) -> None:
    _run_ffmpeg_to(
        [
            "-i", str(speech), "-i", str(ducked_bed),
            "-filter_complex",
            "[0:a][1:a]amix=inputs=2:normalize=0:duration=first[a]",
            "-map", "[a]", "-t", f"{duration:.3f}", "-c:a", "pcm_s16le",
        ],
        dst,
    )


def full_decode(path: Path) -> tuple[bool, str]:
    """Decode every frame and sample of `path`, discarding the output.

    Returns the outcome rather than raising: the production report records what
    was measured, and the contract is what decides whether that is acceptable.
    """
    result = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "null", "-"],
        capture_output=True,
        text=True,
    )
    stderr = result.stderr.strip()
    return result.returncode == 0 and not stderr, stderr[-1000:]


def approval_is_current(ctx: StageContext, draft: DraftResult) -> None:
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




def _polish_fingerprint_extras(ctx: StageContext) -> tuple[str, ...]:
    """The delivery rules this stage is judged against, and the sprites it draws.

    Editing `production-contract.yaml` changes what a valid `final.mp4` is, so it
    has to invalidate the cached one exactly as a config change does. The effects
    library's own content is here too, and hashed all the way down to the PNGs:
    swapping in a better drawing of `comic_starburst` changes the delivered picture
    without changing any artifact this stage reads.
    """
    return (
        f"contract:{contract_hash(ctx.project_dir)}",
        f"effects-library:{library_fingerprint()}",
    )


def _planned_effects(ctx: StageContext) -> tuple[EffectPlan, EffectLibrary]:
    """The accents to composite, or none.

    Effects are seasoning: a missing artifact (an older project, a pipeline run
    that stopped before the effects stage) and `effects.enabled: false` both mean
    no accents, and neither is an error. Delivery has never depended on them and
    must not start now.
    """
    library = load_library()
    if not ctx.config.effects.enabled or not ctx.store.exists("05d-effects"):
        return EffectPlan(), library
    plan = ctx.store.read("05d-effects", EffectPlan)
    unknown = sorted(
        {event.effect_name for event in plan.events}
        - set(library.names())
    )
    if unknown:
        # The plan was validated against the library that existed when it was
        # written. A sprite removed since then cannot be drawn, and quietly
        # skipping it would deliver a video missing accents the artifact promises.
        raise RuntimeError(
            "05d-effects names sprites that are not in "
            f"{library.directory}: {', '.join(unknown)}; restore them or re-run the "
            "effects stage"
        )
    return plan, library


def _resolve_music_dir(ctx: StageContext, configured: str) -> Path:
    music_dir = Path(configured)
    if not music_dir.is_absolute():
        candidate = ctx.project_dir / music_dir
        music_dir = candidate if candidate.is_dir() else music_dir
    return music_dir


def _select_music(
    music_dir: Path, tracks: list[Path], settings, project_dir: Path
) -> Path:
    if settings.music_track:
        track = music_dir / settings.music_track
        if not track.is_file():
            raise RuntimeError(
                f"polish.music_track {settings.music_track!r} is not in {music_dir}; "
                f"available: {', '.join(path.name for path in tracks) or 'nothing'}"
            )
        return track
    track = select_track(tracks, _project_style(project_dir), project_dir.name)
    if track is None:
        raise RuntimeError(
            f"the production contract requires background music and {music_dir} "
            "holds no playable track; add one or point polish.music_dir at a "
            "library that has one"
        )
    return track


def _discard_delivery_outputs(output_dir: Path, output: Path) -> None:
    """Leave nothing behind that would describe a delivery that does not exist.

    A rejected render must not leave `final.srt` beside no video, nor the previous
    run's `production-report.json` claiming this one passed.
    """
    output.unlink(missing_ok=True)
    (output_dir / "final.srt").unlink(missing_ok=True)
    (output_dir / "production-report.json").unlink(missing_ok=True)


POLISH_REQUIRES = (
    "01-manifest", "03-transcript", "05-timeline", "05a-storyplan", "05d-effects",
    "06-draft",
)


@stage(
    id="polish",
    produces="08-final",
    # "01-manifest" names the original files this stage cuts from. "06-draft" is
    # required for its ordering rather than its pixels: the draft is the review
    # the final is only worth building after, and it is what `polish.enabled:
    # false` degrades to a preview of.
    requires=POLISH_REQUIRES,
    version="6",
    model=FinalResult,
    config_keys=(
        "polish.enabled",
        # Read here as well as by the effects stage: turning effects off has to
        # re-render a delivery that already has them composited into its picture.
        "effects.enabled",
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
        "polish.hardware_encode",
        # Read, not inherited: the final does its own cutting now, and the fade at
        # each cut has to be the one the draft was reviewed with.
        "render.audio_fade_seconds",
    ),
    fingerprint_extras=_polish_fingerprint_extras,
)
def polish(ctx: StageContext) -> FinalResult:
    started = time.monotonic()
    settings = ctx.config.polish
    draft = ctx.store.read("06-draft", DraftResult)
    ctx.output_dir.mkdir(parents=True, exist_ok=True)
    contract = load_contract(ctx.project_dir)
    output = ctx.output_dir / "final.mp4"

    if not settings.enabled:
        # Turning polish off is a legitimate request for a cheap preview, and the
        # draft is exactly that. What it is not is a delivery: it is cut from the
        # 540p proxies and has none of the required features. So it goes out under
        # the contract's fallback name, and `final.mp4` — plus anything that would
        # describe one — is removed rather than left pointing at the last render.
        fallback = ctx.output_dir / fallback_output_name(contract)
        source = _draft_path(ctx, draft)
        shutil.copyfile(source, fallback)
        _discard_delivery_outputs(ctx.output_dir, output)
        copied = probe(fallback)
        return FinalResult(
            path=str(fallback), duration=copied.duration,
            width=copied.width, height=copied.height,
            render_seconds=time.monotonic() - started,
            notes=[
                "polish.enabled is false: this is a copy of the review draft, not a "
                "delivery",
                f"written as {fallback.name} because it satisfies none of the "
                "production contract's required features",
                "missing: intro, outro, section titles, captions, music, ducking, "
                "section transitions, delivery resolution",
            ],
        )
    if settings.require_approval:
        approval_is_current(ctx, draft)

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

    music_dir = _resolve_music_dir(ctx, settings.music_dir)
    tracks = list_tracks(music_dir)
    # Read here rather than beside the graphics track, which is built after the
    # picture master: this is a JSON read and a directory listing, and the one way
    # it can fail — a plan naming a sprite the library no longer has — must not
    # cost a full render first.
    effect_plan, library = _planned_effects(ctx)

    # Everything above is a probe, a listing and some arithmetic. Everything below
    # re-encodes gigabytes, so the contract is checked against what is knowable
    # now: an unsatisfiable delivery has to fail in seconds.
    timeline_seconds = (
        sum(clip.dur for clip in timeline.clips)
        + max(0.0, settings.intro_seconds)
        + max(0.0, settings.outro_seconds)
    )
    preflight(
        contract,
        frame=frame,
        output_height=settings.output_height,
        music_dir=music_dir,
        track_count=len(tracks) if not settings.music_track else 1,
        closing_beat=has_closing_beat(timeline),
        estimated_bytes=estimated_delivery_bytes(frame, fps, timeline_seconds),
        available_bytes=free_bytes(ctx.work_dir),
    )
    track = _select_music(music_dir, tracks, settings, ctx.project_dir)

    # Every x264 video encode the delivery path performs, in order, and whether
    # each one actually was lossless. `quality.lossy_video_generations` in the
    # report below is the count of the ones that were not — measured, not
    # asserted, so a future change that makes an intermediate lossy is caught by
    # the contract instead of silently under-reported.
    lossless_encodes: list[bool] = []

    segments: list[Path] = []
    changes = set(section_changes(timeline))
    emitted_transitions = 0
    for index, (clip, source) in enumerate(zip(timeline.clips, sources)):
        target = work_dir / f"segment-{index:03d}.mp4"
        emitted_transitions += _cut_delivery_segment(
            source, clip, frame, fps, ctx.config.render.audio_fade_seconds,
            manifest.by_id(clip.src).has_audio,
            transition,
            fade_in=index == 0 or index in changes,
            fade_out=index == len(timeline.clips) - 1 or index + 1 in changes,
            dst=target,
            lossless_sink=lossless_encodes,
        )
        segments.append(target)
    segment_durations = [probe(path).duration for path in segments]
    starts = cumulative_starts(segment_durations)

    # The paths the segments were really cut from, checked against the disposable
    # proxies the manifest also names. Recording "originals" without looking is
    # exactly how a 540p delivery passed a contract that required 1080p.
    # Ingest always writes the proxy to its own name under work/media, so a
    # delivery input that matches a proxy path really was cut from a proxy —
    # including the case where a manifest claims one file is both.
    proxies = {
        str(Path(clip.proxy_path).resolve())
        for clip in manifest.clips
        if clip.proxy_path
    }
    used = [str(path.resolve()) for path in sources]
    leaked = sorted({path for path in used if path in proxies})

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
    _card_segment(
        intro_image, intro_duration, frame, fps, intro_path,
        lossless_sink=lossless_encodes,
    )
    _card_segment(
        outro_image, outro_duration, frame, fps, outro_path,
        lossless_sink=lossless_encodes,
    )

    picture = work_dir / "picture-master.mp4"
    _concat_copy([intro_path, *segments, outro_path], work_dir, picture)
    total = probe(picture).duration

    intro_offset = probe(intro_path).duration
    captions = clamp_caption_ends(
        build_captions(
            timeline, transcript, starts, [], 0.0, intro_offset,
            settings.caption_words,
        )
    )
    # From here through validation, a raise of any kind must leave output/ exactly
    # as it was before this stage ran rather than an orphan final.srt (or a stale
    # final.mp4 / production-report.json from the previous run) beside whatever
    # step actually failed.
    srt_path = ctx.output_dir / "final.srt"
    try:
        if settings.captions_enabled:
            if not captions:
                raise RuntimeError(
                    "production contract requires captions, but none were generated"
                )
            write_srt_captions(srt_path, captions)
        else:
            # No caption track was asked for, so there must not be one on disk — a
            # stale file from an earlier run would be uploaded as this video's
            # captions.
            srt_path.unlink(missing_ok=True)

        title_overlays = build_section_titles(
            timeline, starts, segment_durations, intro_offset, settings.title_seconds,
        )
        if not title_overlays:
            raise RuntimeError("production contract requires section titles")

        effect_overlays = build_effect_overlays(
            effect_plan.events, library, timeline, starts, intro_offset, frame,
            work_dir,
        )

        graphics = work_dir / "graphics.mov"
        _render_graphics_track(
            graphics, frame, fps, total,
            captions if settings.burn_captions else [],
            title_overlays, work_dir, effect_overlays,
        )

        bed = work_dir / "music-bed.wav"
        ducked = work_dir / "music-bed-ducked.wav"
        speech_track = work_dir / "speech.wav"
        mixed = work_dir / "mixed-audio.wav"
        _extract_speech_track(picture, speech_track)
        _render_music_bed(track, total, settings.music_gain_db, bed)
        duck = duck_bed_under_speech(
            speech_track, bed, total, settings.music_duck_db,
            speech_windows(captions), ducked,
        )
        _mix_delivery_audio(speech_track, ducked, total, mixed)

        hardware = settings.hardware_encode and videotoolbox_available()
        final_args = [
            "-i", str(picture), "-i", str(graphics), "-i", str(mixed),
            "-filter_complex", "[0:v][1:v]overlay=eof_action=pass:shortest=1[v]",
            "-map", "[v]", "-map", "2:a:0", "-t", f"{total:.3f}",
            *h264_encode_args(settings.output_crf, hardware),
            "-pix_fmt", "yuv420p",
            "-c:a", DRAFT_AUDIO_CODEC, "-b:a", DRAFT_AUDIO_BITRATE,
            "-ar", str(DRAFT_AUDIO_SAMPLE_RATE), "-ac", str(DRAFT_AUDIO_CHANNELS),
            "-movflags", "+faststart",
        ]
        lossless_encodes.append(_is_lossless_x264_encode(final_args))
        _run_ffmpeg_to(final_args, output)
        decoded, decode_error = full_decode(output)
        measured = probe(output)
        attribution = attribution_line(track)
        if attribution:
            write_attribution(ctx.output_dir, attribution)

        # What the accents were, on the delivery's own clock. Recorded outside
        # `features` on purpose: the contract's required features are the
        # structure of a finished video, and an effect is seasoning — a delivery
        # with none is as valid as a delivery with eight.
        effect_lines = [
            f"{overlay.name} at {overlay.start:.2f}s for {overlay.duration:.2f}s "
            f"({overlay.cell}, {overlay.animation})"
            for overlay in effect_overlays
        ]
        report_path = ctx.output_dir / "production-report.json"
        # Measured, not asserted: only the encodes that were not lossless count,
        # so a future intermediate that quietly turned lossy would raise this
        # above the contract's `maximum_lossy_video_generations` instead of
        # shipping under a hardcoded `1`.
        lossy_video_generations = sum(
            1 for lossless in lossless_encodes if not lossless
        )
        report = {
            "contract_version": contract.get("version"),
            "status": "passed",
            "output": str(output),
            "duration": measured.duration,
            "width": measured.width,
            "height": measured.height,
            "features": {
                "intro": True,
                "outro": True,
                "section_titles": len(title_overlays),
                "captions": len(captions) if settings.captions_enabled else 0,
                "soft_captions": len(captions) if settings.captions_enabled else 0,
                "burned_in_captions": settings.burn_captions,
                "music": track.name,
                # Measured, in dB, over the speech windows of this very mix.
                "music_ducking": round(duck.attenuation_db, 2),
                # Fade filters actually emitted into the segment cuts, not
                # planned section boundaries.
                "transitions": emitted_transitions,
                "section_boundaries": len(changes),
                "closing_beat": has_closing_beat(timeline),
                "full_decode": decoded,
            },
            "effects": {
                "applied": len(effect_overlays),
                "library": str(library.directory),
                "events": [
                    {
                        "name": event.effect_name,
                        "at_seconds": event.at_seconds,
                        "position": event.screen_position,
                        "scale": event.scale,
                        "text": event.text,
                        "reason": event.reason,
                    }
                    for event in effect_plan.events
                ],
            },
            "quality": {
                "source": "proxies" if leaked else "originals",
                "segment_inputs": used,
                "proxy_inputs": leaked,
                "lossy_video_generations": lossy_video_generations,
            },
        }
        if not decoded:
            report["status"] = "failed"
            report["decode_error"] = decode_error
        validate_production_report(report, ctx.project_dir)
    except BaseException:
        _discard_delivery_outputs(ctx.output_dir, output)
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
        transition_count=emitted_transitions,
        caption_count=len(captions) if settings.captions_enabled else 0,
        burned_in_captions=settings.burn_captions,
        effect_count=len(effect_overlays),
        effects=effect_lines,
        outro=True,
        music_track=track.name,
        music_attribution=attribution,
        music_ducking=True,
        music_ducking_db=round(duck.attenuation_db, 2),
        fully_decoded=decoded,
        production_report=str(report_path),
        notes=[
            "delivery used finite multi-pass rendering",
            "source segments and picture master are lossless x264",
            "final.mp4 is the only lossy video generation",
            f"music bed measured {duck.attenuation_db:.1f} dB down under speech "
            f"(requested {abs(settings.music_duck_db):.1f} dB; sidechain threshold "
            f"{duck.threshold:.5f}, key gain {duck.key_gain:.2f})",
            (
                f"{len(effect_overlays)} cartoon accents composited: "
                + "; ".join(effect_lines)
                if effect_overlays
                else "no cartoon accents were planned for this video"
            ),
            (
                "captions burned into picture"
                if settings.burn_captions
                else "captions delivered as viewer-controlled final.srt"
                if settings.captions_enabled
                else "no caption track was requested"
            ),
        ],
    )
