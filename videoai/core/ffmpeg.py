"""Thin, explicit wrappers over ffmpeg/ffprobe. No hidden defaults."""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from functools import cache
from pathlib import Path

VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".mkv", ".avi"}


@dataclass(frozen=True)
class ProbeResult:
    duration: float
    width: int
    height: int
    fps: float
    has_audio: bool
    created_at: float | None = None
    # Degrees from the container's display matrix. iPhone footage records
    # landscape frames plus a quarter turn rather than portrait frames, so
    # `width`/`height` alone describe a sideways picture.
    rotation: float = 0.0

    @property
    def quarter_turned(self) -> bool:
        return round(abs(self.rotation)) % 180 == 90

    @property
    def display_width(self) -> int:
        return self.height if self.quarter_turned else self.width

    @property
    def display_height(self) -> int:
        return self.width if self.quarter_turned else self.height


def _stream_rotation(video: dict) -> float:
    """Rotation in degrees from either place ffprobe reports it."""
    for entry in video.get("side_data_list") or []:
        if "rotation" in entry:
            try:
                return float(entry["rotation"])
            except (TypeError, ValueError):
                return 0.0
    raw = (video.get("tags") or {}).get("rotate")
    if raw is not None:
        try:
            return float(raw)
        except (TypeError, ValueError):
            return 0.0
    return 0.0


def run_ffmpeg(args: list[str]) -> None:
    result = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", *args],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {' '.join(args)}\n{result.stderr.strip()}")


def probe(path: Path) -> ProbeResult:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-print_format", "json",
            "-show_format", "-show_streams", str(path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {path}: {result.stderr.strip()}")
    data = json.loads(result.stdout)
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        raise RuntimeError(f"no video stream in {path}")

    duration_raw = data.get("format", {}).get("duration")
    if duration_raw is None:
        raise RuntimeError(f"probe({path}): missing format.duration")
    try:
        duration = float(duration_raw)
    except (TypeError, ValueError):
        raise RuntimeError(
            f"probe({path}): unparseable format.duration: {duration_raw!r}"
        ) from None

    frame_rate_raw = video.get("r_frame_rate")
    if not frame_rate_raw:
        raise RuntimeError(f"probe({path}): missing r_frame_rate")
    numerator, _, denominator = frame_rate_raw.partition("/")
    try:
        fps = float(numerator) / float(denominator or 1)
    except (TypeError, ValueError, ZeroDivisionError):
        raise RuntimeError(
            f"probe({path}): unparseable r_frame_rate: {frame_rate_raw!r}"
        ) from None

    if "width" not in video:
        raise RuntimeError(f"probe({path}): missing width")
    if "height" not in video:
        raise RuntimeError(f"probe({path}): missing height")

    created_at: float | None = None
    raw_created = data.get("format", {}).get("tags", {}).get("creation_time")
    if raw_created:
        from datetime import datetime

        try:
            created_at = datetime.fromisoformat(raw_created.replace("Z", "+00:00")).timestamp()
        except ValueError:
            created_at = None

    return ProbeResult(
        duration=duration,
        width=int(video["width"]),
        height=int(video["height"]),
        fps=fps,
        has_audio=any(s.get("codec_type") == "audio" for s in streams),
        created_at=created_at,
        rotation=_stream_rotation(video),
    )


def list_video_files(directory: Path) -> list[Path]:
    """Video files directly inside `directory`, sorted, macOS metadata excluded."""
    if not directory.is_dir():
        return []
    return sorted(
        path for path in directory.iterdir()
        if path.is_file()
        and not path.name.startswith(".")
        and path.suffix.lower() in VIDEO_SUFFIXES
    )


@cache
def _has_videotoolbox_encoder() -> bool:
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True
    )
    return "h264_videotoolbox" in result.stdout


@cache
def _videotoolbox_operational() -> bool:
    """Whether VideoToolbox can create an encoding session right now.

    Listing an encoder only proves that ffmpeg was compiled with it. In headless
    sessions, CI, screen sharing, or while the media engine is busy, macOS may
    still refuse to create a compression session. A one-frame probe is cheap and
    prevents every real encode from failing just because the encoder was listed.
    """
    if not _has_videotoolbox_encoder():
        return False
    result = subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", "color=c=black:s=64x64:r=1",
            "-frames:v", "1", "-c:v", "h264_videotoolbox",
            "-f", "null", "-",
        ],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def videotoolbox_available() -> bool:
    """True when this build and the current session can use VideoToolbox."""
    return _videotoolbox_operational()


@cache
def filter_available(name: str) -> bool:
    """Whether this ffmpeg build exposes a named audio/video filter."""
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-filters"], capture_output=True, text=True
    )
    return any(
        line.split()[1:2] == [name]
        for line in result.stdout.splitlines()
        if line.strip() and not line.lstrip().startswith("-")
    )


# libx264 counts quality in quantiser steps (0 lossless .. 51 unwatchable);
# VideoToolbox counts it as a percentage running the other way. Converting one to
# the other keeps a single `crf` setting meaningful whichever encoder answers.
QP_RANGE = 51


def h264_encode_args(crf: int, hardware: bool) -> list[str]:
    """Encoder arguments for one H.264 output at the quality `crf` asks for.

    VideoToolbox is minutes-per-clip faster than libx264 on this footage, so it
    is preferred when the caller allows it and the build has it; `-b:v 0` is what
    puts it in constant-quality mode rather than its default bitrate target.
    """
    if hardware and videotoolbox_available():
        quality = max(1, min(100, round((QP_RANGE - crf) / QP_RANGE * 100)))
        return ["-c:v", "h264_videotoolbox", "-q:v", str(quality), "-b:v", "0"]
    return ["-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf)]


def hardware_decode_args() -> list[str]:
    """Input arguments that decode through the media engine when it is there.

    Only a speed choice: the frames land back in system memory either way, so
    every filter downstream — including the rotation ffmpeg inserts itself —
    behaves identically.
    """
    return ["-hwaccel", "videotoolbox"] if videotoolbox_available() else []


def _run_ffmpeg_to(args: list[str], dst: Path) -> None:
    """Run ffmpeg with `args` plus an output path, but land the result atomically.

    ffmpeg opens (and truncates) its output file as soon as encoding starts, so a
    crash, OOM, or Ctrl-C mid-encode leaves a truncated file exactly where a plain
    `path.exists()` reuse check would find it and treat it as done. Instead we write
    to a temp file in the same directory as `dst` (same filesystem, so the rename
    below is atomic) and only `os.replace()` it into place once ffmpeg succeeds. On
    failure the temp file is removed, so `dst` never exists in a partial state.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=dst.parent, suffix=dst.suffix)
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        run_ffmpeg([*args, str(tmp_path)])
        os.replace(tmp_path, dst)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def extract_audio(src: Path, dst: Path) -> None:
    """16 kHz mono WAV with EBU R128 loudness normalisation, ready for ASR."""
    _run_ffmpeg_to(
        [
            "-i", str(src),
            "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
            "-ac", "1", "-ar", "16000", "-vn",
        ],
        dst,
    )


def proxy_encoder() -> str:
    """The encoder `make_proxy` would use right now.

    Exposed because the choice is a live capability probe rather than a setting,
    so the caller cannot know it from the config and has to be told: a proxy is
    only reusable for the encoder that made it, and only the caller names the
    file it will be cached under.
    """
    return "h264_videotoolbox" if videotoolbox_available() else "libx264"


def make_proxy(src: Path, dst: Path, height: int, encoder: str | None = None) -> str:
    """Small proxy for analysis and draft renders. Returns the encoder used.

    Sources are 4K HEVC, so decode and encode go through VideoToolbox when the
    build supports it; software encoding is minutes per clip instead of seconds.
    Which one answered comes back to the caller because the two do not produce
    the same picture, and everything measured off a proxy — s02's blur score
    above all — moves with it.
    """
    chosen = encoder or proxy_encoder()
    scale = f"scale=-2:{height}"
    if chosen == "h264_videotoolbox":
        args = [
            "-hwaccel", "videotoolbox", "-i", str(src),
            "-vf", scale,
            "-c:v", "h264_videotoolbox", "-b:v", "2500k",
            "-c:a", "aac", "-b:a", "128k",
        ]
    else:
        args = [
            "-i", str(src),
            "-vf", scale,
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "26",
            "-c:a", "aac", "-b:a", "128k",
        ]
    _run_ffmpeg_to(args, dst)
    return chosen


def extract_frame(src: Path, at: float, dst: Path, height: int = 360) -> None:
    _run_ffmpeg_to(
        [
            "-ss", f"{at:.3f}", "-i", str(src),
            "-frames:v", "1", "-vf", f"scale=-2:{height}",
        ],
        dst,
    )
