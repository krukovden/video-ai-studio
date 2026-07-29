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


def make_proxy(src: Path, dst: Path, height: int) -> None:
    """Small proxy for analysis and draft renders.

    Sources are 4K HEVC, so decode and encode go through VideoToolbox when the
    build supports it; software encoding is minutes per clip instead of seconds.
    """
    scale = f"scale=-2:{height}"
    if _has_videotoolbox_encoder():
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


def extract_frame(src: Path, at: float, dst: Path, height: int = 360) -> None:
    _run_ffmpeg_to(
        [
            "-ss", f"{at:.3f}", "-i", str(src),
            "-frames:v", "1", "-vf", f"scale=-2:{height}",
        ],
        dst,
    )
