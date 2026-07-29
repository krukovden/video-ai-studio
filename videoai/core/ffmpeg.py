"""Thin, explicit wrappers over ffmpeg/ffprobe. No hidden defaults."""
from __future__ import annotations

import json
import subprocess
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
    numerator, _, denominator = video.get("r_frame_rate", "30/1").partition("/")
    fps = float(numerator) / float(denominator or 1)
    return ProbeResult(
        duration=float(data.get("format", {}).get("duration", 0.0)),
        width=int(video["width"]),
        height=int(video["height"]),
        fps=fps,
        has_audio=any(s.get("codec_type") == "audio" for s in streams),
    )


def list_video_files(directory: Path) -> list[Path]:
    """Video files directly inside `directory`, sorted, macOS metadata excluded."""
    if not directory.is_dir():
        return []
    return sorted(
        path for path in directory.iterdir()
        if path.is_file()
        and not path.name.startswith((".", "._"))
        and path.suffix.lower() in VIDEO_SUFFIXES
    )


@cache
def _has_videotoolbox_encoder() -> bool:
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True
    )
    return "h264_videotoolbox" in result.stdout


def extract_audio(src: Path, dst: Path) -> None:
    """16 kHz mono WAV with EBU R128 loudness normalisation, ready for ASR."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    run_ffmpeg([
        "-i", str(src),
        "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
        "-ac", "1", "-ar", "16000", "-vn",
        str(dst),
    ])


def make_proxy(src: Path, dst: Path, height: int) -> None:
    """Small proxy for analysis and draft renders.

    Sources are 4K HEVC, so decode and encode go through VideoToolbox when the
    build supports it; software encoding is minutes per clip instead of seconds.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    scale = f"scale=-2:{height}"
    if _has_videotoolbox_encoder():
        args = [
            "-hwaccel", "videotoolbox", "-i", str(src),
            "-vf", scale,
            "-c:v", "h264_videotoolbox", "-b:v", "2500k",
            "-c:a", "aac", "-b:a", "128k",
            str(dst),
        ]
    else:
        args = [
            "-i", str(src),
            "-vf", scale,
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "26",
            "-c:a", "aac", "-b:a", "128k",
            str(dst),
        ]
    run_ffmpeg(args)


def extract_frame(src: Path, at: float, dst: Path, height: int = 360) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    run_ffmpeg([
        "-ss", f"{at:.3f}", "-i", str(src),
        "-frames:v", "1", "-vf", f"scale=-2:{height}",
        str(dst),
    ])
