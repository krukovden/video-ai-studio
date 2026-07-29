import subprocess
from pathlib import Path

import pytest


def _generate_clip(path: Path, seconds: float, tone_hz: int, size: str = "320x240") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", f"testsrc=size={size}:rate=30:duration={seconds}",
            "-f", "lavfi", "-i", f"sine=frequency={tone_hz}:duration={seconds}",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
            str(path),
        ],
        check=True,
    )
    return path


def _generate_silent_clip(path: Path, seconds: float, size: str = "320x240") -> Path:
    """A clip with no audio input at all — the muted-camera case: a second angle
    with an unusable or absent microphone, which is ordinary for this project's
    two-camera setups."""
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", f"testsrc=size={size}:rate=30:duration={seconds}",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            str(path),
        ],
        check=True,
    )
    return path


def _generate_stereo_clip(path: Path, seconds: float, tone_hz: int, size: str = "320x240") -> Path:
    """A clip with stereo audio at 48000 Hz — this project's real footage (AAC
    stereo from an iPhone), unlike `make_clip`'s mono 44100 Hz `sine=` default,
    which happens to already match the draft's audio target and would hide a
    render bug that only shows up when the source needs resampling/downmixing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", f"testsrc=size={size}:rate=30:duration={seconds}",
            "-f", "lavfi", "-i", f"sine=frequency={tone_hz}:sample_rate=48000:duration={seconds}",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-ac", "2", "-shortest",
            str(path),
        ],
        check=True,
    )
    return path


@pytest.fixture
def make_clip(tmp_path: Path):
    def _make(name: str, seconds: float = 3.0, tone_hz: int = 440, size: str = "320x240") -> Path:
        return _generate_clip(tmp_path / name, seconds, tone_hz, size)

    return _make


@pytest.fixture
def make_silent_clip(tmp_path: Path):
    def _make(name: str, seconds: float = 3.0, size: str = "320x240") -> Path:
        return _generate_silent_clip(tmp_path / name, seconds, size)

    return _make


@pytest.fixture
def make_stereo_clip(tmp_path: Path):
    def _make(name: str, seconds: float = 3.0, tone_hz: int = 440, size: str = "320x240") -> Path:
        return _generate_stereo_clip(tmp_path / name, seconds, tone_hz, size)

    return _make
