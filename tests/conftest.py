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


@pytest.fixture
def make_clip(tmp_path: Path):
    def _make(name: str, seconds: float = 3.0, tone_hz: int = 440, size: str = "320x240") -> Path:
        return _generate_clip(tmp_path / name, seconds, tone_hz, size)

    return _make
