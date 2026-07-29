"""Mock ASR: reads word timings from a sidecar file next to the audio."""
from __future__ import annotations

import json
from pathlib import Path

from videoai.core.models import Word


class MockASR:
    name = "mock"

    def transcribe(self, audio_path: Path) -> list[Word]:
        sidecar = audio_path.with_suffix(".words.json")
        if not sidecar.exists():
            raise FileNotFoundError(
                f"mock ASR needs {sidecar.name} next to the audio file: {sidecar}"
            )
        raw = json.loads(sidecar.read_text(encoding="utf-8"))
        return [Word(**item) for item in raw]
