"""Finding: a real 18-minute clip crashed with an MLX buffer-size error because
`ParakeetASR.transcribe` handed parakeet-mlx the whole file as one buffer.
parakeet-mlx supports chunked transcription (its own CLI defaults to
120s chunks / 15s overlap); `ParakeetASR` must pass configured chunk/overlap
values through on every call so long clips stay inside the GPU memory limit."""
from pathlib import Path

import pytest

from videoai.providers.asr_parakeet import ParakeetASR
from videoai.providers import asr_parakeet


class _FakeSentence:
    tokens: list = []


class _FakeResult:
    sentences: list = []


class _FakeModel:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def transcribe(self, path, **kwargs):
        self.calls.append(kwargs)
        return _FakeResult()


def test_transcribe_passes_configured_chunk_and_overlap_to_the_model(monkeypatch):
    fake_model = _FakeModel()
    monkeypatch.setattr(
        "videoai.providers.asr_parakeet._load_model", lambda model_name: fake_model
    )

    provider = ParakeetASR(chunk_duration_seconds=45.0, overlap_duration_seconds=5.0)
    provider.transcribe(Path("audio.wav"))

    assert len(fake_model.calls) == 1
    assert fake_model.calls[0]["chunk_duration"] == 45.0
    assert fake_model.calls[0]["overlap_duration"] == 5.0


def test_transcribe_defaults_match_the_documented_cli_defaults(monkeypatch):
    fake_model = _FakeModel()
    monkeypatch.setattr(
        "videoai.providers.asr_parakeet._load_model", lambda model_name: fake_model
    )

    ParakeetASR().transcribe(Path("audio.wav"))

    assert fake_model.calls[0]["chunk_duration"] == 120.0
    assert fake_model.calls[0]["overlap_duration"] == 15.0


def test_metal_probe_failure_is_a_managed_python_error(monkeypatch):
    class FailedProbe:
        returncode = 134
        stderr = "No Metal device available"
        stdout = ""

    asr_parakeet._require_usable_metal.cache_clear()
    monkeypatch.setattr(asr_parakeet.subprocess, "run", lambda *args, **kwargs: FailedProbe())

    with pytest.raises(RuntimeError, match="cannot access a Metal device"):
        asr_parakeet._require_usable_metal()
