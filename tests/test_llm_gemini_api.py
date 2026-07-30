"""Gemini through the metered API — the one provider that can watch the footage.

Every other provider in this pipeline is handed stills. This one uploads the
clips themselves, which is the whole reason to reach for Gemini: delivery,
timing and on-screen payoff are things you can only judge by watching.

No test here touches the network. The HTTP calls are three small seams —
upload, wait, generate — and each is replaced.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from videoai.providers.base import resolve_llm
from videoai.providers.llm_gemini_api import GeminiApiLLM, video_token_estimate


def _clip(tmp_path: Path, name: str = "a.mp4") -> Path:
    path = tmp_path / name
    path.write_bytes(b"not really an mp4, but bytes all the same")
    return path


@pytest.fixture
def provider(monkeypatch):
    return GeminiApiLLM(model="gemini-2.5-flash", api_key="test-key")


def test_it_declares_that_it_reads_video():
    assert GeminiApiLLM(api_key="k").reads_video is True


def test_a_missing_key_is_refused_before_any_upload(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        GeminiApiLLM().complete_json("prompt", [], 10)


def test_the_key_is_read_from_the_environment(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "from-env")
    assert GeminiApiLLM().api_key == "from-env"


def test_a_video_is_uploaded_and_referenced_by_uri(tmp_path, provider, monkeypatch):
    clip = _clip(tmp_path)
    uploaded: list[Path] = []
    sent: dict = {}

    monkeypatch.setattr(
        provider, "_upload", lambda path, timeout: (uploaded.append(path), "files/abc")[1]
    )
    monkeypatch.setattr(provider, "_await_active", lambda uri, timeout: None)
    monkeypatch.setattr(
        provider, "_generate",
        lambda body, timeout: (sent.update(body), '{"segments": []}')[1],
    )

    assert provider.complete_json("score this", [], 60, videos=[clip]) == {"segments": []}
    assert uploaded == [clip]

    parts = sent["input"]
    assert parts[0]["type"] == "text" and "score this" in parts[0]["text"]
    video_part = next(part for part in parts if part["type"] == "video")
    assert video_part["uri"] == "files/abc"
    assert video_part["mime_type"] == "video/mp4"
    assert sent["model"] == "gemini-2.5-flash"


def test_stills_are_uploaded_as_images(tmp_path, provider, monkeypatch):
    frame = tmp_path / "f.jpg"
    frame.write_bytes(b"jpeg")
    sent: dict = {}
    monkeypatch.setattr(provider, "_upload", lambda path, timeout: "files/img")
    monkeypatch.setattr(provider, "_await_active", lambda uri, timeout: None)
    monkeypatch.setattr(
        provider, "_generate", lambda body, timeout: (sent.update(body), "{}")[1]
    )

    provider.complete_json("look", [frame], 60)
    image_part = next(part for part in sent["input"] if part["type"] == "image")
    assert image_part["uri"] == "files/img"
    assert image_part["mime_type"] == "image/jpeg"


def test_a_fenced_reply_is_still_read(tmp_path, provider, monkeypatch):
    monkeypatch.setattr(
        provider, "_generate", lambda body, timeout: '```json\n{"a": 1}\n```'
    )
    assert provider.complete_json("p", [], 10) == {"a": 1}


def test_media_resolution_is_sent_when_asked_for(tmp_path, provider, monkeypatch):
    """Low resolution is a third of the tokens; on a long review that is the
    difference between pennies and a dollar, so it must actually be requested."""
    sent: dict = {}
    provider.media_resolution = "low"
    monkeypatch.setattr(provider, "_upload", lambda path, timeout: "files/v")
    monkeypatch.setattr(provider, "_await_active", lambda uri, timeout: None)
    monkeypatch.setattr(
        provider, "_generate", lambda body, timeout: (sent.update(body), "{}")[1]
    )

    provider.complete_json("p", [], 10, videos=[_clip(tmp_path)])
    assert sent["generation_config"]["media_resolution"] == "MEDIA_RESOLUTION_LOW"


def test_no_media_resolution_key_when_left_at_default(tmp_path, provider, monkeypatch):
    sent: dict = {}
    provider.media_resolution = None
    monkeypatch.setattr(
        provider, "_generate", lambda body, timeout: (sent.update(body), "{}")[1]
    )
    provider.complete_json("p", [], 10)
    assert "generation_config" not in sent


def test_an_unreadable_upload_is_named(tmp_path, provider, monkeypatch):
    missing = tmp_path / "gone.mp4"
    with pytest.raises(RuntimeError, match="gone.mp4"):
        provider.complete_json("p", [], 10, videos=[missing])


def test_token_estimate_matches_the_documented_rates():
    # ~300 tokens per second of video at default resolution, ~100 at low.
    assert video_token_estimate(60.0, "default") == 18_000
    assert video_token_estimate(60.0, "low") == 6_000
    # The real project: 28.8 minutes of source.
    assert video_token_estimate(1728.0, "low") == 172_800


def test_the_registry_knows_it():
    provider = resolve_llm("gemini_api", "gemini-2.5-flash")
    assert provider.name == "gemini_api"
    assert provider.reads_video is True
