"""Gemini through the metered API — the one provider that can watch the footage.

Every other provider in this pipeline is handed stills. This one uploads the
clips themselves, which is the whole reason to reach for Gemini: delivery,
timing and on-screen payoff are things you can only judge by watching.

No test here touches the network. The HTTP calls are three small seams —
upload, wait, generate — and each is replaced.
"""
from __future__ import annotations

import io
import json
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
    return GeminiApiLLM(model="gemini-3.1-flash-lite", api_key="test-key")


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
    assert sent["model"] == "gemini-3.1-flash-lite"


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


def test_no_media_resolution_is_sent(tmp_path, provider, monkeypatch):
    """Measured against a live key: the interactions endpoint rejects
    media_resolution in every position — generation_config, top level, and on the
    input part — and sending it fails the whole call."""
    sent: dict = {}
    monkeypatch.setattr(provider, "_upload", lambda path, timeout: "files/v")
    monkeypatch.setattr(provider, "_await_active", lambda uri, timeout: None)
    monkeypatch.setattr(
        provider, "_generate", lambda body, timeout: (sent.update(body), "{}")[1]
    )

    provider.complete_json("p", [], 10, videos=[_clip(tmp_path)])
    assert "media_resolution" not in json.dumps(sent)


def test_sampling_is_pinned_so_one_reel_is_scored_the_same_way_twice(
    tmp_path, provider, monkeypatch
):
    """Left alone the service samples at its own default, near 1.0: two runs
    disagree about the same take, the timeline moves, and nothing in the diff
    says why — which makes every other editorial change unreviewable. A seed as
    well as temperature 0, because greedy decoding still has to break ties."""
    sent: dict = {}
    monkeypatch.setattr(
        provider, "_generate", lambda body, timeout: (sent.update(body), "{}")[1]
    )

    provider.complete_json("score this", [], 10)

    assert sent["generation_config"] == {"temperature": 0.0, "top_p": 1.0, "seed": 7}


def test_the_sampling_controls_are_not_shared_between_calls(provider, monkeypatch):
    """A body handed straight to json.dumps is easy to mutate by accident; the
    next call must still be asked at temperature 0."""
    bodies: list[dict] = []
    monkeypatch.setattr(
        provider, "_generate", lambda body, timeout: (bodies.append(body), "{}")[1]
    )

    provider.complete_json("first", [], 10)
    bodies[0]["generation_config"]["temperature"] = 1.0
    provider.complete_json("second", [], 10)

    assert bodies[1]["generation_config"]["temperature"] == 0.0


def test_the_answer_is_read_from_the_model_output_step(provider, monkeypatch):
    """The interactions API answers in steps: the model's private reasoning is
    one and its answer another, so only the model_output step may be read."""
    monkeypatch.setattr(provider, "_generate", lambda body, timeout: _steps_reply())
    assert provider.complete_json("p", [], 10) == {"ok": True}


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
    provider = resolve_llm("gemini_api", "gemini-3.1-flash-lite")
    assert provider.name == "gemini_api"
    assert provider.reads_video is True


# --- Which model actually answered, and refusing to guess ---


def test_a_claude_model_is_refused_instead_of_quietly_becoming_a_gemini_one():
    """`llm_model: sonnet` is the shipped default, so the old startswith guard
    failed on every run: a creator configured one model and a different one
    scored their footage, with nothing anywhere recording the swap."""
    with pytest.raises(ValueError, match="sonnet") as failure:
        resolve_llm("gemini_api", "sonnet")

    # The diagnostic has to name what to write instead, not just what is wrong.
    assert "gemini-3.1-flash-lite" in str(failure.value)


def test_a_model_can_be_named_with_the_provider_for_one_stage():
    """One `analyze.llm_model` serves every stage, and a config that watches the
    footage with Gemini while the cheap stages stay on Claude needs two."""
    provider = resolve_llm("gemini_api:gemini-3.1-flash-lite", "sonnet")

    assert provider.name == "gemini_api"
    assert provider.model == "gemini-3.1-flash-lite"


def test_the_resolved_model_is_readable_without_building_a_provider():
    from videoai.providers.base import resolved_llm_model
    from videoai.providers.llm_gemini_api import DEFAULT_MODEL

    assert resolved_llm_model("gemini_api", "gemini-3.6-pro") == "gemini-3.6-pro"
    assert resolved_llm_model("gemini_api", "") == DEFAULT_MODEL
    assert resolved_llm_model("gemini_api:gemini-3.6-pro", "sonnet") == "gemini-3.6-pro"
    assert resolved_llm_model("claude_cli", "") == "sonnet"
    # Codex answers as whatever its authenticated profile picked, and the mock
    # has no model at all: neither is something this repository can pin.
    assert resolved_llm_model("codex_cli", "sonnet") == ""
    assert resolved_llm_model("mock", "sonnet") == ""


def _steps_reply() -> str:
    """A real interactions reply, shaped as the live API returns one."""
    from videoai.providers.llm_gemini_api import _reply_text

    return _reply_text({
        "status": "completed",
        "steps": [
            {"type": "thought", "signature": "opaque"},
            {"type": "model_output", "content": [{"type": "text", "text": '{"ok": true}'}]},
        ],
        "usage": {"total_input_tokens": 402},
    })


def test_a_thought_step_is_never_read_as_the_answer():
    from videoai.providers.llm_gemini_api import _reply_text

    text = _reply_text({"steps": [
        {"type": "thought", "content": [{"type": "text", "text": "let me think"}]},
        {"type": "model_output", "content": [{"type": "text", "text": '{"a": 1}'}]},
    ]})
    assert "think" not in text
    assert text == '{"a": 1}'


def test_the_usage_of_the_last_call_is_kept(provider, monkeypatch):
    """The API exposes no balance and no quota headers, so the only honest
    measure of what a run cost is what each reply reports it used."""
    monkeypatch.setattr(provider, "_generate_document", lambda body, timeout: {
        "steps": [{"type": "model_output", "content": [{"type": "text", "text": "{}"}]}],
        "usage": {"total_input_tokens": 402, "total_output_tokens": 10,
                  "input_tokens_by_modality": [{"modality": "video", "tokens": 378},
                                               {"modality": "text", "tokens": 24}]},
    })
    provider.complete_json("p", [], 10)

    assert provider.last_usage is not None
    assert provider.last_usage.input_tokens == 402
    assert provider.last_usage.output_tokens == 10
    assert provider.last_usage.video_tokens == 378


def test_usage_is_absent_until_a_call_is_made():
    assert GeminiApiLLM(api_key="k").last_usage is None


def test_a_reply_without_usage_does_not_invent_one(provider, monkeypatch):
    monkeypatch.setattr(provider, "_generate_document", lambda body, timeout: {
        "steps": [{"type": "model_output", "content": [{"type": "text", "text": "{}"}]}],
    })
    provider.complete_json("p", [], 10)
    assert provider.last_usage is None


def test_a_transient_server_error_is_retried(provider, monkeypatch):
    """Met on a real run: a 500 'high demand' threw away a 15-minute reel that
    had just been uploaded. Transient means try again, not start over."""
    import urllib.error

    attempts = {"n": 0}

    def flaky(request, timeout):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise urllib.error.HTTPError(
                "u", 500, "high demand", {}, io.BytesIO(b'{"error":{"code":"api_error"}}')
            )
        return io.BytesIO(json.dumps({
            "steps": [{"type": "model_output", "content": [{"type": "text", "text": "{}"}]}]
        }).encode())

    monkeypatch.setattr("videoai.providers.llm_gemini_api.time.sleep", lambda s: None)
    monkeypatch.setattr(
        "videoai.providers.llm_gemini_api.urllib.request.urlopen",
        lambda request, timeout=None: _as_context(flaky(request, timeout)),
    )
    assert provider._generate_document({"model": "m"}, 10) == {
        "steps": [{"type": "model_output", "content": [{"type": "text", "text": "{}"}]}]
    }
    assert attempts["n"] == 3


def test_a_bad_request_is_not_retried(provider, monkeypatch):
    """A 400 says the request is wrong; sending it again wastes the upload and
    the creator's time."""
    import urllib.error

    attempts = {"n": 0}

    def always_400(request, timeout=None):
        attempts["n"] += 1
        raise urllib.error.HTTPError(
            "u", 400, "bad", {}, io.BytesIO(b'{"error":{"message":"Unknown parameter"}}')
        )

    monkeypatch.setattr("videoai.providers.llm_gemini_api.time.sleep", lambda s: None)
    monkeypatch.setattr(
        "videoai.providers.llm_gemini_api.urllib.request.urlopen", always_400
    )
    with pytest.raises(RuntimeError, match="400"):
        provider._generate_document({"model": "m"}, 10)
    assert attempts["n"] == 1


class _as_context:
    def __init__(self, stream):
        self._stream = stream

    def __enter__(self):
        return self._stream

    def __exit__(self, *exc):
        return False
