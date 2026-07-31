"""Gemini through its own CLI, on the user's Google account.

A drop-in sibling of the Claude and Codex CLI providers: same protocol, same
JSON envelope handling, no metered API key. It reads text and stills — the CLI
has no way to hand a model a video file — so it does not claim to watch video.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from videoai.providers.base import resolve_llm
from videoai.providers.llm_gemini_cli import GeminiCliLLM


class _FakeResult:
    def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _envelope(response: str) -> str:
    return json.dumps({"response": response})


def test_it_does_not_claim_to_read_video():
    """The whole reason to reach for Gemini is native video. The CLI cannot do
    it, so this must say so rather than let the analyze stage assume otherwise."""
    assert GeminiCliLLM().reads_video is False


def test_a_json_reply_is_unwrapped_from_the_envelope(monkeypatch):
    monkeypatch.setattr(
        subprocess, "run",
        lambda *a, **k: _FakeResult(stdout=_envelope('{"segments": []}')),
    )
    assert GeminiCliLLM().complete_json("prompt", [], 10) == {"segments": []}


def test_a_fenced_reply_is_still_read(monkeypatch):
    monkeypatch.setattr(
        subprocess, "run",
        lambda *a, **k: _FakeResult(stdout=_envelope('```json\n{"a": 1}\n```')),
    )
    assert GeminiCliLLM().complete_json("prompt", [], 10) == {"a": 1}


def test_a_bare_json_stdout_is_accepted(monkeypatch):
    """Not every CLI version wraps its answer; a bare document must still work."""
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: _FakeResult(stdout='{"a": 2}')
    )
    assert GeminiCliLLM().complete_json("prompt", [], 10) == {"a": 2}


def test_a_nonzero_exit_raises_with_the_diagnostic(monkeypatch):
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: _FakeResult(returncode=1, stderr="not logged in")
    )
    with pytest.raises(RuntimeError, match="not logged in"):
        GeminiCliLLM().complete_json("prompt", [], 10)


def test_a_silent_failure_still_explains_itself(monkeypatch):
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: _FakeResult(returncode=2)
    )
    with pytest.raises(RuntimeError, match="exit code 2"):
        GeminiCliLLM().complete_json("prompt", [], 10)


def test_a_timeout_is_named_as_one(monkeypatch):
    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="gemini", timeout=10)

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="timed out"):
        GeminiCliLLM().complete_json("prompt", [], 10)


def test_it_runs_headless_and_asks_for_json(monkeypatch):
    seen: dict[str, list[str]] = {}

    def fake_run(args, **kwargs):
        seen["args"] = args
        return _FakeResult(stdout=_envelope("{}"))

    monkeypatch.setattr(subprocess, "run", fake_run)
    GeminiCliLLM(model="gemini-3.6-flash").complete_json("prompt", [], 10)

    args = seen["args"]
    assert args[0] == "gemini"
    assert "-p" in args                      # headless, not an interactive session
    assert "--output-format" in args and "json" in args
    assert "gemini-3.6-flash" in args
    # A read-only editorial call must never be allowed to edit the workspace.
    assert "--approval-mode" in args and "plan" in args


def test_reference_frames_are_named_in_the_prompt(monkeypatch):
    seen: dict[str, list[str]] = {}

    def fake_run(args, **kwargs):
        seen["args"] = args
        return _FakeResult(stdout=_envelope("{}"))

    monkeypatch.setattr(subprocess, "run", fake_run)
    GeminiCliLLM().complete_json("look at these", [Path("/frames/a.jpg")], 10)

    prompt = seen["args"][seen["args"].index("-p") + 1]
    assert "/frames/a.jpg" in prompt


def test_passing_video_to_a_provider_that_cannot_watch_it_is_refused(monkeypatch):
    """Silently dropping the video would return a text-quality answer while the
    caller believed it had bought video understanding."""
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: _FakeResult(stdout=_envelope("{}"))
    )
    with pytest.raises(ValueError, match="cannot read video"):
        GeminiCliLLM().complete_json("prompt", [], 10, videos=[Path("/clips/a.mp4")])


def test_the_registry_knows_it():
    provider = resolve_llm("gemini_cli", "gemini-3.6-flash")
    assert provider.name == "gemini_cli"
    assert provider.model == "gemini-3.6-flash"


def test_a_claude_model_is_refused_rather_than_dropped_for_the_cli_default():
    """Passing no model let the CLI pick, which is a different model answering
    than the one the creator configured — and nothing said so."""
    with pytest.raises(ValueError, match="not a Gemini model"):
        resolve_llm("gemini_cli", "sonnet")
