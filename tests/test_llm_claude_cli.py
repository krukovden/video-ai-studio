import json
import subprocess

import pytest

from videoai.providers.llm_claude_cli import ClaudeCliLLM, _extract_json


def test_extract_json_bare():
    assert _extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_fenced():
    text = '```json\n{"a": 1}\n```'
    assert _extract_json(text) == {"a": 1}


def test_extract_json_prose_around_json():
    text = 'Sure, here you go:\n{"a": 1}\nHope that helps.'
    assert _extract_json(text) == {"a": 1}


def test_extract_json_two_objects_raises_value_error():
    text = '{"a": 1} and also {"b": 2}'
    with pytest.raises(ValueError):
        _extract_json(text)


class _FakeResult:
    def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_complete_json_nonzero_exit_raises_runtime_error(monkeypatch):
    def fake_run(*args, **kwargs):
        return _FakeResult(returncode=1, stderr="boom")

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="claude CLI failed"):
        ClaudeCliLLM().complete_json("prompt", [], 10)


def test_complete_json_is_error_envelope_raises_runtime_error(monkeypatch):
    envelope = json.dumps({"is_error": True, "result": "something broke"})

    def fake_run(*args, **kwargs):
        return _FakeResult(returncode=0, stdout=envelope)

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="claude CLI returned an error"):
        ClaudeCliLLM().complete_json("prompt", [], 10)


def test_complete_json_timeout_raises_runtime_error(monkeypatch):
    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="claude", timeout=10)

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="timed out"):
        ClaudeCliLLM().complete_json("prompt", [], 10)


def test_complete_json_success_unwraps_result(monkeypatch):
    envelope = json.dumps({"is_error": False, "result": '{"segments": []}'})

    def fake_run(*args, **kwargs):
        return _FakeResult(returncode=0, stdout=envelope)

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert ClaudeCliLLM().complete_json("prompt", [], 10) == {"segments": []}
