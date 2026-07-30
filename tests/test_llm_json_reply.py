"""Reading JSON out of a model reply, and the diagnostic when a CLI fails.

Both live in one shared module because both LLM providers use them: a reply one
provider could parse and the other could not would make the pipeline's behaviour
depend on which subscription is configured.
"""
import subprocess

import pytest

from videoai.providers.json_reply import cli_diagnostic, extract_json


def test_extract_json_bare():
    assert extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_fenced():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_extract_json_prose_around_json():
    assert extract_json('Sure, here you go:\n{"a": 1}\nHope that helps.') == {"a": 1}


def test_extract_json_two_objects_raises_value_error():
    with pytest.raises(ValueError):
        extract_json('{"a": 1} and also {"b": 2}')


def test_extract_json_without_any_object_raises_value_error():
    with pytest.raises(ValueError, match="no JSON object found"):
        extract_json("I would rather not.")


def test_the_cli_diagnostic_prefers_stderr_then_stdout():
    assert "not authenticated" in cli_diagnostic("codex", 1, "not authenticated", "")
    assert "on stdout" in cli_diagnostic("codex", 1, "", "on stdout")


def test_an_empty_diagnostic_explains_the_silent_exit_instead_of_saying_nothing():
    message = cli_diagnostic("codex", 127, "", "")

    assert "exit code 127" in message
    assert "headless or sandboxed" in message


@pytest.mark.parametrize("provider", ["claude", "codex"])
def test_both_providers_report_a_silent_failure_the_same_way(provider, monkeypatch):
    """A CLI blocked by a sandbox exits non-zero and says nothing. Neither provider
    may pass that emptiness through as its whole diagnostic."""
    from videoai.providers.llm_claude_cli import ClaudeCliLLM
    from videoai.providers.llm_codex_cli import CodexCliLLM

    class _Result:
        returncode = 13
        stdout = ""
        stderr = "   "

    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: _Result())
    llm = ClaudeCliLLM() if provider == "claude" else CodexCliLLM()

    with pytest.raises(RuntimeError, match="exit code 13 with no diagnostic output"):
        llm.complete_json("prompt", [], 10)
