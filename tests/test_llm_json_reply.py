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


def test_extract_json_two_objects_reads_the_first_one():
    """The answer comes first and the chatter after it. A greedy match spanning
    both is not valid JSON, so the whole reply used to be thrown away."""
    assert extract_json('{"a": 1} and also {"b": 2}') == {"a": 1}


def test_extract_json_without_any_object_raises_value_error():
    with pytest.raises(ValueError, match="no JSON object found"):
        extract_json("I would rather not.")


# --- Only the fence around the reply is a fence; the ones inside it are content ---


def test_a_second_fenced_block_does_not_damage_the_first():
    """A model that answers and then adds a worked example writes four fence
    markers. Stripping every one of them (a line-anchored strip over the whole
    reply) leaves the two documents run together and unparsable."""
    reply = (
        '```json\n{"summary": "what happens", "events": []}\n```\n\n'
        'For example:\n\n```json\n{"summary": "a different clip"}\n```'
    )

    assert extract_json(reply) == {"summary": "what happens", "events": []}


def test_a_fence_quoted_inside_a_value_survives():
    reply = '```json\n{"summary": "the note said ```pop``` in bold"}\n```'

    assert extract_json(reply) == {"summary": "the note said ```pop``` in bold"}


def test_a_closing_brace_in_the_prose_after_the_answer_is_not_part_of_it():
    reply = 'Here you go: {"a": 1}\nHope that helps (see the {} above).'

    assert extract_json(reply) == {"a": 1}


def test_braces_inside_a_string_value_are_not_structure():
    assert extract_json('Note: {"a": "} not the end", "b": 2} thanks') == {
        "a": "} not the end",
        "b": 2,
    }


def test_an_escaped_quote_does_not_end_the_string_early():
    assert extract_json('prose {"a": "she said \\"} hi\\"", "b": 2} more') == {
        "a": 'she said "} hi"',
        "b": 2,
    }


def test_an_unfinished_object_is_a_diagnostic_not_a_crash():
    with pytest.raises(ValueError, match="no JSON object found"):
        extract_json('{"a": 1, "b": [')


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
