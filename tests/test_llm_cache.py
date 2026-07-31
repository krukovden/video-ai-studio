"""Replaying a model's answer instead of asking it twice.

None of the CLI providers exposes a temperature, so the only way the pipeline
can make the same edit twice from the same footage is to remember what it was
told. These tests are about the two things that makes it: a hit no caller can
tell from a fresh call, and a key that misses the moment anything about the
question changes.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from videoai.providers.base import llm_cache_dir, resolve_llm
from videoai.providers.llm_cache import CachedLLM


class _CountingLLM:
    name = "claude_cli"
    reads_video = False

    def __init__(self, model: str = "sonnet", reply: dict | None = None) -> None:
        self.model = model
        self.reply = reply or {"segments": [{"phrase_id": "clip-01#001", "delivery_score": 7}]}
        self.calls = 0

    def complete_json(self, prompt, images, timeout, videos=None) -> dict:
        self.calls += 1
        return dict(self.reply)


def _cached(tmp_path: Path, inner: object) -> CachedLLM:
    return CachedLLM(inner, tmp_path / "work" / ".llm-cache")


def test_the_same_question_is_only_asked_once(tmp_path: Path):
    inner = _CountingLLM()
    provider = _cached(tmp_path, inner)

    first = provider.complete_json("score this", [], 60)
    second = provider.complete_json("score this", [], 60)

    assert inner.calls == 1
    assert second == first


def test_a_hit_hands_back_a_copy_the_caller_may_edit(tmp_path: Path):
    """A caller that edits the answer must not edit what the next run reads."""
    provider = _cached(tmp_path, _CountingLLM())

    answer = provider.complete_json("score this", [], 60)
    answer["segments"].clear()

    assert provider.complete_json("score this", [], 60)["segments"] != []


def test_a_changed_prompt_is_a_different_question(tmp_path: Path):
    inner = _CountingLLM()
    provider = _cached(tmp_path, inner)

    provider.complete_json("score this", [], 60)
    provider.complete_json("score this, and be strict", [], 60)

    assert inner.calls == 2


def test_a_changed_model_is_a_different_question(tmp_path: Path):
    cache = tmp_path / "work" / ".llm-cache"
    sonnet = _CountingLLM(model="sonnet")
    opus = _CountingLLM(model="opus")

    CachedLLM(sonnet, cache).complete_json("score this", [], 60)
    CachedLLM(opus, cache).complete_json("score this", [], 60)

    assert (sonnet.calls, opus.calls) == (1, 1)


def test_a_changed_provider_is_a_different_question(tmp_path: Path):
    cache = tmp_path / "work" / ".llm-cache"
    claude = _CountingLLM()
    codex = _CountingLLM()
    codex.name = "codex_cli"

    CachedLLM(claude, cache).complete_json("score this", [], 60)
    CachedLLM(codex, cache).complete_json("score this", [], 60)

    assert (claude.calls, codex.calls) == (1, 1)


def test_a_changed_system_preamble_is_a_different_question(tmp_path: Path, monkeypatch):
    """The Claude CLI takes its instruction as a separate argument, so it is not
    in the prompt: keyed on the prompt alone, editing the preamble would re-run
    the stage and replay the answer to the instruction it replaced."""
    import videoai.providers.llm_claude_cli as claude

    inner = _CountingLLM()
    provider = _cached(tmp_path, inner)
    provider.complete_json("score this", [], 60)

    monkeypatch.setattr(claude, "SYSTEM_PROMPT", "You are a blunt video editing analyst.")
    provider.complete_json("score this", [], 60)

    assert inner.calls == 2


def test_footage_that_changed_under_an_identical_prompt_is_asked_again(tmp_path: Path):
    """The reel is rebuilt on every run, so the key has to be the bytes rather
    than the path: keyed by name, a re-shot clip would be answered from the
    cache of the one it replaced."""
    reel = tmp_path / "reel.mp4"
    reel.write_bytes(b"first cut of the reel")
    inner = _CountingLLM()
    provider = _cached(tmp_path, inner)

    provider.complete_json("watch this", [], 60, videos=[reel])
    provider.complete_json("watch this", [], 60, videos=[reel])
    assert inner.calls == 1

    reel.write_bytes(b"a different cut of the same reel")
    provider.complete_json("watch this", [], 60, videos=[reel])

    assert inner.calls == 2


def test_identical_footage_rebuilt_at_a_new_path_still_hits(tmp_path: Path):
    first = tmp_path / "reel-a.mp4"
    second = tmp_path / "reel-b.mp4"
    first.write_bytes(b"the same seconds of footage")
    second.write_bytes(b"the same seconds of footage")
    inner = _CountingLLM()
    provider = _cached(tmp_path, inner)

    provider.complete_json("watch this", [], 60, videos=[first])
    provider.complete_json("watch this", [], 60, videos=[second])

    assert inner.calls == 1


def test_one_project_never_answers_for_another(tmp_path: Path):
    """Two shoots of the same toy produce the same brief and, early on, very
    similar prompts. An answer about one project's footage is not an answer
    about another's."""
    inner = _CountingLLM()

    CachedLLM(inner, tmp_path / "toy-a" / "work" / ".llm-cache").complete_json("p", [], 60)
    CachedLLM(inner, tmp_path / "toy-b" / "work" / ".llm-cache").complete_json("p", [], 60)

    assert inner.calls == 2


def test_a_failed_call_is_not_remembered_as_an_answer(tmp_path: Path):
    class _Flaky(_CountingLLM):
        def complete_json(self, prompt, images, timeout, videos=None):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("claude CLI failed: rate limited")
            return {"segments": []}

    provider = _cached(tmp_path, _Flaky())

    with pytest.raises(RuntimeError, match="rate limited"):
        provider.complete_json("score this", [], 60)

    assert provider.complete_json("score this", [], 60) == {"segments": []}


def test_the_wrapper_is_transparent(tmp_path: Path):
    provider = _cached(tmp_path, _CountingLLM())

    assert provider.name == "claude_cli"
    assert provider.reads_video is False
    assert provider.model == "sonnet"
    assert getattr(provider, "last_usage", None) is None


def test_a_hit_replays_what_the_call_reported_spending(tmp_path: Path):
    """`Analysis` records the token count as content, and every stage downstream
    chains on that content: a cached run that reported spending nothing would
    invalidate the whole edit by being free."""
    from videoai.providers.llm_gemini_api import Usage

    class _Metered(_CountingLLM):
        name = "gemini_api"
        reads_video = True
        last_usage: Usage | None = None

        def complete_json(self, prompt, images, timeout, videos=None):
            self.calls += 1
            self.last_usage = Usage(input_tokens=402, output_tokens=10, video_tokens=378)
            return {"segments": []}

    inner = _Metered(model="gemini-3.1-flash-lite")
    provider = _cached(tmp_path, inner)

    provider.complete_json("watch this", [], 60)
    inner.last_usage = None
    provider.complete_json("watch this", [], 60)

    assert inner.calls == 1
    assert provider.last_usage == Usage(input_tokens=402, output_tokens=10, video_tokens=378)


def test_an_unreadable_attachment_is_left_to_the_provider_to_report(tmp_path: Path):
    inner = _CountingLLM()
    provider = _cached(tmp_path, inner)

    provider.complete_json("watch this", [], 60, videos=[tmp_path / "gone.mp4"])

    assert inner.calls == 1
    assert not (tmp_path / "work" / ".llm-cache").exists()


def test_a_corrupt_entry_asks_the_question_again(tmp_path: Path):
    inner = _CountingLLM()
    provider = _cached(tmp_path, inner)
    provider.complete_json("score this", [], 60)
    for entry in (tmp_path / "work" / ".llm-cache").glob("*.json"):
        entry.write_text("half a file", encoding="utf-8")

    assert provider.complete_json("score this", [], 60) == dict(inner.reply)
    assert inner.calls == 2


def test_the_entry_records_which_model_answered(tmp_path: Path):
    provider = _cached(tmp_path, _CountingLLM())
    provider.complete_json("score this", [], 60)

    entry = json.loads(
        next((tmp_path / "work" / ".llm-cache").glob("*.json")).read_text(encoding="utf-8")
    )

    assert entry["provider"] == "claude_cli"
    assert entry["model"] == "sonnet"


# --- Switching it off, and where it is switched on ---


def test_the_environment_bypass_asks_every_question_again(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("VIDEOAI_LLM_CACHE", "0")

    provider = resolve_llm("claude_cli", "sonnet", cache_dir=llm_cache_dir(tmp_path))

    assert not isinstance(provider, CachedLLM)


def test_resolving_with_a_cache_dir_wraps_the_provider(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("VIDEOAI_LLM_CACHE", raising=False)

    provider = resolve_llm("claude_cli", "sonnet", cache_dir=llm_cache_dir(tmp_path))

    assert isinstance(provider, CachedLLM)
    assert provider.name == "claude_cli"


def test_resolving_without_a_cache_dir_leaves_the_provider_alone(tmp_path: Path):
    assert not isinstance(resolve_llm("claude_cli", "sonnet"), CachedLLM)


def test_the_mock_is_never_cached(tmp_path: Path):
    """Its fixtures answer a repeated question differently on purpose — the
    auto-fix loop plans twice — and a cache would replay round one for ever."""
    assert not isinstance(
        resolve_llm("mock", cache_dir=llm_cache_dir(tmp_path)), CachedLLM
    )


def test_the_cache_lives_inside_the_project(tmp_path: Path):
    assert llm_cache_dir(tmp_path / "work") == tmp_path / "work" / ".llm-cache"
