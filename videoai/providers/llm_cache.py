"""Replaying a model's answer instead of asking for it twice.

Only the metered API exposes sampling controls. The three CLI providers run a
model through somebody else's client and take whatever temperature that client
picked, so two runs over the same footage can score the same phrase differently,
and the edit moves for a reason no diff can show. Reviewing an edit means being
able to change one thing and see what that one thing did; a pipeline whose model
layer answers differently every time cannot be reviewed at all.

So an answer is kept and replayed. The key is everything that went into asking
for it — the provider, the model that will actually answer, the whole prompt,
and the bytes of every attachment — because anything else is a different
question and has to miss. Attachments are hashed by content rather than by path
and mtime: the analyze reel is rebuilt from scratch on every run, so a
timestamp key would never hit even when the footage is identical.

The cache lives under the project's own `work/` directory. That is what stops
one project being served another's answer to an identically worded prompt, and
it means deleting `work/` asks every question again.
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path

from videoai.core.store import hash_file, hash_parts

CACHE_DIRNAME = ".llm-cache"

# The documented bypass: `VIDEOAI_LLM_CACHE=0 videoai produce ...` asks every
# question again without editing config or deleting anything.
DISABLED_VALUES = frozenset({"0", "off", "false", "no"})


def caching_enabled() -> bool:
    return os.getenv("VIDEOAI_LLM_CACHE", "1").strip().lower() not in DISABLED_VALUES


def cache_key(
    provider: str, model: str, prompt: str, images: list[Path], videos: list[Path]
) -> str | None:
    """The digest under which this exact question is filed, or None if it cannot
    be computed — an attachment that is missing or unreadable is the provider's
    diagnostic to give, not something to answer from cache."""
    from videoai.providers.base import llm_system_preamble

    # The preamble is part of what was asked even when it does not travel in the
    # prompt: the Claude CLI takes it as a separate argument, so editing it would
    # otherwise re-run the stage and replay the answer to the old instruction.
    parts = [provider, model, llm_system_preamble(provider), prompt]
    try:
        parts += [f"video:{hash_file(path)}" for path in videos]
        parts += [f"image:{hash_file(path)}" for path in images]
    except OSError:
        return None
    return hash_parts(*parts)


def _read(cache_dir: Path, key: str) -> dict | None:
    path = cache_dir / f"{key}.json"
    if not path.is_file():
        return None
    try:
        entry = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        # A half-written or hand-edited entry is not a reason to fail a run that
        # is perfectly able to ask the question again.
        return None
    return entry if isinstance(entry.get("reply"), dict) else None


def _write(cache_dir: Path, key: str, entry: dict) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=cache_dir, suffix=".tmp")
    with os.fdopen(handle, "w", encoding="utf-8") as stream:
        json.dump(entry, stream, indent=2, sort_keys=True)
    os.replace(temporary, cache_dir / f"{key}.json")


def _replay_usage(provider: object, raw: dict | None) -> None:
    """Put back the token accounting the original call reported.

    A cached run spent nothing, and reporting nothing would still be wrong: the
    usage lands in `04-analysis` as content, every stage downstream chains on
    that content, and a run that answered from cache would invalidate the whole
    edit by being cheaper. What the cache replays is what the question cost when
    it was asked.
    """
    if raw is None or not hasattr(provider, "last_usage"):
        return
    from videoai.providers.llm_gemini_api import Usage

    provider.last_usage = Usage(**raw)


class CachedLLM:
    """A provider that answers from `work/.llm-cache` when it has been asked this
    exact question before.

    Deliberately indistinguishable from what it wraps: the same protocol, the
    same attributes (`name`, `reads_video`, `model`, `last_usage`), and a fresh
    dict per hit so a caller that edits the answer does not edit the cache.
    """

    def __init__(self, inner: object, cache_dir: Path) -> None:
        self._inner = inner
        self._cache_dir = cache_dir

    def __getattr__(self, attribute: str) -> object:
        # Reached only for what this wrapper does not define itself. Private
        # names are refused outright so a lookup during construction cannot
        # recurse through `_inner` before there is one.
        if attribute.startswith("_"):
            raise AttributeError(attribute)
        return getattr(self._inner, attribute)

    def complete_json(
        self,
        prompt: str,
        images: list[Path],
        timeout: int,
        videos: list[Path] | None = None,
    ) -> dict:
        videos = list(videos or [])
        key = cache_key(
            self._inner.name, getattr(self._inner, "model", "") or "", prompt, images, videos
        )
        entry = _read(self._cache_dir, key) if key else None
        if entry is not None:
            _replay_usage(self._inner, entry.get("usage"))
            return entry["reply"]

        reply = self._inner.complete_json(prompt, images, timeout, videos=videos)
        if key is None:
            return reply
        usage = getattr(self._inner, "last_usage", None)
        _write(
            self._cache_dir,
            key,
            {
                "provider": self._inner.name,
                "model": getattr(self._inner, "model", "") or "",
                "reply": reply,
                "usage": asdict(usage) if usage is not None else None,
            },
        )
        return reply
