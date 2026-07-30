"""Stage declaration and registry.

A stage is a pure-ish function: it reads artifacts through the context's store
and returns exactly one artifact model. The registry records what it needs and
what it produces so the runner can order and cache stages without importing them.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from pydantic import BaseModel

from videoai.config import Config
from videoai.core.store import ArtifactStore


@dataclass(frozen=True)
class StageContext:
    project_dir: Path
    input_dir: Path
    work_dir: Path
    output_dir: Path
    config: Config
    store: ArtifactStore


@dataclass(frozen=True)
class StageSpec:
    id: str
    produces: str
    requires: tuple[str, ...]
    provider_key: str | None
    version: str
    model: type[BaseModel]
    fn: Callable[[StageContext], BaseModel]
    uses_brief: bool = False
    # Dotted paths into `Config` (e.g. "render.draft_height") whose values this
    # stage actually reads, and the prompt text it sends to an LLM. Both are
    # mixed into the stage fingerprint: a setting or prompt a stage reads is an
    # input, and changing one has to invalidate that stage's cached artifact.
    config_keys: tuple[str, ...] = ()
    prompt: str | None = None
    # Some stages read only a semantic subset of a required artifact. For
    # example, transcription needs source audio identity from the manifest but
    # not the disposable proxy path. A projection prevents unrelated derived
    # media changes from rerunning an expensive provider.
    fingerprint_inputs: Callable[[StageContext], tuple[str, ...]] | None = None
    # Inputs a stage reads that are neither config nor artifacts — the production
    # contract's rules, for instance. Unlike `fingerprint_inputs` these are always
    # mixed in, whether or not the required artifacts exist yet, because they are
    # not derived from those artifacts.
    fingerprint_extras: Callable[[StageContext], tuple[str, ...]] | None = None


REGISTRY: dict[str, StageSpec] = {}


def stage(
    *,
    id: str,
    produces: str,
    requires: tuple[str, ...] = (),
    provider_key: str | None = None,
    version: str = "1",
    model: type[BaseModel],
    uses_brief: bool = False,
    config_keys: tuple[str, ...] = (),
    prompt: str | None = None,
    fingerprint_inputs: Callable[[StageContext], tuple[str, ...]] | None = None,
    fingerprint_extras: Callable[[StageContext], tuple[str, ...]] | None = None,
):
    def decorator(fn: Callable[[StageContext], BaseModel]):
        if id in REGISTRY:
            raise ValueError(f"stage already registered: {id}")
        for existing in REGISTRY.values():
            if existing.produces == produces:
                raise ValueError(
                    f"duplicate produces artifact '{produces}': "
                    f"stages '{existing.id}' and '{id}' both produce it"
                )
        REGISTRY[id] = StageSpec(
            id=id,
            produces=produces,
            requires=requires,
            provider_key=provider_key,
            version=version,
            model=model,
            fn=fn,
            uses_brief=uses_brief,
            config_keys=config_keys,
            prompt=prompt,
            fingerprint_inputs=fingerprint_inputs,
            fingerprint_extras=fingerprint_extras,
        )
        return fn

    return decorator
