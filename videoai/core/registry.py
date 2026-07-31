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


# The creator's running order and per-accent decisions: an artifact a person
# writes and no stage ever rewrites. A stage may `require` it, which feeds its
# content into that stage's fingerprint — so editing it re-runs the stages that
# read it — without the dependency graph looking for a stage that produces it.
#
# Declaring it is what ended the forgery. While the creator's decisions lived
# inside a model's artifact, a person's edit looked to the runner exactly like a
# stale reply, and the CLI had to recompute and store the fingerprint the stage
# WOULD have written so the work would survive the next run. An input authored by
# one party and never rewritten by another needs no such disguise.
CREATOR_ARTIFACTS: tuple[str, ...] = ("05e-overrides",)

# Artifacts a stage writes alongside its declared output: the story plan the
# planner also produces, and the ids the visual gate refused. Requiring one feeds
# its content into a fingerprint the same way, and it deliberately does not make
# the writing stage a dependency — `05c-rejected` flows backwards, from the gate
# to the planner it re-runs.
SIDE_ARTIFACTS: tuple[str, ...] = ("05a-storyplan", "05c-rejected")


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
