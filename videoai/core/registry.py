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
        )
        return fn

    return decorator
