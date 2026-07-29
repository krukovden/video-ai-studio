"""Stage ordering, fingerprinting and skip logic."""
from __future__ import annotations

from videoai.core.registry import REGISTRY, StageContext, StageSpec
from videoai.core.store import hash_parts


def _ordered_stages() -> list[StageSpec]:
    """Topological order: a stage runs after every stage producing its inputs."""
    produced_by = {spec.produces: spec.id for spec in REGISTRY.values()}
    ordered: list[StageSpec] = []
    placed: set[str] = set()
    pending = list(REGISTRY.values())
    while pending:
        progressed = False
        for spec in list(pending):
            deps = {produced_by[name] for name in spec.requires if name in produced_by}
            if deps <= placed:
                ordered.append(spec)
                placed.add(spec.id)
                pending.remove(spec)
                progressed = True
        if not progressed:
            unresolved = ", ".join(spec.id for spec in pending)
            raise ValueError(f"circular or unsatisfiable stage dependencies: {unresolved}")
    return ordered


def ordered_stages() -> list[StageSpec]:
    """Public alias for `_ordered_stages`, for callers outside this module (e.g. the CLI)."""
    return _ordered_stages()


def _fingerprint(spec: StageSpec, ctx: StageContext, extra_fingerprint: str) -> str:
    parts = [spec.id, spec.version, extra_fingerprint]
    if spec.provider_key:
        parts.append(f"{spec.provider_key}={ctx.config.providers.get(spec.provider_key, '')}")
    for name in spec.requires:
        parts.append(f"{name}:{ctx.store.fingerprint(name) or ''}")
    return hash_parts(*parts)


def run_pipeline(
    ctx: StageContext,
    only: str | None = None,
    force: bool = False,
    extra_fingerprint: str = "",
) -> list[str]:
    """Run stages in order. Returns ids of stages that actually executed."""
    if only is not None and only not in REGISTRY:
        raise KeyError(f"unknown stage: {only}")

    executed: list[str] = []
    for spec in _ordered_stages():
        if only is not None and spec.id != only:
            continue
        fingerprint = _fingerprint(spec, ctx, extra_fingerprint)
        cached = ctx.store.fingerprint(spec.produces)
        # An explicit `only=<id>` request always runs that stage, bypassing the
        # cache: the caller asked for this stage to run now, and skipping it
        # could rely on stale upstream fingerprints that were never re-evaluated
        # (every other stage was skipped over via `continue` above).
        skip_cached = (
            only is None
            and not force
            and cached == fingerprint
            and ctx.store.exists(spec.produces)
        )
        if skip_cached:
            continue
        artifact = spec.fn(ctx)
        if not isinstance(artifact, spec.model):
            raise TypeError(
                f"stage {spec.id} returned {type(artifact).__name__}, expected {spec.model.__name__}"
            )
        ctx.store.write(spec.produces, artifact, fingerprint)
        executed.append(spec.id)
    return executed
