"""Stage ordering, fingerprinting and skip logic."""
from __future__ import annotations

from videoai.config import Config
from videoai.core.registry import (
    CREATOR_ARTIFACTS,
    REGISTRY,
    SIDE_ARTIFACTS,
    StageContext,
    StageSpec,
)
from videoai.core.store import hash_parts


class StageFailure(RuntimeError):
    """A stage raised. Carries the stage id so callers can name it and tell the
    user how to re-run just that stage."""

    def __init__(self, stage_id: str, cause: BaseException) -> None:
        super().__init__(f"stage '{stage_id}' failed: {cause}")
        self.stage_id = stage_id
        self.cause = cause


def _ordered_stages() -> list[StageSpec]:
    """Topological order: a stage runs after every stage producing its inputs."""
    produced_by = {spec.produces: spec.id for spec in REGISTRY.values()}
    # A requirement nobody produces is a declared creator input, a declared side
    # artifact, or a typo — and a typo here is silent: the missing content hashes
    # as '' and the stage stays cached across every change to the artifact it
    # meant to name.
    declared = set(CREATOR_ARTIFACTS) | set(SIDE_ARTIFACTS)
    for spec in REGISTRY.values():
        for name in spec.requires:
            if name not in produced_by and name not in declared:
                raise ValueError(
                    f"stage '{spec.id}' requires '{name}', which no stage produces "
                    "and which registry.py does not declare as a creator input or a "
                    "side artifact"
                )
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


def config_value(config: Config, dotted_key: str) -> object:
    """Resolve a dotted path such as "render.draft_height" against the config.

    A typo raises rather than silently contributing nothing to a fingerprint,
    which would leave the stage cached across a setting it really does read.
    """
    current: object = config
    for part in dotted_key.split("."):
        if not hasattr(current, part):
            raise KeyError(f"unknown config key: {dotted_key}")
        current = getattr(current, part)
    return current


def _fingerprint(
    spec: StageSpec, ctx: StageContext, media_fingerprint: str, brief_fingerprint: str
) -> str:
    # Every stage depends on the source media; only stages that actually read the
    # creator's brief (analyze, plan) also depend on it. Everything downstream of
    # those (e.g. render_draft, via its `requires` on 05-timeline) still picks up
    # a brief change through the `requires`-content chain below, without needing
    # to know about the brief itself.
    parts = [spec.id, spec.version, media_fingerprint]
    if spec.uses_brief:
        parts.append(brief_fingerprint)
    if spec.provider_key:
        # The provider this stage will actually call, which may be its own
        # override rather than the pipeline default: pointing one stage at a
        # different model has to re-run that stage and only that stage.
        provider = (
            ctx.config.llm_for(spec.id)
            if spec.provider_key == "llm"
            else ctx.config.providers.get(spec.provider_key, "")
        )
        parts.append(f"{spec.provider_key}={provider}")
        if spec.provider_key == "llm":
            # The provider's own fixed instruction is as much a part of the prompt
            # as `spec.prompt` is; Codex's preamble is prefixed to every prompt it
            # sends, so editing it has to invalidate the analysis it produced.
            from videoai.providers.base import llm_system_preamble, resolved_llm_model

            parts.append(f"system:{hash_parts(llm_system_preamble(provider))}")
            # The provider name is an alias, not a model: "gemini_api" answers as
            # whatever its DEFAULT_MODEL points at, and one model's judgement of a
            # take is not another's. Resolved without constructing a provider, so
            # fingerprinting still starts no CLI and opens no socket.
            # `analyze.llm_model` is the pipeline-wide setting every model-calling
            # stage hands to its provider.
            parts.append(
                f"model:{resolved_llm_model(provider, ctx.config.analyze.llm_model)}"
            )
    for key in spec.config_keys:
        parts.append(f"{key}={config_value(ctx.config, key)!r}")
    if spec.prompt is not None:
        parts.append(f"prompt:{hash_parts(spec.prompt)}")
    if spec.fingerprint_extras is not None:
        parts.extend(f"extra:{value}" for value in spec.fingerprint_extras(ctx))
    # Chain on upstream CONTENT, not on upstream fingerprints. `analyze` and
    # `plan` are LLM calls: re-running one with `--stage` produces different
    # content under an identical fingerprint, so a fingerprint chain would leave
    # everything downstream silently stale. Hashing the artifact on disk also
    # means a hand-edited artifact invalidates its dependents.
    #
    # A stage may narrow that to a projection of the content it really reads (see
    # `fingerprint_inputs`), which is only consulted once every required artifact
    # exists — a projection reads those artifacts to compute itself. Until then the
    # whole-artifact hashes below stand in, and they are '' for what is missing.
    if (
        spec.fingerprint_inputs is not None
        and all(ctx.store.exists(name) for name in spec.requires)
    ):
        parts.extend(f"input:{value}" for value in spec.fingerprint_inputs(ctx))
    else:
        for name in spec.requires:
            parts.append(f"{name}:{ctx.store.content_hash(name) or ''}")
    return hash_parts(*parts)


def run_pipeline(
    ctx: StageContext,
    only: str | None = None,
    stop_after: str | None = None,
    force: bool = False,
    media_fingerprint: str = "",
    brief_fingerprint: str = "",
) -> list[str]:
    """Run stages in order. Returns ids of stages that actually executed."""
    if only is not None and only not in REGISTRY:
        raise KeyError(f"unknown stage: {only}")
    if stop_after is not None and stop_after not in REGISTRY:
        raise KeyError(f"unknown stage: {stop_after}")

    executed: list[str] = []
    for spec in _ordered_stages():
        if only is not None and spec.id != only:
            continue
        fingerprint = _fingerprint(spec, ctx, media_fingerprint, brief_fingerprint)
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
            if stop_after == spec.id:
                break
            continue
        try:
            artifact = spec.fn(ctx)
        except Exception as error:
            raise StageFailure(spec.id, error) from error
        if not isinstance(artifact, spec.model):
            raise TypeError(
                f"stage {spec.id} returned {type(artifact).__name__}, expected {spec.model.__name__}"
            )
        ctx.store.write(spec.produces, artifact, fingerprint)
        executed.append(spec.id)
        if stop_after == spec.id:
            break
    return executed


def stale_downstream(
    ctx: StageContext, ran_id: str, media_fingerprint: str, brief_fingerprint: str
) -> list[str]:
    """After running a single stage via `only=`, which cached stages (in pipeline
    order) now disagree with a freshly recomputed fingerprint — i.e. whose artifact
    was built against an input that has since moved on. `run_pipeline` always
    re-executes an explicitly requested stage regardless of cache state, which can
    leave everything downstream of it silently stale."""
    stale: list[str] = []
    for spec in _ordered_stages():
        if spec.id == ran_id or not ctx.store.exists(spec.produces):
            continue
        fingerprint = _fingerprint(spec, ctx, media_fingerprint, brief_fingerprint)
        if ctx.store.fingerprint(spec.produces) != fingerprint:
            stale.append(spec.id)
    return stale
