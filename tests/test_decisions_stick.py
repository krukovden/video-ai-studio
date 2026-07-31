"""Applying decisions must not put them anywhere a stage will overwrite.

The decisions were reaching the plan and then being wiped minutes later: they
were written into the effects stage's own artifact, so the runner compared its
fingerprint against the real one, decided the stage was out of date and
re-planned — discarding exactly what the creator had just approved. Twice this
cost real work. The CLI's answer was to forge the fingerprint the stage WOULD
have written, so a person's edit could impersonate a model's reply.

The answer now is that they are not in the model's artifact at all. Nothing
forges anything, and there is nothing for the runner to call stale.
"""
from __future__ import annotations

from pathlib import Path

import videoai.stages  # noqa: F401  (registers every stage)
from videoai.config import Config
from videoai.core.models import EffectEvent, EffectPlan, Overrides
from videoai.core.registry import REGISTRY, StageContext
from videoai.core.runner import _fingerprint
from videoai.core.store import ArtifactStore
from videoai.logic.approval import apply_decisions
from videoai.logic.decisions import OVERRIDES_ARTIFACT


def _context(tmp_path: Path) -> StageContext:
    work = tmp_path / "work"
    return StageContext(
        project_dir=tmp_path, input_dir=tmp_path, work_dir=work,
        output_dir=tmp_path / "output", config=Config(), store=ArtifactStore(work),
    )


def _plan() -> EffectPlan:
    return EffectPlan(provider="mock", library="lib", events=[
        EffectEvent(at_seconds=10.0, effect_name="badge_boom", clip_ref="clip-01#001",
                    at_in_clip=3.0, screen_position="center", scale="medium", reason="r"),
    ])


def _refuse_it(ctx: StageContext) -> None:
    apply_decisions(
        ctx.store, {"events": [{"index": 0, "keep": False, "effect_name": "badge_boom"}]}
    )


def test_the_model_s_own_artifact_is_left_exactly_as_the_stage_wrote_it(tmp_path: Path):
    ctx = _context(tmp_path)
    ctx.store.write("05d-effects", _plan(), fingerprint="whatever")
    before = ctx.store.content_hash("05d-effects")

    _refuse_it(ctx)

    assert ctx.store.content_hash("05d-effects") == before
    assert ctx.store.fingerprint("05d-effects") == "whatever"


def test_the_decision_is_written_where_the_creator_owns_it(tmp_path: Path):
    ctx = _context(tmp_path)
    ctx.store.write("05d-effects", _plan(), fingerprint="whatever")

    _refuse_it(ctx)

    decided = ctx.store.read(OVERRIDES_ARTIFACT, Overrides).effects
    assert [(item.clip_ref, item.keep) for item in decided] == [("clip-01#001", False)]


def test_no_caller_has_to_forge_a_stage_fingerprint():
    """`videoai.core.runner._fingerprint` is private, and the CLI importing it to
    recompute what a stage would have stored is the design fault this replaced."""
    import ast
    import inspect

    from videoai import cli

    assert "_effects_fingerprint" not in dir(cli)
    private = [
        alias.name
        for node in ast.walk(ast.parse(inspect.getsource(cli)))
        if isinstance(node, ast.ImportFrom) and node.module == "videoai.core.runner"
        for alias in node.names
        if alias.name.startswith("_")
    ]
    assert private == []


def test_a_decision_does_not_make_the_effects_stage_look_stale(tmp_path: Path):
    """The comparison the runner makes before deciding to re-run. The overrides
    are not among the effects stage's inputs, so it is untouched by them."""
    ctx = _context(tmp_path)
    ctx.store.write("05d-effects", _plan(), fingerprint="whatever")
    before = _fingerprint(REGISTRY["effects"], ctx, "media", "brief")

    _refuse_it(ctx)

    assert _fingerprint(REGISTRY["effects"], ctx, "media", "brief") == before


def test_a_decision_re_renders_the_stages_that_read_it(tmp_path: Path):
    """Cached is not the same as ignored: the deterministic assembly and the
    delivery both read the overrides, so both have to notice a new one."""
    ctx = _context(tmp_path)
    ctx.store.write("05d-effects", _plan(), fingerprint="whatever")
    before = {
        stage_id: _fingerprint(REGISTRY[stage_id], ctx, "media", "brief")
        for stage_id in ("assemble", "polish")
    }

    _refuse_it(ctx)

    for stage_id, value in before.items():
        assert _fingerprint(REGISTRY[stage_id], ctx, "media", "brief") != value, stage_id
