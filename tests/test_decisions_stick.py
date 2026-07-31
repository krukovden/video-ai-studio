"""Applying decisions must also stop the stage from redoing itself.

The decisions were reaching the plan and then being wiped minutes later: the
artifact was stored under a made-up fingerprint, the runner compared it against
the real one, decided the stage was out of date and re-planned — discarding
exactly what the creator had just approved. Twice this cost real work.
"""
from __future__ import annotations

from pathlib import Path

import videoai.stages  # noqa: F401  (registers every stage)
from videoai.config import Config
from videoai.core.models import EffectEvent, EffectPlan
from videoai.core.registry import REGISTRY, StageContext
from videoai.core.runner import _fingerprint
from videoai.core.store import ArtifactStore
from videoai.logic.approval import apply_decisions


def _context(tmp_path: Path) -> StageContext:
    work = tmp_path / "work"
    return StageContext(
        project_dir=tmp_path, input_dir=tmp_path, work_dir=work,
        output_dir=tmp_path / "output", config=Config(), store=ArtifactStore(work),
    )


def _plan() -> EffectPlan:
    return EffectPlan(provider="mock", library="lib", events=[
        EffectEvent(at_seconds=10.0, effect_name="badge_boom",
                    screen_position="center", scale="medium", reason="r"),
    ])


def test_the_stage_is_up_to_date_after_decisions_are_applied(tmp_path: Path):
    ctx = _context(tmp_path)
    ctx.store.write("05d-effects", _plan(), fingerprint="whatever")

    apply_decisions(
        ctx.store,
        {"events": [{"index": 0, "keep": False, "effect_name": "badge_boom"}]},
        stage_fingerprint=_fingerprint(REGISTRY["effects"], ctx, "media", "brief"),
    )

    # This is the comparison the runner makes before deciding to re-run.
    assert ctx.store.fingerprint("05d-effects") == _fingerprint(
        REGISTRY["effects"], ctx, "media", "brief"
    )


def test_the_decision_itself_is_still_written(tmp_path: Path):
    ctx = _context(tmp_path)
    ctx.store.write("05d-effects", _plan(), fingerprint="whatever")
    apply_decisions(
        ctx.store,
        {"events": [{"index": 0, "keep": False, "effect_name": "badge_boom"}]},
        stage_fingerprint="pinned",
    )
    assert ctx.store.read("05d-effects", EffectPlan).events[0].keep is False


def test_without_a_fingerprint_it_still_applies(tmp_path: Path):
    """The caller may not know the fingerprint — a decisions file applied by
    hand, say. That must not stop the decision landing."""
    ctx = _context(tmp_path)
    ctx.store.write("05d-effects", _plan(), fingerprint="whatever")
    result = apply_decisions(
        ctx.store, {"events": [{"index": 0, "keep": False, "effect_name": "badge_boom"}]}
    )
    assert result.dropped == 1
