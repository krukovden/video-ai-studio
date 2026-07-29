from pathlib import Path

import pytest
from pydantic import BaseModel

from videoai.config import Config
from videoai.core.registry import REGISTRY, StageContext, StageSpec, stage
from videoai.core.runner import run_pipeline, stale_downstream
from videoai.core.store import ArtifactStore


class Payload(BaseModel):
    value: int


@pytest.fixture
def clean_registry():
    saved = dict(REGISTRY)
    REGISTRY.clear()
    yield REGISTRY
    REGISTRY.clear()
    REGISTRY.update(saved)


@pytest.fixture
def ctx(tmp_path: Path) -> StageContext:
    for name in ("input", "work", "output"):
        (tmp_path / name).mkdir()
    return StageContext(
        project_dir=tmp_path,
        input_dir=tmp_path / "input",
        work_dir=tmp_path / "work",
        output_dir=tmp_path / "output",
        config=Config(),
        store=ArtifactStore(tmp_path / "work"),
    )


def _register_two_stages(calls: list[str]) -> None:
    @stage(id="first", produces="01-first", requires=(), model=Payload)
    def first(ctx: StageContext) -> Payload:
        calls.append("first")
        return Payload(value=1)

    @stage(id="second", produces="02-second", requires=("01-first",), model=Payload)
    def second(ctx: StageContext) -> Payload:
        calls.append("second")
        previous = ctx.store.read("01-first", Payload)
        return Payload(value=previous.value + 1)


def test_stages_run_in_dependency_order(clean_registry, ctx: StageContext):
    calls: list[str] = []
    _register_two_stages(calls)
    executed = run_pipeline(ctx, only=None, force=False, media_fingerprint="src")
    assert calls == ["first", "second"]
    assert executed == ["first", "second"]
    assert ctx.store.read("02-second", Payload).value == 2


def test_second_run_skips_everything_when_inputs_unchanged(clean_registry, ctx: StageContext):
    calls: list[str] = []
    _register_two_stages(calls)
    run_pipeline(ctx, only=None, force=False, media_fingerprint="src")
    executed = run_pipeline(ctx, only=None, force=False, media_fingerprint="src")
    assert executed == []
    assert calls == ["first", "second"]


def test_changed_source_fingerprint_reruns_all(clean_registry, ctx: StageContext):
    calls: list[str] = []
    _register_two_stages(calls)
    run_pipeline(ctx, only=None, force=False, media_fingerprint="src")
    executed = run_pipeline(ctx, only=None, force=False, media_fingerprint="src-changed")
    assert executed == ["first", "second"]


def test_force_reruns_even_when_cached(clean_registry, ctx: StageContext):
    calls: list[str] = []
    _register_two_stages(calls)
    run_pipeline(ctx, only=None, force=False, media_fingerprint="src")
    executed = run_pipeline(ctx, only=None, force=True, media_fingerprint="src")
    assert executed == ["first", "second"]


def test_only_runs_a_single_stage(clean_registry, ctx: StageContext):
    calls: list[str] = []
    _register_two_stages(calls)
    run_pipeline(ctx, only=None, force=False, media_fingerprint="src")
    calls.clear()
    executed = run_pipeline(ctx, only="second", force=True, media_fingerprint="src")
    assert executed == ["second"]
    assert calls == ["second"]


def test_only_with_unknown_stage_id_raises(clean_registry, ctx: StageContext):
    _register_two_stages([])
    with pytest.raises(KeyError, match="unknown stage"):
        run_pipeline(ctx, only="nope", force=False, media_fingerprint="src")


def test_provider_change_invalidates_stage(clean_registry, ctx: StageContext):
    calls: list[str] = []

    @stage(id="only", produces="01-only", requires=(), provider_key="asr", model=Payload)
    def only_stage(ctx: StageContext) -> Payload:
        calls.append("only")
        return Payload(value=1)

    run_pipeline(ctx, only=None, force=False, media_fingerprint="src")
    switched = StageContext(
        project_dir=ctx.project_dir,
        input_dir=ctx.input_dir,
        work_dir=ctx.work_dir,
        output_dir=ctx.output_dir,
        config=Config(providers={"asr": "mock", "llm": "claude_cli"}),
        store=ctx.store,
    )
    executed = run_pipeline(switched, only=None, force=False, media_fingerprint="src")
    assert executed == ["only"]


def test_duplicate_produces_raises(clean_registry):
    @stage(id="first", produces="01-shared", requires=(), model=Payload)
    def first(ctx: StageContext) -> Payload:
        return Payload(value=1)

    with pytest.raises(ValueError, match="01-shared"):

        @stage(id="second", produces="01-shared", requires=(), model=Payload)
        def second(ctx: StageContext) -> Payload:
            return Payload(value=2)


def test_only_runs_even_when_cached(clean_registry, ctx: StageContext):
    calls: list[str] = []
    _register_two_stages(calls)
    run_pipeline(ctx, only=None, force=False, media_fingerprint="src")
    calls.clear()
    executed = run_pipeline(ctx, only="second", force=False, media_fingerprint="src")
    assert executed == ["second"]
    assert calls == ["second"]


def test_version_bump_invalidates_cache(clean_registry, ctx: StageContext):
    calls: list[str] = []

    @stage(id="only", produces="01-only", requires=(), version="1", model=Payload)
    def only_stage(ctx: StageContext) -> Payload:
        calls.append("only")
        return Payload(value=1)

    run_pipeline(ctx, only=None, force=False, media_fingerprint="src")
    REGISTRY["only"] = StageSpec(
        id="only",
        produces="01-only",
        requires=(),
        provider_key=None,
        version="2",
        model=Payload,
        fn=only_stage,
    )
    executed = run_pipeline(ctx, only=None, force=False, media_fingerprint="src")
    assert executed == ["only"]


def test_missing_artifact_with_stale_fingerprint_reruns(clean_registry, ctx: StageContext):
    calls: list[str] = []
    _register_two_stages(calls)
    run_pipeline(ctx, only=None, force=False, media_fingerprint="src")
    ctx.store.path("01-first").unlink()
    calls.clear()
    executed = run_pipeline(ctx, only=None, force=False, media_fingerprint="src")
    assert executed == ["first"]
    assert calls == ["first"]


def _register_media_and_brief_stages(calls: list[str]) -> None:
    # Not `calls`: the tests clear that list between runs.
    runs: list[int] = []

    @stage(id="media_only", produces="01-media", requires=(), model=Payload)
    def media_only(ctx: StageContext) -> Payload:
        calls.append("media_only")
        return Payload(value=1)

    @stage(id="brief_reader", produces="02-brief", requires=(), model=Payload, uses_brief=True)
    def brief_reader(ctx: StageContext) -> Payload:
        # Non-deterministic on purpose: `analyze` and `plan` are LLM calls, so a
        # re-run of a brief reader normally yields different content.
        calls.append("brief_reader")
        runs.append(1)
        return Payload(value=100 + len(runs))

    @stage(id="downstream", produces="03-downstream", requires=("02-brief",), model=Payload)
    def downstream(ctx: StageContext) -> Payload:
        calls.append("downstream")
        return Payload(value=3)


def test_brief_fingerprint_change_only_reruns_stages_that_use_it(clean_registry, ctx: StageContext):
    calls: list[str] = []
    _register_media_and_brief_stages(calls)
    run_pipeline(ctx, only=None, force=False, media_fingerprint="media-1", brief_fingerprint="brief-1")
    calls.clear()

    executed = run_pipeline(
        ctx, only=None, force=False, media_fingerprint="media-1", brief_fingerprint="brief-2"
    )

    # brief_reader directly consumes the brief; downstream picks up the change only
    # through the `requires`-fingerprint chain on 02-brief. media_only never reads
    # the brief and must not be touched by a brief-only edit.
    assert executed == ["brief_reader", "downstream"]
    assert calls == ["brief_reader", "downstream"]


def test_media_fingerprint_change_reruns_every_stage_including_brief_readers(
    clean_registry, ctx: StageContext
):
    calls: list[str] = []
    _register_media_and_brief_stages(calls)
    run_pipeline(ctx, only=None, force=False, media_fingerprint="media-1", brief_fingerprint="brief-1")
    calls.clear()

    executed = run_pipeline(
        ctx, only=None, force=False, media_fingerprint="media-2", brief_fingerprint="brief-1"
    )

    assert executed == ["media_only", "brief_reader", "downstream"]


def test_unchanged_media_and_brief_reruns_nothing(clean_registry, ctx: StageContext):
    calls: list[str] = []
    _register_media_and_brief_stages(calls)
    run_pipeline(ctx, only=None, force=False, media_fingerprint="media-1", brief_fingerprint="brief-1")
    calls.clear()

    executed = run_pipeline(
        ctx, only=None, force=False, media_fingerprint="media-1", brief_fingerprint="brief-1"
    )

    assert executed == []
    assert calls == []


def test_stale_downstream_names_stages_left_behind_by_a_single_stage_run(
    clean_registry, ctx: StageContext
):
    calls: list[str] = []
    _register_media_and_brief_stages(calls)
    run_pipeline(ctx, only=None, force=False, media_fingerprint="media-1", brief_fingerprint="brief-1")

    # Simulate the source media changing, then only re-running the stage that reads
    # it directly (as `--stage <id>` does) instead of a full run.
    run_pipeline(ctx, only="media_only", force=False, media_fingerprint="media-2", brief_fingerprint="brief-1")

    stale = stale_downstream(ctx, "media_only", media_fingerprint="media-2", brief_fingerprint="brief-1")

    # brief_reader and downstream still hold artifacts fingerprinted against the old
    # media fingerprint, even though neither of them was touched by this run.
    assert stale == ["brief_reader", "downstream"]


def test_stale_downstream_reports_nothing_when_single_stage_run_changed_nothing(
    clean_registry, ctx: StageContext
):
    calls: list[str] = []
    _register_media_and_brief_stages(calls)
    run_pipeline(ctx, only=None, force=False, media_fingerprint="media-1", brief_fingerprint="brief-1")

    run_pipeline(ctx, only="media_only", force=False, media_fingerprint="media-1", brief_fingerprint="brief-1")

    stale = stale_downstream(ctx, "media_only", media_fingerprint="media-1", brief_fingerprint="brief-1")

    assert stale == []


# --- Finding C3: chain on artifact CONTENT, not on upstream fingerprints ---


def _register_nondeterministic_chain(calls: list[str]) -> None:
    """`analyze` and `plan` are LLM calls: re-running one with identical inputs
    legitimately produces different content under an identical fingerprint."""
    runs: list[int] = []

    @stage(id="upstream", produces="01-upstream", requires=(), model=Payload)
    def upstream(ctx: StageContext) -> Payload:
        calls.append("upstream")
        runs.append(1)
        return Payload(value=len(runs))

    @stage(id="downstream", produces="02-downstream", requires=("01-upstream",), model=Payload)
    def downstream(ctx: StageContext) -> Payload:
        calls.append("downstream")
        return Payload(value=ctx.store.read("01-upstream", Payload).value * 10)


def test_single_stage_rerun_of_a_nondeterministic_stage_marks_downstream_stale(
    clean_registry, ctx: StageContext
):
    calls: list[str] = []
    _register_nondeterministic_chain(calls)
    run_pipeline(ctx, only=None, force=False, media_fingerprint="media-1")
    assert ctx.store.read("02-downstream", Payload).value == 10

    run_pipeline(ctx, only="upstream", force=False, media_fingerprint="media-1")
    assert ctx.store.read("01-upstream", Payload).value == 2

    assert stale_downstream(ctx, "upstream", media_fingerprint="media-1", brief_fingerprint="") == [
        "downstream"
    ]

    calls.clear()
    executed = run_pipeline(ctx, only=None, force=False, media_fingerprint="media-1")

    assert executed == ["downstream"]
    assert ctx.store.read("02-downstream", Payload).value == 20


def test_upstream_rerun_producing_identical_content_leaves_downstream_cached(
    clean_registry, ctx: StageContext
):
    calls: list[str] = []
    _register_two_stages(calls)
    run_pipeline(ctx, only=None, force=False, media_fingerprint="media-1")

    run_pipeline(ctx, only="first", force=True, media_fingerprint="media-1")
    calls.clear()

    assert stale_downstream(ctx, "first", media_fingerprint="media-1", brief_fingerprint="") == []
    assert run_pipeline(ctx, only=None, force=False, media_fingerprint="media-1") == []
    assert calls == []


def test_hand_edited_artifact_reruns_its_dependents(clean_registry, ctx: StageContext):
    """Artifacts are readable JSON on purpose, so editing one by hand is a
    supported move; its dependents must notice."""
    calls: list[str] = []
    _register_two_stages(calls)
    run_pipeline(ctx, only=None, force=False, media_fingerprint="media-1")
    calls.clear()

    ctx.store.path("01-first").write_text(
        Payload(value=41).model_dump_json(indent=2), encoding="utf-8"
    )

    executed = run_pipeline(ctx, only=None, force=False, media_fingerprint="media-1")

    assert executed == ["second"]
    assert calls == ["second"]
    assert ctx.store.read("02-second", Payload).value == 42
