# VideoAI Core Pipeline (Plan 1 of 4) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the atomic-stage skeleton and the vertical slice that turns a folder of raw clips into a watchable draft video with an LLM-chosen edit.

**Architecture:** Stages are independent units that read and write JSON artifacts in `projects/<name>/work/`; there is no shared in-memory state. A registry declares each stage's inputs, output artifact and provider key; a runner orders them and skips any stage whose input fingerprint and provider are unchanged. External services (ASR, LLM) sit behind provider protocols with mock implementations so the whole pipeline is testable offline.

**Tech Stack:** Python 3.13 (uv-managed venv), Pydantic v2, Typer, ffmpeg 8.1.1, OpenCV (headless), rapidfuzz, parakeet-mlx (ASR), Claude Code CLI in headless mode (LLM), pytest.

## Scope

This is **Plan 1 of 4**. It delivers: scaffold → ingest → quality gate → transcribe → phrase/take analysis → LLM analysis → story plan → timeline → **draft render**. Running `videoai run <project>` produces `output/draft.mp4`.

Out of scope here (later plans): Plan 2 — review web page + review decisions + re-assembly; Plan 3 — Remotion render (captions, graphics library, B-roll sourcing, music); Plan 4 — shorts (reframe), thumbnail, packaging.

Reference spec: `docs/superpowers/specs/2026-07-28-videoai-pipeline-design.md`.

## Global Constraints

- Python version: `requires-python = ">=3.13,<3.14"`. Create the venv with `uv venv --python 3.13`.
- Package name is `videoai` (the spec sketched `pipeline/`; `videoai/` is used so imports read `from videoai.core...` and match the CLI name).
- Every artifact is a Pydantic v2 model serialized to `work/<NN>-<name>.json` with `indent=2` and `ensure_ascii=False`.
- All time values in artifacts are **float seconds**, absolute within their source clip.
- No network calls and no LLM calls in tests. Every provider has a `mock` implementation; tests use only mocks.
- Secrets live in `.env` (already git-ignored). Never write keys into artifacts, logs, or committed files.
- CLI output and code identifiers are English. Comments only where they state a non-obvious constraint.
- Never add Claude/Anthropic co-author or "generated with" attribution to commits.
- Commit after every task using the exact command given in the task's final step.
- Test media is generated at test time with ffmpeg `lavfi`; no binary media is committed to git.

## Real Project Layout (authoritative)

The creator's folders already exist and the pipeline adapts to them, not the other way round.
A project directory looks like this — `video/` and `description/` are supplied by the creator,
the rest the pipeline creates:

```
assets/1.Toy_Pimple_Popping/
├── video/          # source clips (.MOV/.mp4), supplied
├── description/    # brief: .docx/.md/.txt plus product photos, supplied
├── work/           # artifacts and cache, created
└── output/
    ├── draft.mp4   # review draft
    ├── video/      # approved final video (later plans)
    └── shorts/     # shorts, created after approval (later plans)
```

Clip source directory resolution order: `input/` if it exists, else `video/`, else the project
directory itself. macOS metadata files (`.DS_Store`, any dotfile, AppleDouble `._*`) are ignored
everywhere. Real footage for the first project is 56 clips / 44.8 minutes of 4K HEVC 30 fps with
stereo AAC, so anything that touches every clip must be cached and must not decode 4K when a
proxy would do.

## File Structure

| File | Responsibility |
|---|---|
| `pyproject.toml` | uv project metadata, dependencies, pytest config |
| `config.yaml` | Provider selection per stage + tunables |
| `videoai/config.py` | `Config` model, YAML + `.env` loading |
| `videoai/core/models.py` | All artifact models (Manifest, Quality, Transcript, Phrases, TakeGroups, Analysis, StoryPlan, Timeline) |
| `videoai/core/store.py` | `ArtifactStore` — artifact read/write plus fingerprint cache |
| `videoai/core/registry.py` | `StageSpec`, `StageContext`, `@stage` decorator, registry |
| `videoai/core/runner.py` | Stage ordering, fingerprinting, skip/force logic |
| `videoai/core/ffmpeg.py` | ffprobe/ffmpeg subprocess helpers |
| `videoai/core/project.py` | Project layout resolution and creator-brief reading |
| `videoai/logic/phrases.py` | Words → phrases → packed transcript (pure) |
| `videoai/logic/takes.py` | Repeated-take detection (pure) |
| `videoai/logic/timeline.py` | Analysis + StoryPlan → Timeline (pure) |
| `videoai/logic/validate.py` | Timeline hard-rule validator (pure) |
| `videoai/providers/base.py` | `ASRProvider`, `LLMProvider` protocols + resolver |
| `videoai/providers/asr_mock.py`, `asr_parakeet.py` | ASR implementations |
| `videoai/providers/llm_mock.py`, `llm_claude_cli.py` | LLM implementations |
| `videoai/stages/s01_ingest.py` … `s06_render_draft.py` | One stage per file |
| `videoai/cli.py` | Typer app: `run`, `stages`, `show` |
| `tests/` | One test module per source module |

---

### Task 1: Project scaffold and configuration

**Files:**
- Create: `pyproject.toml`, `config.yaml`, `videoai/__init__.py`, `videoai/config.py`, `videoai/cli.py`, `videoai/core/__init__.py`, `videoai/logic/__init__.py`, `videoai/providers/__init__.py`, `videoai/stages/__init__.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `Config` (Pydantic model) with fields `providers: dict[str, str]`, `transcribe: TranscribeSettings`, `analyze: AnalyzeSettings`, `render: RenderSettings`; `load_config(path: Path | None) -> Config`. CLI entry point `videoai`.

- [ ] **Step 1: Create the uv project and install dependencies**

```bash
cd /Users/denyskriukov/Documents/TestDocAI/VideoAI
uv venv --python 3.13
uv add "pydantic>=2.13" "typer>=0.27" "pyyaml>=6.0" "python-dotenv>=1.0" "rapidfuzz>=3.9" "numpy>=2.1" "opencv-python-headless>=4.10"
uv add --dev "pytest>=8.3"
```

- [ ] **Step 2: Write `pyproject.toml` additions**

Append to `pyproject.toml` (keep the `[project]` block uv generated, add these):

```toml
[project.scripts]
videoai = "videoai.cli:app"

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"

[tool.hatch.build.targets.wheel]
packages = ["videoai"]
```

Ensure `[project]` contains `requires-python = ">=3.13,<3.14"`.

- [ ] **Step 3: Create package directories**

```bash
mkdir -p videoai/core videoai/logic videoai/providers videoai/stages tests
touch videoai/__init__.py videoai/core/__init__.py videoai/logic/__init__.py videoai/providers/__init__.py videoai/stages/__init__.py
```

- [ ] **Step 4: Write the failing test**

`tests/test_config.py`:

```python
from pathlib import Path

import pytest

from videoai.config import Config, load_config


def test_load_config_returns_defaults_when_file_missing(tmp_path: Path):
    config = load_config(tmp_path / "nope.yaml")
    assert config.providers["asr"] == "parakeet"
    assert config.providers["llm"] == "claude_cli"
    assert config.transcribe.phrase_gap_seconds == 0.5


def test_load_config_overrides_from_yaml(tmp_path: Path):
    path = tmp_path / "config.yaml"
    path.write_text(
        "providers:\n  asr: mock\n  llm: mock\ntranscribe:\n  phrase_gap_seconds: 0.9\n",
        encoding="utf-8",
    )
    config = load_config(path)
    assert config.providers["asr"] == "mock"
    assert config.providers["llm"] == "mock"
    assert config.transcribe.phrase_gap_seconds == 0.9
    assert config.render.draft_height == 720


def test_unknown_provider_key_is_rejected(tmp_path: Path):
    path = tmp_path / "config.yaml"
    path.write_text("providers:\n  telepathy: yes\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown provider key"):
        load_config(path)


def test_config_is_immutable():
    config = Config()
    with pytest.raises(Exception):
        config.providers = {}
```

- [ ] **Step 5: Run the test to verify it fails**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'videoai.config'`

- [ ] **Step 6: Implement `videoai/config.py`**

```python
"""Pipeline configuration: provider selection per stage plus tunables."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, model_validator

PROVIDER_KEYS = ("asr", "llm")


class TranscribeSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    phrase_gap_seconds: float = 0.5
    max_words_per_phrase: int = 30
    cut_padding_seconds: float = 0.15


class AnalyzeSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    keyframes_per_phrase: int = 1
    llm_model: str = "sonnet"
    llm_timeout_seconds: int = 600


class RenderSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    draft_height: int = 720
    draft_crf: int = 23
    audio_fade_seconds: float = 0.03


class Config(BaseModel):
    model_config = ConfigDict(frozen=True)

    providers: dict[str, str] = Field(
        default_factory=lambda: {"asr": "parakeet", "llm": "claude_cli"}
    )
    transcribe: TranscribeSettings = TranscribeSettings()
    analyze: AnalyzeSettings = AnalyzeSettings()
    render: RenderSettings = RenderSettings()

    @model_validator(mode="after")
    def _check_provider_keys(self) -> "Config":
        for key in self.providers:
            if key not in PROVIDER_KEYS:
                raise ValueError(f"unknown provider key: {key}")
        return self


def load_config(path: Path | None = None) -> Config:
    """Load config.yaml if present, merged over defaults. Also loads .env."""
    load_dotenv()
    if path is None or not path.exists():
        return Config()
    raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    defaults = Config()
    providers = {**defaults.providers, **(raw.get("providers") or {})}
    return Config(
        providers=providers,
        transcribe=TranscribeSettings(**(raw.get("transcribe") or {})),
        analyze=AnalyzeSettings(**(raw.get("analyze") or {})),
        render=RenderSettings(**(raw.get("render") or {})),
    )
```

- [ ] **Step 7: Write the repository `config.yaml`**

```yaml
# Provider selection per stage. Switching a value re-runs only the affected
# stages (fingerprints include the provider name).
providers:
  asr: parakeet      # parakeet | mock
  llm: claude_cli    # claude_cli | mock

transcribe:
  phrase_gap_seconds: 0.5
  max_words_per_phrase: 30
  cut_padding_seconds: 0.15

analyze:
  keyframes_per_phrase: 1
  llm_model: sonnet
  llm_timeout_seconds: 600

render:
  draft_height: 720
  draft_crf: 23
  audio_fade_seconds: 0.03
```

- [ ] **Step 8: Write a minimal CLI so the entry point exists**

`videoai/cli.py`:

```python
"""VideoAI command line interface."""
from __future__ import annotations

from pathlib import Path

import typer

from videoai.config import load_config

app = typer.Typer(add_completion=False, help="Automated video pipeline.")


@app.command()
def config(path: Path = typer.Option(Path("config.yaml"), help="Config file path")) -> None:
    """Print the effective configuration."""
    typer.echo(load_config(path).model_dump_json(indent=2))


if __name__ == "__main__":
    app()
```

- [ ] **Step 9: Run the tests to verify they pass**

Run: `uv run pytest tests/test_config.py -v && uv run videoai config`
Expected: 4 passed, then the effective config printed as JSON.

- [ ] **Step 10: Commit**

```bash
git add pyproject.toml uv.lock config.yaml videoai tests
git commit -m "feat: project scaffold with configuration model and CLI entry point"
```

---

### Task 2: Artifact store with fingerprint cache

**Files:**
- Create: `videoai/core/store.py`
- Test: `tests/test_store.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `ArtifactStore(work_dir: Path)` with `write(name: str, model: BaseModel, fingerprint: str) -> Path`, `read(name: str, model_cls: type[T]) -> T`, `exists(name: str) -> bool`, `fingerprint(name: str) -> str | None`, `path(name: str) -> Path`; module function `hash_parts(*parts: str) -> str`.

- [ ] **Step 1: Write the failing test**

`tests/test_store.py`:

```python
from pathlib import Path

import pytest
from pydantic import BaseModel

from videoai.core.store import ArtifactStore, hash_parts


class Sample(BaseModel):
    name: str
    value: int


def test_write_then_read_roundtrip(tmp_path: Path):
    store = ArtifactStore(tmp_path)
    store.write("01-sample", Sample(name="a", value=1), fingerprint="fp1")
    loaded = store.read("01-sample", Sample)
    assert loaded.name == "a"
    assert loaded.value == 1


def test_artifact_is_human_readable_json(tmp_path: Path):
    store = ArtifactStore(tmp_path)
    store.write("01-sample", Sample(name="привет", value=2), fingerprint="fp1")
    text = (tmp_path / "01-sample.json").read_text(encoding="utf-8")
    assert "привет" in text
    assert "\n  " in text


def test_fingerprint_is_recorded_and_returned(tmp_path: Path):
    store = ArtifactStore(tmp_path)
    assert store.fingerprint("01-sample") is None
    store.write("01-sample", Sample(name="a", value=1), fingerprint="fp1")
    assert store.fingerprint("01-sample") == "fp1"


def test_exists_is_false_before_write(tmp_path: Path):
    store = ArtifactStore(tmp_path)
    assert store.exists("01-sample") is False
    store.write("01-sample", Sample(name="a", value=1), fingerprint="fp1")
    assert store.exists("01-sample") is True


def test_read_missing_artifact_raises(tmp_path: Path):
    store = ArtifactStore(tmp_path)
    with pytest.raises(FileNotFoundError):
        store.read("01-sample", Sample)


def test_hash_parts_is_stable_and_order_sensitive():
    assert hash_parts("a", "b") == hash_parts("a", "b")
    assert hash_parts("a", "b") != hash_parts("b", "a")
    assert len(hash_parts("a")) == 16
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_store.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'videoai.core.store'`

- [ ] **Step 3: Implement `videoai/core/store.py`**

```python
"""Artifact persistence: JSON files plus a sidecar fingerprint used for caching."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def hash_parts(*parts: str) -> str:
    """Short stable digest over ordered string parts."""
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()[:16]


class ArtifactStore:
    def __init__(self, work_dir: Path) -> None:
        self.work_dir = work_dir
        self.meta_dir = work_dir / ".meta"
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.meta_dir.mkdir(parents=True, exist_ok=True)

    def path(self, name: str) -> Path:
        return self.work_dir / f"{name}.json"

    def _meta_path(self, name: str) -> Path:
        return self.meta_dir / f"{name}.json"

    def exists(self, name: str) -> bool:
        return self.path(name).exists()

    def write(self, name: str, model: BaseModel, fingerprint: str) -> Path:
        target = self.path(name)
        target.write_text(
            model.model_dump_json(indent=2).encode("utf-8").decode("utf-8"),
            encoding="utf-8",
        )
        self._meta_path(name).write_text(
            json.dumps({"fingerprint": fingerprint}), encoding="utf-8"
        )
        return target

    def read(self, name: str, model_cls: type[T]) -> T:
        target = self.path(name)
        if not target.exists():
            raise FileNotFoundError(f"artifact not found: {target}")
        return model_cls.model_validate_json(target.read_text(encoding="utf-8"))

    def fingerprint(self, name: str) -> str | None:
        meta = self._meta_path(name)
        if not meta.exists():
            return None
        return json.loads(meta.read_text(encoding="utf-8")).get("fingerprint")
```

Note: `model_dump_json` already emits non-ASCII characters unescaped, which satisfies the readability test.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_store.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add videoai/core/store.py tests/test_store.py
git commit -m "feat: artifact store with fingerprint sidecars"
```

---

### Task 3: Stage registry and runner

**Files:**
- Create: `videoai/core/registry.py`, `videoai/core/runner.py`
- Modify: `videoai/cli.py`
- Test: `tests/test_runner.py`

**Interfaces:**
- Consumes: `ArtifactStore`, `hash_parts` (Task 2); `Config` (Task 1).
- Produces:
  - `StageContext` dataclass with `project_dir, input_dir, work_dir, output_dir, config, store`.
  - `StageSpec` dataclass with `id: str`, `produces: str`, `requires: tuple[str, ...]`, `provider_key: str | None`, `version: str`, `fn: Callable[[StageContext], BaseModel]`, `model: type[BaseModel]`.
  - `@stage(id=..., produces=..., requires=(...), provider_key=..., version=..., model=...)` decorator registering into `REGISTRY: dict[str, StageSpec]`.
  - `run_pipeline(ctx: StageContext, only: str | None, force: bool, extra_fingerprint: str) -> list[str]` returning the ids of stages that actually executed.

- [ ] **Step 1: Write the failing test**

`tests/test_runner.py`:

```python
from pathlib import Path

import pytest
from pydantic import BaseModel

from videoai.config import Config
from videoai.core.registry import REGISTRY, StageContext, StageSpec, stage
from videoai.core.runner import run_pipeline
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
    executed = run_pipeline(ctx, only=None, force=False, extra_fingerprint="src")
    assert calls == ["first", "second"]
    assert executed == ["first", "second"]
    assert ctx.store.read("02-second", Payload).value == 2


def test_second_run_skips_everything_when_inputs_unchanged(clean_registry, ctx: StageContext):
    calls: list[str] = []
    _register_two_stages(calls)
    run_pipeline(ctx, only=None, force=False, extra_fingerprint="src")
    executed = run_pipeline(ctx, only=None, force=False, extra_fingerprint="src")
    assert executed == []
    assert calls == ["first", "second"]


def test_changed_source_fingerprint_reruns_all(clean_registry, ctx: StageContext):
    calls: list[str] = []
    _register_two_stages(calls)
    run_pipeline(ctx, only=None, force=False, extra_fingerprint="src")
    executed = run_pipeline(ctx, only=None, force=False, extra_fingerprint="src-changed")
    assert executed == ["first", "second"]


def test_force_reruns_even_when_cached(clean_registry, ctx: StageContext):
    calls: list[str] = []
    _register_two_stages(calls)
    run_pipeline(ctx, only=None, force=False, extra_fingerprint="src")
    executed = run_pipeline(ctx, only=None, force=True, extra_fingerprint="src")
    assert executed == ["first", "second"]


def test_only_runs_a_single_stage(clean_registry, ctx: StageContext):
    calls: list[str] = []
    _register_two_stages(calls)
    run_pipeline(ctx, only=None, force=False, extra_fingerprint="src")
    calls.clear()
    executed = run_pipeline(ctx, only="second", force=True, extra_fingerprint="src")
    assert executed == ["second"]
    assert calls == ["second"]


def test_only_with_unknown_stage_id_raises(clean_registry, ctx: StageContext):
    _register_two_stages([])
    with pytest.raises(KeyError, match="unknown stage"):
        run_pipeline(ctx, only="nope", force=False, extra_fingerprint="src")


def test_provider_change_invalidates_stage(clean_registry, ctx: StageContext):
    calls: list[str] = []

    @stage(id="only", produces="01-only", requires=(), provider_key="asr", model=Payload)
    def only_stage(ctx: StageContext) -> Payload:
        calls.append("only")
        return Payload(value=1)

    run_pipeline(ctx, only=None, force=False, extra_fingerprint="src")
    switched = StageContext(
        project_dir=ctx.project_dir,
        input_dir=ctx.input_dir,
        work_dir=ctx.work_dir,
        output_dir=ctx.output_dir,
        config=Config(providers={"asr": "mock", "llm": "claude_cli"}),
        store=ctx.store,
    )
    executed = run_pipeline(switched, only=None, force=False, extra_fingerprint="src")
    assert executed == ["only"]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_runner.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'videoai.core.registry'`

- [ ] **Step 3: Implement `videoai/core/registry.py`**

```python
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


REGISTRY: dict[str, StageSpec] = {}


def stage(
    *,
    id: str,
    produces: str,
    requires: tuple[str, ...] = (),
    provider_key: str | None = None,
    version: str = "1",
    model: type[BaseModel],
):
    def decorator(fn: Callable[[StageContext], BaseModel]):
        if id in REGISTRY:
            raise ValueError(f"stage already registered: {id}")
        REGISTRY[id] = StageSpec(
            id=id,
            produces=produces,
            requires=requires,
            provider_key=provider_key,
            version=version,
            model=model,
            fn=fn,
        )
        return fn

    return decorator
```

- [ ] **Step 4: Implement `videoai/core/runner.py`**

```python
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
        if not force and cached == fingerprint and ctx.store.exists(spec.produces):
            continue
        artifact = spec.fn(ctx)
        if not isinstance(artifact, spec.model):
            raise TypeError(
                f"stage {spec.id} returned {type(artifact).__name__}, expected {spec.model.__name__}"
            )
        ctx.store.write(spec.produces, artifact, fingerprint)
        executed.append(spec.id)
    return executed
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_runner.py -v`
Expected: 7 passed

- [ ] **Step 6: Commit**

```bash
git add videoai/core/registry.py videoai/core/runner.py tests/test_runner.py
git commit -m "feat: stage registry and caching runner"
```

---

### Task 4: ffmpeg helpers, project layout and the ingest stage

**Files:**
- Create: `videoai/core/ffmpeg.py`, `videoai/core/project.py`, `videoai/core/models.py`, `videoai/stages/s01_ingest.py`, `tests/conftest.py`
- Test: `tests/test_ffmpeg.py`, `tests/test_project.py`, `tests/test_stage_ingest.py`

**Interfaces:**
- Consumes: `StageContext`, `stage` (Task 3).
- Produces:
  - `videoai/core/ffmpeg.py`: `ProbeResult` dataclass (`duration: float, width: int, height: int, fps: float, has_audio: bool`), `probe(path) -> ProbeResult`, `run_ffmpeg(args: list[str]) -> None`, `extract_audio(src, dst) -> None`, `make_proxy(src, dst, height) -> None`, `extract_frame(src, at, dst, height=360) -> None`, `list_video_files(directory: Path) -> list[Path]`, and the constant `VIDEO_SUFFIXES`.
  - `videoai/core/project.py`: `resolve_clip_dir(project_dir: Path) -> Path`, `read_brief(project_dir: Path) -> str`.
  - `videoai/core/models.py`: `ClipInfo`, `Manifest`.
  - Stage id `ingest`, artifact `01-manifest`, model `Manifest`.
  - `tests/conftest.py`: fixture `make_clip(name, seconds, tone_hz, size)` returning a `Path` to a generated mp4 in `tmp_path`.

**Layout adaptation (this task owns it).** `StageContext.input_dir` is the *project* directory.
Clips live in `resolve_clip_dir(project_dir)`, which prefers `input/`, then `video/`, then the
project directory itself. `read_brief` concatenates, in this order and skipping whatever is
absent: `project.yaml`, `notes.md`, and every `.md`, `.txt` and `.docx` inside `description/`.
Files whose name starts with `.` or `._` are ignored everywhere.

**Performance requirement.** Sources are 4K HEVC (56 clips, 44.8 minutes for the first real
project). `make_proxy` decodes with `-hwaccel videotoolbox` and encodes with
`h264_videotoolbox`, falling back to `libx264` when the hardware encoder is unavailable.

- [ ] **Step 1: Add the docx dependency**

```bash
uv add python-docx
```

- [ ] **Step 2: Write the test fixture helper**

`tests/conftest.py`:

```python
import subprocess
from pathlib import Path

import pytest


def _generate_clip(path: Path, seconds: float, tone_hz: int, size: str = "320x240") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", f"testsrc=size={size}:rate=30:duration={seconds}",
            "-f", "lavfi", "-i", f"sine=frequency={tone_hz}:duration={seconds}",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
            str(path),
        ],
        check=True,
    )
    return path


@pytest.fixture
def make_clip(tmp_path: Path):
    def _make(name: str, seconds: float = 3.0, tone_hz: int = 440, size: str = "320x240") -> Path:
        return _generate_clip(tmp_path / name, seconds, tone_hz, size)

    return _make
```

- [ ] **Step 3: Write the failing ffmpeg tests**

`tests/test_ffmpeg.py`:

```python
from pathlib import Path

from videoai.core.ffmpeg import (
    extract_audio,
    extract_frame,
    list_video_files,
    make_proxy,
    probe,
)


def test_probe_reads_stream_properties(make_clip):
    clip = make_clip("a.mp4", seconds=3.0)
    result = probe(clip)
    assert 2.8 < result.duration < 3.3
    assert (result.width, result.height) == (320, 240)
    assert 29.0 < result.fps < 31.0
    assert result.has_audio is True


def test_extract_audio_writes_wav(make_clip, tmp_path: Path):
    clip = make_clip("a.mp4", seconds=2.0)
    wav = tmp_path / "out" / "a.wav"
    extract_audio(clip, wav)
    assert wav.exists() and wav.stat().st_size > 1000


def test_make_proxy_scales_height(make_clip, tmp_path: Path):
    clip = make_clip("a.mp4", seconds=2.0, size="640x480")
    proxy = tmp_path / "out" / "a-proxy.mp4"
    make_proxy(clip, proxy, height=240)
    assert probe(proxy).height == 240


def test_make_proxy_keeps_audio(make_clip, tmp_path: Path):
    clip = make_clip("a.mp4", seconds=2.0, size="640x480")
    proxy = tmp_path / "out" / "a-proxy.mp4"
    make_proxy(clip, proxy, height=240)
    assert probe(proxy).has_audio is True


def test_extract_frame_writes_image(make_clip, tmp_path: Path):
    clip = make_clip("a.mp4", seconds=3.0)
    frame = tmp_path / "frames" / "f.jpg"
    extract_frame(clip, at=1.0, dst=frame, height=180)
    assert frame.exists() and frame.stat().st_size > 500


def test_list_video_files_sorts_and_filters(tmp_path: Path, make_clip):
    make_clip("b.MOV", seconds=1.0).rename(tmp_path / "b.MOV")
    make_clip("a.mp4", seconds=1.0).rename(tmp_path / "a.mp4")
    (tmp_path / ".DS_Store").write_bytes(b"junk")
    (tmp_path / "._a.mp4").write_bytes(b"junk")
    (tmp_path / "notes.md").write_text("hello", encoding="utf-8")
    (tmp_path / "sub").mkdir()

    found = list_video_files(tmp_path)

    assert [path.name for path in found] == ["a.mp4", "b.MOV"]


def test_list_video_files_on_missing_directory_returns_empty(tmp_path: Path):
    assert list_video_files(tmp_path / "nope") == []
```

- [ ] **Step 4: Run the ffmpeg tests to verify they fail**

Run: `uv run pytest tests/test_ffmpeg.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'videoai.core.ffmpeg'`

- [ ] **Step 5: Implement `videoai/core/ffmpeg.py`**

```python
"""Thin, explicit wrappers over ffmpeg/ffprobe. No hidden defaults."""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from functools import cache
from pathlib import Path

VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".mkv", ".avi"}


@dataclass(frozen=True)
class ProbeResult:
    duration: float
    width: int
    height: int
    fps: float
    has_audio: bool


def run_ffmpeg(args: list[str]) -> None:
    result = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", *args],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {' '.join(args)}\n{result.stderr.strip()}")


def probe(path: Path) -> ProbeResult:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-print_format", "json",
            "-show_format", "-show_streams", str(path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {path}: {result.stderr.strip()}")
    data = json.loads(result.stdout)
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        raise RuntimeError(f"no video stream in {path}")
    numerator, _, denominator = video.get("r_frame_rate", "30/1").partition("/")
    fps = float(numerator) / float(denominator or 1)
    return ProbeResult(
        duration=float(data.get("format", {}).get("duration", 0.0)),
        width=int(video["width"]),
        height=int(video["height"]),
        fps=fps,
        has_audio=any(s.get("codec_type") == "audio" for s in streams),
    )


def list_video_files(directory: Path) -> list[Path]:
    """Video files directly inside `directory`, sorted, macOS metadata excluded."""
    if not directory.is_dir():
        return []
    return sorted(
        path for path in directory.iterdir()
        if path.is_file()
        and not path.name.startswith((".", "._"))
        and path.suffix.lower() in VIDEO_SUFFIXES
    )


@cache
def _has_videotoolbox_encoder() -> bool:
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True
    )
    return "h264_videotoolbox" in result.stdout


def extract_audio(src: Path, dst: Path) -> None:
    """16 kHz mono WAV with EBU R128 loudness normalisation, ready for ASR."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    run_ffmpeg([
        "-i", str(src),
        "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
        "-ac", "1", "-ar", "16000", "-vn",
        str(dst),
    ])


def make_proxy(src: Path, dst: Path, height: int) -> None:
    """Small proxy for analysis and draft renders.

    Sources are 4K HEVC, so decode and encode go through VideoToolbox when the
    build supports it; software encoding is minutes per clip instead of seconds.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    scale = f"scale=-2:{height}"
    if _has_videotoolbox_encoder():
        args = [
            "-hwaccel", "videotoolbox", "-i", str(src),
            "-vf", scale,
            "-c:v", "h264_videotoolbox", "-b:v", "2500k",
            "-c:a", "aac", "-b:a", "128k",
            str(dst),
        ]
    else:
        args = [
            "-i", str(src),
            "-vf", scale,
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "26",
            "-c:a", "aac", "-b:a", "128k",
            str(dst),
        ]
    run_ffmpeg(args)


def extract_frame(src: Path, at: float, dst: Path, height: int = 360) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    run_ffmpeg([
        "-ss", f"{at:.3f}", "-i", str(src),
        "-frames:v", "1", "-vf", f"scale=-2:{height}",
        str(dst),
    ])
```

- [ ] **Step 6: Run the ffmpeg tests to verify they pass**

Run: `uv run pytest tests/test_ffmpeg.py -v`
Expected: 7 passed

- [ ] **Step 7: Write the failing project-layout tests**

`tests/test_project.py`:

```python
from pathlib import Path

import docx

from videoai.core.project import read_brief, resolve_clip_dir


def test_resolve_clip_dir_prefers_input(tmp_path: Path):
    (tmp_path / "input").mkdir()
    (tmp_path / "video").mkdir()
    assert resolve_clip_dir(tmp_path) == tmp_path / "input"


def test_resolve_clip_dir_falls_back_to_video(tmp_path: Path):
    (tmp_path / "video").mkdir()
    assert resolve_clip_dir(tmp_path) == tmp_path / "video"


def test_resolve_clip_dir_falls_back_to_project_dir(tmp_path: Path):
    assert resolve_clip_dir(tmp_path) == tmp_path


def test_read_brief_returns_empty_string_when_nothing_present(tmp_path: Path):
    assert read_brief(tmp_path) == ""


def test_read_brief_includes_project_yaml_and_notes(tmp_path: Path):
    (tmp_path / "project.yaml").write_text("title: Slime review\n", encoding="utf-8")
    (tmp_path / "notes.md").write_text("Keep the laugh at the end.", encoding="utf-8")

    brief = read_brief(tmp_path)

    assert "Slime review" in brief
    assert "Keep the laugh at the end." in brief


def test_read_brief_reads_docx_from_description(tmp_path: Path):
    description = tmp_path / "description"
    description.mkdir()
    document = docx.Document()
    document.add_paragraph("Pimple Popping Stress Toy")
    document.add_paragraph("Refillable, two in one.")
    document.save(description / "product.docx")

    brief = read_brief(tmp_path)

    assert "Pimple Popping Stress Toy" in brief
    assert "Refillable, two in one." in brief


def test_read_brief_reads_markdown_and_text_from_description(tmp_path: Path):
    description = tmp_path / "description"
    description.mkdir()
    (description / "a.md").write_text("markdown content", encoding="utf-8")
    (description / "b.txt").write_text("plain content", encoding="utf-8")

    brief = read_brief(tmp_path)

    assert "markdown content" in brief
    assert "plain content" in brief


def test_read_brief_ignores_macos_metadata_and_images(tmp_path: Path):
    description = tmp_path / "description"
    description.mkdir()
    (description / ".DS_Store").write_bytes(b"junk")
    (description / "._notes.md").write_bytes(b"junk")
    (description / "photo.jpeg").write_bytes(b"\xff\xd8\xff")
    (description / "real.md").write_text("real content", encoding="utf-8")

    brief = read_brief(tmp_path)

    assert brief.strip() == "real content"


def test_read_brief_survives_an_unreadable_docx(tmp_path: Path):
    description = tmp_path / "description"
    description.mkdir()
    (description / "broken.docx").write_bytes(b"not a real docx")
    (description / "real.md").write_text("real content", encoding="utf-8")

    brief = read_brief(tmp_path)

    assert "real content" in brief
    assert "broken.docx" in brief
```

- [ ] **Step 8: Run the project tests to verify they fail**

Run: `uv run pytest tests/test_project.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'videoai.core.project'`

- [ ] **Step 9: Implement `videoai/core/project.py`**

```python
"""Project folder conventions.

The creator's folders came first, so the pipeline adapts to them: clips may sit
in `input/` or `video/`, and the brief is whatever prose lives in `description/`.
"""
from __future__ import annotations

from pathlib import Path

BRIEF_SUFFIXES = {".md", ".txt", ".docx"}


def resolve_clip_dir(project_dir: Path) -> Path:
    for name in ("input", "video"):
        candidate = project_dir / name
        if candidate.is_dir():
            return candidate
    return project_dir


def _read_docx(path: Path) -> str:
    import docx

    document = docx.Document(str(path))
    return "\n".join(paragraph.text for paragraph in document.paragraphs)


def _read_one(path: Path) -> str:
    if path.suffix.lower() == ".docx":
        try:
            return _read_docx(path)
        except Exception:
            return f"[could not read {path.name}]"
    try:
        return path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return f"[could not read {path.name}]"


def read_brief(project_dir: Path) -> str:
    """Everything the creator wrote about this video, concatenated."""
    parts: list[str] = []
    for name in ("project.yaml", "notes.md"):
        path = project_dir / name
        if path.is_file():
            parts.append(_read_one(path))

    description = project_dir / "description"
    if description.is_dir():
        for path in sorted(description.iterdir()):
            if (
                path.is_file()
                and not path.name.startswith((".", "._"))
                and path.suffix.lower() in BRIEF_SUFFIXES
            ):
                parts.append(_read_one(path))

    return "\n\n".join(part.strip() for part in parts if part.strip())
```

- [ ] **Step 10: Run the project tests to verify they pass**

Run: `uv run pytest tests/test_project.py -v`
Expected: 9 passed

- [ ] **Step 11: Write the failing ingest tests**

`tests/test_stage_ingest.py`:

```python
from pathlib import Path

from videoai.config import Config
from videoai.core.models import Manifest
from videoai.core.registry import StageContext
from videoai.core.store import ArtifactStore
from videoai.stages.s01_ingest import ingest


def _context(project: Path) -> StageContext:
    (project / "work").mkdir(parents=True, exist_ok=True)
    (project / "output").mkdir(parents=True, exist_ok=True)
    return StageContext(
        project_dir=project,
        input_dir=project,
        work_dir=project / "work",
        output_dir=project / "output",
        config=Config(),
        store=ArtifactStore(project / "work"),
    )


def test_ingest_indexes_clips_from_video_folder(tmp_path: Path, make_clip):
    project = tmp_path / "project"
    clips = project / "video"
    clips.mkdir(parents=True)
    make_clip("b.mp4", seconds=2.0).rename(clips / "b.mp4")
    make_clip("a.mp4", seconds=2.0).rename(clips / "a.mp4")

    manifest = ingest(_context(project))

    assert isinstance(manifest, Manifest)
    assert [clip.clip_id for clip in manifest.clips] == ["clip-01", "clip-02"]
    assert [Path(clip.path).name for clip in manifest.clips] == ["a.mp4", "b.mp4"]
    assert all(clip.duration > 1.5 for clip in manifest.clips)


def test_ingest_creates_audio_and_proxy_files(tmp_path: Path, make_clip):
    project = tmp_path / "project"
    clips = project / "video"
    clips.mkdir(parents=True)
    make_clip("a.mp4", seconds=2.0).rename(clips / "a.mp4")

    manifest = ingest(_context(project))

    clip = manifest.clips[0]
    assert Path(clip.audio_path).exists()
    assert Path(clip.proxy_path).exists()


def test_ingest_ignores_non_video_and_macos_metadata(tmp_path: Path, make_clip):
    project = tmp_path / "project"
    clips = project / "video"
    clips.mkdir(parents=True)
    make_clip("a.mp4", seconds=2.0).rename(clips / "a.mp4")
    (clips / ".DS_Store").write_bytes(b"junk")
    (clips / "IMG_8195.JPG").write_bytes(b"\xff\xd8\xff")
    (project / "project.yaml").write_text("title: test\n", encoding="utf-8")

    manifest = ingest(_context(project))

    assert len(manifest.clips) == 1


def test_ingest_is_idempotent_and_reuses_existing_media(tmp_path: Path, make_clip):
    project = tmp_path / "project"
    clips = project / "video"
    clips.mkdir(parents=True)
    make_clip("a.mp4", seconds=2.0).rename(clips / "a.mp4")
    ctx = _context(project)

    first = ingest(ctx)
    proxy = Path(first.clips[0].proxy_path)
    marker = proxy.stat().st_mtime_ns

    second = ingest(ctx)

    assert second.clips[0].proxy_path == first.clips[0].proxy_path
    assert Path(second.clips[0].proxy_path).stat().st_mtime_ns == marker


def test_ingest_raises_when_no_clips_found(tmp_path: Path):
    project = tmp_path / "project"
    (project / "video").mkdir(parents=True)

    try:
        ingest(_context(project))
    except RuntimeError as error:
        assert "no video files" in str(error)
    else:
        raise AssertionError("expected RuntimeError")
```

- [ ] **Step 12: Run the ingest tests to verify they fail**

Run: `uv run pytest tests/test_stage_ingest.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'videoai.core.models'`

- [ ] **Step 13: Implement `videoai/core/models.py`**

```python
"""Artifact models. Every stage input and output is defined here."""
from __future__ import annotations

from pydantic import BaseModel, Field


class ClipInfo(BaseModel):
    clip_id: str
    path: str
    duration: float
    width: int
    height: int
    fps: float
    has_audio: bool
    audio_path: str | None = None
    proxy_path: str | None = None


class Manifest(BaseModel):
    clips: list[ClipInfo] = Field(default_factory=list)

    def by_id(self, clip_id: str) -> ClipInfo:
        for clip in self.clips:
            if clip.clip_id == clip_id:
                return clip
        raise KeyError(f"unknown clip_id: {clip_id}")
```

- [ ] **Step 14: Implement `videoai/stages/s01_ingest.py`**

```python
"""s01 ingest: index the source clips, normalise audio, build proxies.

Derived media is expensive (44 minutes of 4K HEVC in the first real project), so
anything already on disk is reused; deleting `work/media` forces a rebuild.
"""
from __future__ import annotations

from pathlib import Path

from videoai.core.ffmpeg import extract_audio, list_video_files, make_proxy, probe
from videoai.core.models import ClipInfo, Manifest
from videoai.core.project import resolve_clip_dir
from videoai.core.registry import StageContext, stage


@stage(id="ingest", produces="01-manifest", requires=(), model=Manifest)
def ingest(ctx: StageContext) -> Manifest:
    clip_dir = resolve_clip_dir(ctx.input_dir)
    sources = list_video_files(clip_dir)
    if not sources:
        raise RuntimeError(f"no video files found in {clip_dir}")

    media_dir = ctx.work_dir / "media"
    clips: list[ClipInfo] = []
    for index, source in enumerate(sources, start=1):
        clip_id = f"clip-{index:02d}"
        info = probe(source)

        audio_path: Path | None = None
        if info.has_audio:
            audio_path = media_dir / f"{clip_id}.wav"
            if not audio_path.exists():
                extract_audio(source, audio_path)

        proxy_path = media_dir / f"{clip_id}-proxy.mp4"
        if not proxy_path.exists():
            make_proxy(source, proxy_path, height=ctx.config.render.draft_height)

        clips.append(
            ClipInfo(
                clip_id=clip_id,
                path=str(source),
                duration=info.duration,
                width=info.width,
                height=info.height,
                fps=info.fps,
                has_audio=info.has_audio,
                audio_path=str(audio_path) if audio_path else None,
                proxy_path=str(proxy_path),
            )
        )
    return Manifest(clips=clips)
```

- [ ] **Step 15: Run every test written so far**

Run: `uv run pytest -v`
Expected: all tests pass (25 from Tasks 1-3 plus 21 new)

- [ ] **Step 16: Commit**

```bash
git add videoai/core/ffmpeg.py videoai/core/project.py videoai/core/models.py videoai/stages/s01_ingest.py tests/conftest.py tests/test_ffmpeg.py tests/test_project.py tests/test_stage_ingest.py pyproject.toml uv.lock
git commit -m "feat: ffmpeg helpers, project layout resolution and ingest stage"
```

### Task 5: Quality gate stage

**Files:**
- Modify: `videoai/core/models.py`
- Create: `videoai/stages/s02_quality.py`
- Test: `tests/test_stage_quality.py`

**Interfaces:**
- Consumes: `Manifest` artifact `01-manifest` (Task 4); `probe`, `extract_frame`.
- Produces: models `ClipQuality` (fields `clip_id: str`, `blur: float`, `motion: float`, `black_ratio: float`, `usable: bool`, `scored: bool`) and `QualityReport` (`clips: list[ClipQuality]`, method `by_id`); stage id `quality`, artifact `02-quality`.

Blur is the mean Laplacian variance across sampled frames, inverted and normalised to 0..1 where higher means blurrier. Shake is the mean absolute difference between consecutive sampled frames, normalised to 0..1.

- [ ] **Step 1: Write the failing test**

`tests/test_stage_quality.py`:

```python
import subprocess
from pathlib import Path

from videoai.config import Config
from videoai.core.models import Manifest, QualityReport
from videoai.core.registry import StageContext
from videoai.core.store import ArtifactStore
from videoai.stages.s01_ingest import ingest
from videoai.stages.s02_quality import quality


def _context(tmp_path: Path) -> StageContext:
    for name in ("input", "work", "output"):
        (tmp_path / name).mkdir(exist_ok=True)
    return StageContext(
        project_dir=tmp_path,
        input_dir=tmp_path / "input",
        work_dir=tmp_path / "work",
        output_dir=tmp_path / "output",
        config=Config(),
        store=ArtifactStore(tmp_path / "work"),
    )


def _blurred_clip(path: Path, seconds: float = 2.0) -> Path:
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", f"testsrc=size=320x240:rate=30:duration={seconds}",
            "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
            "-vf", "boxblur=10:1",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
            str(path),
        ],
        check=True,
    )
    return path


def _black_clip(path: Path, seconds: float = 2.0) -> Path:
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", f"color=c=black:size=320x240:rate=30:duration={seconds}",
            "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
            str(path),
        ],
        check=True,
    )
    return path


def test_blurred_clip_scores_higher_blur_than_sharp(tmp_path: Path, make_clip):
    ctx = _context(tmp_path)
    make_clip("a.mp4", seconds=2.0).rename(ctx.input_dir / "a.mp4")
    _blurred_clip(ctx.input_dir / "b.mp4", seconds=2.0)
    ctx.store.write("01-manifest", ingest(ctx), fingerprint="fp")

    report = quality(ctx)

    assert isinstance(report, QualityReport)
    sharp = report.by_id("clip-01")
    blurred = report.by_id("clip-02")
    assert blurred.blur > sharp.blur


def test_black_clip_is_flagged_unusable(tmp_path: Path):
    ctx = _context(tmp_path)
    _black_clip(ctx.input_dir / "a.mp4", seconds=2.0)
    ctx.store.write("01-manifest", ingest(ctx), fingerprint="fp")

    report = quality(ctx)

    assert report.by_id("clip-01").black_ratio > 0.5
    assert report.by_id("clip-01").usable is False


def test_every_clip_appears_in_report(tmp_path: Path, make_clip):
    ctx = _context(tmp_path)
    make_clip("a.mp4", seconds=2.0).rename(ctx.input_dir / "a.mp4")
    make_clip("b.mp4", seconds=2.0).rename(ctx.input_dir / "b.mp4")
    manifest: Manifest = ingest(ctx)
    ctx.store.write("01-manifest", manifest, fingerprint="fp")

    report = quality(ctx)

    assert {c.clip_id for c in report.clips} == {c.clip_id for c in manifest.clips}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_stage_quality.py -v`
Expected: FAIL with `ImportError: cannot import name 'QualityReport'`

- [ ] **Step 3: Add quality models to `videoai/core/models.py`**

```python
class ClipQuality(BaseModel):
    clip_id: str
    blur: float
    motion: float
    black_ratio: float
    usable: bool


class QualityReport(BaseModel):
    clips: list[ClipQuality] = Field(default_factory=list)

    def by_id(self, clip_id: str) -> ClipQuality:
        for clip in self.clips:
            if clip.clip_id == clip_id:
                return clip
        raise KeyError(f"unknown clip_id: {clip_id}")
```

- [ ] **Step 4: Implement `videoai/stages/s02_quality.py`**

```python
"""s02 quality gate: deterministic technical scoring before any model sees the footage."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from videoai.core.models import ClipQuality, Manifest, QualityReport
from videoai.core.registry import StageContext, stage

SAMPLE_COUNT = 12
BLUR_REFERENCE = 500.0  # Laplacian variance of a comfortably sharp frame
BLACK_LUMA_THRESHOLD = 12.0


def _sample_frames(path: Path, duration: float, count: int) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    frames: list[np.ndarray] = []
    try:
        for index in range(count):
            position = duration * (index + 0.5) / count
            capture.set(cv2.CAP_PROP_POS_MSEC, position * 1000.0)
            ok, frame = capture.read()
            if ok and frame is not None:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    finally:
        capture.release()
    return frames


def _score_clip(clip_id: str, path: Path, duration: float) -> ClipQuality:
    frames = _sample_frames(path, duration, SAMPLE_COUNT)
    if not frames:
        return ClipQuality(clip_id=clip_id, blur=0.0, motion=0.0, black_ratio=0.0,
                           usable=False, scored=False)

    sharpness = [float(cv2.Laplacian(frame, cv2.CV_64F).var()) for frame in frames]
    blur = 1.0 - min(1.0, float(np.mean(sharpness)) / BLUR_REFERENCE)

    differences = [
        float(np.mean(np.abs(frames[i].astype(np.int16) - frames[i - 1].astype(np.int16))))
        for i in range(1, len(frames))
    ]
    motion = min(1.0, float(np.mean(differences)) / 64.0) if differences else 0.0

    black_ratio = sum(1 for frame in frames if float(np.mean(frame)) < BLACK_LUMA_THRESHOLD) / len(frames)
    usable = black_ratio < 0.5 and blur < 0.95
    return ClipQuality(
        clip_id=clip_id, blur=blur, motion=motion, black_ratio=black_ratio, usable=usable
    )


@stage(id="quality", produces="02-quality", requires=("01-manifest",), model=QualityReport)
def quality(ctx: StageContext) -> QualityReport:
    manifest = ctx.store.read("01-manifest", Manifest)
    return QualityReport(
        clips=[
            _score_clip(clip.clip_id, Path(clip.proxy_path or clip.path), clip.duration)
            for clip in manifest.clips
        ]
    )
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_stage_quality.py -v`
Expected: 3 passed

- [ ] **Step 6: Commit**

```bash
git add videoai/core/models.py videoai/stages/s02_quality.py tests/test_stage_quality.py
git commit -m "feat: deterministic quality gate stage"
```

---

### Task 6: Transcription stage with ASR providers

**Files:**
- Modify: `videoai/core/models.py`
- Create: `videoai/providers/base.py`, `videoai/providers/asr_mock.py`, `videoai/providers/asr_parakeet.py`, `videoai/stages/s03_transcribe.py`
- Test: `tests/test_stage_transcribe.py`

**Interfaces:**
- Consumes: `Manifest` artifact `01-manifest`; `Config.providers["asr"]`.
- Produces:
  - Models `Word` (`text: str`, `start: float`, `end: float`, `confidence: float = 1.0`), `SpeechSpan` (`start`, `end`), `ClipTranscript` (`clip_id`, `words`, `speech_spans`), `Transcript` (`provider: str`, `clips: list[ClipTranscript]`, method `by_id`).
  - `ASRProvider` protocol: `name: str`, `transcribe(audio_path: Path) -> list[Word]`.
  - `resolve_asr(name: str) -> ASRProvider`, `resolve_llm(name: str) -> LLMProvider` in `videoai/providers/base.py`.
  - `derive_speech_spans(words: list[Word], gap: float) -> list[SpeechSpan]`.
  - Stage id `transcribe`, artifact `03-transcript`, provider_key `asr`.

The mock ASR reads a sidecar JSON next to the audio file (`<clip_id>.words.json`) so tests are deterministic and offline.

- [ ] **Step 1: Write the failing test**

`tests/test_stage_transcribe.py`:

```python
import json
from pathlib import Path

import pytest

from videoai.config import Config
from videoai.core.models import Manifest, Transcript, Word
from videoai.core.registry import StageContext
from videoai.core.store import ArtifactStore
from videoai.stages.s03_transcribe import derive_speech_spans, transcribe


def _context(tmp_path: Path) -> StageContext:
    for name in ("input", "work", "output"):
        (tmp_path / name).mkdir(exist_ok=True)
    return StageContext(
        project_dir=tmp_path,
        input_dir=tmp_path / "input",
        work_dir=tmp_path / "work",
        output_dir=tmp_path / "output",
        config=Config(providers={"asr": "mock", "llm": "mock"}),
        store=ArtifactStore(tmp_path / "work"),
    )


def _words(*triples: tuple[str, float, float]) -> list[dict]:
    return [{"text": t, "start": s, "end": e} for t, s, e in triples]


def test_derive_speech_spans_splits_on_long_gaps():
    words = [
        Word(text="a", start=0.0, end=0.4),
        Word(text="b", start=0.5, end=0.9),
        Word(text="c", start=2.5, end=2.9),
    ]
    spans = derive_speech_spans(words, gap=0.5)
    assert len(spans) == 2
    assert spans[0].start == 0.0 and spans[0].end == 0.9
    assert spans[1].start == 2.5 and spans[1].end == 2.9


def test_derive_speech_spans_returns_empty_for_no_words():
    assert derive_speech_spans([], gap=0.5) == []


def test_mock_provider_produces_transcript_artifact(tmp_path: Path, make_clip):
    ctx = _context(tmp_path)
    make_clip("a.mp4", seconds=2.0).rename(ctx.input_dir / "a.mp4")
    from videoai.stages.s01_ingest import ingest

    manifest: Manifest = ingest(ctx)
    ctx.store.write("01-manifest", manifest, fingerprint="fp")
    sidecar = Path(manifest.clips[0].audio_path).with_suffix(".words.json")
    sidecar.write_text(
        json.dumps(_words(("Look", 0.2, 0.5), ("here", 0.55, 0.9))), encoding="utf-8"
    )

    result = transcribe(ctx)

    assert isinstance(result, Transcript)
    assert result.provider == "mock"
    clip = result.by_id("clip-01")
    assert [w.text for w in clip.words] == ["Look", "here"]
    assert len(clip.speech_spans) == 1


def test_missing_sidecar_raises_clear_error(tmp_path: Path, make_clip):
    ctx = _context(tmp_path)
    make_clip("a.mp4", seconds=2.0).rename(ctx.input_dir / "a.mp4")
    from videoai.stages.s01_ingest import ingest

    ctx.store.write("01-manifest", ingest(ctx), fingerprint="fp")
    with pytest.raises(FileNotFoundError, match="words.json"):
        transcribe(ctx)


def test_clip_without_audio_yields_empty_transcript(tmp_path: Path):
    ctx = _context(tmp_path)
    import subprocess

    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", "testsrc=size=320x240:rate=30:duration=2",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", str(ctx.input_dir / "a.mp4"),
        ],
        check=True,
    )
    from videoai.stages.s01_ingest import ingest

    ctx.store.write("01-manifest", ingest(ctx), fingerprint="fp")

    result = transcribe(ctx)

    assert result.by_id("clip-01").words == []
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_stage_transcribe.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'videoai.stages.s03_transcribe'`

- [ ] **Step 3: Add transcript models to `videoai/core/models.py`**

```python
class Word(BaseModel):
    text: str
    start: float
    end: float
    confidence: float = 1.0


class SpeechSpan(BaseModel):
    start: float
    end: float


class ClipTranscript(BaseModel):
    clip_id: str
    words: list[Word] = Field(default_factory=list)
    speech_spans: list[SpeechSpan] = Field(default_factory=list)


class Transcript(BaseModel):
    provider: str
    clips: list[ClipTranscript] = Field(default_factory=list)

    def by_id(self, clip_id: str) -> ClipTranscript:
        for clip in self.clips:
            if clip.clip_id == clip_id:
                return clip
        raise KeyError(f"unknown clip_id: {clip_id}")
```

- [ ] **Step 4: Implement `videoai/providers/base.py`**

```python
"""Provider protocols and resolution. Providers are the only place external
services are touched; every one has a mock twin so the pipeline runs offline."""
from __future__ import annotations

from pathlib import Path
from typing import Protocol

from videoai.core.models import Word


class ASRProvider(Protocol):
    name: str

    def transcribe(self, audio_path: Path) -> list[Word]: ...


class LLMProvider(Protocol):
    name: str

    def complete_json(self, prompt: str, images: list[Path], timeout: int) -> dict: ...


def resolve_asr(name: str) -> ASRProvider:
    if name == "mock":
        from videoai.providers.asr_mock import MockASR

        return MockASR()
    if name == "parakeet":
        from videoai.providers.asr_parakeet import ParakeetASR

        return ParakeetASR()
    raise ValueError(f"unknown asr provider: {name}")


def resolve_llm(name: str) -> LLMProvider:
    if name == "mock":
        from videoai.providers.llm_mock import MockLLM

        return MockLLM()
    if name == "claude_cli":
        from videoai.providers.llm_claude_cli import ClaudeCliLLM

        return ClaudeCliLLM()
    raise ValueError(f"unknown llm provider: {name}")
```

- [ ] **Step 5: Implement `videoai/providers/asr_mock.py`**

```python
"""Mock ASR: reads word timings from a sidecar file next to the audio."""
from __future__ import annotations

import json
from pathlib import Path

from videoai.core.models import Word


class MockASR:
    name = "mock"

    def transcribe(self, audio_path: Path) -> list[Word]:
        sidecar = audio_path.with_suffix(".words.json")
        if not sidecar.exists():
            raise FileNotFoundError(
                f"mock ASR needs {sidecar.name} next to the audio file: {sidecar}"
            )
        raw = json.loads(sidecar.read_text(encoding="utf-8"))
        return [Word(**item) for item in raw]
```

- [ ] **Step 6: Implement `videoai/providers/asr_parakeet.py`**

```python
"""Local ASR on Apple Silicon via parakeet-mlx (verbatim output, word timestamps).

Verbatim matters: disfluencies are the signal used to find failed takes, so the
transcript must not be cleaned up.
"""
from __future__ import annotations

import os
from functools import cache
from pathlib import Path

from videoai.core.models import Word

DEFAULT_MODEL = os.getenv("PARAKEET_MODEL", "mlx-community/parakeet-tdt-0.6b-v3")


@cache
def _load_model(model_name: str):
    from parakeet_mlx import from_pretrained

    return from_pretrained(model_name)


class ParakeetASR:
    name = "parakeet"

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self.model_name = model_name

    def transcribe(self, audio_path: Path) -> list[Word]:
        model = _load_model(self.model_name)
        result = model.transcribe(str(audio_path))
        words: list[Word] = []
        for sentence in result.sentences:
            for token in sentence.tokens:
                text = token.text.strip()
                if text:
                    words.append(Word(text=text, start=float(token.start), end=float(token.end)))
        return words
```

- [ ] **Step 7: Implement `videoai/stages/s03_transcribe.py`**

```python
"""s03 transcribe: word-level timings plus speech spans derived from word gaps."""
from __future__ import annotations

from pathlib import Path

from videoai.core.models import ClipTranscript, Manifest, SpeechSpan, Transcript, Word
from videoai.core.registry import StageContext, stage
from videoai.providers.base import resolve_asr


def derive_speech_spans(words: list[Word], gap: float) -> list[SpeechSpan]:
    """Contiguous speech regions: split wherever the pause between words exceeds `gap`."""
    if not words:
        return []
    spans: list[SpeechSpan] = []
    start = words[0].start
    previous_end = words[0].end
    for word in words[1:]:
        if word.start - previous_end > gap:
            spans.append(SpeechSpan(start=start, end=previous_end))
            start = word.start
        previous_end = word.end
    spans.append(SpeechSpan(start=start, end=previous_end))
    return spans


@stage(
    id="transcribe",
    produces="03-transcript",
    requires=("01-manifest",),
    provider_key="asr",
    model=Transcript,
)
def transcribe(ctx: StageContext) -> Transcript:
    manifest = ctx.store.read("01-manifest", Manifest)
    provider = resolve_asr(ctx.config.providers["asr"])
    gap = ctx.config.transcribe.phrase_gap_seconds

    clips: list[ClipTranscript] = []
    for clip in manifest.clips:
        if not clip.audio_path:
            clips.append(ClipTranscript(clip_id=clip.clip_id))
            continue
        words = provider.transcribe(Path(clip.audio_path))
        clips.append(
            ClipTranscript(
                clip_id=clip.clip_id,
                words=words,
                speech_spans=derive_speech_spans(words, gap),
            )
        )
    return Transcript(provider=provider.name, clips=clips)
```

- [ ] **Step 8: Run the tests to verify they pass**

Run: `uv run pytest tests/test_stage_transcribe.py -v`
Expected: 5 passed

- [ ] **Step 9: Install the ASR dependency and smoke-test it against real audio**

```bash
uv add parakeet-mlx
uv run python -c "
from pathlib import Path
import subprocess
subprocess.run(['ffmpeg','-y','-loglevel','error','-f','lavfi','-i','sine=frequency=440:duration=2','-ac','1','-ar','16000','/tmp/tone.wav'], check=True)
from videoai.providers.asr_parakeet import ParakeetASR
print('words:', ParakeetASR().transcribe(Path('/tmp/tone.wav')))
"
```

Expected: the model downloads once, then prints `words: []` (a sine tone has no speech). Any `AttributeError` here means the parakeet-mlx result shape changed — inspect `result.sentences[0]` and adjust `asr_parakeet.py` before continuing.

- [ ] **Step 10: Commit**

```bash
git add videoai/core/models.py videoai/providers videoai/stages/s03_transcribe.py tests/test_stage_transcribe.py pyproject.toml uv.lock
git commit -m "feat: transcription stage with parakeet and mock ASR providers"
```

---

### Task 7: Phrase segmentation and packed transcript

**Files:**
- Create: `videoai/logic/phrases.py`
- Modify: `videoai/core/models.py`
- Test: `tests/test_phrases.py`

**Interfaces:**
- Consumes: `Word`, `ClipTranscript`, `Transcript` (Task 6).
- Produces:
  - Models `Phrase` (`phrase_id: str`, `clip_id: str`, `start: float`, `end: float`, `text: str`, `word_start: int`, `word_end: int`) and `PhraseIndex` (`phrases: list[Phrase]`, method `by_id`).
  - `build_phrases(transcript: Transcript, gap: float, max_words: int) -> PhraseIndex`.
  - `pack_transcript(index: PhraseIndex) -> str` producing the compact markdown given to the LLM.

`phrase_id` format is `<clip_id>#<ordinal>`, e.g. `clip-01#003`. `word_start`/`word_end` are indices into the clip's `words` list, `word_end` exclusive.

- [ ] **Step 1: Write the failing test**

`tests/test_phrases.py`:

```python
from videoai.core.models import ClipTranscript, Transcript, Word
from videoai.logic.phrases import build_phrases, pack_transcript


def _transcript(*clips: tuple[str, list[tuple[str, float, float]]]) -> Transcript:
    return Transcript(
        provider="mock",
        clips=[
            ClipTranscript(
                clip_id=clip_id,
                words=[Word(text=t, start=s, end=e) for t, s, e in words],
            )
            for clip_id, words in clips
        ],
    )


def test_phrases_split_on_gap():
    transcript = _transcript(
        ("clip-01", [("Look", 0.0, 0.3), ("here", 0.35, 0.7), ("Wow", 2.0, 2.4)])
    )
    index = build_phrases(transcript, gap=0.5, max_words=30)
    assert [p.text for p in index.phrases] == ["Look here", "Wow"]
    assert index.phrases[0].phrase_id == "clip-01#001"
    assert index.phrases[1].phrase_id == "clip-01#002"


def test_phrase_carries_time_and_word_range():
    transcript = _transcript(
        ("clip-01", [("Look", 0.0, 0.3), ("here", 0.35, 0.7), ("Wow", 2.0, 2.4)])
    )
    index = build_phrases(transcript, gap=0.5, max_words=30)
    first = index.phrases[0]
    assert first.start == 0.0 and first.end == 0.7
    assert (first.word_start, first.word_end) == (0, 2)
    second = index.phrases[1]
    assert (second.word_start, second.word_end) == (2, 3)


def test_phrases_split_on_max_words():
    words = [(f"w{i}", i * 0.1, i * 0.1 + 0.05) for i in range(10)]
    transcript = _transcript(("clip-01", words))
    index = build_phrases(transcript, gap=5.0, max_words=4)
    assert [len(p.text.split()) for p in index.phrases] == [4, 4, 2]


def test_phrases_are_numbered_per_clip():
    transcript = _transcript(
        ("clip-01", [("a", 0.0, 0.2)]),
        ("clip-02", [("b", 0.0, 0.2)]),
    )
    index = build_phrases(transcript, gap=0.5, max_words=30)
    assert [p.phrase_id for p in index.phrases] == ["clip-01#001", "clip-02#001"]


def test_clip_without_words_produces_no_phrases():
    transcript = _transcript(("clip-01", []))
    assert build_phrases(transcript, gap=0.5, max_words=30).phrases == []


def test_by_id_raises_for_unknown_phrase():
    index = build_phrases(_transcript(("clip-01", [("a", 0.0, 0.2)])), gap=0.5, max_words=30)
    assert index.by_id("clip-01#001").text == "a"
    try:
        index.by_id("clip-09#001")
    except KeyError as error:
        assert "clip-09#001" in str(error)
    else:
        raise AssertionError("expected KeyError")


def test_pack_transcript_is_compact_and_labelled():
    transcript = _transcript(
        ("clip-01", [("Look", 0.0, 0.3), ("here", 0.35, 0.7), ("Wow", 2.0, 2.4)])
    )
    packed = pack_transcript(build_phrases(transcript, gap=0.5, max_words=30))
    assert "## clip-01" in packed
    assert "[clip-01#001] 0.00-0.70 Look here" in packed
    assert "[clip-01#002] 2.00-2.40 Wow" in packed
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_phrases.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'videoai.logic.phrases'`

- [ ] **Step 3: Add phrase models to `videoai/core/models.py`**

```python
class Phrase(BaseModel):
    phrase_id: str
    clip_id: str
    start: float
    end: float
    text: str
    word_start: int
    word_end: int


class PhraseIndex(BaseModel):
    phrases: list[Phrase] = Field(default_factory=list)

    def by_id(self, phrase_id: str) -> Phrase:
        for phrase in self.phrases:
            if phrase.phrase_id == phrase_id:
                return phrase
        raise KeyError(f"unknown phrase_id: {phrase_id}")
```

- [ ] **Step 4: Implement `videoai/logic/phrases.py`**

```python
"""Words to phrases, and phrases to the compact view handed to the LLM.

Phrases are the unit every later stage reasons about: they are short enough to
score individually and long enough to carry meaning, and their boundaries sit in
silence, which makes them safe cut points.
"""
from __future__ import annotations

from videoai.core.models import Phrase, PhraseIndex, Transcript, Word


def _flush(
    clip_id: str, ordinal: int, words: list[Word], word_start: int
) -> Phrase:
    return Phrase(
        phrase_id=f"{clip_id}#{ordinal:03d}",
        clip_id=clip_id,
        start=words[0].start,
        end=words[-1].end,
        text=" ".join(word.text for word in words),
        word_start=word_start,
        word_end=word_start + len(words),
    )


def build_phrases(transcript: Transcript, gap: float, max_words: int) -> PhraseIndex:
    phrases: list[Phrase] = []
    for clip in transcript.clips:
        ordinal = 0
        buffer: list[Word] = []
        buffer_start = 0
        for index, word in enumerate(clip.words):
            if buffer:
                too_long = len(buffer) >= max_words
                long_pause = word.start - buffer[-1].end > gap
                if too_long or long_pause:
                    ordinal += 1
                    phrases.append(_flush(clip.clip_id, ordinal, buffer, buffer_start))
                    buffer = []
            if not buffer:
                buffer_start = index
            buffer.append(word)
        if buffer:
            ordinal += 1
            phrases.append(_flush(clip.clip_id, ordinal, buffer, buffer_start))
    return PhraseIndex(phrases=phrases)


def pack_transcript(index: PhraseIndex) -> str:
    """Compact, id-addressable transcript. Roughly a tenth of raw JSON in tokens."""
    lines: list[str] = []
    current_clip: str | None = None
    for phrase in index.phrases:
        if phrase.clip_id != current_clip:
            current_clip = phrase.clip_id
            lines.append(f"\n## {current_clip}")
        lines.append(
            f"[{phrase.phrase_id}] {phrase.start:.2f}-{phrase.end:.2f} {phrase.text}"
        )
    return "\n".join(lines).strip()
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_phrases.py -v`
Expected: 7 passed

- [ ] **Step 6: Commit**

```bash
git add videoai/logic/phrases.py videoai/core/models.py tests/test_phrases.py
git commit -m "feat: phrase segmentation and packed transcript"
```

---

### Task 8: Repeated-take detection

**Files:**
- Create: `videoai/logic/takes.py`
- Modify: `videoai/core/models.py`
- Test: `tests/test_takes.py`

**Interfaces:**
- Consumes: `Phrase`, `PhraseIndex` (Task 7).
- Produces:
  - Models `TakeGroup` (`group_id: str`, `phrase_ids: list[str]`) and `TakeGroups` (`groups: list[TakeGroup]`, method `group_of(phrase_id) -> str | None`).
  - `detect_take_groups(index: PhraseIndex, similarity: int = 80, window: int = 6) -> TakeGroups`.

Two phrases belong to the same take group when their normalised text similarity (rapidfuzz `token_sort_ratio`) is at least `similarity` and they are within `window` phrases of each other, **including across clips** (clip boundaries are re-take boundaries in this workflow). Group ids are `take-01`, `take-02`, …

- [ ] **Step 1: Write the failing test**

`tests/test_takes.py`:

```python
from videoai.core.models import Phrase, PhraseIndex
from videoai.logic.takes import detect_take_groups


def _index(*items: tuple[str, str, str]) -> PhraseIndex:
    phrases = []
    for ordinal, (phrase_id, clip_id, text) in enumerate(items):
        phrases.append(
            Phrase(
                phrase_id=phrase_id,
                clip_id=clip_id,
                start=float(ordinal),
                end=float(ordinal) + 0.8,
                text=text,
                word_start=ordinal,
                word_end=ordinal + 1,
            )
        )
    return PhraseIndex(phrases=phrases)


def test_exact_repeat_is_grouped():
    index = _index(
        ("clip-01#001", "clip-01", "This is the mega wrex truck"),
        ("clip-01#002", "clip-01", "This is the mega wrex truck"),
    )
    groups = detect_take_groups(index)
    assert len(groups.groups) == 1
    assert groups.groups[0].phrase_ids == ["clip-01#001", "clip-01#002"]


def test_paraphrased_repeat_is_grouped():
    index = _index(
        ("clip-01#001", "clip-01", "this is the mega wrex monster truck"),
        ("clip-01#002", "clip-01", "this is the mega wrex truck monster"),
    )
    groups = detect_take_groups(index, similarity=80)
    assert len(groups.groups) == 1


def test_unrelated_phrases_are_not_grouped():
    index = _index(
        ("clip-01#001", "clip-01", "look at the wheels"),
        ("clip-01#002", "clip-01", "now I will open the box"),
    )
    assert detect_take_groups(index).groups == []


def test_repeats_are_detected_across_clips():
    index = _index(
        ("clip-01#001", "clip-01", "welcome to my channel"),
        ("clip-02#001", "clip-02", "welcome to my channel"),
    )
    groups = detect_take_groups(index)
    assert len(groups.groups) == 1
    assert groups.groups[0].phrase_ids == ["clip-01#001", "clip-02#001"]


def test_repeat_beyond_window_is_ignored():
    items = [("clip-01#001", "clip-01", "welcome to my channel")]
    items += [(f"clip-01#{i:03d}", "clip-01", f"filler sentence number {i}") for i in range(2, 10)]
    items.append(("clip-01#010", "clip-01", "welcome to my channel"))
    assert detect_take_groups(_index(*items), window=3).groups == []


def test_three_attempts_form_one_group():
    index = _index(
        ("clip-01#001", "clip-01", "hello everyone welcome back"),
        ("clip-01#002", "clip-01", "hello everyone welcome back"),
        ("clip-01#003", "clip-01", "hello everyone welcome back"),
    )
    groups = detect_take_groups(index)
    assert len(groups.groups) == 1
    assert len(groups.groups[0].phrase_ids) == 3


def test_group_of_returns_group_id_or_none():
    index = _index(
        ("clip-01#001", "clip-01", "hello everyone welcome back"),
        ("clip-01#002", "clip-01", "hello everyone welcome back"),
        ("clip-01#003", "clip-01", "completely different content here"),
    )
    groups = detect_take_groups(index)
    assert groups.group_of("clip-01#001") == "take-01"
    assert groups.group_of("clip-01#003") is None


def test_short_phrases_are_skipped():
    index = _index(
        ("clip-01#001", "clip-01", "yes"),
        ("clip-01#002", "clip-01", "yes"),
    )
    assert detect_take_groups(index).groups == []
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_takes.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'videoai.logic.takes'`

- [ ] **Step 3: Add take models to `videoai/core/models.py`**

```python
class TakeGroup(BaseModel):
    group_id: str
    phrase_ids: list[str] = Field(default_factory=list)


class TakeGroups(BaseModel):
    groups: list[TakeGroup] = Field(default_factory=list)

    def group_of(self, phrase_id: str) -> str | None:
        for group in self.groups:
            if phrase_id in group.phrase_ids:
                return group.group_id
        return None
```

- [ ] **Step 4: Implement `videoai/logic/takes.py`**

```python
"""Detect repeated attempts at the same line.

Recall is deterministic and cheap here; precision is the model's job later. The
detector only proposes candidate groups — which attempt is best is an editorial
judgement made during analysis.
"""
from __future__ import annotations

import re

from rapidfuzz import fuzz

from videoai.core.models import PhraseIndex, TakeGroup, TakeGroups

MIN_WORDS = 4


def _normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", text.lower()).strip()


def detect_take_groups(
    index: PhraseIndex, similarity: int = 80, window: int = 6
) -> TakeGroups:
    phrases = index.phrases
    normalised = [_normalise(phrase.text) for phrase in phrases]
    eligible = [len(text.split()) >= MIN_WORDS for text in normalised]

    parent = list(range(len(phrases)))

    def find(item: int) -> int:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    for i in range(len(phrases)):
        if not eligible[i]:
            continue
        for j in range(i + 1, min(i + 1 + window, len(phrases))):
            if not eligible[j]:
                continue
            if fuzz.token_sort_ratio(normalised[i], normalised[j]) >= similarity:
                union(i, j)

    members: dict[int, list[int]] = {}
    for position in range(len(phrases)):
        if eligible[position]:
            members.setdefault(find(position), []).append(position)

    groups: list[TakeGroup] = []
    for root in sorted(members):
        positions = members[root]
        if len(positions) < 2:
            continue
        groups.append(
            TakeGroup(
                group_id=f"take-{len(groups) + 1:02d}",
                phrase_ids=[phrases[position].phrase_id for position in positions],
            )
        )
    return TakeGroups(groups=groups)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_takes.py -v`
Expected: 8 passed

- [ ] **Step 6: Commit**

```bash
git add videoai/logic/takes.py videoai/core/models.py tests/test_takes.py
git commit -m "feat: fuzzy repeated-take detection"
```

---

### Task 9: Analysis stage with LLM providers

**Files:**
- Modify: `videoai/core/models.py`
- Create: `videoai/providers/llm_mock.py`, `videoai/providers/llm_claude_cli.py`, `videoai/stages/s04_analyze.py`
- Test: `tests/test_stage_analyze.py`

**Interfaces:**
- Consumes: `Manifest`, `QualityReport`, `Transcript`; `build_phrases`, `pack_transcript`; `detect_take_groups`; `resolve_llm`; `extract_frame`.
- Produces:
  - Models `SegmentAnalysis` (`phrase_id: str`, `clip_id: str`, `start: float`, `end: float`, `text: str`, `content: str`, `delivery_score: int`, `visual_score: int`, `emotion: str`, `is_failed_take: bool`, `take_group: str | None`, `shorts_candidate: bool`) and `Analysis` (`provider: str`, `segments: list[SegmentAnalysis]`, method `by_phrase`).
  - `MockLLM` reading canned JSON from the `VIDEOAI_MOCK_LLM` environment variable (a file path).
  - `ClaudeCliLLM.complete_json(prompt, images, timeout)` shelling out to `claude -p ... --output-format json`.
  - `build_analysis_prompt(packed: str, takes: TakeGroups, quality: QualityReport, brief: str) -> str`.
  - Stage id `analyze`, artifact `04-analysis`, provider_key `llm`, requires `01-manifest`, `02-quality`, `03-transcript`.

- [ ] **Step 1: Write the failing test**

`tests/test_stage_analyze.py`:

```python
import json
from pathlib import Path

import pytest

from videoai.config import Config
from videoai.core.models import (
    Analysis,
    ClipQuality,
    ClipTranscript,
    Manifest,
    QualityReport,
    Transcript,
    Word,
)
from videoai.core.registry import StageContext
from videoai.core.store import ArtifactStore
from videoai.stages.s04_analyze import analyze, build_analysis_prompt


def _context(tmp_path: Path, monkeypatch, llm_payload: dict) -> StageContext:
    for name in ("input", "work", "output"):
        (tmp_path / name).mkdir(exist_ok=True)
    payload_path = tmp_path / "llm.json"
    payload_path.write_text(json.dumps(llm_payload), encoding="utf-8")
    monkeypatch.setenv("VIDEOAI_MOCK_LLM", str(payload_path))
    return StageContext(
        project_dir=tmp_path,
        input_dir=tmp_path / "input",
        work_dir=tmp_path / "work",
        output_dir=tmp_path / "output",
        config=Config(providers={"asr": "mock", "llm": "mock"}),
        store=ArtifactStore(tmp_path / "work"),
    )


def _seed_artifacts(ctx: StageContext) -> None:
    ctx.store.write(
        "01-manifest",
        Manifest(clips=[{
            "clip_id": "clip-01", "path": "/tmp/a.mp4", "duration": 10.0,
            "width": 320, "height": 240, "fps": 30.0, "has_audio": True,
        }]),
        fingerprint="fp",
    )
    ctx.store.write(
        "02-quality",
        QualityReport(clips=[ClipQuality(clip_id="clip-01", blur=0.1, motion=0.2, black_ratio=0.0, usable=True)]),
        fingerprint="fp",
    )
    ctx.store.write(
        "03-transcript",
        Transcript(
            provider="mock",
            clips=[ClipTranscript(
                clip_id="clip-01",
                words=[
                    Word(text="Look", start=0.0, end=0.3),
                    Word(text="here", start=0.35, end=0.7),
                    Word(text="Wow", start=2.0, end=2.4),
                ],
            )],
        ),
        fingerprint="fp",
    )


def test_analysis_maps_llm_scores_onto_phrases(tmp_path: Path, monkeypatch):
    payload = {"segments": [
        {"phrase_id": "clip-01#001", "content": "intro", "delivery_score": 8,
         "visual_score": 7, "emotion": "excited", "is_failed_take": False,
         "shorts_candidate": True},
        {"phrase_id": "clip-01#002", "content": "reaction", "delivery_score": 5,
         "visual_score": 6, "emotion": "calm", "is_failed_take": False,
         "shorts_candidate": False},
    ]}
    ctx = _context(tmp_path, monkeypatch, payload)
    _seed_artifacts(ctx)

    result = analyze(ctx)

    assert isinstance(result, Analysis)
    assert result.provider == "mock"
    first = result.by_phrase("clip-01#001")
    assert first.delivery_score == 8
    assert first.shorts_candidate is True
    assert first.start == 0.0 and first.end == 0.7
    assert first.text == "Look here"


def test_unknown_phrase_id_from_llm_is_rejected(tmp_path: Path, monkeypatch):
    payload = {"segments": [
        {"phrase_id": "clip-99#001", "content": "ghost", "delivery_score": 5,
         "visual_score": 5, "emotion": "calm", "is_failed_take": False,
         "shorts_candidate": False},
    ]}
    ctx = _context(tmp_path, monkeypatch, payload)
    _seed_artifacts(ctx)

    with pytest.raises(ValueError, match="clip-99#001"):
        analyze(ctx)


def test_missing_phrase_in_llm_response_gets_neutral_defaults(tmp_path: Path, monkeypatch):
    payload = {"segments": [
        {"phrase_id": "clip-01#001", "content": "intro", "delivery_score": 8,
         "visual_score": 7, "emotion": "excited", "is_failed_take": False,
         "shorts_candidate": True},
    ]}
    ctx = _context(tmp_path, monkeypatch, payload)
    _seed_artifacts(ctx)

    result = analyze(ctx)

    second = result.by_phrase("clip-01#002")
    assert second.delivery_score == 5
    assert second.is_failed_take is False


def test_prompt_contains_packed_transcript_and_rules(tmp_path: Path, monkeypatch):
    from videoai.core.models import TakeGroups

    prompt = build_analysis_prompt(
        packed="[clip-01#001] 0.00-0.70 Look here",
        takes=TakeGroups(),
        quality=QualityReport(clips=[ClipQuality(clip_id="clip-01", blur=0.9, motion=0.1, black_ratio=0.0, usable=False)]),
        brief="Toy review",
    )
    assert "clip-01#001" in prompt
    assert "Toy review" in prompt
    assert "JSON" in prompt
    assert "clip-01" in prompt
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_stage_analyze.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'videoai.stages.s04_analyze'`

- [ ] **Step 3: Add analysis models to `videoai/core/models.py`**

```python
class SegmentAnalysis(BaseModel):
    phrase_id: str
    clip_id: str
    start: float
    end: float
    text: str
    content: str = ""
    delivery_score: int = 5
    visual_score: int = 5
    emotion: str = "neutral"
    is_failed_take: bool = False
    take_group: str | None = None
    shorts_candidate: bool = False


class Analysis(BaseModel):
    provider: str
    segments: list[SegmentAnalysis] = Field(default_factory=list)

    def by_phrase(self, phrase_id: str) -> SegmentAnalysis:
        for segment in self.segments:
            if segment.phrase_id == phrase_id:
                return segment
        raise KeyError(f"unknown phrase_id: {phrase_id}")
```

- [ ] **Step 4: Implement `videoai/providers/llm_mock.py`**

```python
"""Mock LLM: returns a canned JSON document named by VIDEOAI_MOCK_LLM."""
from __future__ import annotations

import json
import os
from pathlib import Path


class MockLLM:
    name = "mock"

    def complete_json(self, prompt: str, images: list[Path], timeout: int) -> dict:
        path = os.getenv("VIDEOAI_MOCK_LLM")
        if not path:
            raise RuntimeError("VIDEOAI_MOCK_LLM must point to a canned response file")
        return json.loads(Path(path).read_text(encoding="utf-8"))
```

- [ ] **Step 5: Implement `videoai/providers/llm_claude_cli.py`**

```python
"""LLM via the Claude Code CLI in headless mode.

Runs under the user's Claude subscription rather than an API key. The CLI is
asked for JSON and its envelope is unwrapped; the model's own text lands in
`result`.
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

SYSTEM_PROMPT = (
    "You are a video editing analyst. You reply with a single JSON document and "
    "nothing else: no prose, no markdown fences."
)


def _extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise ValueError(f"no JSON object found in model reply: {text[:300]}")
        return json.loads(match.group(0))


class ClaudeCliLLM:
    name = "claude_cli"

    def __init__(self, model: str = "sonnet") -> None:
        self.model = model

    def complete_json(self, prompt: str, images: list[Path], timeout: int) -> dict:
        if images:
            listing = "\n".join(f"- {path}" for path in images)
            prompt = f"{prompt}\n\nReference frames (read them if useful):\n{listing}"
        result = subprocess.run(
            [
                "claude", "-p", prompt,
                "--output-format", "json",
                "--model", self.model,
                "--system-prompt", SYSTEM_PROMPT,
                "--strict-mcp-config",
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            raise RuntimeError(f"claude CLI failed: {result.stderr.strip()[:500]}")
        envelope = json.loads(result.stdout)
        if envelope.get("is_error"):
            raise RuntimeError(f"claude CLI returned an error: {envelope.get('result')}")
        return _extract_json(envelope["result"])
```

- [ ] **Step 6: Implement `videoai/stages/s04_analyze.py`**

```python
"""s04 analyze: score every phrase for delivery, visuals and shorts potential."""
from __future__ import annotations

from pathlib import Path

from videoai.core.ffmpeg import extract_frame
from videoai.core.models import (
    Analysis,
    Manifest,
    PhraseIndex,
    QualityReport,
    SegmentAnalysis,
    TakeGroups,
    Transcript,
)
from videoai.core.registry import StageContext, stage
from videoai.logic.phrases import build_phrases, pack_transcript
from videoai.logic.takes import detect_take_groups
from videoai.providers.base import resolve_llm

INSTRUCTIONS = """You are editing a short YouTube video in which a child reviews a toy.

Below is the full transcript of every take, split into phrases. Each phrase has an
id like clip-01#003, a time range and its verbatim text (disfluencies included).

Score every phrase. Reply with one JSON document:

{"segments": [
  {"phrase_id": "clip-01#001", "content": "one short clause describing what happens",
   "delivery_score": 1-10, "visual_score": 1-10, "emotion": "excited|calm|funny|neutral",
   "is_failed_take": true|false, "shorts_candidate": true|false}
]}

Rules:
- Include every phrase id exactly once. Invent no ids.
- delivery_score rates energy and clarity of speech; visual_score rates how
  interesting the phrase is likely to look on screen.
- is_failed_take is true for restarts, cut-off sentences and obvious mistakes.
- shorts_candidate is true only for self-contained, high-energy moments.
"""


def build_analysis_prompt(
    packed: str, takes: TakeGroups, quality: QualityReport, brief: str
) -> str:
    sections = [INSTRUCTIONS]
    if brief.strip():
        sections.append(f"Creator brief:\n{brief.strip()}")
    if takes.groups:
        lines = [
            f"- {group.group_id}: {', '.join(group.phrase_ids)}" for group in takes.groups
        ]
        sections.append(
            "Phrases suspected to be repeated attempts at the same line "
            "(pick the best one, mark the others as failed takes):\n" + "\n".join(lines)
        )
    quality_lines = [
        f"- {clip.clip_id}: blur={clip.blur:.2f} motion={clip.motion:.2f} "
        f"usable={clip.usable} scored={clip.scored}"
        for clip in quality.clips
    ]
    sections.append("Technical quality per clip:\n" + "\n".join(quality_lines))
    sections.append(f"Transcript:\n{packed}")
    return "\n\n".join(sections)


def _read_brief(input_dir: Path) -> str:
    parts: list[str] = []
    for name in ("project.yaml", "notes.md"):
        path = input_dir / name
        if path.exists():
            parts.append(path.read_text(encoding="utf-8"))
    return "\n\n".join(parts)


def _keyframes(ctx: StageContext, manifest: Manifest, index: PhraseIndex) -> list[Path]:
    frames_dir = ctx.work_dir / "keyframes"
    paths: list[Path] = []
    for phrase in index.phrases:
        clip = manifest.by_id(phrase.clip_id)
        target = frames_dir / f"{phrase.phrase_id.replace('#', '-')}.jpg"
        if not target.exists():
            source = Path(clip.proxy_path or clip.path)
            if source.exists():
                extract_frame(source, at=(phrase.start + phrase.end) / 2, dst=target)
        if target.exists():
            paths.append(target)
    return paths


@stage(
    id="analyze",
    produces="04-analysis",
    requires=("01-manifest", "02-quality", "03-transcript"),
    provider_key="llm",
    model=Analysis,
)
def analyze(ctx: StageContext) -> Analysis:
    manifest = ctx.store.read("01-manifest", Manifest)
    quality = ctx.store.read("02-quality", QualityReport)
    transcript = ctx.store.read("03-transcript", Transcript)

    settings = ctx.config.transcribe
    index = build_phrases(transcript, settings.phrase_gap_seconds, settings.max_words_per_phrase)
    takes = detect_take_groups(index)
    ctx.store.write("03b-phrases", index, fingerprint="derived")
    ctx.store.write("03c-takes", takes, fingerprint="derived")

    provider = resolve_llm(ctx.config.providers["llm"])
    prompt = build_analysis_prompt(pack_transcript(index), takes, quality, _read_brief(ctx.input_dir))
    response = provider.complete_json(
        prompt, _keyframes(ctx, manifest, index), ctx.config.analyze.llm_timeout_seconds
    )

    known = {phrase.phrase_id for phrase in index.phrases}
    scored: dict[str, dict] = {}
    for item in response.get("segments", []):
        phrase_id = item.get("phrase_id")
        if phrase_id not in known:
            raise ValueError(f"model returned an unknown phrase_id: {phrase_id}")
        scored[phrase_id] = item

    segments: list[SegmentAnalysis] = []
    for phrase in index.phrases:
        item = scored.get(phrase.phrase_id, {})
        segments.append(
            SegmentAnalysis(
                phrase_id=phrase.phrase_id,
                clip_id=phrase.clip_id,
                start=phrase.start,
                end=phrase.end,
                text=phrase.text,
                content=item.get("content", ""),
                delivery_score=int(item.get("delivery_score", 5)),
                visual_score=int(item.get("visual_score", 5)),
                emotion=item.get("emotion", "neutral"),
                is_failed_take=bool(item.get("is_failed_take", False)),
                take_group=takes.group_of(phrase.phrase_id),
                shorts_candidate=bool(item.get("shorts_candidate", False)),
            )
        )
    return Analysis(provider=provider.name, segments=segments)
```

- [ ] **Step 7: Run the tests to verify they pass**

Run: `uv run pytest tests/test_stage_analyze.py -v`
Expected: 4 passed

- [ ] **Step 8: Commit**

```bash
git add videoai/core/models.py videoai/providers/llm_mock.py videoai/providers/llm_claude_cli.py videoai/stages/s04_analyze.py tests/test_stage_analyze.py
git commit -m "feat: analysis stage with Claude CLI and mock LLM providers"
```

---

### Task 10: Story plan, timeline builder and validator

**Files:**
- Modify: `videoai/core/models.py`
- Create: `videoai/logic/timeline.py`, `videoai/logic/validate.py`, `videoai/stages/s05_plan.py`
- Test: `tests/test_timeline.py`, `tests/test_validate.py`, `tests/test_stage_plan.py`

**Interfaces:**
- Consumes: `Analysis`, `Manifest`, `Transcript`, `PhraseIndex`, `resolve_llm`.
- Produces:
  - Models `PlanSection` (`name: str`, `goal: str`, `phrase_ids: list[str]`), `StoryPlan` (`sections`, `title`, `description`, `tags`), `TimelineClip` (`src: str`, `offset: float`, `dur: float`, `start: float`, `quote: str`, `reason: str`, `beat: str`), `Timeline` (`fps: float`, `width: int`, `height: int`, `clips: list[TimelineClip]`, property `duration`).
  - `build_timeline(plan: StoryPlan, analysis: Analysis, manifest: Manifest, padding: float, fps: float) -> Timeline`.
  - `validate_timeline(timeline: Timeline, manifest: Manifest, transcript: Transcript) -> list[str]` returning human-readable violations (empty list means valid).
  - Stage id `plan`, artifact `05-timeline`, provider_key `llm`, requires `01-manifest`, `03-transcript`, `04-analysis`. It also writes `05a-storyplan`.

- [ ] **Step 1: Write the failing tests for the builder**

`tests/test_timeline.py`:

```python
from videoai.core.models import (
    Analysis,
    ClipInfo,
    Manifest,
    PlanSection,
    SegmentAnalysis,
    StoryPlan,
)
from videoai.logic.timeline import build_timeline


def _manifest() -> Manifest:
    return Manifest(clips=[
        ClipInfo(clip_id="clip-01", path="/tmp/a.mp4", duration=30.0, width=1920,
                 height=1080, fps=30.0, has_audio=True),
        ClipInfo(clip_id="clip-02", path="/tmp/b.mp4", duration=30.0, width=1920,
                 height=1080, fps=30.0, has_audio=True),
    ])


def _analysis() -> Analysis:
    return Analysis(provider="mock", segments=[
        SegmentAnalysis(phrase_id="clip-01#001", clip_id="clip-01", start=2.0, end=5.0,
                        text="hello everyone", content="intro", delivery_score=9),
        SegmentAnalysis(phrase_id="clip-02#001", clip_id="clip-02", start=10.0, end=14.0,
                        text="look at the wheels", content="wheels", delivery_score=7),
    ])


def _plan() -> StoryPlan:
    return StoryPlan(
        sections=[
            PlanSection(name="Hook", goal="open strong", phrase_ids=["clip-01#001"]),
            PlanSection(name="Body", goal="show details", phrase_ids=["clip-02#001"]),
        ],
        title="T", description="D", tags=["toy"],
    )


def test_timeline_clips_follow_plan_order():
    timeline = build_timeline(_plan(), _analysis(), _manifest(), padding=0.15, fps=30.0)
    assert [clip.src for clip in timeline.clips] == ["clip-01", "clip-02"]


def test_timeline_positions_are_contiguous():
    timeline = build_timeline(_plan(), _analysis(), _manifest(), padding=0.15, fps=30.0)
    assert timeline.clips[0].start == 0.0
    expected = timeline.clips[0].start + timeline.clips[0].dur
    assert abs(timeline.clips[1].start - expected) < 1e-6


def test_padding_extends_each_segment_on_both_sides():
    timeline = build_timeline(_plan(), _analysis(), _manifest(), padding=0.15, fps=30.0)
    first = timeline.clips[0]
    assert abs(first.offset - 1.85) < 1e-6
    assert abs(first.dur - 3.3) < 1e-6


def test_padding_is_clamped_to_clip_bounds():
    analysis = Analysis(provider="mock", segments=[
        SegmentAnalysis(phrase_id="clip-01#001", clip_id="clip-01", start=0.05, end=29.98,
                        text="whole clip", content="all", delivery_score=8),
    ])
    plan = StoryPlan(
        sections=[PlanSection(name="All", goal="everything", phrase_ids=["clip-01#001"])],
        title="T", description="D", tags=[],
    )
    timeline = build_timeline(plan, analysis, _manifest(), padding=0.5, fps=30.0)
    clip = timeline.clips[0]
    assert clip.offset == 0.0
    assert clip.offset + clip.dur <= 30.0


def test_clip_carries_provenance_fields():
    timeline = build_timeline(_plan(), _analysis(), _manifest(), padding=0.15, fps=30.0)
    first = timeline.clips[0]
    assert first.quote == "hello everyone"
    assert first.beat == "Hook"
    assert "intro" in first.reason


def test_resolution_comes_from_first_source_clip():
    timeline = build_timeline(_plan(), _analysis(), _manifest(), padding=0.15, fps=30.0)
    assert (timeline.width, timeline.height) == (1920, 1080)
    assert timeline.fps == 30.0


def test_unknown_phrase_id_in_plan_raises():
    plan = StoryPlan(
        sections=[PlanSection(name="Hook", goal="x", phrase_ids=["clip-09#001"])],
        title="T", description="D", tags=[],
    )
    try:
        build_timeline(plan, _analysis(), _manifest(), padding=0.15, fps=30.0)
    except KeyError as error:
        assert "clip-09#001" in str(error)
    else:
        raise AssertionError("expected KeyError")
```

- [ ] **Step 2: Write the failing tests for the validator**

`tests/test_validate.py`:

```python
from videoai.core.models import (
    ClipInfo,
    ClipTranscript,
    Manifest,
    Timeline,
    TimelineClip,
    Transcript,
    Word,
)
from videoai.logic.validate import validate_timeline


def _manifest() -> Manifest:
    return Manifest(clips=[ClipInfo(clip_id="clip-01", path="/tmp/a.mp4", duration=30.0,
                                    width=1920, height=1080, fps=30.0, has_audio=True)])


def _transcript() -> Transcript:
    return Transcript(provider="mock", clips=[ClipTranscript(
        clip_id="clip-01",
        words=[Word(text="hello", start=2.0, end=2.5), Word(text="world", start=2.6, end=3.0)],
    )])


def _timeline(*clips: TimelineClip) -> Timeline:
    return Timeline(fps=30.0, width=1920, height=1080, clips=list(clips))


def test_valid_timeline_has_no_violations():
    timeline = _timeline(
        TimelineClip(src="clip-01", offset=1.8, dur=1.4, start=0.0, quote="hello world")
    )
    assert validate_timeline(timeline, _manifest(), _transcript()) == []


def test_gap_between_clips_is_reported():
    timeline = _timeline(
        TimelineClip(src="clip-01", offset=1.8, dur=1.4, start=0.0, quote="hello world"),
        TimelineClip(src="clip-01", offset=5.0, dur=1.0, start=2.0, quote=""),
    )
    violations = validate_timeline(timeline, _manifest(), _transcript())
    assert any("contiguous" in v for v in violations)


def test_segment_beyond_source_duration_is_reported():
    timeline = _timeline(TimelineClip(src="clip-01", offset=29.0, dur=5.0, start=0.0, quote=""))
    violations = validate_timeline(timeline, _manifest(), _transcript())
    assert any("exceeds source duration" in v for v in violations)


def test_unknown_source_is_reported():
    timeline = _timeline(TimelineClip(src="clip-99", offset=0.0, dur=1.0, start=0.0, quote=""))
    violations = validate_timeline(timeline, _manifest(), _transcript())
    assert any("unknown source" in v for v in violations)


def test_too_short_segment_is_reported():
    timeline = _timeline(TimelineClip(src="clip-01", offset=2.0, dur=0.1, start=0.0, quote=""))
    violations = validate_timeline(timeline, _manifest(), _transcript())
    assert any("shorter than" in v for v in violations)


def test_cut_inside_a_word_is_reported():
    timeline = _timeline(TimelineClip(src="clip-01", offset=2.2, dur=0.6, start=0.0, quote=""))
    violations = validate_timeline(timeline, _manifest(), _transcript())
    assert any("inside word" in v for v in violations)


def test_quote_not_present_in_segment_is_reported():
    timeline = _timeline(
        TimelineClip(src="clip-01", offset=1.8, dur=1.4, start=0.0, quote="totally invented")
    )
    violations = validate_timeline(timeline, _manifest(), _transcript())
    assert any("quote not found" in v for v in violations)


def test_negative_offset_is_reported():
    timeline = _timeline(TimelineClip(src="clip-01", offset=-0.5, dur=1.0, start=0.0, quote=""))
    violations = validate_timeline(timeline, _manifest(), _transcript())
    assert any("negative offset" in v for v in violations)
```

- [ ] **Step 3: Run both test files to verify they fail**

Run: `uv run pytest tests/test_timeline.py tests/test_validate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'videoai.logic.timeline'`

- [ ] **Step 4: Add plan and timeline models to `videoai/core/models.py`**

```python
class PlanSection(BaseModel):
    name: str
    goal: str = ""
    phrase_ids: list[str] = Field(default_factory=list)


class StoryPlan(BaseModel):
    sections: list[PlanSection] = Field(default_factory=list)
    title: str = ""
    description: str = ""
    tags: list[str] = Field(default_factory=list)


class TimelineClip(BaseModel):
    src: str
    offset: float
    dur: float
    start: float
    quote: str = ""
    reason: str = ""
    beat: str = ""


class Timeline(BaseModel):
    fps: float
    width: int
    height: int
    clips: list[TimelineClip] = Field(default_factory=list)

    @property
    def duration(self) -> float:
        return sum(clip.dur for clip in self.clips)
```

- [ ] **Step 5: Implement `videoai/logic/timeline.py`**

```python
"""Turn an editorial plan into an exact timeline.

The model chooses phrases; geometry is computed here so timestamps are always
arithmetic on real transcript data rather than numbers a model wrote down.
"""
from __future__ import annotations

from videoai.core.models import Analysis, Manifest, StoryPlan, Timeline, TimelineClip


def build_timeline(
    plan: StoryPlan,
    analysis: Analysis,
    manifest: Manifest,
    padding: float,
    fps: float,
) -> Timeline:
    first_clip = manifest.clips[0]
    timeline = Timeline(fps=fps, width=first_clip.width, height=first_clip.height)

    position = 0.0
    for section in plan.sections:
        for phrase_id in section.phrase_ids:
            segment = analysis.by_phrase(phrase_id)
            source = manifest.by_id(segment.clip_id)
            offset = max(0.0, segment.start - padding)
            end = min(source.duration, segment.end + padding)
            duration = max(0.0, end - offset)
            if duration <= 0:
                continue
            timeline.clips.append(
                TimelineClip(
                    src=segment.clip_id,
                    offset=offset,
                    dur=duration,
                    start=position,
                    quote=segment.text,
                    reason=segment.content or section.goal,
                    beat=section.name,
                )
            )
            position += duration
    return timeline
```

- [ ] **Step 6: Implement `videoai/logic/validate.py`**

```python
"""Hard rules every timeline must satisfy before anything is rendered.

These encode the cut-safety knowledge that is expensive to rediscover: never cut
inside a word, keep segments inside their source, keep the timeline contiguous,
and prove each segment really contains the line it claims to.
"""
from __future__ import annotations

import re

from videoai.core.models import Manifest, Timeline, Transcript

MIN_SEGMENT_SECONDS = 0.3
EPSILON = 0.01


def _normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", text.lower())


def validate_timeline(
    timeline: Timeline, manifest: Manifest, transcript: Transcript
) -> list[str]:
    violations: list[str] = []
    known_clips = {clip.clip_id: clip for clip in manifest.clips}

    expected_position = 0.0
    for index, clip in enumerate(timeline.clips):
        label = f"clip {index} ({clip.src} @ {clip.offset:.2f})"

        source = known_clips.get(clip.src)
        if source is None:
            violations.append(f"{label}: unknown source {clip.src}")
            expected_position += clip.dur
            continue

        if clip.offset < 0:
            violations.append(f"{label}: negative offset")
        if clip.dur < MIN_SEGMENT_SECONDS:
            violations.append(
                f"{label}: shorter than {MIN_SEGMENT_SECONDS}s ({clip.dur:.2f}s)"
            )
        if clip.offset + clip.dur > source.duration + EPSILON:
            violations.append(
                f"{label}: exceeds source duration ({source.duration:.2f}s)"
            )
        if abs(clip.start - expected_position) > EPSILON:
            violations.append(
                f"{label}: timeline is not contiguous, expected start "
                f"{expected_position:.2f} but got {clip.start:.2f}"
            )
        expected_position += clip.dur

        words = transcript.by_id(clip.src).words if clip.src in {c.clip_id for c in transcript.clips} else []
        cut_in, cut_out = clip.offset, clip.offset + clip.dur
        for word in words:
            if word.start + EPSILON < cut_in < word.end - EPSILON:
                violations.append(f"{label}: cut starts inside word '{word.text}'")
            if word.start + EPSILON < cut_out < word.end - EPSILON:
                violations.append(f"{label}: cut ends inside word '{word.text}'")

        if clip.quote.strip():
            spoken = " ".join(
                word.text for word in words if word.start >= cut_in - EPSILON and word.end <= cut_out + EPSILON
            )
            if _normalise(clip.quote) .strip() not in _normalise(spoken):
                violations.append(f"{label}: quote not found in segment audio range")
    return violations
```

- [ ] **Step 7: Run the two test files to verify they pass**

Run: `uv run pytest tests/test_timeline.py tests/test_validate.py -v`
Expected: 15 passed

- [ ] **Step 8: Write the failing test for the plan stage**

`tests/test_stage_plan.py`:

```python
import json
from pathlib import Path

import pytest

from videoai.config import Config
from videoai.core.models import (
    Analysis,
    ClipInfo,
    ClipTranscript,
    Manifest,
    SegmentAnalysis,
    StoryPlan,
    Timeline,
    Transcript,
    Word,
)
from videoai.core.registry import StageContext
from videoai.core.store import ArtifactStore
from videoai.stages.s05_plan import plan


def _context(tmp_path: Path, monkeypatch, payload: dict) -> StageContext:
    for name in ("input", "work", "output"):
        (tmp_path / name).mkdir(exist_ok=True)
    payload_path = tmp_path / "llm.json"
    payload_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setenv("VIDEOAI_MOCK_LLM", str(payload_path))
    ctx = StageContext(
        project_dir=tmp_path,
        input_dir=tmp_path / "input",
        work_dir=tmp_path / "work",
        output_dir=tmp_path / "output",
        config=Config(providers={"asr": "mock", "llm": "mock"}),
        store=ArtifactStore(tmp_path / "work"),
    )
    ctx.store.write("01-manifest", Manifest(clips=[
        ClipInfo(clip_id="clip-01", path="/tmp/a.mp4", duration=30.0, width=1920,
                 height=1080, fps=30.0, has_audio=True)
    ]), fingerprint="fp")
    ctx.store.write("03-transcript", Transcript(provider="mock", clips=[ClipTranscript(
        clip_id="clip-01",
        words=[Word(text="hello", start=2.0, end=2.5), Word(text="world", start=2.6, end=3.0)],
    )]), fingerprint="fp")
    ctx.store.write("04-analysis", Analysis(provider="mock", segments=[
        SegmentAnalysis(phrase_id="clip-01#001", clip_id="clip-01", start=2.0, end=3.0,
                        text="hello world", content="greeting", delivery_score=9),
    ]), fingerprint="fp")
    return ctx


def test_plan_stage_produces_valid_timeline(tmp_path: Path, monkeypatch):
    payload = {
        "title": "Mega Wrex Review",
        "description": "A review",
        "tags": ["toys"],
        "sections": [{"name": "Hook", "goal": "start strong", "phrase_ids": ["clip-01#001"]}],
    }
    ctx = _context(tmp_path, monkeypatch, payload)

    timeline = plan(ctx)

    assert isinstance(timeline, Timeline)
    assert len(timeline.clips) == 1
    assert timeline.clips[0].beat == "Hook"
    saved = ctx.store.read("05a-storyplan", StoryPlan)
    assert saved.title == "Mega Wrex Review"


def test_plan_stage_rejects_failed_validation(tmp_path: Path, monkeypatch):
    payload = {
        "title": "T", "description": "D", "tags": [],
        "sections": [{"name": "Hook", "goal": "x", "phrase_ids": ["clip-01#001", "clip-01#001"]}],
    }
    ctx = _context(tmp_path, monkeypatch, payload)
    monkeypatch.setattr(
        "videoai.stages.s05_plan.validate_timeline", lambda *args, **kwargs: ["boom"]
    )
    with pytest.raises(ValueError, match="boom"):
        plan(ctx)


def test_plan_stage_rejects_unknown_phrase_id(tmp_path: Path, monkeypatch):
    payload = {
        "title": "T", "description": "D", "tags": [],
        "sections": [{"name": "Hook", "goal": "x", "phrase_ids": ["clip-77#001"]}],
    }
    ctx = _context(tmp_path, monkeypatch, payload)
    with pytest.raises(KeyError, match="clip-77#001"):
        plan(ctx)
```

- [ ] **Step 9: Implement `videoai/stages/s05_plan.py`**

```python
"""s05 plan: the model picks and orders phrases; geometry and validation are ours."""
from __future__ import annotations

from videoai.core.models import (
    Analysis,
    Manifest,
    PlanSection,
    StoryPlan,
    Timeline,
    Transcript,
)
from videoai.core.registry import StageContext, stage
from videoai.logic.timeline import build_timeline
from videoai.logic.validate import validate_timeline
from videoai.providers.base import resolve_llm

INSTRUCTIONS = """You are assembling a short YouTube video from a child's toy review.

Below is every usable phrase with its scores. Build the edit by selecting and
ordering phrase ids. Reply with one JSON document:

{"title": "...", "description": "...", "tags": ["..."],
 "sections": [{"name": "Hook", "goal": "why this section exists",
               "phrase_ids": ["clip-01#004"]}]}

Rules:
- Use only phrase ids from the list. Never invent ids and never repeat one.
- Drop phrases marked as failed takes; when several phrases share a take group,
  keep exactly one — the best delivery.
- Open with the single strongest moment, then tell the review in a sensible order.
- Aim for the target duration in the brief if one is given.
- title is punchy and under 70 characters; description is two or three sentences.
"""


def _segments_view(analysis: Analysis) -> str:
    lines = []
    for segment in analysis.segments:
        flags = []
        if segment.is_failed_take:
            flags.append("FAILED")
        if segment.take_group:
            flags.append(segment.take_group)
        if segment.shorts_candidate:
            flags.append("shorts")
        suffix = f" [{' '.join(flags)}]" if flags else ""
        lines.append(
            f"{segment.phrase_id} ({segment.end - segment.start:.1f}s, "
            f"delivery={segment.delivery_score}, visual={segment.visual_score}, "
            f"{segment.emotion}){suffix}: {segment.text}"
        )
    return "\n".join(lines)


@stage(
    id="plan",
    produces="05-timeline",
    requires=("01-manifest", "03-transcript", "04-analysis"),
    provider_key="llm",
    model=Timeline,
)
def plan(ctx: StageContext) -> Timeline:
    manifest = ctx.store.read("01-manifest", Manifest)
    transcript = ctx.store.read("03-transcript", Transcript)
    analysis = ctx.store.read("04-analysis", Analysis)

    brief_path = ctx.input_dir / "project.yaml"
    brief = brief_path.read_text(encoding="utf-8") if brief_path.exists() else ""

    provider = resolve_llm(ctx.config.providers["llm"])
    prompt = "\n\n".join([
        INSTRUCTIONS,
        f"Creator brief:\n{brief}" if brief.strip() else "",
        f"Phrases:\n{_segments_view(analysis)}",
    ]).strip()
    response = provider.complete_json(prompt, [], ctx.config.analyze.llm_timeout_seconds)

    story = StoryPlan(
        title=response.get("title", ""),
        description=response.get("description", ""),
        tags=list(response.get("tags", [])),
        sections=[
            PlanSection(
                name=section.get("name", "Section"),
                goal=section.get("goal", ""),
                phrase_ids=list(section.get("phrase_ids", [])),
            )
            for section in response.get("sections", [])
        ],
    )
    ctx.store.write("05a-storyplan", story, fingerprint="derived")

    timeline = build_timeline(
        story,
        analysis,
        manifest,
        padding=ctx.config.transcribe.cut_padding_seconds,
        fps=manifest.clips[0].fps,
    )
    violations = validate_timeline(timeline, manifest, transcript)
    if violations:
        raise ValueError("timeline validation failed:\n" + "\n".join(violations))
    return timeline
```

- [ ] **Step 10: Run the plan stage tests to verify they pass**

Run: `uv run pytest tests/test_stage_plan.py -v`
Expected: 3 passed

- [ ] **Step 11: Commit**

```bash
git add videoai/core/models.py videoai/logic/timeline.py videoai/logic/validate.py videoai/stages/s05_plan.py tests/test_timeline.py tests/test_validate.py tests/test_stage_plan.py
git commit -m "feat: story plan stage with timeline builder and hard-rule validator"
```

---

### Task 11: Draft render and end-to-end CLI

**Files:**
- Create: `videoai/stages/s06_render_draft.py`
- Modify: `videoai/cli.py`, `videoai/stages/__init__.py`
- Test: `tests/test_stage_render.py`, `tests/test_cli_end_to_end.py`

**Interfaces:**
- Consumes: `Timeline`, `Manifest`; `run_ffmpeg`, `probe`.
- Produces:
  - Model `DraftResult` (`path: str`, `duration: float`, `segment_count: int`).
  - Stage id `render_draft`, artifact `06-draft`, requires `01-manifest`, `05-timeline`.
  - CLI commands `videoai run <project> [--stage ID] [--force] [--config PATH]` and `videoai stages`.
  - `videoai/stages/__init__.py` imports every stage module so the registry is populated by importing the package.

- [ ] **Step 1: Write the failing test**

`tests/test_stage_render.py`:

```python
from pathlib import Path

from videoai.config import Config
from videoai.core.ffmpeg import probe
from videoai.core.models import ClipInfo, DraftResult, Manifest, Timeline, TimelineClip
from videoai.core.registry import StageContext
from videoai.core.store import ArtifactStore
from videoai.stages.s06_render_draft import render_draft


def _context(tmp_path: Path) -> StageContext:
    for name in ("input", "work", "output"):
        (tmp_path / name).mkdir(exist_ok=True)
    return StageContext(
        project_dir=tmp_path,
        input_dir=tmp_path / "input",
        work_dir=tmp_path / "work",
        output_dir=tmp_path / "output",
        config=Config(),
        store=ArtifactStore(tmp_path / "work"),
    )


def test_draft_is_rendered_from_two_segments(tmp_path: Path, make_clip):
    ctx = _context(tmp_path)
    source = make_clip("a.mp4", seconds=6.0)
    ctx.store.write("01-manifest", Manifest(clips=[
        ClipInfo(clip_id="clip-01", path=str(source), duration=6.0, width=320,
                 height=240, fps=30.0, has_audio=True, proxy_path=str(source)),
    ]), fingerprint="fp")
    ctx.store.write("05-timeline", Timeline(fps=30.0, width=320, height=240, clips=[
        TimelineClip(src="clip-01", offset=0.5, dur=1.5, start=0.0),
        TimelineClip(src="clip-01", offset=3.0, dur=2.0, start=1.5),
    ]), fingerprint="fp")

    result = render_draft(ctx)

    assert isinstance(result, DraftResult)
    assert result.segment_count == 2
    output = Path(result.path)
    assert output.exists()
    assert 3.0 < probe(output).duration < 4.2


def test_draft_output_lands_in_output_directory(tmp_path: Path, make_clip):
    ctx = _context(tmp_path)
    source = make_clip("a.mp4", seconds=4.0)
    ctx.store.write("01-manifest", Manifest(clips=[
        ClipInfo(clip_id="clip-01", path=str(source), duration=4.0, width=320,
                 height=240, fps=30.0, has_audio=True, proxy_path=str(source)),
    ]), fingerprint="fp")
    ctx.store.write("05-timeline", Timeline(fps=30.0, width=320, height=240, clips=[
        TimelineClip(src="clip-01", offset=0.5, dur=1.0, start=0.0),
    ]), fingerprint="fp")

    result = render_draft(ctx)

    assert Path(result.path).parent == ctx.output_dir
    assert Path(result.path).name == "draft.mp4"


def test_empty_timeline_raises(tmp_path: Path, make_clip):
    ctx = _context(tmp_path)
    source = make_clip("a.mp4", seconds=2.0)
    ctx.store.write("01-manifest", Manifest(clips=[
        ClipInfo(clip_id="clip-01", path=str(source), duration=2.0, width=320,
                 height=240, fps=30.0, has_audio=True, proxy_path=str(source)),
    ]), fingerprint="fp")
    ctx.store.write("05-timeline", Timeline(fps=30.0, width=320, height=240, clips=[]),
                    fingerprint="fp")

    try:
        render_draft(ctx)
    except RuntimeError as error:
        assert "empty timeline" in str(error)
    else:
        raise AssertionError("expected RuntimeError")
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_stage_render.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'videoai.stages.s06_render_draft'`

- [ ] **Step 3: Add the draft model to `videoai/core/models.py`**

```python
class DraftResult(BaseModel):
    path: str
    duration: float
    segment_count: int
```

- [ ] **Step 4: Implement `videoai/stages/s06_render_draft.py`**

```python
"""s06 draft render: cut the approved segments and concatenate them.

Segments are re-encoded from the proxy so cuts land exactly where the timeline
says, and each boundary gets a short audio fade to avoid clicks.
"""
from __future__ import annotations

from pathlib import Path

from videoai.core.ffmpeg import probe, run_ffmpeg
from videoai.core.models import DraftResult, Manifest, Timeline
from videoai.core.registry import StageContext, stage


def _render_segment(source: Path, offset: float, duration: float, fade: float, crf: int, dst: Path) -> None:
    fade_out_start = max(0.0, duration - fade)
    run_ffmpeg([
        "-ss", f"{offset:.3f}", "-i", str(source), "-t", f"{duration:.3f}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
        "-c:a", "aac", "-b:a", "160k",
        "-af", f"afade=t=in:st=0:d={fade},afade=t=out:st={fade_out_start:.3f}:d={fade}",
        "-avoid_negative_ts", "make_zero",
        str(dst),
    ])


@stage(
    id="render_draft",
    produces="06-draft",
    requires=("01-manifest", "05-timeline"),
    model=DraftResult,
)
def render_draft(ctx: StageContext) -> DraftResult:
    manifest = ctx.store.read("01-manifest", Manifest)
    timeline = ctx.store.read("05-timeline", Timeline)
    if not timeline.clips:
        raise RuntimeError("cannot render an empty timeline")

    segments_dir = ctx.work_dir / "segments"
    segments_dir.mkdir(parents=True, exist_ok=True)
    fade = ctx.config.render.audio_fade_seconds
    crf = ctx.config.render.draft_crf

    segment_paths: list[Path] = []
    for index, clip in enumerate(timeline.clips):
        source_info = manifest.by_id(clip.src)
        source = Path(source_info.proxy_path or source_info.path)
        target = segments_dir / f"seg-{index:03d}.mp4"
        _render_segment(source, clip.offset, clip.dur, fade, crf, target)
        segment_paths.append(target)

    list_file = segments_dir / "concat.txt"
    list_file.write_text(
        "\n".join(f"file '{path.name}'" for path in segment_paths) + "\n",
        encoding="utf-8",
    )

    ctx.output_dir.mkdir(parents=True, exist_ok=True)
    output = ctx.output_dir / "draft.mp4"
    run_ffmpeg([
        "-f", "concat", "-safe", "0", "-i", str(list_file),
        "-c", "copy", str(output),
    ])

    return DraftResult(
        path=str(output),
        duration=probe(output).duration,
        segment_count=len(segment_paths),
    )
```

- [ ] **Step 5: Populate the stage package so importing it registers every stage**

`videoai/stages/__init__.py`:

```python
"""Importing this package registers every stage in the registry."""
from videoai.stages import (  # noqa: F401
    s01_ingest,
    s02_quality,
    s03_transcribe,
    s04_analyze,
    s05_plan,
    s06_render_draft,
)
```

- [ ] **Step 6: Write the failing end-to-end test**

`tests/test_cli_end_to_end.py`:

```python
import json
from pathlib import Path

from typer.testing import CliRunner

from videoai.cli import app

runner = CliRunner()


def test_run_produces_draft_from_a_folder_of_clips(tmp_path: Path, make_clip, monkeypatch):
    project = tmp_path / "project"
    (project / "input").mkdir(parents=True)
    make_clip("a.mp4", seconds=6.0).rename(project / "input" / "a.mp4")
    (project / "input" / "project.yaml").write_text("title: Test review\n", encoding="utf-8")

    config_path = tmp_path / "config.yaml"
    config_path.write_text("providers:\n  asr: mock\n  llm: mock\n", encoding="utf-8")

    words_payload = [
        {"text": "hello", "start": 0.5, "end": 0.9},
        {"text": "everyone", "start": 0.95, "end": 1.5},
        {"text": "look", "start": 3.0, "end": 3.3},
        {"text": "here", "start": 3.35, "end": 3.8},
    ]
    llm_payload = {
        "segments": [
            {"phrase_id": "clip-01#001", "content": "intro", "delivery_score": 9,
             "visual_score": 8, "emotion": "excited", "is_failed_take": False,
             "shorts_candidate": True},
            {"phrase_id": "clip-01#002", "content": "demo", "delivery_score": 7,
             "visual_score": 7, "emotion": "calm", "is_failed_take": False,
             "shorts_candidate": False},
        ],
        "title": "Test Review",
        "description": "A test review.",
        "tags": ["toys"],
        "sections": [
            {"name": "Hook", "goal": "open", "phrase_ids": ["clip-01#001"]},
            {"name": "Body", "goal": "show", "phrase_ids": ["clip-01#002"]},
        ],
    }
    llm_path = tmp_path / "llm.json"
    llm_path.write_text(json.dumps(llm_payload), encoding="utf-8")
    monkeypatch.setenv("VIDEOAI_MOCK_LLM", str(llm_path))

    # The mock ASR reads its sidecar next to the extracted audio, which ingest
    # creates on the first run; seed it by running ingest alone first.
    result = runner.invoke(app, ["run", str(project), "--config", str(config_path), "--stage", "ingest"])
    assert result.exit_code == 0, result.output
    sidecar = project / "work" / "media" / "clip-01.words.json"
    sidecar.write_text(json.dumps(words_payload), encoding="utf-8")

    result = runner.invoke(app, ["run", str(project), "--config", str(config_path)])

    assert result.exit_code == 0, result.output
    draft = project / "output" / "draft.mp4"
    assert draft.exists()
    assert (project / "work" / "05-timeline.json").exists()
    assert "render_draft" in result.output


def test_second_run_skips_cached_stages(tmp_path: Path, make_clip, monkeypatch):
    project = tmp_path / "project"
    (project / "input").mkdir(parents=True)
    make_clip("a.mp4", seconds=6.0).rename(project / "input" / "a.mp4")
    config_path = tmp_path / "config.yaml"
    config_path.write_text("providers:\n  asr: mock\n  llm: mock\n", encoding="utf-8")
    llm_path = tmp_path / "llm.json"
    llm_path.write_text(json.dumps({
        "segments": [{"phrase_id": "clip-01#001", "content": "intro", "delivery_score": 9,
                      "visual_score": 8, "emotion": "excited", "is_failed_take": False,
                      "shorts_candidate": False}],
        "title": "T", "description": "D", "tags": [],
        "sections": [{"name": "Hook", "goal": "open", "phrase_ids": ["clip-01#001"]}],
    }), encoding="utf-8")
    monkeypatch.setenv("VIDEOAI_MOCK_LLM", str(llm_path))

    runner.invoke(app, ["run", str(project), "--config", str(config_path), "--stage", "ingest"])
    (project / "work" / "media" / "clip-01.words.json").write_text(
        json.dumps([{"text": "hello", "start": 0.5, "end": 0.9},
                    {"text": "everyone", "start": 0.95, "end": 1.5}]),
        encoding="utf-8",
    )
    runner.invoke(app, ["run", str(project), "--config", str(config_path)])

    result = runner.invoke(app, ["run", str(project), "--config", str(config_path)])

    assert result.exit_code == 0, result.output
    assert "nothing to do" in result.output.lower()


def test_stages_command_lists_pipeline_order():
    result = runner.invoke(app, ["stages"])
    assert result.exit_code == 0
    for stage_id in ("ingest", "quality", "transcribe", "analyze", "plan", "render_draft"):
        assert stage_id in result.output
```

- [ ] **Step 7: Implement the full CLI in `videoai/cli.py`**

```python
"""VideoAI command line interface."""
from __future__ import annotations

from pathlib import Path

import typer

import videoai.stages  # noqa: F401  (imports register every stage)
from videoai.config import load_config
from videoai.core.ffmpeg import VIDEO_SUFFIXES
from videoai.core.registry import REGISTRY, StageContext
from videoai.core.runner import _ordered_stages, run_pipeline
from videoai.core.store import ArtifactStore, hash_parts

app = typer.Typer(add_completion=False, help="Automated video pipeline.")


def _source_fingerprint(input_dir: Path) -> str:
    parts: list[str] = []
    for path in sorted(input_dir.rglob("*")):
        if path.is_file() and (path.suffix.lower() in VIDEO_SUFFIXES or path.name in {"project.yaml", "notes.md"}):
            stat = path.stat()
            parts.append(f"{path.name}:{stat.st_size}:{int(stat.st_mtime)}")
    return hash_parts(*parts)


@app.command()
def run(
    project: Path = typer.Argument(..., help="Project directory containing input/"),
    config_path: Path = typer.Option(Path("config.yaml"), "--config", help="Config file"),
    stage_id: str | None = typer.Option(None, "--stage", help="Run a single stage by id"),
    force: bool = typer.Option(False, "--force", help="Ignore the cache and re-run"),
) -> None:
    """Run the pipeline over a project folder."""
    input_dir = project / "input"
    if not input_dir.is_dir():
        raise typer.BadParameter(f"no input directory: {input_dir}")

    work_dir = project / "work"
    output_dir = project / "output"
    work_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    ctx = StageContext(
        project_dir=project,
        input_dir=input_dir,
        work_dir=work_dir,
        output_dir=output_dir,
        config=load_config(config_path),
        store=ArtifactStore(work_dir),
    )

    executed = run_pipeline(
        ctx, only=stage_id, force=force, extra_fingerprint=_source_fingerprint(input_dir)
    )
    if executed:
        typer.echo("Executed: " + ", ".join(executed))
    else:
        typer.echo("Nothing to do — every stage is up to date.")


@app.command()
def stages() -> None:
    """List pipeline stages in execution order."""
    for spec in _ordered_stages():
        requires = ", ".join(spec.requires) or "-"
        typer.echo(f"{spec.id:<14} produces={spec.produces:<16} requires={requires}")


@app.command()
def config(path: Path = typer.Option(Path("config.yaml"), help="Config file path")) -> None:
    """Print the effective configuration."""
    typer.echo(load_config(path).model_dump_json(indent=2))


if __name__ == "__main__":
    app()
```

- [ ] **Step 8: Run the render and end-to-end tests**

Run: `uv run pytest tests/test_stage_render.py tests/test_cli_end_to_end.py -v`
Expected: 6 passed

- [ ] **Step 9: Run the whole suite**

Run: `uv run pytest -v`
Expected: all tests pass (approximately 60)

- [ ] **Step 10: Commit**

```bash
git add videoai/core/models.py videoai/stages/s06_render_draft.py videoai/stages/__init__.py videoai/cli.py tests/test_stage_render.py tests/test_cli_end_to_end.py
git commit -m "feat: draft render stage and end-to-end pipeline CLI"
```

---

### Task 12: Stack documentation and a real-footage smoke run

**Files:**
- Create: `docs/state/STACK.md`, `docs/state/UPGRADES.md`, `README.md`
- Test: manual smoke run against real clips

**Interfaces:**
- Consumes: everything above.
- Produces: no code interfaces; documentation the upgrade workflow depends on.

- [ ] **Step 1: Write `docs/state/STACK.md`**

```markdown
# Current stack

| Stage | Artifact | Provider now | Cost per video |
|---|---|---|---|
| ingest | 01-manifest | ffmpeg (local) | $0 |
| quality | 02-quality | OpenCV (local) | $0 |
| transcribe | 03-transcript | parakeet-mlx (local) | $0 |
| analyze | 04-analysis | Claude Code CLI (Max subscription) | $0 |
| plan | 05-timeline | Claude Code CLI (Max subscription) | $0 |
| render_draft | 06-draft | ffmpeg (local) | $0 |

Switching a provider: edit `config.yaml`, re-run `videoai run <project>`. Only the
affected stage and everything downstream of it re-runs; earlier artifacts are reused.

Mock providers (`asr: mock`, `llm: mock`) run the whole pipeline offline and are what
the test suite uses.
```

- [ ] **Step 2: Write `docs/state/UPGRADES.md`**

```markdown
# Upgrade map

Each row is a stage whose quality can be raised by paying for a better provider.
Prices are per video for a typical project (15–20 minutes of source footage).

| Stage | Now | Paid upgrade | Price | What improves | How to switch |
|---|---|---|---|---|---|
| analyze | Claude reads transcript + keyframes | Gemini API watches the video natively | $0.30–1.00 | Hears delivery, energy and laughter, so best-take choice stops relying on text alone | Add `GEMINI_API_KEY` to `.env`, implement `providers/llm_gemini.py`, set `providers.llm: gemini` |
| transcribe | parakeet-mlx locally | AssemblyAI | ~$0.04 | Fewer errors on child speech, better punctuation | Add `ASSEMBLYAI_API_KEY` to `.env`, implement `providers/asr_assemblyai.py`, set `providers.asr: assemblyai` |
| b-roll | not implemented yet (Plan 3) | Kling / Hailuo / Veo via fal.ai | $0.30–3.00 per clip | Photoreal shots of the actual toy, animated from a photo | Plan 3 |
| music | local library (Plan 3) | ElevenLabs Music | ~$0.60 | Custom-length score that fits the edit | Plan 3 |

Always compare before adopting: run the stage with both providers and diff the
artifacts (`work/04-analysis.json` versus a copy) before switching permanently.
```

- [ ] **Step 3: Write `README.md`**

```markdown
# VideoAI

Automated editing pipeline for a kid's toy-review channel: a folder of raw clips
becomes a cut draft video, with every intermediate decision stored as a readable
JSON artifact.

## Requirements

- macOS on Apple Silicon
- ffmpeg 8.x (`brew install ffmpeg`)
- Python 3.13 via uv (`brew install uv`)
- Claude Code CLI, authenticated

## Setup

```bash
uv venv --python 3.13
uv sync
cp .env.example .env   # then fill in the keys you have
```

## Use

```bash
mkdir -p projects/my-review/input
# copy clips into projects/my-review/input/, optionally add project.yaml

uv run videoai run projects/my-review
open projects/my-review/output/draft.mp4
```

Useful commands:

```bash
uv run videoai stages                              # pipeline order
uv run videoai run projects/my-review --stage plan --force   # re-run one stage
uv run videoai config                              # effective configuration
```

## Layout

- `videoai/stages/` — one file per stage; each reads and writes artifacts only
- `videoai/providers/` — swappable implementations (local, subscription, paid)
- `videoai/logic/` — pure functions: phrases, take detection, timeline, validation
- `projects/<name>/work/` — artifacts and cache; safe to delete, everything rebuilds
- `docs/state/` — what the stack is now and how to upgrade it
```

- [ ] **Step 4: Create `.env.example`**

```bash
cat > .env.example <<'EOF'
# Stock footage (free tiers)
PEXELS_API_KEY=
PIXABAY_API_KEY=

# Optional paid upgrades — see docs/state/UPGRADES.md
GEMINI_API_KEY=
ASSEMBLYAI_API_KEY=
EOF
```

- [ ] **Step 5: Smoke-run the real pipeline against real footage**

Place three real clips in `projects/smoke/input/` (plus `project.yaml` with a title), then:

```bash
uv run videoai run projects/smoke
```

Expected: every stage executes, `projects/smoke/output/draft.mp4` plays, and
`projects/smoke/work/05-timeline.json` shows segments whose `quote` fields match
what is actually said. If the plan stage raises a validation error, read the listed
violations — they name the offending segment and rule.

- [ ] **Step 6: Commit**

```bash
git add docs/state README.md .env.example
git commit -m "docs: stack state, upgrade map and README"
```

---

## Self-Review

**Spec coverage (Plan 1 scope only):**

| Spec requirement | Task |
|---|---|
| Atomic stages, artifacts on disk, no shared state | 2, 3 |
| Provider plugins selected by config | 1, 6, 9 |
| Content-based cache, `--force`, single-stage runs | 3, 11 |
| s01 ingest (ffprobe, loudnorm, proxy) | 4 |
| s02 quality gate before any model | 5 |
| s03 verbatim word-level transcription, speech spans | 6 |
| Packed transcript (~1/10 tokens) | 7 |
| Two-stage take detection: deterministic recall, model judgement | 8, 9 |
| s04 analysis with keyframes, take groups, quality in context | 9 |
| StoryPlan separate from Timeline | 10 |
| Timeline hard rules (word boundaries, contiguity, bounds, quote re-anchoring) | 10 |
| Draft render with audio fades | 11 |
| `docs/state/STACK.md` and `UPGRADES.md` | 12 |

Deferred to later plans by design: review web page (Plan 2), Remotion captions/graphics/B-roll/music (Plan 3), shorts, thumbnail, packaging, AI reviewer loop (Plan 4). `auto-editor` integration for NLE export is deferred to Plan 3 — Plan 1 renders with ffmpeg directly so nothing depends on an external timeline format yet.

**Placeholder scan:** no TBD/TODO markers; every code step contains complete, runnable code; every test step contains real assertions.

**Type consistency:** `phrase_id` format `<clip-id>#<NNN>` is produced in Task 7 and consumed unchanged in Tasks 8–10. `Analysis.by_phrase`, `Manifest.by_id`, `Transcript.by_id`, `PhraseIndex.by_id`, `TakeGroups.group_of` are defined once and used with those exact names throughout. `TimelineClip` fields (`src`, `offset`, `dur`, `start`, `quote`, `reason`, `beat`) are identical in Tasks 10 and 11. `StageContext` field names match across Tasks 3–11. `complete_json(prompt, images, timeout)` has one signature used by both LLM providers and both calling stages.
