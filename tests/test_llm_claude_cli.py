import json
import subprocess

import pytest

from videoai.providers.llm_claude_cli import ClaudeCliLLM


class _FakeResult:
    def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_complete_json_nonzero_exit_raises_runtime_error(monkeypatch):
    def fake_run(*args, **kwargs):
        return _FakeResult(returncode=1, stderr="boom")

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="claude CLI failed"):
        ClaudeCliLLM().complete_json("prompt", [], 10)


def test_complete_json_is_error_envelope_raises_runtime_error(monkeypatch):
    envelope = json.dumps({"is_error": True, "result": "something broke"})

    def fake_run(*args, **kwargs):
        return _FakeResult(returncode=0, stdout=envelope)

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="claude CLI returned an error"):
        ClaudeCliLLM().complete_json("prompt", [], 10)


def test_complete_json_timeout_raises_runtime_error(monkeypatch):
    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="claude", timeout=10)

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="timed out"):
        ClaudeCliLLM().complete_json("prompt", [], 10)


def test_complete_json_success_unwraps_result(monkeypatch):
    envelope = json.dumps({"is_error": False, "result": '{"segments": []}'})

    def fake_run(*args, **kwargs):
        return _FakeResult(returncode=0, stdout=envelope)

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert ClaudeCliLLM().complete_json("prompt", [], 10) == {"segments": []}


# --- Finding I2: analyze.llm_model was config nothing read ---


def test_resolve_llm_passes_the_configured_model_to_the_provider():
    from videoai.providers.base import resolve_llm

    assert resolve_llm("claude_cli", "opus").model == "opus"
    assert resolve_llm("claude_cli").model == "sonnet"


def test_configured_model_reaches_the_claude_cli_invocation(monkeypatch):
    seen: dict[str, list[str]] = {}

    def fake_run(args, **kwargs):
        seen["args"] = args
        return _FakeResult(returncode=0, stdout=json.dumps({"is_error": False, "result": "{}"}))

    monkeypatch.setattr(subprocess, "run", fake_run)
    from videoai.providers.base import resolve_llm

    resolve_llm("claude_cli", "opus").complete_json("prompt", [], 10)

    assert "opus" in seen["args"]


def test_analyze_stage_resolves_the_llm_with_the_configured_model(tmp_path, monkeypatch):
    """The stage, not just the resolver, has to hand the setting over."""
    from videoai.config import AnalyzeSettings, Config
    from videoai.core.models import (
        Analysis,
        ClipInfo,
        Manifest,
        QualityReport,
        SyncMap,
        Transcript,
    )
    from videoai.core.registry import StageContext
    from videoai.core.store import ArtifactStore
    from videoai.providers.llm_mock import MockLLM
    from videoai.stages import s04_analyze, s05_plan

    seen: list[tuple[str, str | None]] = []

    def fake_resolve(name: str, model: str | None = None, cache_dir=None):
        seen.append((name, model))
        return MockLLM()

    monkeypatch.setattr(s04_analyze, "resolve_llm", fake_resolve)
    monkeypatch.setattr(s05_plan, "resolve_llm", fake_resolve)
    monkeypatch.setenv("VIDEOAI_MOCK_LLM", str(tmp_path / "llm.json"))
    (tmp_path / "llm.json").write_text(json.dumps({"segments": [], "sections": []}), encoding="utf-8")

    for name in ("work", "output"):
        (tmp_path / name).mkdir(exist_ok=True)
    ctx = StageContext(
        project_dir=tmp_path,
        input_dir=tmp_path,
        work_dir=tmp_path / "work",
        output_dir=tmp_path / "output",
        config=Config(
            providers={"asr": "mock", "llm": "mock"},
            analyze=AnalyzeSettings(llm_model="haiku"),
        ),
        store=ArtifactStore(tmp_path / "work"),
    )
    manifest = Manifest(clips=[ClipInfo(
        clip_id="clip-01", path=str(tmp_path / "a.mp4"), duration=10.0, width=320, height=240,
        fps=30.0, has_audio=True,
    )])
    ctx.store.write("01-manifest", manifest, fingerprint="fp")
    ctx.store.write("01b-sync", SyncMap(), fingerprint="fp")
    ctx.store.write("02-quality", QualityReport(), fingerprint="fp")
    ctx.store.write("03-transcript", Transcript(provider="mock"), fingerprint="fp")
    ctx.store.write("04-analysis", Analysis(provider="mock"), fingerprint="fp")

    s04_analyze.analyze(ctx)
    s05_plan.plan(ctx)

    assert seen == [("mock", "haiku"), ("mock", "haiku")]
