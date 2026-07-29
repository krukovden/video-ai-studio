import subprocess
from pathlib import Path

import pytest

from videoai.providers.llm_codex_cli import CodexCliLLM


class _FakeResult:
    def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_codex_cli_reads_the_last_message_and_attaches_images(monkeypatch, tmp_path: Path):
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"frame")
    seen: list[str] = []

    def fake_run(args, **kwargs):
        seen.extend(args)
        output = Path(args[args.index("--output-last-message") + 1])
        output.write_text('{"segments": []}', encoding="utf-8")
        return _FakeResult()

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert CodexCliLLM().complete_json("score this", [image], 10) == {"segments": []}
    assert "--ephemeral" in seen
    assert "--sandbox" in seen
    assert str(image) in seen


def test_codex_cli_failure_is_clear(monkeypatch):
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: _FakeResult(returncode=1, stderr="not authenticated"),
    )
    with pytest.raises(RuntimeError, match="codex CLI failed.*not authenticated"):
        CodexCliLLM().complete_json("prompt", [], 10)


def test_codex_cli_timeout_is_clear(monkeypatch):
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="codex", timeout=10)

    monkeypatch.setattr(subprocess, "run", timeout)
    with pytest.raises(RuntimeError, match="timed out"):
        CodexCliLLM().complete_json("prompt", [], 10)
