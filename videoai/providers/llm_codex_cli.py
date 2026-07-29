"""LLM through the authenticated Codex CLI and the user's OpenAI subscription."""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from videoai.providers.llm_claude_cli import _extract_json


class CodexCliLLM:
    name = "codex_cli"

    def complete_json(self, prompt: str, images: list[Path], timeout: int) -> dict:
        system = (
            "Act only as a video editing analyst. Do not change files or run commands. "
            "Return one JSON document and nothing else.\n\n"
        )
        with tempfile.TemporaryDirectory(prefix="videoai-codex-") as directory:
            output = Path(directory) / "response.txt"
            args = [
                "codex", "exec",
                "--ephemeral",
                "--sandbox", "read-only",
                "--skip-git-repo-check",
                "--color", "never",
                "--output-last-message", str(output),
            ]
            for image in images:
                args.extend(["--image", str(image)])
            args.append(system + prompt)
            try:
                result = subprocess.run(
                    args,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(f"codex CLI timed out after {timeout} seconds") from exc
            if result.returncode != 0:
                diagnostic = (result.stderr or result.stdout).strip()[:500]
                raise RuntimeError(f"codex CLI failed: {diagnostic}")
            if not output.is_file():
                raise RuntimeError("codex CLI completed without a final response")
            return _extract_json(output.read_text(encoding="utf-8"))
