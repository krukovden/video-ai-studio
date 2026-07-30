"""LLM via the Claude Code CLI in headless mode.

Runs under the user's Claude subscription rather than an API key. The CLI is
asked for JSON and its envelope is unwrapped; the model's own text lands in
`result`.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from videoai.providers.json_reply import cli_diagnostic, extract_json

SYSTEM_PROMPT = (
    "You are a video editing analyst. You reply with a single JSON document and "
    "nothing else: no prose, no markdown fences."
)


class ClaudeCliLLM:
    name = "claude_cli"

    def __init__(self, model: str = "sonnet") -> None:
        self.model = model

    def complete_json(self, prompt: str, images: list[Path], timeout: int) -> dict:
        if images:
            listing = "\n".join(f"- {path}" for path in images)
            prompt = f"{prompt}\n\nReference frames (read them if useful):\n{listing}"
        try:
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
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"claude CLI timed out after {timeout} seconds") from exc
        if result.returncode != 0:
            raise RuntimeError(
                cli_diagnostic("claude", result.returncode, result.stderr, result.stdout)
            )
        envelope = json.loads(result.stdout)
        if envelope.get("is_error"):
            raise RuntimeError(f"claude CLI returned an error: {envelope.get('result')}")
        return extract_json(envelope["result"])
