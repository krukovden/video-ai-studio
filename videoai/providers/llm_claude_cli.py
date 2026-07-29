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
