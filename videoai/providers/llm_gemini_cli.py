"""LLM through the Gemini CLI, on the user's own Google account.

The third interchangeable editorial voice, alongside Claude and Codex: same
protocol, same JSON reader, no metered API key. Useful for the same reason the
Codex provider is — comparing how two models read the same footage costs nothing
but a config line.

What it is NOT is the video upgrade. The reason to want Gemini in this pipeline
is that its models can watch a video natively, and this CLI has no way to hand a
model a video file: it is an agent with file tools, and `.mp4` is not something
those tools can turn into frames. So `reads_video` is False here and the analyze
stage keeps sending stills. Native video lives in `llm_gemini_api`, which needs
a metered key.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from videoai.providers.json_reply import cli_diagnostic, extract_json

SYSTEM_PROMPT = (
    "You are a video editing analyst. You reply with a single JSON document and "
    "nothing else: no prose, no markdown fences."
)

# Read-only by construction. The CLI is an agent that can edit files and run
# commands, and an editorial call has no business doing either; "plan" is its
# read-only approval mode.
APPROVAL_MODE = "plan"


class GeminiCliLLM:
    name = "gemini_cli"
    # Stills only — see the module docstring.
    reads_video = False

    def __init__(self, model: str | None = None) -> None:
        self.model = model

    def complete_json(
        self,
        prompt: str,
        images: list[Path],
        timeout: int,
        videos: list[Path] | None = None,
    ) -> dict:
        if videos:
            raise ValueError(
                "gemini_cli cannot read video: the CLI has no way to pass a video "
                "file to the model. Use the gemini_api provider for native video."
            )
        body = f"{SYSTEM_PROMPT}\n\n{prompt}"
        if images:
            listing = "\n".join(f"- {path}" for path in images)
            body = f"{body}\n\nReference frames (read them if useful):\n{listing}"

        args = ["gemini", "-p", body, "--output-format", "json",
                "--approval-mode", APPROVAL_MODE]
        if self.model:
            args += ["--model", self.model]

        try:
            result = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"gemini CLI timed out after {timeout} seconds") from exc
        if result.returncode != 0:
            raise RuntimeError(
                cli_diagnostic("gemini", result.returncode, result.stderr, result.stdout)
            )
        return _unwrap(result.stdout)


def _unwrap(stdout: str) -> dict:
    """The model's own document, out of whatever envelope the CLI put it in.

    `--output-format json` wraps the answer in an object whose `response` holds
    the model's text. Versions differ on the key, and a bare document is also
    possible, so the envelope is unwrapped when it is recognised and the whole
    reply is read as the answer when it is not — rather than tying this provider
    to one CLI release.
    """
    document = extract_json(stdout)
    for key in ("response", "result", "output", "text"):
        value = document.get(key)
        if isinstance(value, str):
            return extract_json(value)
        if isinstance(value, dict):
            return value
    if document.get("error"):
        raise RuntimeError(f"gemini CLI returned an error: {document['error']}")
    return document
