"""LLM through the authenticated Codex CLI and the user's OpenAI subscription."""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from videoai.providers.json_reply import cli_diagnostic, extract_json

# Codex has no `--system-prompt`, so the preamble is prefixed to the prompt itself.
# It is a module constant rather than a local string because it is part of what the
# model is asked, and the stages that call this provider mix it into their
# fingerprints: editing it changes the analysis and has to invalidate the cached one.
SYSTEM_PROMPT = (
    "Act only as a video editing analyst. Do not change files or run commands. "
    "Return one JSON document and nothing else.\n\n"
)


class CodexCliLLM:
    name = "codex_cli"
    # Stills only: no CLI here can hand a model a video file.
    reads_video = False
    system_preamble = SYSTEM_PROMPT

    def complete_json(
        self,
        prompt: str,
        images: list[Path],
        timeout: int,
        videos: list[Path] | None = None,
    ) -> dict:
        if videos:
            raise ValueError(
                "codex_cli cannot read video: it is given stills only. Use a "
                "video-capable provider to submit clips."
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
            args.append(SYSTEM_PROMPT + prompt)
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
                raise RuntimeError(
                    cli_diagnostic("codex", result.returncode, result.stderr, result.stdout)
                )
            if not output.is_file():
                raise RuntimeError("codex CLI completed without a final response")
            return extract_json(output.read_text(encoding="utf-8"))
