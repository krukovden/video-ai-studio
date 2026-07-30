"""Reading one JSON document out of an LLM's reply.

Shared by every LLM provider: the models are interchangeable editorial voices and
they all wrap, fence or narrate their JSON in the same handful of ways, so the
unwrapping must not differ per provider — a reply that one provider parses and
another rejects would make the pipeline's behaviour depend on the subscription.
"""
from __future__ import annotations

import json
import re


def extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise ValueError(f"no JSON object found in model reply: {text[:300]}")
        # The greedy match can span two sibling JSON objects (e.g. "{...} and {...}"),
        # which is not valid JSON on its own; surface that as the same diagnostic
        # ValueError rather than letting a raw JSONDecodeError escape.
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise ValueError(f"could not parse JSON out of model reply: {text[:300]}") from exc


def cli_diagnostic(name: str, returncode: int, stderr: str, stdout: str) -> str:
    """What to tell the creator when a provider CLI exits non-zero.

    A CLI that fails silently is the common case in a headless or sandboxed
    session, and "claude CLI failed: " with nothing after it has sent people
    looking for a bug in this repository more than once.
    """
    diagnostic = (stderr or stdout).strip()[:500]
    if not diagnostic:
        diagnostic = (
            f"exit code {returncode} with no diagnostic output; the CLI may be "
            "blocked by a headless or sandboxed session"
        )
    return f"{name} CLI failed: {diagnostic}"
