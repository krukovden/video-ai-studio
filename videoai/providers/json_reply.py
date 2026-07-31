"""Reading one JSON document out of an LLM's reply.

Shared by every LLM provider: the models are interchangeable editorial voices and
they all wrap, fence or narrate their JSON in the same handful of ways, so the
unwrapping must not differ per provider — a reply that one provider parses and
another rejects would make the pipeline's behaviour depend on the subscription.
"""
from __future__ import annotations

import json

FENCE = "```"


def _strip_outer_fence(text: str) -> str:
    """Remove one fence wrapping the whole reply, and nothing else.

    Stripping every fence in the document (a line-anchored regex over the whole
    text) also deletes the ones a model wrote *inside* the answer — a second
    worked example after the JSON, or a fence quoted in a value. What comes back
    from that is a document with holes punched in the middle of it, which then
    fails to parse for a reason nothing in the reply explains.
    """
    if not text.startswith(FENCE):
        return text
    # The opening fence carries an optional language tag ("```json") and runs to
    # the end of its line; a reply that is nothing but a fence has no body.
    body = text.split("\n", 1)[1] if "\n" in text else ""
    stripped = body.rstrip()
    if stripped.endswith(FENCE):
        body = stripped[: -len(FENCE)]
    return body.strip()


def _first_object(text: str) -> str | None:
    """The first balanced `{...}` in `text`, or None if there is not one.

    Brace-counting rather than a greedy regex: `\\{.*\\}` spans from the first
    brace of the answer to the last brace of whatever the model said afterwards,
    so one trailing "hope that helps {see above}" turns a perfectly good reply
    into a parse error. Strings and their escapes are respected, because a brace
    inside a value is not structure.
    """
    depth = 0
    start = -1
    in_string = False
    escaped = False
    for position, character in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character == "{":
            if depth == 0:
                start = position
            depth += 1
        elif character == "}" and depth:
            depth -= 1
            if depth == 0:
                return text[start : position + 1]
    return None


def extract_json(text: str) -> dict:
    text = _strip_outer_fence(text.strip())
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        candidate = _first_object(text)
        if candidate is None:
            raise ValueError(f"no JSON object found in model reply: {text[:300]}")
        try:
            return json.loads(candidate)
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
