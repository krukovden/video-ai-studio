"""Mock LLM: returns a canned JSON document named by VIDEOAI_MOCK_LLM.

The file is either one document, returned to every call (the original and still
the common case), or a mapping of call kinds to documents. A kind may map to a
LIST of documents, consumed one per call with the last one repeating: a single
`videoai run --auto-fix` calls the planner more than once, and a fixture that
cannot answer the second round differently cannot exercise the loop at all.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

# Opening words of each stage's prompt. Matching on the prompt keeps the
# production `LLMProvider` protocol free of test-only parameters.
KIND_MARKERS = (
    ("visual", "You are checking frames from segments already chosen"),
    ("inserts", "You are looking at silent visual insert clips"),
    ("plan", "You are assembling a short YouTube video"),
    ("analyze", "You are editing a short YouTube video"),
)

_consumed: dict[tuple[str, str], int] = {}


def _kind(prompt: str) -> str:
    for kind, marker in KIND_MARKERS:
        if marker in prompt:
            return kind
    return ""


class MockLLM:
    name = "mock"

    def complete_json(self, prompt: str, images: list[Path], timeout: int) -> dict:
        path = os.getenv("VIDEOAI_MOCK_LLM")
        if not path:
            raise RuntimeError("VIDEOAI_MOCK_LLM must point to a canned response file")
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        kind = _kind(prompt)
        if kind not in payload:
            return payload
        answer = payload[kind]
        if not isinstance(answer, list):
            return answer
        position = _consumed.get((path, kind), 0)
        _consumed[(path, kind)] = position + 1
        return answer[min(position, len(answer) - 1)]
