"""Mock LLM: returns a canned JSON document named by VIDEOAI_MOCK_LLM."""
from __future__ import annotations

import json
import os
from pathlib import Path


class MockLLM:
    name = "mock"

    def complete_json(self, prompt: str, images: list[Path], timeout: int) -> dict:
        path = os.getenv("VIDEOAI_MOCK_LLM")
        if not path:
            raise RuntimeError("VIDEOAI_MOCK_LLM must point to a canned response file")
        return json.loads(Path(path).read_text(encoding="utf-8"))
