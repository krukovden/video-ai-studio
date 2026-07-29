import json
from pathlib import Path

import pytest

from videoai.config import Config
from videoai.core.models import (
    Analysis,
    ClipInfo,
    ClipTranscript,
    Manifest,
    SegmentAnalysis,
    StoryPlan,
    Timeline,
    Transcript,
    Word,
)
from videoai.core.registry import StageContext
from videoai.core.store import ArtifactStore
from videoai.stages.s05_plan import plan


def _context(tmp_path: Path, monkeypatch, payload: dict) -> StageContext:
    for name in ("input", "work", "output"):
        (tmp_path / name).mkdir(exist_ok=True)
    payload_path = tmp_path / "llm.json"
    payload_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setenv("VIDEOAI_MOCK_LLM", str(payload_path))
    ctx = StageContext(
        project_dir=tmp_path,
        input_dir=tmp_path / "input",
        work_dir=tmp_path / "work",
        output_dir=tmp_path / "output",
        config=Config(providers={"asr": "mock", "llm": "mock"}),
        store=ArtifactStore(tmp_path / "work"),
    )
    ctx.store.write("01-manifest", Manifest(clips=[
        ClipInfo(clip_id="clip-01", path="/tmp/a.mp4", duration=30.0, width=1920,
                 height=1080, fps=30.0, has_audio=True)
    ]), fingerprint="fp")
    ctx.store.write("03-transcript", Transcript(provider="mock", clips=[ClipTranscript(
        clip_id="clip-01",
        words=[Word(text="hello", start=2.0, end=2.5), Word(text="world", start=2.6, end=3.0)],
    )]), fingerprint="fp")
    ctx.store.write("04-analysis", Analysis(provider="mock", segments=[
        SegmentAnalysis(phrase_id="clip-01#001", clip_id="clip-01", start=2.0, end=3.0,
                        text="hello world", content="greeting", delivery_score=9),
    ]), fingerprint="fp")
    return ctx


def test_plan_stage_produces_valid_timeline(tmp_path: Path, monkeypatch):
    payload = {
        "title": "Mega Wrex Review",
        "description": "A review",
        "tags": ["toys"],
        "sections": [{"name": "Hook", "goal": "start strong", "phrase_ids": ["clip-01#001"]}],
    }
    ctx = _context(tmp_path, monkeypatch, payload)

    timeline = plan(ctx)

    assert isinstance(timeline, Timeline)
    assert len(timeline.clips) == 1
    assert timeline.clips[0].beat == "Hook"
    saved = ctx.store.read("05a-storyplan", StoryPlan)
    assert saved.title == "Mega Wrex Review"


def test_plan_stage_rejects_failed_validation(tmp_path: Path, monkeypatch):
    payload = {
        "title": "T", "description": "D", "tags": [],
        "sections": [{"name": "Hook", "goal": "x", "phrase_ids": ["clip-01#001", "clip-01#001"]}],
    }
    ctx = _context(tmp_path, monkeypatch, payload)
    monkeypatch.setattr(
        "videoai.stages.s05_plan.validate_timeline", lambda *args, **kwargs: ["boom"]
    )
    with pytest.raises(ValueError, match="boom"):
        plan(ctx)


def test_plan_stage_rejects_unknown_phrase_id(tmp_path: Path, monkeypatch):
    """A hallucinated phrase id is the likeliest LLM failure here; it must name the
    id and the section rather than escaping as a bare KeyError from the timeline."""
    payload = {
        "title": "T", "description": "D", "tags": [],
        "sections": [{"name": "Hook", "goal": "x", "phrase_ids": ["clip-77#001"]}],
    }
    ctx = _context(tmp_path, monkeypatch, payload)
    with pytest.raises(RuntimeError) as error:
        plan(ctx)

    assert "clip-77#001" in str(error.value)
    assert "Hook" in str(error.value)
    assert not isinstance(error.value, KeyError)
