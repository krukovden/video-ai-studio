import json
from pathlib import Path

import pytest

from videoai.config import Config
from videoai.core.models import (
    Analysis,
    ClipQuality,
    ClipTranscript,
    Manifest,
    QualityReport,
    Transcript,
    Word,
)
from videoai.core.registry import StageContext
from videoai.core.store import ArtifactStore
from videoai.stages.s04_analyze import analyze, build_analysis_prompt


def _context(tmp_path: Path, monkeypatch, llm_payload: dict) -> StageContext:
    for name in ("input", "work", "output"):
        (tmp_path / name).mkdir(exist_ok=True)
    payload_path = tmp_path / "llm.json"
    payload_path.write_text(json.dumps(llm_payload), encoding="utf-8")
    monkeypatch.setenv("VIDEOAI_MOCK_LLM", str(payload_path))
    return StageContext(
        project_dir=tmp_path,
        input_dir=tmp_path / "input",
        work_dir=tmp_path / "work",
        output_dir=tmp_path / "output",
        config=Config(providers={"asr": "mock", "llm": "mock"}),
        store=ArtifactStore(tmp_path / "work"),
    )


def _seed_artifacts(ctx: StageContext) -> None:
    ctx.store.write(
        "01-manifest",
        Manifest(clips=[{
            "clip_id": "clip-01", "path": "/tmp/a.mp4", "duration": 10.0,
            "width": 320, "height": 240, "fps": 30.0, "has_audio": True,
        }]),
        fingerprint="fp",
    )
    ctx.store.write(
        "02-quality",
        QualityReport(clips=[ClipQuality(clip_id="clip-01", blur=0.1, motion=0.2, black_ratio=0.0, usable=True)]),
        fingerprint="fp",
    )
    ctx.store.write(
        "03-transcript",
        Transcript(
            provider="mock",
            clips=[ClipTranscript(
                clip_id="clip-01",
                words=[
                    Word(text="Look", start=0.0, end=0.3),
                    Word(text="here", start=0.35, end=0.7),
                    Word(text="Wow", start=2.0, end=2.4),
                ],
            )],
        ),
        fingerprint="fp",
    )


def test_analysis_maps_llm_scores_onto_phrases(tmp_path: Path, monkeypatch):
    payload = {"segments": [
        {"phrase_id": "clip-01#001", "content": "intro", "delivery_score": 8,
         "visual_score": 7, "emotion": "excited", "is_failed_take": False,
         "shorts_candidate": True},
        {"phrase_id": "clip-01#002", "content": "reaction", "delivery_score": 5,
         "visual_score": 6, "emotion": "calm", "is_failed_take": False,
         "shorts_candidate": False},
    ]}
    ctx = _context(tmp_path, monkeypatch, payload)
    _seed_artifacts(ctx)

    result = analyze(ctx)

    assert isinstance(result, Analysis)
    assert result.provider == "mock"
    first = result.by_phrase("clip-01#001")
    assert first.delivery_score == 8
    assert first.shorts_candidate is True
    assert first.start == 0.0 and first.end == 0.7
    assert first.text == "Look here"


def test_unknown_phrase_id_from_llm_is_rejected(tmp_path: Path, monkeypatch):
    payload = {"segments": [
        {"phrase_id": "clip-99#001", "content": "ghost", "delivery_score": 5,
         "visual_score": 5, "emotion": "calm", "is_failed_take": False,
         "shorts_candidate": False},
    ]}
    ctx = _context(tmp_path, monkeypatch, payload)
    _seed_artifacts(ctx)

    with pytest.raises(ValueError, match="clip-99#001"):
        analyze(ctx)


def test_missing_phrase_in_llm_response_gets_neutral_defaults(tmp_path: Path, monkeypatch):
    payload = {"segments": [
        {"phrase_id": "clip-01#001", "content": "intro", "delivery_score": 8,
         "visual_score": 7, "emotion": "excited", "is_failed_take": False,
         "shorts_candidate": True},
    ]}
    ctx = _context(tmp_path, monkeypatch, payload)
    _seed_artifacts(ctx)

    result = analyze(ctx)

    second = result.by_phrase("clip-01#002")
    assert second.delivery_score == 5
    assert second.is_failed_take is False


def test_prompt_contains_packed_transcript_and_rules(tmp_path: Path, monkeypatch):
    from videoai.core.models import TakeGroups

    prompt = build_analysis_prompt(
        packed="[clip-01#001] 0.00-0.70 Look here",
        takes=TakeGroups(),
        quality=QualityReport(clips=[ClipQuality(clip_id="clip-01", blur=0.9, motion=0.1, black_ratio=0.0, usable=False)]),
        brief="Toy review",
    )
    assert "clip-01#001" in prompt
    assert "Toy review" in prompt
    assert "JSON" in prompt
    assert "clip-01" in prompt
