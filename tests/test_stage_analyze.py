import json
from pathlib import Path

import pytest

from videoai.config import AnalyzeSettings, Config
from videoai.core.models import (
    Analysis,
    ClipInfo,
    ClipQuality,
    ClipTranscript,
    Manifest,
    Phrase,
    PhraseIndex,
    QualityReport,
    Transcript,
    Word,
)
from videoai.core.registry import StageContext
from videoai.core.store import ArtifactStore
from videoai.stages.s04_analyze import _keyframes, analyze, build_analysis_prompt


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


def _seed_many_phrases(ctx: StageContext, count: int) -> None:
    ctx.store.write(
        "01-manifest",
        Manifest(clips=[{
            "clip_id": "clip-01", "path": "/tmp/a.mp4", "duration": 60.0,
            "width": 320, "height": 240, "fps": 30.0, "has_audio": True,
        }]),
        fingerprint="fp",
    )
    ctx.store.write(
        "02-quality",
        QualityReport(clips=[ClipQuality(clip_id="clip-01", blur=0.1, motion=0.2, black_ratio=0.0, usable=True)]),
        fingerprint="fp",
    )
    # Gaps of ~1.7s comfortably exceed the default 0.5s phrase-gap threshold,
    # so each word becomes its own phrase.
    words = [Word(text=f"word{i}", start=float(i) * 2.0, end=float(i) * 2.0 + 0.3) for i in range(count)]
    ctx.store.write(
        "03-transcript",
        Transcript(provider="mock", clips=[ClipTranscript(clip_id="clip-01", words=words)]),
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


# --- Finding 1: an omitted phrase must be distinguishable from a genuine 5/5 ---


def test_omitted_phrase_is_marked_not_scored(tmp_path: Path, monkeypatch):
    payload = {"segments": [
        {"phrase_id": "clip-01#001", "content": "intro", "delivery_score": 8,
         "visual_score": 7, "emotion": "excited", "is_failed_take": False,
         "shorts_candidate": True},
    ]}
    ctx = _context(tmp_path, monkeypatch, payload)
    _seed_artifacts(ctx)

    result = analyze(ctx)

    answered = result.by_phrase("clip-01#001")
    assert answered.scored is True
    omitted = result.by_phrase("clip-01#002")
    assert omitted.scored is False
    assert omitted.delivery_score == 5
    assert omitted.visual_score == 5


# --- Finding 2: an unusable reply must fail loudly, not silently score nothing ---


def test_missing_segments_key_raises(tmp_path: Path, monkeypatch):
    ctx = _context(tmp_path, monkeypatch, {})
    _seed_artifacts(ctx)

    with pytest.raises(RuntimeError, match="segments"):
        analyze(ctx)


def test_segments_not_a_list_raises(tmp_path: Path, monkeypatch):
    ctx = _context(tmp_path, monkeypatch, {"segments": "oops"})
    _seed_artifacts(ctx)

    with pytest.raises(RuntimeError, match="list"):
        analyze(ctx)


def test_empty_segments_with_phrases_present_raises(tmp_path: Path, monkeypatch):
    ctx = _context(tmp_path, monkeypatch, {"segments": []})
    _seed_artifacts(ctx)

    with pytest.raises(RuntimeError, match="empty"):
        analyze(ctx)


def test_reply_covering_only_a_quarter_of_phrases_raises(tmp_path: Path, monkeypatch):
    payload = {"segments": [
        {"phrase_id": "clip-01#001", "content": "intro", "delivery_score": 8,
         "visual_score": 7, "emotion": "excited", "is_failed_take": False,
         "shorts_candidate": True},
    ]}
    ctx = _context(tmp_path, monkeypatch, payload)
    _seed_many_phrases(ctx, 4)

    with pytest.raises(RuntimeError, match=r"1 of 4"):
        analyze(ctx)


# --- Minor: malformed score fields must never crash and must clamp/flag ---


def test_null_delivery_score_falls_back_to_neutral_and_marks_unscored(tmp_path: Path, monkeypatch):
    payload = {"segments": [
        {"phrase_id": "clip-01#001", "content": "intro", "delivery_score": None,
         "visual_score": 7, "emotion": "excited", "is_failed_take": False,
         "shorts_candidate": True},
        {"phrase_id": "clip-01#002", "content": "reaction", "delivery_score": 5,
         "visual_score": 6, "emotion": "calm", "is_failed_take": False,
         "shorts_candidate": False},
    ]}
    ctx = _context(tmp_path, monkeypatch, payload)
    _seed_artifacts(ctx)

    result = analyze(ctx)

    first = result.by_phrase("clip-01#001")
    assert first.delivery_score == 5
    assert first.scored is False
    second = result.by_phrase("clip-01#002")
    assert second.scored is True


def test_malformed_and_out_of_range_scores_are_handled_defensively(tmp_path: Path, monkeypatch):
    payload = {"segments": [
        {"phrase_id": "clip-01#001", "content": "intro", "delivery_score": "high",
         "visual_score": 15, "emotion": "excited", "is_failed_take": False,
         "shorts_candidate": True},
        {"phrase_id": "clip-01#002", "content": "reaction", "delivery_score": 0,
         "visual_score": 6, "emotion": "calm", "is_failed_take": False,
         "shorts_candidate": False},
    ]}
    ctx = _context(tmp_path, monkeypatch, payload)
    _seed_artifacts(ctx)

    result = analyze(ctx)

    first = result.by_phrase("clip-01#001")
    assert first.delivery_score == 5  # unparseable string -> neutral fallback
    assert first.visual_score == 10  # clamped down from 15
    assert first.scored is False  # delivery_score was unparseable

    second = result.by_phrase("clip-01#002")
    assert second.delivery_score == 1  # clamped up from 0
    assert second.scored is True  # both scores parsed fine, just out of range


# --- Minor: is_failed_take must not be inverted by a JSON string "false" ---


def test_string_booleans_are_parsed_not_truthy_cast(tmp_path: Path, monkeypatch):
    payload = {"segments": [
        {"phrase_id": "clip-01#001", "content": "intro", "delivery_score": 8,
         "visual_score": 7, "emotion": "excited", "is_failed_take": "false",
         "shorts_candidate": "TRUE"},
        {"phrase_id": "clip-01#002", "content": "reaction", "delivery_score": 5,
         "visual_score": 6, "emotion": "calm", "is_failed_take": "not-a-bool",
         "shorts_candidate": False},
    ]}
    ctx = _context(tmp_path, monkeypatch, payload)
    _seed_artifacts(ctx)

    result = analyze(ctx)

    first = result.by_phrase("clip-01#001")
    assert first.is_failed_take is False
    assert first.shorts_candidate is True

    second = result.by_phrase("clip-01#002")
    assert second.is_failed_take is False  # unparseable string falls back to default


# --- Minor: keyframes_per_phrase honoured, max_keyframes cap respected ---


def test_keyframes_per_phrase_zero_extracts_nothing(tmp_path: Path, make_clip):
    clip_path = make_clip("a.mp4", seconds=3.0)
    manifest = Manifest(clips=[ClipInfo(
        clip_id="clip-01", path=str(clip_path), duration=3.0,
        width=320, height=240, fps=30.0, has_audio=True,
    )])
    index = PhraseIndex(phrases=[
        Phrase(phrase_id="clip-01#001", clip_id="clip-01", start=0.0, end=1.0,
               text="hi", word_start=0, word_end=1),
    ])
    (tmp_path / "work").mkdir(exist_ok=True)
    ctx = StageContext(
        project_dir=tmp_path,
        input_dir=tmp_path,
        work_dir=tmp_path / "work",
        output_dir=tmp_path,
        config=Config(analyze=AnalyzeSettings(keyframes_per_phrase=0)),
        store=ArtifactStore(tmp_path / "work"),
    )

    frames, truncated = _keyframes(ctx, manifest, index, ctx.config.analyze)

    assert frames == []
    assert truncated is False
    assert not (ctx.work_dir / "keyframes").exists()


def test_keyframes_cap_is_respected(tmp_path: Path, make_clip):
    clip_path = make_clip("a.mp4", seconds=5.0)
    manifest = Manifest(clips=[ClipInfo(
        clip_id="clip-01", path=str(clip_path), duration=5.0,
        width=320, height=240, fps=30.0, has_audio=True,
    )])
    phrases = [
        Phrase(
            phrase_id=f"clip-01#{i:03d}", clip_id="clip-01",
            start=float(i) * 0.5, end=float(i) * 0.5 + 0.4,
            text="hi", word_start=i, word_end=i + 1,
        )
        for i in range(1, 6)
    ]
    index = PhraseIndex(phrases=phrases)
    (tmp_path / "work").mkdir(exist_ok=True)
    ctx = StageContext(
        project_dir=tmp_path,
        input_dir=tmp_path,
        work_dir=tmp_path / "work",
        output_dir=tmp_path,
        config=Config(analyze=AnalyzeSettings(max_keyframes=2)),
        store=ArtifactStore(tmp_path / "work"),
    )

    frames, truncated = _keyframes(ctx, manifest, index, ctx.config.analyze)

    assert len(frames) == 2
    assert truncated is True
