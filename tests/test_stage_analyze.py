import json
from pathlib import Path

import pytest

from videoai.config import AnalyzeSettings, Config
from videoai.core.models import (
    Analysis,
    ClipInfo,
    ClipQuality,
    ClipTranscript,
    InsertClip,
    Manifest,
    Phrase,
    PhraseIndex,
    QualityReport,
    Transcript,
    Word,
)
from videoai.core.registry import StageContext
from videoai.core.store import ArtifactStore
from videoai.stages.s04_analyze import (
    INSTRUCTIONS,
    _describe_inserts,
    _insert_keyframes,
    _keyframes,
    analyze,
    build_analysis_prompt,
    build_insert_description_prompt,
)


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
            "clip_id": "clip-01", "path": str(ctx.project_dir / "a.mp4"), "duration": 10.0,
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
            "clip_id": "clip-01", "path": str(ctx.project_dir / "a.mp4"), "duration": 60.0,
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


# --- Finding C1: cached keyframes must be keyed by source identity, not by the
# phrase ordinals that shift whenever the transcript or the clip numbering moves ---


def test_cached_keyframes_are_not_reused_across_sources(tmp_path: Path, make_clip):
    """`clip-01#001` names a different moment of a different file once a clip that
    sorts earlier is added. Reusing the cached frame would show the model footage
    from the wrong source."""
    from videoai.core.ffmpeg import probe
    from videoai.core.store import source_key

    four_three = make_clip("a.mp4", seconds=3.0, size="320x240")
    wide = make_clip("b.mp4", seconds=3.0, size="640x360")
    (tmp_path / "work").mkdir(exist_ok=True)
    ctx = StageContext(
        project_dir=tmp_path,
        input_dir=tmp_path,
        work_dir=tmp_path / "work",
        output_dir=tmp_path,
        config=Config(),
        store=ArtifactStore(tmp_path / "work"),
    )
    index = PhraseIndex(phrases=[
        Phrase(phrase_id="clip-01#001", clip_id="clip-01", start=0.0, end=1.0,
               text="hi", word_start=0, word_end=1),
    ])

    def _manifest_for(source: Path, width: int, height: int) -> Manifest:
        return Manifest(clips=[ClipInfo(
            clip_id="clip-01", path=str(source), duration=3.0, width=width, height=height,
            fps=30.0, has_audio=True, source_key=source_key(source),
        )])

    before, _ = _keyframes(ctx, _manifest_for(four_three, 320, 240), index, ctx.config.analyze)
    assert probe(before[0]).width == 480  # 4:3 scaled to height 360

    after, _ = _keyframes(ctx, _manifest_for(wide, 640, 360), index, ctx.config.analyze)

    assert probe(after[0]).width == 640  # 16:9 scaled to height 360
    assert after[0] != before[0]


# --- Silent visual inserts: clips with (almost) no narration are recorded in the
# analysis artifact, because nothing phrase-based downstream could ever find them ---


def _seed_with_a_silent_clip(ctx: StageContext) -> None:
    ctx.store.write(
        "01-manifest",
        Manifest(clips=[
            # 3 words over 10s (0.3 words/s) is below the 0.5 default.
            ClipInfo(clip_id="clip-01", path=str(ctx.project_dir / "a.mp4"), duration=10.0,
                     width=320, height=240, fps=30.0, has_audio=True),
            # 3 words over 2s (1.5 words/s) is ordinary narration.
            ClipInfo(clip_id="clip-02", path=str(ctx.project_dir / "b.mp4"), duration=2.0,
                     width=320, height=240, fps=30.0, has_audio=True),
            # The close-up of the bubble popping: nobody says anything.
            ClipInfo(clip_id="clip-10", path=str(ctx.project_dir / "c.mp4"), duration=7.0,
                     width=320, height=240, fps=30.0, has_audio=True),
        ]),
        fingerprint="fp",
    )
    ctx.store.write(
        "02-quality",
        QualityReport(clips=[
            ClipQuality(clip_id=clip_id, blur=0.1, motion=0.2, black_ratio=0.0, usable=True)
            for clip_id in ("clip-01", "clip-02", "clip-10")
        ]),
        fingerprint="fp",
    )
    words = [
        Word(text="Look", start=0.0, end=0.3),
        Word(text="here", start=0.35, end=0.7),
        Word(text="Wow", start=2.0, end=2.4),
    ]
    ctx.store.write(
        "03-transcript",
        Transcript(provider="mock", clips=[
            ClipTranscript(clip_id="clip-01", words=words),
            ClipTranscript(clip_id="clip-02", words=words),
            ClipTranscript(clip_id="clip-10", words=[]),
        ]),
        fingerprint="fp",
    )


def _payload_for(*phrase_ids: str) -> dict:
    return {"segments": [
        {"phrase_id": phrase_id, "content": "x", "delivery_score": 7, "visual_score": 7,
         "emotion": "calm", "is_failed_take": False, "shorts_candidate": False}
        for phrase_id in phrase_ids
    ]}


def test_quiet_and_silent_clips_are_recorded_as_inserts(tmp_path: Path, monkeypatch):
    ctx = _context(tmp_path, monkeypatch, _payload_for(
        "clip-01#001", "clip-01#002", "clip-02#001", "clip-02#002",
    ))
    _seed_with_a_silent_clip(ctx)

    result = analyze(ctx)

    inserts = {insert.clip_id: insert for insert in result.inserts}
    assert "clip-01" in inserts  # 0.3 words/s, below the threshold
    assert "clip-10" in inserts  # no words at all
    assert "clip-02" not in inserts  # 1.5 words/s, ordinary narration
    assert abs(inserts["clip-01"].speech_density - 0.3) < 1e-6
    assert inserts["clip-10"].duration == 7.0
    assert inserts["clip-10"].speech_density == 0.0


def test_insert_threshold_is_configurable(tmp_path: Path, monkeypatch):
    ctx = _context(tmp_path, monkeypatch, _payload_for(
        "clip-01#001", "clip-01#002", "clip-02#001", "clip-02#002",
    ))
    _seed_with_a_silent_clip(ctx)
    ctx = StageContext(
        project_dir=ctx.project_dir,
        input_dir=ctx.input_dir,
        work_dir=ctx.work_dir,
        output_dir=ctx.output_dir,
        config=Config(
            providers={"asr": "mock", "llm": "mock"},
            analyze=AnalyzeSettings(insert_max_words_per_second=0.1),
        ),
        store=ctx.store,
    )

    result = analyze(ctx)

    # Only the wordless clip survives a threshold below clip-01's 0.3 words/s.
    assert [insert.clip_id for insert in result.inserts] == ["clip-10"]


# --- Insert descriptions: the planner cannot see a silent clip's content, so the
# model is shown its keyframes and asked to name what is physically happening ---


def _seed_with_a_describable_insert(ctx: StageContext, insert_path: Path, duration: float) -> None:
    ctx.store.write(
        "01-manifest",
        Manifest(clips=[
            ClipInfo(clip_id="clip-01", path=str(ctx.project_dir / "a.mp4"), duration=10.0,
                     width=320, height=240, fps=30.0, has_audio=True),
            ClipInfo(clip_id="clip-10", path=str(insert_path), duration=duration,
                     width=320, height=240, fps=30.0, has_audio=True),
        ]),
        fingerprint="fp",
    )
    ctx.store.write(
        "02-quality",
        QualityReport(clips=[
            ClipQuality(clip_id=clip_id, blur=0.1, motion=0.2, black_ratio=0.0, usable=True)
            for clip_id in ("clip-01", "clip-10")
        ]),
        fingerprint="fp",
    )
    words = [
        Word(text="Look", start=0.0, end=0.3),
        Word(text="here", start=0.35, end=0.7),
        Word(text="Wow", start=2.0, end=2.4),
    ]
    ctx.store.write(
        "03-transcript",
        Transcript(provider="mock", clips=[
            ClipTranscript(clip_id="clip-01", words=words),
            ClipTranscript(clip_id="clip-10", words=[]),  # silent: the insert
        ]),
        fingerprint="fp",
    )


def _segments_payload_for_clip_01() -> list[dict]:
    return [
        {"phrase_id": "clip-01#001", "content": "intro", "delivery_score": 8,
         "visual_score": 7, "emotion": "excited", "is_failed_take": False,
         "shorts_candidate": True},
        {"phrase_id": "clip-01#002", "content": "reaction", "delivery_score": 5,
         "visual_score": 6, "emotion": "calm", "is_failed_take": False,
         "shorts_candidate": False},
    ]


def test_insert_description_reaches_analysis_inserts(tmp_path: Path, monkeypatch, make_clip):
    clip_path = make_clip("insert.mp4", seconds=6.0)
    payload = {
        "segments": _segments_payload_for_clip_01(),
        "descriptions": {"clip-10": "Filling the toy with paint using the syringe."},
    }
    ctx = _context(tmp_path, monkeypatch, payload)
    _seed_with_a_describable_insert(ctx, clip_path, duration=6.0)

    result = analyze(ctx)

    insert = next(i for i in result.inserts if i.clip_id == "clip-10")
    assert insert.description == "Filling the toy with paint using the syringe."


def test_clip_missing_from_description_reply_ends_up_empty_and_stage_succeeds(
    tmp_path: Path, monkeypatch, make_clip
):
    clip_path = make_clip("insert.mp4", seconds=6.0)
    payload = {
        "segments": _segments_payload_for_clip_01(),
        "descriptions": {},  # the model never answered about clip-10
    }
    ctx = _context(tmp_path, monkeypatch, payload)
    _seed_with_a_describable_insert(ctx, clip_path, duration=6.0)

    result = analyze(ctx)

    insert = next(i for i in result.inserts if i.clip_id == "clip-10")
    assert insert.description == ""


def test_unextractable_keyframe_leaves_analyze_succeeding_with_empty_description(
    tmp_path: Path, monkeypatch
):
    """A source file that exists but ffmpeg refuses to read (corrupt, truncated,
    an unsupported codec) must degrade the same way a missing model reply does:
    the insert goes undescribed, and analyze must not raise over it."""
    corrupt_path = tmp_path / "corrupt-insert.mp4"
    corrupt_path.write_bytes(b"not a real video file, just garbage bytes")
    payload = {
        "segments": _segments_payload_for_clip_01(),
        "descriptions": {"clip-10": "unreachable: no keyframe ever gets extracted"},
    }
    ctx = _context(tmp_path, monkeypatch, payload)
    _seed_with_a_describable_insert(ctx, corrupt_path, duration=6.0)

    result = analyze(ctx)

    insert = next(i for i in result.inserts if i.clip_id == "clip-10")
    assert insert.description == ""


def _insert_manifest_and_context(tmp_path: Path, clip_path: Path, config: Config) -> tuple[Manifest, StageContext]:
    manifest = Manifest(clips=[ClipInfo(
        clip_id="clip-10", path=str(clip_path), duration=6.0,
        width=320, height=240, fps=30.0, has_audio=True,
    )])
    (tmp_path / "work").mkdir(exist_ok=True)
    ctx = StageContext(
        project_dir=tmp_path, input_dir=tmp_path, work_dir=tmp_path / "work",
        output_dir=tmp_path, config=config, store=ArtifactStore(tmp_path / "work"),
    )
    return manifest, ctx


def test_insert_keyframes_extracted_even_when_keyframes_per_phrase_zero(tmp_path: Path, make_clip):
    clip_path = make_clip("insert.mp4", seconds=6.0)
    manifest, ctx = _insert_manifest_and_context(
        tmp_path, clip_path, Config(analyze=AnalyzeSettings(keyframes_per_phrase=0))
    )
    inserts = [InsertClip(clip_id="clip-10", duration=6.0, speech_density=0.0)]

    frames_by_clip = _insert_keyframes(ctx, manifest, inserts, ctx.config.analyze)

    assert len(frames_by_clip["clip-10"]) == 3


def test_insert_keyframes_not_extracted_when_describe_inserts_false(tmp_path: Path, make_clip):
    clip_path = make_clip("insert.mp4", seconds=6.0)
    manifest, ctx = _insert_manifest_and_context(
        tmp_path, clip_path, Config(analyze=AnalyzeSettings(describe_inserts=False))
    )
    inserts = [InsertClip(clip_id="clip-10", duration=6.0, speech_density=0.0)]

    frames_by_clip = _insert_keyframes(ctx, manifest, inserts, ctx.config.analyze)

    assert frames_by_clip == {}
    assert not (ctx.work_dir / "keyframes").exists()


def test_insert_keyframes_already_present_are_not_re_extracted(
    tmp_path: Path, make_clip, monkeypatch
):
    clip_path = make_clip("insert.mp4", seconds=6.0)
    manifest, ctx = _insert_manifest_and_context(tmp_path, clip_path, Config())
    inserts = [InsertClip(clip_id="clip-10", duration=6.0, speech_density=0.0)]

    first = _insert_keyframes(ctx, manifest, inserts, ctx.config.analyze)
    assert len(first["clip-10"]) == 3

    import videoai.stages.s04_analyze as s04

    calls: list[object] = []
    real_extract_frame = s04.extract_frame

    def _counting_extract_frame(*args, **kwargs):
        calls.append((args, kwargs))
        return real_extract_frame(*args, **kwargs)

    monkeypatch.setattr(s04, "extract_frame", _counting_extract_frame)

    second = _insert_keyframes(ctx, manifest, inserts, ctx.config.analyze)

    assert second["clip-10"] == first["clip-10"]
    assert calls == []  # every frame was already cached on disk


def test_build_insert_description_prompt_lists_frame_paths_per_clip(tmp_path: Path):
    inserts = [
        InsertClip(clip_id="clip-09", duration=5.0, speech_density=0.0),
        InsertClip(clip_id="clip-10", duration=7.0, speech_density=0.0),
    ]
    kf = tmp_path / "kf"
    frames_by_clip = {
        "clip-09": [kf / "a1.jpg", kf / "a2.jpg", kf / "a3.jpg"],
        "clip-10": [kf / "b1.jpg", kf / "b2.jpg", kf / "b3.jpg"],
    }

    prompt = build_insert_description_prompt(inserts, frames_by_clip, brief="A pop-toy review.")

    assert "clip-09" in prompt
    assert str(kf / "a1.jpg") in prompt
    assert "clip-10" in prompt
    assert str(kf / "b3.jpg") in prompt
    assert "A pop-toy review." in prompt


def test_describe_inserts_returns_empty_when_no_inserts(tmp_path: Path):
    manifest = Manifest(clips=[])
    (tmp_path / "work").mkdir(exist_ok=True)
    ctx = StageContext(
        project_dir=tmp_path, input_dir=tmp_path, work_dir=tmp_path / "work",
        output_dir=tmp_path, config=Config(), store=ArtifactStore(tmp_path / "work"),
    )

    assert _describe_inserts(ctx, manifest, [], "brief") == {}


# --- The prompt must tell the model that a phrase addressed to the camera
# operator (pause/stop/cut/restart/delete) is a failed take even when the
# recogniser garbles the words - this is fingerprint material, so dropping it
# silently would not be caught by any artifact-shape test above. ---


def test_prompt_instructs_model_to_reject_operator_instructions():
    lowered = INSTRUCTIONS.lower()
    assert "pause" in lowered
    assert "camera" in lowered or "filming" in lowered
    assert "garbled" in lowered or "mangled" in lowered or "mangles" in lowered
    assert "delete" in lowered


# --- A transient error must not cost every insert its description in silence ---


class _FailingDescriber:
    """A provider that scores phrases normally and fails the description call."""

    name = "flaky"
    reads_video = False

    def __init__(self, failures: int) -> None:
        self.failures = failures
        self.description_calls = 0

    def complete_json(self, prompt, images, timeout, videos=None) -> dict:
        if "silent visual insert clips" not in prompt:
            return {"segments": _segments_payload_for_clip_01()}
        self.description_calls += 1
        if self.description_calls <= self.failures:
            raise RuntimeError("claude CLI failed: rate limited")
        return {"descriptions": {"clip-10": "Squeezing the toy until it pops."}}


def _with_provider(ctx: StageContext, monkeypatch, provider: object) -> None:
    import videoai.stages.s04_analyze as s04

    monkeypatch.setattr(s04, "resolve_llm", lambda *args, **kwargs: provider)


def test_a_transient_description_failure_is_retried_once(
    tmp_path: Path, monkeypatch, make_clip
):
    clip_path = make_clip("insert.mp4", seconds=6.0)
    ctx = _context(tmp_path, monkeypatch, {})
    _seed_with_a_describable_insert(ctx, clip_path, duration=6.0)
    provider = _FailingDescriber(failures=1)
    _with_provider(ctx, monkeypatch, provider)

    result = analyze(ctx)

    assert provider.description_calls == 2
    insert = next(i for i in result.inserts if i.clip_id == "clip-10")
    assert insert.description == "Squeezing the toy until it pops."


def test_a_description_call_that_keeps_failing_fails_the_stage(
    tmp_path: Path, monkeypatch, make_clip
):
    """Swallowing it left every silent clip placed by chronology alone, with
    nothing in the artifact recording that the descriptions went missing."""
    clip_path = make_clip("insert.mp4", seconds=6.0)
    ctx = _context(tmp_path, monkeypatch, {})
    _seed_with_a_describable_insert(ctx, clip_path, duration=6.0)
    provider = _FailingDescriber(failures=2)
    _with_provider(ctx, monkeypatch, provider)

    with pytest.raises(RuntimeError, match="could not describe the silent insert clips"):
        analyze(ctx)

    assert provider.description_calls == 2


def test_a_reply_without_a_descriptions_object_is_asked_again(
    tmp_path: Path, monkeypatch, make_clip
):
    """A model answering under the wrong key gets a second chance, and if it
    answers the same way twice the inserts keep an empty description: that is
    the model having answered, and the artifact records what it did not say."""
    clip_path = make_clip("insert.mp4", seconds=6.0)
    ctx = _context(tmp_path, monkeypatch, {})
    _seed_with_a_describable_insert(ctx, clip_path, duration=6.0)

    class _WrongShape(_FailingDescriber):
        def complete_json(self, prompt, images, timeout, videos=None) -> dict:
            if "silent visual insert clips" not in prompt:
                return {"segments": _segments_payload_for_clip_01()}
            self.description_calls += 1
            return {"clip-10": "not under the key that was asked for"}

    provider = _WrongShape(failures=0)
    _with_provider(ctx, monkeypatch, provider)

    result = analyze(ctx)

    assert provider.description_calls == 2
    assert next(i for i in result.inserts if i.clip_id == "clip-10").description == ""
