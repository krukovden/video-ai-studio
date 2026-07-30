import json
from pathlib import Path

import pytest

from videoai.config import Config, PlanSettings
from videoai.core.models import (
    Analysis,
    ClipInfo,
    ClipSync,
    ClipTranscript,
    InsertClip,
    Manifest,
    RejectedPhrases,
    SegmentAnalysis,
    StoryPlan,
    SyncMap,
    Timeline,
    Transcript,
    Word,
)
from videoai.core.registry import StageContext
from videoai.core.store import ArtifactStore
from videoai.stages.s05_plan import INSTRUCTIONS, plan


def _context(
    tmp_path: Path,
    monkeypatch,
    payload: dict,
    config: Config | None = None,
    segments: list[SegmentAnalysis] | None = None,
    extra_clips: list[ClipInfo] | None = None,
    inserts: list[InsertClip] | None = None,
) -> StageContext:
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
        config=config or Config(providers={"asr": "mock", "llm": "mock"}),
        store=ArtifactStore(tmp_path / "work"),
    )
    clips = [
        ClipInfo(clip_id="clip-01", path=str(tmp_path / "a.mp4"), duration=30.0, width=1920,
                 height=1080, fps=30.0, has_audio=True),
        *(extra_clips or []),
    ]
    ctx.store.write("01-manifest", Manifest(clips=clips), fingerprint="fp")
    ctx.store.write("01b-sync", SyncMap(clips=[
        ClipSync(clip_id=clip.clip_id, camera=clip.camera,
                 global_start=float(index) * 60.0, method="sequential")
        for index, clip in enumerate(clips)
    ]), fingerprint="fp")
    ctx.store.write("03-transcript", Transcript(provider="mock", clips=[ClipTranscript(
        clip_id="clip-01",
        words=[
            Word(text="hello", start=2.0, end=2.5), Word(text="world", start=2.6, end=3.0),
            Word(text="bad", start=4.0, end=4.4), Word(text="moment", start=4.5, end=5.0),
        ],
    )]), fingerprint="fp")
    ctx.store.write("04-analysis", Analysis(
        provider="mock",
        segments=segments or [
            SegmentAnalysis(phrase_id="clip-01#001", clip_id="clip-01", start=2.0, end=3.0,
                            text="hello world", content="greeting", delivery_score=9),
        ],
        inserts=inserts or [],
    ), fingerprint="fp")
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


def _two_segments() -> list[SegmentAnalysis]:
    return [
        SegmentAnalysis(phrase_id="clip-01#001", clip_id="clip-01", start=2.0, end=3.0,
                        text="hello world", content="greeting", delivery_score=9),
        SegmentAnalysis(phrase_id="clip-01#002", clip_id="clip-01", start=4.0, end=5.0,
                        text="bad moment", content="an adult walks through frame",
                        delivery_score=2),
    ]


def test_excluded_phrase_never_appears_in_the_resulting_timeline(tmp_path: Path, monkeypatch):
    """The creator named a bad moment as excluded; the planner must never be given
    the chance to pick it, and none of the phrase's own words may show up in the
    final edit."""
    payload = {
        "title": "T", "description": "D", "tags": [],
        "sections": [{"name": "Hook", "goal": "x", "phrase_ids": ["clip-01#001"]}],
    }
    config = Config(
        providers={"asr": "mock", "llm": "mock"},
        plan=PlanSettings(exclude_phrases=["clip-01#002"]),
    )
    ctx = _context(tmp_path, monkeypatch, payload, config=config, segments=_two_segments())

    timeline = plan(ctx)

    assert [clip.src for clip in timeline.clips] == ["clip-01"]
    assert all(clip.quote != "bad moment" for clip in timeline.clips)


def test_excluded_phrase_is_hidden_from_the_prompt(tmp_path: Path, monkeypatch):
    from videoai.stages.s05_plan import _segments_view

    analysis = Analysis(provider="mock", segments=_two_segments())
    view = _segments_view(analysis, {"clip-01#002"})

    assert "clip-01#001" in view
    assert "clip-01#002" not in view
    assert "bad moment" not in view


def test_model_reply_naming_an_excluded_phrase_raises(tmp_path: Path, monkeypatch):
    """A stale cached LLM reply could still name a phrase the creator has since
    excluded; filtering it out of the prompt is not enough on its own."""
    payload = {
        "title": "T", "description": "D", "tags": [],
        "sections": [{"name": "Hook", "goal": "x", "phrase_ids": ["clip-01#002"]}],
    }
    config = Config(
        providers={"asr": "mock", "llm": "mock"},
        plan=PlanSettings(exclude_phrases=["clip-01#002"]),
    )
    ctx = _context(tmp_path, monkeypatch, payload, config=config, segments=_two_segments())

    with pytest.raises(RuntimeError) as error:
        plan(ctx)

    assert "clip-01#002" in str(error.value)


# --- Silent visual inserts: footage with no narration reaches the timeline only
# through an `insert:` reference, since there is no phrase to select it by ---


def _silent_clip(tmp_path: Path) -> ClipInfo:
    return ClipInfo(clip_id="clip-10", path=str(tmp_path / "c.mp4"), duration=7.0, width=1920,
                    height=1080, fps=30.0, has_audio=True)


def _insert() -> InsertClip:
    return InsertClip(clip_id="clip-10", duration=7.0, speech_density=0.0)


def test_planner_can_place_a_whole_insert_into_the_timeline(tmp_path: Path, monkeypatch):
    payload = {
        "title": "T", "description": "D", "tags": [],
        "sections": [{"name": "Popping", "goal": "show the bubble",
                      "phrase_ids": ["clip-01#001", "insert:clip-10"]}],
    }
    ctx = _context(tmp_path, monkeypatch, payload,
                   extra_clips=[_silent_clip(tmp_path)], inserts=[_insert()])

    timeline = plan(ctx)

    insert_clip = timeline.clips[1]
    assert insert_clip.src == "clip-10"
    assert insert_clip.is_insert is True
    assert insert_clip.quote == ""
    assert abs(insert_clip.dur - 7.0) < 1e-6


def test_planner_can_trim_an_insert_to_a_range(tmp_path: Path, monkeypatch):
    payload = {
        "title": "T", "description": "D", "tags": [],
        "sections": [{"name": "Popping", "goal": "show the bubble",
                      "phrase_ids": ["insert:clip-10@2-5"]}],
    }
    ctx = _context(tmp_path, monkeypatch, payload,
                   extra_clips=[_silent_clip(tmp_path)], inserts=[_insert()])

    timeline = plan(ctx)

    assert timeline.clips[0].offset == 2.0
    assert abs(timeline.clips[0].dur - 3.0) < 1e-6


def test_unknown_insert_clip_id_from_the_planner_raises(tmp_path: Path, monkeypatch):
    payload = {
        "title": "T", "description": "D", "tags": [],
        "sections": [{"name": "Popping", "goal": "x", "phrase_ids": ["insert:clip-77"]}],
    }
    ctx = _context(tmp_path, monkeypatch, payload,
                   extra_clips=[_silent_clip(tmp_path)], inserts=[_insert()])

    with pytest.raises(RuntimeError) as error:
        plan(ctx)

    assert "insert:clip-77" in str(error.value)


def test_insert_range_outside_the_clip_from_the_planner_raises(tmp_path: Path, monkeypatch):
    payload = {
        "title": "T", "description": "D", "tags": [],
        "sections": [{"name": "Popping", "goal": "x", "phrase_ids": ["insert:clip-10@4-12"]}],
    }
    ctx = _context(tmp_path, monkeypatch, payload,
                   extra_clips=[_silent_clip(tmp_path)], inserts=[_insert()])

    with pytest.raises(RuntimeError) as error:
        plan(ctx)

    assert "insert:clip-10@4-12" in str(error.value)


def test_inverted_insert_range_from_the_planner_raises(tmp_path: Path, monkeypatch):
    payload = {
        "title": "T", "description": "D", "tags": [],
        "sections": [{"name": "Popping", "goal": "x", "phrase_ids": ["insert:clip-10@5-2"]}],
    }
    ctx = _context(tmp_path, monkeypatch, payload,
                   extra_clips=[_silent_clip(tmp_path)], inserts=[_insert()])

    with pytest.raises(RuntimeError) as error:
        plan(ctx)

    assert "insert:clip-10@5-2" in str(error.value)


def test_inserts_are_listed_with_their_place_in_recording_order():
    """A wordless clip cannot be read; the narration either side of it is the only
    thing that tells the planner which moment of the shoot it belongs to."""
    from videoai.stages.s05_plan import _inserts_view

    analysis = Analysis(
        provider="mock",
        segments=[
            SegmentAnalysis(phrase_id="clip-01#001", clip_id="clip-01", start=2.0, end=3.0,
                            text="hello world", content="greeting", delivery_score=9),
            SegmentAnalysis(phrase_id="clip-02#001", clip_id="clip-02", start=1.0, end=2.0,
                            text="all done", content="outro", delivery_score=8),
        ],
        inserts=[InsertClip(clip_id="clip-10", duration=7.0, speech_density=0.0)],
    )
    sync = SyncMap(clips=[
        ClipSync(clip_id="clip-01", camera="main", global_start=0.0, method="metadata"),
        ClipSync(clip_id="clip-10", camera="main", global_start=60.0, method="metadata"),
        ClipSync(clip_id="clip-02", camera="main", global_start=120.0, method="metadata"),
    ])

    view = _inserts_view(analysis, sync, set())

    assert "insert:clip-10" in view
    assert "7.0s" in view
    assert "after clip-01#001" in view
    assert "before clip-02#001" in view


def test_insert_listing_reaches_the_prompt(tmp_path: Path, monkeypatch):
    payload = {
        "title": "T", "description": "D", "tags": [],
        "sections": [{"name": "Hook", "goal": "x", "phrase_ids": ["clip-01#001"]}],
    }
    ctx = _context(tmp_path, monkeypatch, payload,
                   extra_clips=[_silent_clip(tmp_path)], inserts=[_insert()])
    from videoai.providers.llm_mock import MockLLM

    seen: list[str] = []
    original = MockLLM.complete_json

    def _capture(self, prompt, images, timeout):
        seen.append(prompt)
        return original(self, prompt, images, timeout)

    monkeypatch.setattr(MockLLM, "complete_json", _capture)

    plan(ctx)

    assert "Silent visual inserts" in seen[0]
    assert "insert:clip-10" in seen[0]
    assert "no narration" in seen[0]


def test_insert_description_reaches_the_plan_prompt(tmp_path: Path, monkeypatch):
    payload = {
        "title": "T", "description": "D", "tags": [],
        "sections": [{"name": "Hook", "goal": "x", "phrase_ids": ["clip-01#001"]}],
    }
    described_insert = InsertClip(
        clip_id="clip-10", duration=7.0, speech_density=0.0,
        description="The bubbles are already formed and a hand is reaching in to pop them.",
    )
    ctx = _context(tmp_path, monkeypatch, payload,
                   extra_clips=[_silent_clip(tmp_path)], inserts=[described_insert])
    from videoai.providers.llm_mock import MockLLM

    seen: list[str] = []
    original = MockLLM.complete_json

    def _capture(self, prompt, images, timeout):
        seen.append(prompt)
        return original(self, prompt, images, timeout)

    monkeypatch.setattr(MockLLM, "complete_json", _capture)

    plan(ctx)

    assert "The bubbles are already formed and a hand is reaching in to pop them." in seen[0]


def test_inserts_view_includes_description_when_present():
    from videoai.stages.s05_plan import _inserts_view

    analysis = Analysis(
        provider="mock",
        segments=[],
        inserts=[InsertClip(
            clip_id="clip-10", duration=7.0, speech_density=0.0,
            description="Filling the toy with paint using the syringe.",
        )],
    )
    sync = SyncMap(clips=[ClipSync(clip_id="clip-10", camera="main", global_start=0.0, method="metadata")])

    view = _inserts_view(analysis, sync, set())

    assert "Filling the toy with paint using the syringe." in view


# --- The visual gate's half of the loop: what the check rejected for what it
# SHOWS must never be offered to the planner again ---


def _reject(ctx: StageContext, *phrase_ids: str) -> None:
    ctx.store.write("05c-rejected", RejectedPhrases(phrase_ids=list(phrase_ids)),
                    fingerprint="derived")


def _captured_prompt(monkeypatch) -> list[str]:
    from videoai.providers.llm_mock import MockLLM

    seen: list[str] = []
    original = MockLLM.complete_json

    def _capture(self, prompt, images, timeout):
        seen.append(prompt)
        return original(self, prompt, images, timeout)

    monkeypatch.setattr(MockLLM, "complete_json", _capture)
    return seen


def test_visually_rejected_phrase_is_hidden_from_the_planner(tmp_path: Path, monkeypatch):
    """Excluding one bad phrase by hand only moved the problem: the planner picked
    its neighbour, which had the same adult in frame. The rejection has to reach
    the candidate list itself, before the model ever sees it."""
    payload = {
        "title": "T", "description": "D", "tags": [],
        "sections": [{"name": "Hook", "goal": "x", "phrase_ids": ["clip-01#001"]}],
    }
    ctx = _context(tmp_path, monkeypatch, payload, segments=_two_segments())
    _reject(ctx, "clip-01#002")
    seen = _captured_prompt(monkeypatch)

    timeline = plan(ctx)

    assert "clip-01#002" not in seen[0]
    assert "bad moment" not in seen[0]
    assert "clip-01#001" in seen[0]
    assert all(clip.quote != "bad moment" for clip in timeline.clips)


def test_model_reply_naming_a_visually_rejected_phrase_raises(tmp_path: Path, monkeypatch):
    payload = {
        "title": "T", "description": "D", "tags": [],
        "sections": [{"name": "Hook", "goal": "x", "phrase_ids": ["clip-01#002"]}],
    }
    ctx = _context(tmp_path, monkeypatch, payload, segments=_two_segments())
    _reject(ctx, "clip-01#002")

    with pytest.raises(RuntimeError) as error:
        plan(ctx)

    assert "clip-01#002" in str(error.value)


def test_visually_rejected_insert_is_hidden_and_refused(tmp_path: Path, monkeypatch):
    """An insert is rejected by its reference, since it has no phrase of its own."""
    payload = {
        "title": "T", "description": "D", "tags": [],
        "sections": [{"name": "Popping", "goal": "x",
                      "phrase_ids": ["clip-01#001", "insert:clip-10@2-5"]}],
    }
    ctx = _context(tmp_path, monkeypatch, payload,
                   extra_clips=[_silent_clip(tmp_path)], inserts=[_insert()])
    _reject(ctx, "insert:clip-10")
    seen = _captured_prompt(monkeypatch)

    with pytest.raises(RuntimeError) as error:
        plan(ctx)

    # The instructions use `insert:clip-10` in their own worked example, so the
    # listing line is what has to be gone: "insert:<id> (<length>s, ...".
    assert "insert:clip-10 (" not in seen[0]
    assert "insert:clip-10@2-5" in str(error.value)


def test_no_rejection_file_withholds_nothing(tmp_path: Path, monkeypatch):
    payload = {
        "title": "T", "description": "D", "tags": [],
        "sections": [{"name": "Hook", "goal": "x",
                      "phrase_ids": ["clip-01#001", "clip-01#002"]}],
    }
    ctx = _context(tmp_path, monkeypatch, payload, segments=_two_segments())

    timeline = plan(ctx)

    assert len(timeline.clips) == 2


# --- The prompt must tell the model the video needs a proper ending: this is
# fingerprint material, so dropping it silently would not be caught by any
# artifact-shape test above. ---


def test_prompt_instructs_model_that_the_video_needs_an_ending():
    lowered = INSTRUCTIONS.lower()
    assert "closing" in lowered or "close it out" in lowered
    assert "end of the shoot" in lowered or "end of recording" in lowered or (
        "end" in lowered and "recording order" in lowered
    )
    assert "mid-thought" in lowered or "fragment" in lowered
    assert "no closing material" in lowered or "description field" in lowered
