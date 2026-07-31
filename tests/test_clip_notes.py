"""A description of every source clip, written once and kept.

Watching footage costs money, and the answer does not change unless the footage
does — or unless the question does. So each clip is described once, keyed by the
identity of the file rather than by its position and by the question that was put
to the model, and a later run reuses that description instead of paying for it
again.
"""
from __future__ import annotations

from pathlib import Path

from videoai.core.models import ClipEvent, ClipNote, ClipNotes
from videoai.logic.clip_notes import (
    clips_needing_description,
    merge_notes,
    render_notes_markdown,
)


def _note(clip_id: str, key: str, what: str = "a summary") -> ClipNote:
    return ClipNote(clip_id=clip_id, source_key=key, duration=10.0, summary=what)


def test_a_clip_already_described_is_not_sent_again():
    cached = ClipNotes(notes=[_note("clip-01", "aaa")])
    manifest_keys = [("clip-01", "aaa"), ("clip-02", "bbb")]
    assert clips_needing_description(cached, manifest_keys) == ["clip-02"]


def test_nothing_is_sent_when_everything_is_described():
    cached = ClipNotes(notes=[_note("clip-01", "aaa"), _note("clip-02", "bbb")])
    assert clips_needing_description(cached, [("clip-01", "aaa"), ("clip-02", "bbb")]) == []


def test_a_reshoot_is_described_again():
    """Identity is the file, not the slot: replacing the footage behind clip-01
    has to be noticed, or the notes would describe a video that is gone."""
    cached = ClipNotes(notes=[_note("clip-01", "aaa")])
    assert clips_needing_description(cached, [("clip-01", "zzz")]) == ["clip-01"]


def test_renumbering_alone_does_not_cost_anything():
    """Adding a clip that sorts first renumbers everything after it. The files
    are unchanged, so nothing should be re-sent."""
    cached = ClipNotes(notes=[_note("clip-01", "aaa"), _note("clip-02", "bbb")])
    renumbered = [("clip-01", "new"), ("clip-02", "aaa"), ("clip-03", "bbb")]
    assert clips_needing_description(cached, renumbered) == ["clip-01"]


def test_merging_keeps_what_was_known_and_adds_what_is_new():
    cached = ClipNotes(notes=[_note("clip-01", "aaa", "the old summary")])
    fresh = [_note("clip-02", "bbb", "the new one")]
    merged = merge_notes(cached, fresh)
    assert [note.source_key for note in merged.notes] == ["aaa", "bbb"]
    assert merged.by_key("aaa").summary == "the old summary"


def test_re_describing_a_clip_replaces_its_note():
    cached = ClipNotes(notes=[_note("clip-01", "aaa", "stale")])
    merged = merge_notes(cached, [_note("clip-01", "aaa", "fresh")])
    assert len(merged.notes) == 1
    assert merged.by_key("aaa").summary == "fresh"


def test_the_written_file_reads_as_a_log_of_each_video():
    notes = ClipNotes(notes=[
        ClipNote(
            clip_id="clip-02", source_key="aaa", duration=48.1,
            source_name="IMG_5663.MOV",
            summary="The child opens the package at the table.",
            events=[
                ClipEvent(at=18.1, kind="action", what="the box lid comes off",
                          where="center"),
                ClipEvent(at=39.1, kind="action", what="gloves go on",
                          where="bottom-center"),
            ],
        )
    ])
    text = render_notes_markdown(notes)
    assert "clip-02" in text
    assert "IMG_5663.MOV" in text
    assert "The child opens the package" in text
    # The chronology is the point: times, in order, with where to look.
    assert "18.1" in text and "39.1" in text
    assert "the box lid comes off" in text
    assert "center" in text


def test_an_undescribed_clip_is_still_listed():
    """A file that could not be read must appear, or the log would quietly imply
    the footage does not exist."""
    notes = ClipNotes(notes=[
        ClipNote(clip_id="clip-09", source_key="zzz", duration=14.6,
                 source_name="IMG_8199.MOV", summary="", events=[])
    ])
    text = render_notes_markdown(notes)
    assert "clip-09" in text
    assert "not described" in text.lower()


def test_events_are_written_in_time_order():
    notes = ClipNotes(notes=[ClipNote(
        clip_id="clip-01", source_key="a", duration=30.0,
        summary="s",
        events=[ClipEvent(at=20.0, kind="action", what="later"),
                ClipEvent(at=5.0, kind="action", what="earlier")],
    )])
    text = render_notes_markdown(notes)
    assert text.index("earlier") < text.index("later")


# --- The stage: a note answers a question, and the question is editable ---


class _WatchingLLM:
    name = "gemini_api"
    reads_video = True

    def __init__(self) -> None:
        self.watched: list[str] = []

    def complete_json(self, prompt, images, timeout, videos=None) -> dict:
        self.watched.append(str(videos[0]))
        return {"summary": "a hand squeezes the toy", "events": []}


def _describe_context(tmp_path, clip_names: tuple[str, ...], model: str = "gemini-3.1-flash-lite"):
    from videoai.config import AnalyzeSettings, Config, DescribeSettings
    from videoai.core.models import ClipInfo, Manifest
    from videoai.core.registry import StageContext
    from videoai.core.store import ArtifactStore

    for name in ("work", "output"):
        (tmp_path / name).mkdir(exist_ok=True)
    clips = []
    for index, name in enumerate(clip_names, start=1):
        path = tmp_path / name
        path.write_bytes(f"pretend footage of {name}".encode("utf-8"))
        clips.append(ClipInfo(
            clip_id=f"clip-{index:02d}", path=str(path), duration=12.0,
            width=320, height=240, fps=30.0, has_audio=True, source_key=name,
        ))
    ctx = StageContext(
        project_dir=tmp_path,
        input_dir=tmp_path,
        work_dir=tmp_path / "work",
        output_dir=tmp_path / "output",
        config=Config(
            providers={"asr": "mock", "llm": "gemini_api"},
            describe=DescribeSettings(enabled=True),
            analyze=AnalyzeSettings(llm_model=model),
        ),
        store=ArtifactStore(tmp_path / "work"),
    )
    ctx.store.write("01-manifest", Manifest(clips=clips), fingerprint="fp")
    return ctx


def _run_describe(ctx, monkeypatch, provider):
    import videoai.stages.s04b_describe as s04b

    monkeypatch.setattr(s04b, "resolve_llm", lambda *args, **kwargs: provider)
    return s04b.describe(ctx)


def test_footage_already_watched_is_not_watched_again(tmp_path, monkeypatch):
    ctx = _describe_context(tmp_path, ("a.mp4",))
    provider = _WatchingLLM()

    first = _run_describe(ctx, monkeypatch, provider)
    second = _run_describe(ctx, monkeypatch, provider)

    assert len(provider.watched) == 1
    assert second == first


def test_only_new_footage_is_watched(tmp_path, monkeypatch):
    ctx = _describe_context(tmp_path, ("a.mp4",))
    provider = _WatchingLLM()
    _run_describe(ctx, monkeypatch, provider)

    grown = _describe_context(tmp_path, ("a.mp4", "b.mp4"))
    notes = _run_describe(grown, monkeypatch, provider)

    assert [Path(path).name for path in provider.watched] == ["a.mp4", "b.mp4"]
    assert len(notes.notes) == 2


def test_a_changed_prompt_makes_every_clip_undescribed(tmp_path, monkeypatch):
    """Keyed by the file alone, the stage "ran", found nothing new and returned
    the old notes byte for byte — under the new fingerprint, so everything
    downstream stayed cached against an answer to a question nobody asked."""
    import videoai.stages.s04b_describe as s04b

    ctx = _describe_context(tmp_path, ("a.mp4",))
    provider = _WatchingLLM()
    before = _run_describe(ctx, monkeypatch, provider)

    monkeypatch.setattr(s04b, "INSTRUCTIONS", s04b.INSTRUCTIONS + "\n- Also name the colours.")
    after = _run_describe(ctx, monkeypatch, provider)

    assert len(provider.watched) == 2
    assert [note.clip_id for note in after.notes] == ["clip-01"]
    assert after.notes[0].source_key != before.notes[0].source_key


def test_a_changed_model_makes_every_clip_undescribed(tmp_path, monkeypatch):
    """Flash-lite's account of a clip is not Pro's, and the notes are what the
    planner reads when it decides where a silent shot belongs."""
    ctx = _describe_context(tmp_path, ("a.mp4",))
    provider = _WatchingLLM()
    _run_describe(ctx, monkeypatch, provider)

    switched = _describe_context(tmp_path, ("a.mp4",), model="gemini-3.6-pro")
    notes = _run_describe(switched, monkeypatch, provider)

    assert len(provider.watched) == 2
    assert [note.clip_id for note in notes.notes] == ["clip-01"]


def test_notes_about_footage_that_left_the_project_are_kept(tmp_path, monkeypatch):
    ctx = _describe_context(tmp_path, ("a.mp4", "b.mp4"))
    provider = _WatchingLLM()
    _run_describe(ctx, monkeypatch, provider)

    shrunk = _describe_context(tmp_path, ("a.mp4",))
    notes = _run_describe(shrunk, monkeypatch, provider)

    assert len(provider.watched) == 2  # nothing was re-watched
    assert len(notes.notes) == 2
