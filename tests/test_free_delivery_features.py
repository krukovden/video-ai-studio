from pathlib import Path

import pytest
from typer.testing import CliRunner

from videoai.cli import app
from videoai.core.models import (
    Approval,
    ClipTranscript,
    DraftResult,
    Timeline,
    TimelineClip,
    Transcript,
    Word,
)
from videoai.core.store import ArtifactStore
from videoai.stages.s08_polish import build_captions, write_ass_captions


def test_captions_map_source_words_to_the_assembled_timeline(tmp_path: Path):
    timeline = Timeline(
        fps=30,
        width=1920,
        height=1080,
        clips=[
            TimelineClip(src="clip-01", offset=10, dur=3, start=0),
            TimelineClip(src="clip-01", offset=20, dur=3, start=3),
        ],
    )
    transcript = Transcript(
        provider="test",
        clips=[
            ClipTranscript(
                clip_id="clip-01",
                words=[
                    Word(text="Hello", start=10.5, end=10.9),
                    Word(text="there", start=11.0, end=11.4),
                    Word(text="Big", start=20.2, end=20.5),
                    Word(text="reveal", start=20.6, end=21.0),
                ],
            )
        ],
    )

    captions = build_captions(
        timeline,
        transcript,
        measured_starts=[0.0, 3.0],
        split_indices=[1],
        transition=0.25,
        intro_offset=2.0,
        words_per_caption=4,
    )

    assert [caption.text for caption in captions] == ["Hello there", "Big reveal"]
    assert captions[0].start == pytest.approx(2.46)
    assert captions[1].start == pytest.approx(4.91)

    output = tmp_path / "captions.ass"
    write_ass_captions(output, captions, 1920, 1080)
    text = output.read_text(encoding="utf-8")
    assert "PlayResX: 1920" in text
    assert "Hello there" in text


def test_approve_binds_the_review_to_the_current_timeline(tmp_path: Path):
    project = tmp_path / "project"
    store = ArtifactStore(project / "work")
    timeline = Timeline(fps=30, width=320, height=240)
    store.write("05-timeline", timeline, fingerprint="timeline")
    store.write(
        "06-draft",
        DraftResult(path=str(project / "output" / "draft.mp4"), duration=0, segment_count=0),
        fingerprint="draft",
    )

    result = CliRunner().invoke(app, ["approve", str(project)])

    assert result.exit_code == 0, result.output
    approval = store.read("06-approval", Approval)
    assert approval.timeline_hash == store.content_hash("05-timeline")
