from pathlib import Path
import subprocess

import pytest
from typer.testing import CliRunner

from videoai.cli import app
from videoai.config import Config, PolishSettings
from videoai.core.models import (
    Approval,
    ClipInfo,
    ClipTranscript,
    DraftResult,
    Manifest,
    StoryPlan,
    Timeline,
    TimelineClip,
    Transcript,
    Word,
)
from videoai.core.registry import StageContext
from videoai.core.store import ArtifactStore
from videoai.stages.s06_render_draft import render_draft
from videoai.stages.s08_polish import (
    build_captions,
    build_cartoon_effects,
    build_major_action_titles,
    polish,
    write_ass_captions,
)


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
    draft_path = project / "output" / "draft.mp4"
    draft_path.parent.mkdir(parents=True)
    draft_path.write_bytes(b"reviewed draft")
    store.write(
        "06-draft",
        DraftResult(
            path=str(draft_path),
            duration=0,
            segment_count=0,
            timeline_hash=store.content_hash("05-timeline") or "",
        ),
        fingerprint="draft",
    )

    result = CliRunner().invoke(app, ["approve", str(project)])

    assert result.exit_code == 0, result.output
    approval = store.read("06-approval", Approval)
    assert approval.timeline_hash == store.content_hash("05-timeline")
    assert approval.draft_hash
    assert approval.config_hash


def test_titles_are_limited_to_major_actions_and_effects_follow_story_beats():
    beats = [
        ("Hook - Box Opens", False),
        ("Meet the Toy", False),
        ("Filling With Paint", False),
        ("Filling With Paint", True),
        ("Covered in Bubbles", True),
        ("Popping Time", False),
        ("Closing", False),
    ]
    timeline = Timeline(
        fps=30,
        width=1920,
        height=1080,
        clips=[
            TimelineClip(
                src="clip-01", offset=index * 4, dur=4, start=index * 4,
                beat=beat, is_insert=is_insert,
            )
            for index, (beat, is_insert) in enumerate(beats)
        ],
    )
    starts = [index * 4.0 for index in range(len(beats))]
    durations = [4.0] * len(beats)

    titles = build_major_action_titles(
        timeline, starts, durations, 2.5, 2.0, 1920, 1080,
    )
    effects = build_cartoon_effects(timeline, starts, durations, 2.5, 2.0)

    assert [title.text for title in titles] == [
        "OPENING THE BOX!", "INJECTION TIME!", "POPPING TIME!",
    ]
    assert {effect.kind for effect in effects} >= {
        "box_fly", "injection_burst", "toy_reaction", "pop",
    }
    assert sum(effect.kind == "pop" for effect in effects) <= 5


def test_strict_delivery_applies_every_required_local_feature(
    tmp_path: Path, make_clip
):
    (tmp_path / "production-contract.yaml").write_text(
        """
version: 1
required_output: {width: 426, height: 240, video_codec: h264, audio: true, full_decode: true}
required_features:
  intro: true
  outro: true
  section_titles: true
  captions: true
  music: true
  music_ducking: true
  transitions: true
  closing_beat: true
quality:
  source: originals
  lossless_intermediates: true
  maximum_lossy_video_generations: 1
""".strip()
        + "\n",
        encoding="utf-8",
    )
    source = make_clip("source.mp4", seconds=7.0, size="640x360")
    music_dir = tmp_path / "music"
    music_dir.mkdir()
    track = music_dir / "bed.mp3"
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", "sine=frequency=330:duration=3",
            "-c:a", "libmp3lame", str(track),
        ],
        check=True,
    )
    work = tmp_path / "work"
    output = tmp_path / "output"
    work.mkdir()
    output.mkdir()
    ctx = StageContext(
        project_dir=tmp_path,
        input_dir=tmp_path,
        work_dir=work,
        output_dir=output,
        config=Config(
            polish=PolishSettings(
                strict_contract=True,
                require_approval=False,
                intro_seconds=0.5,
                outro_seconds=0.5,
                title_seconds=0.5,
                captions_enabled=True,
                caption_words=2,
                transition_frames=3,
                music_dir=str(music_dir),
                output_height=240,
                lossless_intermediates=True,
                hardware_encode=False,
            )
        ),
        store=ArtifactStore(work),
    )
    ctx.store.write(
        "01-manifest",
        Manifest(
            clips=[
                ClipInfo(
                    clip_id="clip-01",
                    path=str(source),
                    proxy_path=str(source),
                    duration=7,
                    width=640,
                    height=360,
                    fps=30,
                    has_audio=True,
                )
            ]
        ),
        fingerprint="manifest",
    )
    timeline = Timeline(
        fps=30,
        width=640,
        height=360,
        clips=[
            TimelineClip(
                src="clip-01", offset=0, dur=2, start=0, beat="Hook - Box Opens"
            ),
            TimelineClip(
                src="clip-01", offset=2, dur=2, start=2,
                beat="Closing", quote="Thanks for watching",
            ),
        ],
    )
    ctx.store.write("05-timeline", timeline, fingerprint="timeline")
    ctx.store.write(
        "05a-storyplan", StoryPlan(title="Color Pop Review"), fingerprint="story"
    )
    ctx.store.write(
        "03-transcript",
        Transcript(
            provider="test",
            clips=[
                ClipTranscript(
                    clip_id="clip-01",
                    words=[
                        Word(text="Hello", start=0.2, end=0.6),
                        Word(text="everyone", start=0.7, end=1.1),
                        Word(text="Look", start=2.2, end=2.5),
                        Word(text="inside", start=2.6, end=3.0),
                    ],
                )
            ],
        ),
        fingerprint="transcript",
    )
    draft = render_draft(ctx)
    ctx.store.write("06-draft", draft, fingerprint="draft")

    result = polish(ctx)

    assert Path(result.path).is_file()
    assert result.intro and result.outro
    assert result.title_count == 1
    assert result.caption_count == 2
    assert result.cartoon_effect_count == 1
    assert result.transition_count == 1
    assert result.music_track == track.name
    assert result.music_ducking is True
    assert result.fully_decoded is True
    assert Path(result.production_report).is_file()
    assert (output / "final.srt").is_file()
