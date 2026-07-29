"""The polish stage: the finished cut on top of the reviewed draft.

Every fixture here is generated with ffmpeg's `lavfi` sources, including the
music — the repository's real bensound mp3s are read-only inputs to the project,
not test material, and a test that decoded them would be measuring a licensed
file instead of the stage.
"""
import json
import re
import subprocess
from pathlib import Path

import pytest

from videoai.config import Config, PolishSettings
from videoai.core.ffmpeg import probe
from videoai.core.models import (
    ClipInfo,
    DraftResult,
    FinalResult,
    Manifest,
    StoryPlan,
    Timeline,
    TimelineClip,
)
from videoai.core.registry import StageContext
from videoai.core.store import ArtifactStore
from videoai.logic.music import attribution_line, list_tracks, select_track
from videoai.stages.s06_render_draft import (
    DRAFT_AUDIO_CHANNELS,
    DRAFT_AUDIO_CODEC,
    DRAFT_AUDIO_SAMPLE_RATE,
    render_draft,
)
from videoai.stages.s08_polish import duck_ratio, polish, section_changes, write_attribution


def _probe_streams(path: Path) -> list[dict]:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", str(path)],
        capture_output=True, text=True, check=True,
    )
    return json.loads(result.stdout)["streams"]


def _mean_volume(path: Path) -> float:
    result = subprocess.run(
        ["ffmpeg", "-i", str(path), "-af", "volumedetect", "-f", "null", "-"],
        capture_output=True, text=True,
    )
    match = re.search(r"mean_volume:\s*(-?[\d.]+) dB", result.stderr)
    if not match:
        raise AssertionError(f"mean_volume not found in ffmpeg stderr:\n{result.stderr}")
    return float(match.group(1))


def _make_tone(path: Path, seconds: float = 5.0, hz: int = 660) -> Path:
    """A short generated tone standing in for a music file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "lavfi", "-i", f"sine=frequency={hz}:duration={seconds}",
        "-c:a", "libmp3lame", str(path),
    ], check=True)
    return path


def _context(tmp_path: Path, settings: PolishSettings | None = None) -> StageContext:
    for name in ("input", "work", "output"):
        (tmp_path / name).mkdir(exist_ok=True)
    return StageContext(
        project_dir=tmp_path,
        input_dir=tmp_path / "input",
        work_dir=tmp_path / "work",
        output_dir=tmp_path / "output",
        config=Config(polish=settings or PolishSettings()),
        store=ArtifactStore(tmp_path / "work"),
    )


def _seed(ctx: StageContext, source: Path, beats: list[str], title: str = "A Test Review") -> None:
    """A manifest, a timeline of 3-second clips carrying `beats`, and a story plan."""
    duration = 3.0 * len(beats) + 3.0
    ctx.store.write("01-manifest", Manifest(clips=[
        ClipInfo(clip_id="clip-01", path=str(source), duration=duration, width=320,
                 height=240, fps=30.0, has_audio=True, proxy_path=str(source)),
    ]), fingerprint="fp")
    ctx.store.write("05-timeline", Timeline(fps=30.0, width=320, height=240, clips=[
        TimelineClip(src="clip-01", offset=3.0 * index, dur=3.0, start=3.0 * index, beat=beat)
        for index, beat in enumerate(beats)
    ]), fingerprint="fp")
    ctx.store.write("05a-storyplan", StoryPlan(title=title), fingerprint="fp")


def _render(ctx: StageContext) -> DraftResult:
    draft = render_draft(ctx)
    ctx.store.write("06-draft", draft, fingerprint="fp")
    return draft


@pytest.fixture
def project(tmp_path: Path, make_clip):
    """A rendered draft plus a generated music library, ready to polish."""
    def _build(beats: list[str], *, music: bool = True, **overrides) -> tuple[StageContext, DraftResult]:
        music_dir = tmp_path / "music"
        if music:
            _make_tone(music_dir / "bensound-funday.mp3")
            _make_tone(music_dir / "bensound-energy.mp3", hz=440)
        overrides.setdefault("music_dir", str(music_dir))
        ctx = _context(tmp_path, PolishSettings(**overrides))
        source = make_clip("a.mp4", seconds=3.0 * len(beats) + 3.0)
        _seed(ctx, source, beats)
        return ctx, _render(ctx)

    return _build


def test_final_is_the_draft_plus_the_intro(project):
    """One beat, so no section dissolves eat into the length: the final is longer
    than the draft by the intro, less the dissolve that opens it."""
    ctx, draft = project(["Hook", "Hook"])

    result = polish(ctx)

    assert isinstance(result, FinalResult)
    output = Path(result.path)
    assert output.name == "final.mp4"
    assert output.parent == ctx.output_dir
    assert Path(draft.path).exists(), "the draft must be left alone"

    measured = probe(output)
    grown = measured.duration - draft.duration
    assert 1.9 < grown < 2.6, f"grew by {grown:.2f}s, expected about the 2.5s intro"
    assert result.intro is True
    assert result.intro_title == "A Test Review"


def test_final_decodes_cleanly(project):
    ctx, _ = project(["Hook", "Middle"])

    result = polish(ctx)

    decode = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", result.path, "-f", "null", "-"],
        capture_output=True, text=True,
    )
    assert decode.returncode == 0
    assert decode.stderr.strip() == "", decode.stderr


def test_final_keeps_the_drafts_single_video_and_audio_stream_layout(project):
    """The project's homogeneity invariant: one video stream, one audio stream,
    and the same codec, sample rate and channel count the draft settled on."""
    ctx, draft = project(["Hook", "Middle", "End"])

    result = polish(ctx)

    streams = _probe_streams(Path(result.path))
    video = [s for s in streams if s["codec_type"] == "video"]
    audio = [s for s in streams if s["codec_type"] == "audio"]
    assert len(video) == 1
    assert len(audio) == 1

    draft_audio = next(
        s for s in _probe_streams(Path(draft.path)) if s["codec_type"] == "audio"
    )
    assert audio[0]["codec_name"] == draft_audio["codec_name"] == DRAFT_AUDIO_CODEC
    assert int(audio[0]["sample_rate"]) == int(draft_audio["sample_rate"]) == DRAFT_AUDIO_SAMPLE_RATE
    assert int(audio[0]["channels"]) == int(draft_audio["channels"]) == DRAFT_AUDIO_CHANNELS


def test_disabled_polish_copies_the_draft_through_unchanged(project):
    ctx, draft = project(["Hook", "Middle"], enabled=False)

    result = polish(ctx)

    assert Path(result.path).read_bytes() == Path(draft.path).read_bytes()
    assert result.music_track is None
    assert result.title_count == 0
    assert any("enabled" in note for note in result.notes)


def test_a_missing_music_directory_still_renders_and_says_so(project, tmp_path: Path):
    ctx, draft = project(["Hook", "Middle"], music=False,
                         music_dir=str(tmp_path / "nowhere"))

    result = polish(ctx)

    assert Path(result.path).exists()
    assert probe(Path(result.path)).duration > draft.duration
    assert result.music_track is None
    assert result.music_attribution == ""
    assert any("no music" in note for note in result.notes)
    assert not (ctx.output_dir / "metadata.md").exists()


def test_an_empty_music_directory_still_renders_and_says_so(project, tmp_path: Path):
    empty = tmp_path / "empty-music"
    empty.mkdir()
    ctx, _ = project(["Hook", "Middle"], music=False, music_dir=str(empty))

    result = polish(ctx)

    assert Path(result.path).exists()
    assert result.music_track is None
    assert any("no music" in note for note in result.notes)


def test_the_same_project_always_selects_the_same_track(tmp_path: Path):
    """Selection must not depend on the process: `hash()` is salted per run and
    would hand the same project different music every time."""
    tracks = [Path(f"/music/bensound-{name}.mp3") for name in
              ("angelsbymyside", "dawnofchange", "moonlightdrive", "rhythmmagnet")]
    first = select_track(tracks, style="", project_name="1.Toy_Pimple_Popping")
    again = select_track(list(reversed(tracks)), style="", project_name="1.Toy_Pimple_Popping")

    assert first == again
    assert first in tracks
    # A different project may legitimately get a different track, but the same
    # name must always land in the same place.
    assert select_track(tracks, "", "1.Toy_Pimple_Popping") == first


def test_a_known_style_picks_its_track_and_an_unknown_one_falls_back(tmp_path: Path):
    tracks = [Path("/music/bensound-funday.mp3"), Path("/music/bensound-slowlife.mp3")]
    assert select_track(tracks, "playful", "project").name == "bensound-funday.mp3"
    assert select_track(tracks, "calm", "project").name == "bensound-slowlife.mp3"
    assert select_track(tracks, "no-such-style", "project") in tracks


def test_style_from_project_yaml_chooses_the_track(project, tmp_path: Path):
    ctx, _ = project(["Hook", "Middle"])
    (tmp_path / "project.yaml").write_text("style: energetic\n", encoding="utf-8")

    result = polish(ctx)

    assert result.music_track == "bensound-energy.mp3"


def test_an_explicit_track_overrides_the_automatic_choice(project):
    ctx, _ = project(["Hook", "Middle"], music_track="bensound-energy.mp3")

    result = polish(ctx)

    assert result.music_track == "bensound-energy.mp3"


def test_an_explicit_track_that_is_not_there_is_refused(project):
    ctx, _ = project(["Hook", "Middle"], music_track="bensound-nothing.mp3")

    with pytest.raises(RuntimeError, match="bensound-nothing.mp3"):
        polish(ctx)


def test_the_attribution_reaches_metadata_md(project):
    ctx, _ = project(["Hook", "Middle"])

    result = polish(ctx)

    metadata = (ctx.output_dir / "metadata.md").read_text(encoding="utf-8")
    assert "Bensound" in metadata
    assert result.music_track is not None
    assert result.music_track in metadata
    assert result.music_attribution in metadata


def test_the_attribution_is_appended_once_however_often_polish_runs(tmp_path: Path):
    output_dir = tmp_path / "output"
    line = attribution_line(Path("/music/bensound-funday.mp3"))

    assert write_attribution(output_dir, line) is True
    assert write_attribution(output_dir, line) is False

    metadata = (output_dir / "metadata.md").read_text(encoding="utf-8")
    assert metadata.count(line) == 1


def test_the_attribution_is_appended_to_an_existing_metadata_file(tmp_path: Path):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "metadata.md").write_text("# Notes\n\nSomething else.\n", encoding="utf-8")

    write_attribution(output_dir, attribution_line(Path("/music/bensound-funday.mp3")))

    metadata = (output_dir / "metadata.md").read_text(encoding="utf-8")
    assert "Something else." in metadata
    assert "Bensound" in metadata


def test_section_changes_become_titles_and_dissolves(project):
    """A title per beat change and a dissolve on each of those cuts — never on the
    cuts inside a section, where a straight cut on speech is what is wanted."""
    ctx, _ = project(["Hook", "Hook", "Middle", "Middle", "End"])

    result = polish(ctx)

    assert result.title_count == 2
    assert result.transition_count == 2


def test_the_music_bed_sits_far_below_the_speech(project):
    """Measured, not eyeballed: the polished mix must stay within a decibel or so
    of the speech-only draft. A bed that pushed the mix up would be a bed the
    viewer hears over the child."""
    ctx, draft = project(["Hook", "Middle"])

    result = polish(ctx)

    speech = _mean_volume(Path(draft.path))
    mixed = _mean_volume(Path(result.path))
    assert mixed < speech + 1.5, (
        f"mixed {mixed:.1f} dB against speech-only {speech:.1f} dB: the bed is audible"
    )


def test_section_changes_ignores_the_first_clip_and_repeated_beats():
    timeline = Timeline(fps=30.0, width=320, height=240, clips=[
        TimelineClip(src="clip-01", offset=0.0, dur=1.0, start=0.0, beat="Hook"),
        TimelineClip(src="clip-01", offset=1.0, dur=1.0, start=1.0, beat="Hook"),
        TimelineClip(src="clip-01", offset=2.0, dur=1.0, start=2.0, beat="Middle"),
        TimelineClip(src="clip-01", offset=3.0, dur=1.0, start=3.0, beat="Hook"),
    ])
    assert section_changes(timeline) == [2, 3]


def test_duck_ratio_grows_with_the_requested_attenuation():
    assert duck_ratio(0.0) == 1.0
    assert duck_ratio(-6.0) < duck_ratio(-12.0) < duck_ratio(-15.0)
    # ffmpeg refuses a ratio outside 1..20, so an extreme setting is clamped
    # rather than passed through into a filter that will not build.
    assert 1.0 <= duck_ratio(-400.0) <= 20.0


def test_list_tracks_ignores_non_audio_and_dotfiles(tmp_path: Path):
    music = tmp_path / "music"
    music.mkdir()
    (music / "bensound-funday.mp3").write_bytes(b"")
    (music / "readme.txt").write_text("not music", encoding="utf-8")
    (music / ".DS_Store").write_bytes(b"")

    assert [path.name for path in list_tracks(music)] == ["bensound-funday.mp3"]
