"""The polish stage: the delivered cut on top of the reviewed draft.

There is one renderer and `production-contract.yaml` always applies to it, so
every test that renders writes a contract next to the project describing the
delivery it expects. That is the same file the repository ships, resized: a test
that quietly relaxed the rules would be testing a pipeline nobody runs.

Every fixture here is generated with ffmpeg's `lavfi` sources, including the
music — the repository's real bensound mp3s are read-only inputs to the project,
not test material, and a test that decoded them would be measuring a licensed
file instead of the stage.
"""
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from videoai.config import Config, PolishSettings
from videoai.core.ffmpeg import probe
from videoai.core.models import (
    ClipInfo,
    ClipTranscript,
    DraftResult,
    FinalResult,
    Manifest,
    StoryPlan,
    Timeline,
    TimelineClip,
    Transcript,
    Word,
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
from videoai.stages.s08_polish import (
    _Caption,
    _conform_intro_clip,
    _cut_delivery_segment,
    clamp_caption_ends,
    delivery_frame,
    duck_ratio,
    polish,
    section_changes,
    write_attribution,
)

CLOSING_BEAT = "Closing"


def _write_contract(
    project_dir: Path,
    frame: tuple[int, int],
    *,
    minimum_duck_db: float = 3.0,
    captions: bool = True,
) -> None:
    """The shipped contract, resized to the delivery this project can produce."""
    width, height = frame
    (project_dir / "production-contract.yaml").write_text(
        f"""
version: 1
required_output:
  width: {width}
  height: {height}
  full_decode: true
required_features:
  intro: true
  outro: true
  section_titles: true
  captions: {str(captions).lower()}
  transitions: true
  music: true
  music_ducking:
    minimum_db: {minimum_duck_db}
  closing_beat: true
quality:
  source: originals
  maximum_lossy_video_generations: 1
failure_policy:
  fallback_output_name: preview-fallback.mp4
""".lstrip(),
        encoding="utf-8",
    )


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


def _mean_brightness(path: Path, at: float) -> float:
    """The average luma of the frame at `at` seconds, 0-255."""
    import numpy as np

    result = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", f"{at:.3f}", "-i", str(path),
         "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "gray", "-"],
        capture_output=True, check=True,
    )
    return float(np.frombuffer(result.stdout, dtype=np.uint8).mean())


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


def _seed(
    ctx: StageContext, source: Path, beats: list[str], title: str = "A Test Review",
    proxy: Path | None = None, clip_seconds: float = 3.0,
) -> None:
    """A manifest, a timeline carrying `beats`, a story plan and a transcript.

    The transcript is not decoration: the contract requires captions, and captions
    are mapped from word timings, so a project with no words cannot be delivered.

    `proxy` must be a real file — the draft is cut from it — and must not be the
    source, because the delivery contract fails a final cut from a proxy.
    """
    duration = clip_seconds * len(beats) + clip_seconds
    ctx.store.write("01-manifest", Manifest(clips=[
        ClipInfo(clip_id="clip-01", path=str(source), duration=duration, width=320,
                 height=240, fps=30.0, has_audio=True,
                 proxy_path=str(proxy or source)),
    ]), fingerprint="fp")
    timeline = Timeline(fps=30.0, width=320, height=240, clips=[
        TimelineClip(
            ref=f"clip-01#{index:03d}",
            src="clip-01", offset=clip_seconds * index, dur=clip_seconds,
            start=clip_seconds * index, beat=beat,
        )
        for index, beat in enumerate(beats)
    ])
    # The proposal and the edit are the same until a creator says otherwise, but
    # they are separate artifacts: a test that seeded only one would not notice a
    # stage reading the wrong one.
    ctx.store.write("05-proposal", timeline, fingerprint="fp")
    ctx.store.write("05-timeline", timeline, fingerprint="fp")
    ctx.store.write("05a-storyplan", StoryPlan(title=title), fingerprint="fp")
    words = [
        Word(
            text=f"word{index}",
            start=clip_seconds * index + 0.2,
            end=clip_seconds * index + 0.2 + min(0.4, clip_seconds / 4),
        )
        for index in range(len(beats))
    ]
    ctx.store.write(
        "03-transcript",
        Transcript(provider="test", clips=[
            ClipTranscript(clip_id="clip-01", words=words),
        ]),
        fingerprint="fp",
    )


def _render(ctx: StageContext) -> DraftResult:
    draft = render_draft(ctx)
    ctx.store.write("06-draft", draft, fingerprint="fp")
    return draft


@pytest.fixture
def project(tmp_path: Path, make_clip):
    """A rendered draft plus a generated music library, ready to polish."""
    def _build(beats: list[str], *, music: bool = True, contract: bool = True,
               **overrides) -> tuple[StageContext, DraftResult]:
        music_dir = tmp_path / "music"
        if music:
            _make_tone(music_dir / "bensound-funday.mp3")
            _make_tone(music_dir / "bensound-energy.mp3", hz=440)
        overrides.setdefault("music_dir", str(music_dir))
        overrides.setdefault("output_height", 240)
        ctx = _context(tmp_path, PolishSettings(**overrides))
        source = make_clip("a.mp4", seconds=3.0 * len(beats) + 3.0)
        # A distinct proxy file, as ingest always builds: the delivery report
        # records which of the two each segment was cut from.
        proxy = Path(shutil.copyfile(source, tmp_path / "a-proxy.mp4"))
        _seed(ctx, source, beats, proxy=proxy)
        if contract:
            _write_contract(
                tmp_path, delivery_frame(probe(source), ctx.config.polish.output_height)
            )
        return ctx, _render(ctx)

    return _build


def _scale_to(source: Path, dst: Path, height: int) -> Path:
    """A proxy of `source` the way ingest builds one: scaled down and re-encoded."""
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-i", str(source),
        "-vf", f"scale=-2:{height}", "-c:v", "libx264", "-preset", "veryfast",
        "-crf", "28", "-c:a", "aac", "-b:a", "128k", str(dst),
    ], check=True)
    return dst


def _turned(source: Path, dst: Path, degrees: int) -> Path:
    """A copy carrying a display matrix, the way an iPhone records a portrait clip:
    landscape frames plus a quarter turn, with nothing rotated in the pixels."""
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-display_rotation", str(degrees),
        "-i", str(source), "-c", "copy", str(dst),
    ], check=True)
    return dst


def _video_bitrate(path: Path) -> float:
    """The picture's bitrate, not the file's. Both files carry the same 160 kbit/s
    of audio, which in a 180p draft is most of the container's bitrate and would
    swamp the very difference being measured."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=bit_rate", "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(result.stdout.strip())


def _make_detailed_clip(path: Path, seconds: float, size: str) -> Path:
    """`testsrc2` rather than the shared `make_clip`'s `testsrc`: it carries fine
    detail and real motion, so scaling it to proxy height genuinely destroys
    picture the way it does with the creator's footage."""
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "lavfi", "-i", f"testsrc2=size={size}:rate=30:duration={seconds}",
        "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "12", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-shortest", str(path),
    ], check=True)
    return path


@pytest.fixture
def delivery(tmp_path: Path):
    """A project whose original is big and whose proxy is small.

    That gap is the whole point: while the manifest's `path` and `proxy_path`
    point at the same file, a final built from the proxy and a final built from
    the source are indistinguishable, and the defect this stage was rewritten to
    fix would pass every assertion.
    """
    def _build(
        beats: list[str], *, size: str = "1280x720", proxy_height: int = 180,
        source: Path | None = None, clip_seconds: float = 3.0,
        music: bool = True, contract: bool = True, **overrides,
    ) -> tuple[StageContext, Path, Path, DraftResult]:
        music_dir = tmp_path / "music"
        if music:
            _make_tone(music_dir / "bensound-funday.mp3", seconds=2.0)
        overrides.setdefault("music_dir", str(music_dir))
        overrides.setdefault("output_height", 720)
        ctx = _context(tmp_path, PolishSettings(**overrides))
        original = source or _make_detailed_clip(
            tmp_path / "original.mp4", clip_seconds * len(beats) + clip_seconds, size
        )
        proxy = _scale_to(original, tmp_path / "proxy.mp4", proxy_height)
        _seed(ctx, original, beats, proxy=proxy, clip_seconds=clip_seconds)
        if contract:
            _write_contract(
                tmp_path,
                delivery_frame(probe(original), ctx.config.polish.output_height),
            )
        return ctx, original, proxy, _render(ctx)

    return _build


# --- delivery quality -------------------------------------------------------


def test_the_final_is_cut_at_delivery_height_while_the_draft_stays_at_proxy_height(delivery):
    ctx, _, proxy, draft = delivery(["Hook", CLOSING_BEAT])

    result = polish(ctx)

    assert probe(proxy).height == 180
    assert probe(Path(draft.path)).height == 180, "the review draft must stay cheap"
    final = probe(Path(result.path))
    assert (final.width, final.height) == (1280, 720)
    assert (result.width, result.height) == (1280, 720)


def test_the_final_is_materially_better_than_the_draft(delivery):
    """Resolution and bitrate, not file size: the final is longer than the draft by
    the cards, so a bigger file on its own would prove nothing."""
    ctx, _, _, draft = delivery(["Hook", CLOSING_BEAT])

    result = polish(ctx)

    draft_height = probe(Path(draft.path)).height
    final_height = probe(Path(result.path)).height
    assert final_height >= draft_height * 4

    draft_bitrate = _video_bitrate(Path(draft.path))
    final_bitrate = _video_bitrate(Path(result.path))
    assert final_bitrate > draft_bitrate * 8, (
        f"final {final_bitrate / 1e6:.2f} Mbit/s against draft "
        f"{draft_bitrate / 1e6:.2f} Mbit/s: the final is still proxy-grade"
    )


def test_the_delivered_audio_still_matches_the_draft(delivery):
    """The concat homogeneity invariant: raising the picture and mixing a bed under
    it must not move the audio target every segment already agrees on."""
    ctx, _, _, draft = delivery(["Hook", CLOSING_BEAT])

    result = polish(ctx)

    final_audio = next(
        s for s in _probe_streams(Path(result.path)) if s["codec_type"] == "audio"
    )
    draft_audio = next(
        s for s in _probe_streams(Path(draft.path)) if s["codec_type"] == "audio"
    )
    assert final_audio["codec_name"] == draft_audio["codec_name"] == DRAFT_AUDIO_CODEC
    assert int(final_audio["sample_rate"]) == DRAFT_AUDIO_SAMPLE_RATE
    assert int(draft_audio["sample_rate"]) == DRAFT_AUDIO_SAMPLE_RATE
    assert int(final_audio["channels"]) == int(draft_audio["channels"]) == DRAFT_AUDIO_CHANNELS


def test_output_height_2160_delivers_true_4k(delivery):
    ctx, _, _, _ = delivery(
        ["Hook", CLOSING_BEAT], size="3840x2160", proxy_height=540, output_height=2160,
        clip_seconds=1.0, intro_seconds=0.4, outro_seconds=0.4, title_seconds=0.4,
    )

    result = polish(ctx)

    assert probe(Path(result.path)).height == 2160
    assert result.height == 2160


def test_lossy_video_generations_is_measured_not_a_constant(project):
    """The report's `quality.lossy_video_generations` used to be a literal `1`.
    It has to come from actually checking each delivery-path encode's own
    arguments: today that means every intermediate is `-crf 0` and only the
    final encode is not, so the measured count still lands on 1."""
    ctx, _ = project(["Hook", CLOSING_BEAT])

    result = polish(ctx)

    report = json.loads(Path(result.production_report).read_text())
    assert report["quality"]["lossy_video_generations"] == 1


def test_a_future_lossy_intermediate_would_fail_the_contract(project, monkeypatch):
    """Proof the count above is really measured: if a later change made every
    delivery-path encode lossy (not just the final one), the contract's
    `maximum_lossy_video_generations: 1` must catch it instead of the report
    still claiming 1."""
    ctx, _ = project(["Hook", CLOSING_BEAT])
    monkeypatch.setattr(
        "videoai.stages.s08_polish._is_lossless_x264_encode", lambda args: False
    )

    with pytest.raises(RuntimeError, match="lossy video generations"):
        polish(ctx)

    assert not (ctx.output_dir / "final.mp4").exists()
    assert not (ctx.output_dir / "final.srt").exists()
    assert not (ctx.output_dir / "production-report.json").exists()


def test_a_missing_original_is_refused_by_name_instead_of_falling_back_to_the_proxy(delivery):
    """Silently delivering the proxy is the defect. A source that has moved has to
    stop the stage and say which file it is."""
    ctx, original, proxy, _ = delivery(["Hook", CLOSING_BEAT])
    original.unlink()
    assert proxy.is_file(), "the proxy is still there, and must not be used"

    with pytest.raises(RuntimeError, match="original.mp4"):
        polish(ctx)

    assert not (ctx.output_dir / "final.mp4").exists()


def test_the_report_names_the_files_each_segment_was_really_cut_from(delivery):
    ctx, original, proxy, _ = delivery(["Hook", CLOSING_BEAT])

    result = polish(ctx)

    report = json.loads(Path(result.production_report).read_text(encoding="utf-8"))
    assert report["quality"]["source"] == "originals"
    assert report["quality"]["proxy_inputs"] == []
    assert report["quality"]["segment_inputs"] == [str(original.resolve())] * 2
    assert str(proxy.resolve()) not in report["quality"]["segment_inputs"]


def test_a_proxy_that_reaches_the_delivery_cut_fails_the_contract(delivery, tmp_path: Path):
    """The contract's `quality.source` has to be measured. Pointing the manifest's
    `path` at the proxy is exactly the regression it exists to catch."""
    ctx, original, proxy, _ = delivery(["Hook", CLOSING_BEAT], output_height=180)
    manifest = ctx.store.read("01-manifest", Manifest)
    ctx.store.write(
        "01-manifest",
        Manifest(clips=[manifest.clips[0].model_copy(update={"path": str(proxy)})]),
        fingerprint="fp",
    )
    _write_contract(tmp_path, delivery_frame(probe(proxy), 180))

    with pytest.raises(RuntimeError, match="cut from proxies"):
        polish(ctx)

    assert not (ctx.output_dir / "final.mp4").exists()


def test_a_turned_original_is_delivered_upright(delivery, tmp_path: Path, make_clip):
    """iPhone portrait footage is landscape frames plus a display matrix. The
    proxies had the turn baked in by their scale operation; the originals have
    not, so the final has to come out of the source upright rather than sideways."""
    landscape = make_clip("landscape.mov", seconds=9.0, size="640x480")
    turned = _turned(landscape, tmp_path / "turned.mov", 90)
    assert probe(turned).rotation == 90
    ctx, _, _, _ = delivery(
        ["Hook", CLOSING_BEAT], source=turned, output_height=240,
    )

    result = polish(ctx)

    final = probe(Path(result.path))
    assert (final.width, final.height) == (180, 240), "delivered sideways"
    # ffmpeg's autorotation resolves the turn into the pixels, so the finished
    # file carries no display matrix a player could apply a second time.
    assert final.rotation == 0


# --- what the delivery contains --------------------------------------------


def test_the_final_is_the_draft_plus_the_two_cards(project):
    ctx, draft = project(["Hook", CLOSING_BEAT])

    result = polish(ctx)

    assert isinstance(result, FinalResult)
    output = Path(result.path)
    assert output.name == "final.mp4"
    assert output.parent == ctx.output_dir
    assert Path(draft.path).exists(), "the draft must be left alone"

    measured = probe(output)
    grown = measured.duration - draft.duration
    assert 4.4 < grown < 5.6, f"grew by {grown:.2f}s, expected the 2.5s+2.5s cards"
    assert result.intro is True
    assert result.outro is True
    assert result.intro_title == "A Test Review"


def test_final_decodes_cleanly_and_says_so(project):
    ctx, _ = project(["Hook", CLOSING_BEAT])

    result = polish(ctx)

    decode = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", result.path, "-f", "null", "-"],
        capture_output=True, text=True,
    )
    assert decode.returncode == 0
    assert decode.stderr.strip() == "", decode.stderr
    assert result.fully_decoded is True
    report = json.loads(Path(result.production_report).read_text(encoding="utf-8"))
    assert report["features"]["full_decode"] is True


def test_final_keeps_the_drafts_single_video_and_audio_stream_layout(project):
    """The project's homogeneity invariant: one video stream, one audio stream,
    and the same codec, sample rate and channel count the draft settled on."""
    ctx, draft = project(["Hook", "Middle", CLOSING_BEAT])

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


def test_every_section_change_gets_a_title_and_an_emitted_transition(project):
    """A title per beat change, and a transition count that is the number of fade
    filters actually emitted rather than the number of boundaries planned."""
    ctx, _ = project(["Hook", "Hook", "Middle", "Middle", CLOSING_BEAT])

    result = polish(ctx)

    assert result.title_count == 2
    report = json.loads(Path(result.production_report).read_text(encoding="utf-8"))
    assert report["features"]["section_titles"] == 2
    assert report["features"]["section_boundaries"] == 2
    # Five segments: the first fades in, the last fades out, and each of the two
    # boundaries fades out of one segment and into the next.
    assert result.transition_count == 6
    assert report["features"]["transitions"] == 6


def test_a_transition_length_of_zero_frames_fails_the_contract(project):
    """`transitions: true` means transitions were rendered. Zero emitted filters
    with the feature required is a production failure, not a silent 'passed'."""
    ctx, _ = project(["Hook", CLOSING_BEAT], transition_frames=0)

    with pytest.raises(RuntimeError, match="required feature missing: transitions"):
        polish(ctx)

    assert not (ctx.output_dir / "final.mp4").exists()


# --- preflight: fail in seconds, not after a 4K render ----------------------


def _delivery_segments(ctx: StageContext) -> list[Path]:
    return sorted((ctx.work_dir / "delivery").glob("segment-*.mp4"))


def test_a_delivery_the_contract_cannot_accept_is_refused_before_any_cutting(
    project, tmp_path: Path
):
    """The contract wants 1080p and the config asks for 240p. Discovering that
    after cutting every segment costs an hour; discovering it here costs a probe."""
    ctx, _ = project(["Hook", CLOSING_BEAT], contract=False)
    _write_contract(tmp_path, (1920, 1080))

    with pytest.raises(RuntimeError, match="preflight failed"):
        polish(ctx)

    assert _delivery_segments(ctx) == [], "preflight ran after the cutting started"
    assert not (ctx.output_dir / "final.mp4").exists()


def test_preflight_names_the_setting_and_the_rule_that_disagree(project, tmp_path: Path):
    ctx, _ = project(["Hook", CLOSING_BEAT], contract=False)
    _write_contract(tmp_path, (1920, 1080))

    with pytest.raises(RuntimeError) as failure:
        polish(ctx)

    message = str(failure.value)
    assert "1080px-tall" in message
    assert "polish.output_height=240" in message


def test_a_timeline_without_a_closing_beat_is_refused_before_any_cutting(project):
    ctx, _ = project(["Hook", "Middle"])

    with pytest.raises(RuntimeError, match="closing story beat"):
        polish(ctx)

    assert _delivery_segments(ctx) == []


def test_a_full_disk_is_refused_before_any_cutting(project, monkeypatch):
    ctx, _ = project(["Hook", CLOSING_BEAT])
    monkeypatch.setattr("videoai.stages.s08_polish.free_bytes", lambda _: 4_000_000)

    with pytest.raises(RuntimeError, match="scratch space"):
        polish(ctx)

    assert _delivery_segments(ctx) == []


def test_a_missing_music_library_is_refused_before_any_cutting(project, tmp_path: Path):
    ctx, _ = project(["Hook", CLOSING_BEAT], music=False,
                     music_dir=str(tmp_path / "nowhere"))

    with pytest.raises(RuntimeError, match="requires background music"):
        polish(ctx)

    assert _delivery_segments(ctx) == []


# --- music -----------------------------------------------------------------


def test_a_missing_music_directory_is_refused_and_names_the_directory(project, tmp_path: Path):
    """The contract requires a music bed, so a library that is not there is a
    production failure. Delivering silently without music is the degradation the
    contract exists to forbid."""
    ctx, _ = project(["Hook", CLOSING_BEAT], music=False,
                     music_dir=str(tmp_path / "nowhere"))

    with pytest.raises(RuntimeError, match="nowhere"):
        polish(ctx)

    assert not (ctx.output_dir / "final.mp4").exists()
    assert not (ctx.output_dir / "metadata.md").exists()


def test_an_empty_music_directory_is_refused(project, tmp_path: Path):
    empty = tmp_path / "empty-music"
    empty.mkdir()
    ctx, _ = project(["Hook", CLOSING_BEAT], music=False, music_dir=str(empty))

    with pytest.raises(RuntimeError, match="music"):
        polish(ctx)

    assert not (ctx.output_dir / "final.mp4").exists()


def test_the_same_project_always_selects_the_same_track():
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


def test_a_known_style_picks_its_track_and_an_unknown_one_falls_back():
    tracks = [Path("/music/bensound-funday.mp3"), Path("/music/bensound-slowlife.mp3")]
    assert select_track(tracks, "playful", "project").name == "bensound-funday.mp3"
    assert select_track(tracks, "calm", "project").name == "bensound-slowlife.mp3"
    assert select_track(tracks, "no-such-style", "project") in tracks


def test_style_from_project_yaml_chooses_the_track(project, tmp_path: Path):
    ctx, _ = project(["Hook", CLOSING_BEAT])
    (tmp_path / "project.yaml").write_text("style: energetic\n", encoding="utf-8")

    result = polish(ctx)

    assert result.music_track == "bensound-energy.mp3"


def test_an_explicit_track_overrides_the_automatic_choice(project):
    ctx, _ = project(["Hook", CLOSING_BEAT], music_track="bensound-energy.mp3")

    result = polish(ctx)

    assert result.music_track == "bensound-energy.mp3"


def test_an_explicit_track_that_is_not_there_is_refused(project):
    ctx, _ = project(["Hook", CLOSING_BEAT], music_track="bensound-nothing.mp3")

    with pytest.raises(RuntimeError, match="bensound-nothing.mp3"):
        polish(ctx)


def test_the_attribution_reaches_metadata_md(project):
    ctx, _ = project(["Hook", CLOSING_BEAT])

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


def test_the_music_bed_sits_far_below_the_speech(project):
    """Measured, not eyeballed: the polished mix must stay within a decibel or so
    of the speech-only draft. A bed that pushed the mix up would be a bed the
    viewer hears over the child."""
    ctx, draft = project(["Hook", CLOSING_BEAT])

    result = polish(ctx)

    speech = _mean_volume(Path(draft.path))
    mixed = _mean_volume(Path(result.path))
    assert mixed < speech + 1.5, (
        f"mixed {mixed:.1f} dB against speech-only {speech:.1f} dB: the bed is audible"
    )


def test_the_duck_is_measured_from_the_delivered_mix_and_recorded_in_db(project):
    """`music_ducking: true` used to be a literal in the report. It is now the
    measured difference between the bed with and without the sidechain, over the
    speech windows of this very render."""
    ctx, _ = project(["Hook", CLOSING_BEAT])

    result = polish(ctx)

    report = json.loads(Path(result.production_report).read_text(encoding="utf-8"))
    measured = report["features"]["music_ducking"]
    assert isinstance(measured, (int, float))
    assert measured >= 3.0, f"the bed only dropped {measured} dB under speech"
    assert result.music_ducking_db == pytest.approx(measured)
    assert any("dB down under speech" in note for note in result.notes)


def test_the_duck_lands_on_the_requested_attenuation_rather_than_a_fixed_setting(
    project,
):
    """A fixed sidechain threshold encodes an assumption about the recording level.
    Two different requests against the same material must produce two different
    measured ducks, or `polish.music_duck_db` is decoration."""
    ctx, _ = project(["Hook", CLOSING_BEAT], music_duck_db=-6.0)
    gentle = polish(ctx).music_ducking_db

    ctx, _ = project(["Hook", CLOSING_BEAT], music_duck_db=-18.0)
    deep = polish(ctx).music_ducking_db

    assert gentle == pytest.approx(6.0, abs=1.5), gentle
    assert deep > gentle + 4.0, f"{deep} dB is not deeper than {gentle} dB"


def test_the_reported_duck_is_measured_from_the_bed_that_reaches_the_mix(project):
    """The search renders several candidate beds. The figure in the report has to
    belong to the one left on disk, not to whichever attempt happened to be last."""
    from videoai.stages.s08_polish import (
        _windowed_rms_db,
        build_captions,
        clamp_caption_ends,
        cumulative_starts,
        speech_windows,
    )

    ctx, _ = project(["Hook", CLOSING_BEAT], music_duck_db=-9.0)

    result = polish(ctx)

    delivery = ctx.work_dir / "delivery"
    timeline = ctx.store.read("05-timeline", Timeline)
    transcript = ctx.store.read("03-transcript", Transcript)
    durations = [probe(path).duration for path in sorted(delivery.glob("segment-*.mp4"))]
    captions = clamp_caption_ends(build_captions(
        timeline, transcript, cumulative_starts(durations), [], 0.0,
        probe(delivery / "intro.mp4").duration, 4,
    ))
    windows = speech_windows(captions)
    on_disk = (
        _windowed_rms_db(delivery / "music-bed.wav", windows)
        - _windowed_rms_db(delivery / "music-bed-ducked.wav", windows)
    )

    assert result.music_ducking_db == pytest.approx(on_disk, abs=0.05)


def test_a_duck_below_the_contracts_floor_fails_and_leaves_no_delivery(
    project, tmp_path: Path
):
    ctx, _ = project(["Hook", CLOSING_BEAT], contract=False)
    _write_contract(tmp_path, (320, 240), minimum_duck_db=200.0)

    with pytest.raises(RuntimeError, match="music_ducking measured"):
        polish(ctx)

    assert not (ctx.output_dir / "final.mp4").exists()
    assert not (ctx.output_dir / "final.srt").exists()
    assert not (ctx.output_dir / "production-report.json").exists()


def test_a_contract_failure_removes_a_stale_report_and_srt(project, tmp_path: Path):
    """`output/` must never describe a delivery that is not there."""
    ctx, _ = project(["Hook", CLOSING_BEAT], contract=False)
    _write_contract(tmp_path, (320, 240), minimum_duck_db=200.0)
    (ctx.output_dir / "production-report.json").write_text(
        '{"status": "passed"}\n', encoding="utf-8"
    )
    (ctx.output_dir / "final.srt").write_text("1\n", encoding="utf-8")
    (ctx.output_dir / "final.mp4").write_bytes(b"stale")

    with pytest.raises(RuntimeError):
        polish(ctx)

    assert not (ctx.output_dir / "production-report.json").exists()
    assert not (ctx.output_dir / "final.srt").exists()
    assert not (ctx.output_dir / "final.mp4").exists()


def test_a_failure_after_the_srt_write_leaves_no_delivery_output(project, monkeypatch):
    """final.srt is written before every later step that can still raise
    (section titles, effect overlays, the graphics track, duck measurement, the
    final encode, validation). A crash among those must not leave a freshly
    written final.srt beside a stale final.mp4/production-report.json from an
    earlier, successful run."""
    ctx, _ = project(["Hook", CLOSING_BEAT])
    (ctx.output_dir / "production-report.json").write_text(
        '{"status": "passed"}\n', encoding="utf-8"
    )
    (ctx.output_dir / "final.mp4").write_bytes(b"stale")

    def _boom(*args, **kwargs):
        raise RuntimeError("boom: simulated failure after the srt write")

    monkeypatch.setattr("videoai.stages.s08_polish.build_section_titles", _boom)

    with pytest.raises(RuntimeError, match="boom"):
        polish(ctx)

    assert not (ctx.output_dir / "final.srt").exists()
    assert not (ctx.output_dir / "final.mp4").exists()
    assert not (ctx.output_dir / "production-report.json").exists()


# --- the creator's running order --------------------------------------------


def _reorder(ctx: StageContext, refs: list[str], off: list[str] = ()) -> None:
    """Put `refs` in that order through the creator's own artifact and reassemble.

    The whole path, not a hand-written timeline: `assemble` is what turns the
    planner's proposal into the edit, and the point of the test is that polish
    delivers what assemble produced.
    """
    from videoai.core.models import ClipOverride, Overrides
    from videoai.logic.decisions import OVERRIDES_ARTIFACT
    from videoai.stages.s05_plan import assemble

    ctx.store.write(
        OVERRIDES_ARTIFACT,
        Overrides(clips=[
            *(ClipOverride(ref=ref) for ref in refs),
            *(ClipOverride(ref=ref, enabled=False) for ref in off),
        ]),
        fingerprint="creator",
    )
    ctx.store.write("05-timeline", assemble(ctx), fingerprint="fp")


def test_a_reordered_edit_is_delivered_in_the_creator_s_order(project, tmp_path: Path):
    """Section titles, transitions, captions and the closing beat are all derived
    from clip order, so all of them have to be recomputed rather than carried."""
    ctx, _ = project(["Hook", "Filling", CLOSING_BEAT])
    _reorder(ctx, ["clip-01#001", "clip-01#000", "clip-01#002"])
    _render(ctx)

    result = polish(ctx)

    report = json.loads(Path(result.production_report).read_text(encoding="utf-8"))
    beats = [clip.beat for clip in ctx.store.read("05-timeline", Timeline).clips]
    assert beats == ["Filling", "Hook", CLOSING_BEAT]
    # One title per beat change, and the first clip never gets one.
    assert report["features"]["section_titles"] == len(beats) - 1
    assert report["features"]["closing_beat"] is True
    assert report["features"]["transitions"] > 0


def test_disabling_the_sign_off_is_caught_by_the_contract(project):
    """A reorder may legitimately change everything derived from clip order —
    including whether the video still ends. Switching the last shot off must not
    quietly deliver a video with no closing beat."""
    ctx, _ = project(["Hook", "Filling", CLOSING_BEAT])
    _reorder(ctx, ["clip-01#000", "clip-01#001"], off=["clip-01#002"])

    with pytest.raises(RuntimeError, match="closing story beat"):
        polish(ctx)


def test_an_accent_is_delivered_over_the_shot_it_was_planned_for(project, tmp_path: Path):
    """The middle shot moves to the front; its accent goes with it. Matching on
    absolute time would have left the badge over whatever slid underneath."""
    from videoai.core.models import EffectEvent, EffectPlan

    ctx, _ = project(["Hook", "Filling", CLOSING_BEAT])
    ctx.store.write("05d-effects", EffectPlan(library="", events=[
        EffectEvent(clip_ref="clip-01#001", at_in_clip=1.0, at_seconds=4.0,
                    effect_name="comic_starburst", screen_position="center",
                    scale="small", seconds=0.8, reason="the syringe goes in"),
    ]), fingerprint="fp")
    _reorder(ctx, ["clip-01#001", "clip-01#000", "clip-01#002"])
    _render(ctx)

    result = polish(ctx)

    report = json.loads(Path(result.production_report).read_text(encoding="utf-8"))
    assert report["effects"]["applied"] == 1
    # The shot now opens the video, so the accent is a second into the edit
    # rather than four.
    assert report["effects"]["events"][0]["at_seconds"] == pytest.approx(1.0, abs=0.05)
    assert result.effects[0].startswith("comic_starburst at ")


# --- the branded intro clip -------------------------------------------------


def _make_branded_clip(
    path: Path, seconds: float = 1.5, size: str = "320x240", rate: int = 30,
    colour: str = "white", tone_hz: int = 880, channels: int = 1,
    sample_rate: int = 44100,
) -> Path:
    """A stand-in for the channel's own intro: a flat bright frame and a tone.

    Flat and bright on purpose. The generated title card is a dark gradient, so a
    single luma sample says which of the two the delivery actually opens with —
    without depending on anything the card happens to draw.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "lavfi", "-i", f"color=c={colour}:s={size}:r={rate}:d={seconds}",
        "-f", "lavfi", "-i",
        f"sine=frequency={tone_hz}:sample_rate={sample_rate}:duration={seconds}",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-ac", str(channels),
        "-shortest", str(path),
    ], check=True)
    return path


def _with_intro_clip(ctx: StageContext, clip: Path | str) -> StageContext:
    """The same project, delivered behind `clip`."""
    from dataclasses import replace

    return replace(
        ctx,
        config=ctx.config.model_copy(
            update={"polish": ctx.config.polish.model_copy(
                update={"intro_clip": str(clip)}
            )}
        ),
    )


def _graphics_calls(monkeypatch) -> list[dict]:
    """Record what each graphics-track render was asked to draw, and when.

    Section titles and cartoon accents are not timestamped anywhere in the
    delivered package, so this is where their delivery-time placement can be read:
    the overlays handed to the renderer are exactly the ones composited over the
    picture.
    """
    from videoai.stages import s08_polish

    calls: list[dict] = []
    real = s08_polish._render_graphics_track

    def spy(path, frame, fps, duration, captions, titles, work_dir, effects=()):
        calls.append({
            "duration": duration,
            "titles": [overlay.start for overlay in titles],
            "effects": [overlay.start for overlay in effects],
        })
        return real(path, frame, fps, duration, captions, titles, work_dir, effects)

    monkeypatch.setattr(s08_polish, "_render_graphics_track", spy)
    return calls


def _cue_starts(srt: Path) -> list[float]:
    stamps = re.findall(
        r"(\d\d):(\d\d):(\d\d),(\d\d\d) -->", srt.read_text(encoding="utf-8")
    )
    return [
        int(hours) * 3600 + int(minutes) * 60 + int(whole) + int(milli) / 1000
        for hours, minutes, whole, milli in stamps
    ]


def _seed_accent(ctx: StageContext) -> None:
    from videoai.core.models import EffectEvent, EffectPlan

    ctx.store.write("05d-effects", EffectPlan(library="", events=[
        EffectEvent(clip_ref="clip-01#001", at_in_clip=1.0, at_seconds=4.0,
                    effect_name="comic_starburst", screen_position="center",
                    scale="small", seconds=0.8, reason="the syringe goes in"),
    ]), fingerprint="fp")


def test_no_intro_clip_delivers_exactly_what_it_did_before(project):
    """The default. `polish.intro_clip` empty has to mean the title card alone:
    no extra part in the picture master, and nothing added to the report."""
    ctx, draft = project(["Hook", CLOSING_BEAT])

    result = polish(ctx)

    assert result.intro_clip == ""
    assert result.intro_clip_seconds == 0.0
    assert not (ctx.work_dir / "delivery" / "intro-clip.mp4").exists()
    report = json.loads(Path(result.production_report).read_text(encoding="utf-8"))
    assert "intro_clip" not in report["quality"]
    grown = probe(Path(result.path)).duration - draft.duration
    assert 4.4 < grown < 5.6, f"grew by {grown:.2f}s, expected the 2.5s+2.5s cards"


def test_the_branded_clip_opens_the_delivery_and_lengthens_it_by_its_own_duration(
    project, tmp_path: Path
):
    """The clip plays first, ahead of the generated title card, and the delivery
    is longer than the same delivery without it by exactly the clip's length."""
    ctx, _ = project(["Hook", CLOSING_BEAT], intro_seconds=1.2, outro_seconds=0.6,
                     title_seconds=0.5)

    plain = probe(Path(polish(ctx).path)).duration

    clip = _make_branded_clip(tmp_path / "brand.mp4", seconds=1.5)
    result = polish(_with_intro_clip(ctx, clip))

    branded = probe(Path(result.path))
    assert result.intro_clip == str(clip)
    assert result.intro_clip_seconds == pytest.approx(1.5, abs=0.05)
    assert branded.duration - plain == pytest.approx(
        result.intro_clip_seconds, abs=0.1
    ), f"{branded.duration:.2f}s against {plain:.2f}s without the clip"
    # Half a second in is inside the clip; the title card follows it, and its own
    # fade from black is over by then.
    inside_clip = _mean_brightness(Path(result.path), 0.5)
    inside_card = _mean_brightness(Path(result.path), result.intro_clip_seconds + 0.6)
    assert inside_clip > 200, f"the delivery does not open on the clip ({inside_clip:.0f})"
    assert inside_clip > inside_card + 60, (
        f"clip {inside_clip:.0f} against card {inside_card:.0f}: the title card is "
        "still first"
    )
    report = json.loads(Path(result.production_report).read_text(encoding="utf-8"))
    # The intro is neither an original nor a proxy, so `source: originals` cannot
    # speak for it and the report has to name it separately.
    assert report["quality"]["source"] == "originals"
    assert report["quality"]["intro_clip"] == {
        "path": str(clip.resolve()),
        "duration": pytest.approx(result.intro_clip_seconds, abs=0.001),
        "shoot_footage": False,
    }
    assert any("branded intro" in note for note in result.notes)


def test_a_branded_intro_moves_every_caption_title_and_accent_with_it(
    project, tmp_path: Path, monkeypatch
):
    """The failure this feature can have that nobody notices: the clip is added to
    the picture and not to the offset every overlay is placed on, so the whole
    graphics layer runs early by the clip's length — and the result still plays,
    still decodes and still passes the contract.

    Measured as a difference: the same project delivered with and without the
    clip, cue for cue, title for title, accent for accent.
    """
    ctx, _ = project(["Hook", "Filling", CLOSING_BEAT], intro_seconds=1.2,
                     outro_seconds=0.6, title_seconds=0.5)
    _seed_accent(ctx)
    calls = _graphics_calls(monkeypatch)

    polish(ctx)
    plain_cues = _cue_starts(ctx.output_dir / "final.srt")

    clip = _make_branded_clip(tmp_path / "brand.mp4", seconds=1.5)
    result = polish(_with_intro_clip(ctx, clip))
    branded_cues = _cue_starts(ctx.output_dir / "final.srt")

    shift = result.intro_clip_seconds
    assert shift == pytest.approx(1.5, abs=0.05)
    plain_graphics, branded_graphics = calls
    assert plain_cues and plain_graphics["titles"] and plain_graphics["effects"]
    assert branded_cues == pytest.approx([cue + shift for cue in plain_cues], abs=0.01)
    assert branded_graphics["titles"] == pytest.approx(
        [start + shift for start in plain_graphics["titles"]], abs=0.01
    )
    assert branded_graphics["effects"] == pytest.approx(
        [start + shift for start in plain_graphics["effects"]], abs=0.01
    )
    # The alpha track has to cover the longer picture too, or the last accent of a
    # long video would fall off the end of it.
    assert branded_graphics["duration"] == pytest.approx(
        plain_graphics["duration"] + shift, abs=0.1
    )


def test_a_clip_that_disagrees_on_geometry_rate_and_channels_is_conformed(
    project, tmp_path: Path
):
    """The picture master is a raw stream copy, which does not check that its
    parts agree. A 640x480 25 fps stereo clip in front of a 320x240 30 fps mono
    delivery has to come out as one homogeneous file that decodes end to end —
    and through a lossless intermediate, so the delivery encode stays the only
    lossy generation."""
    ctx, _ = project(["Hook", CLOSING_BEAT], intro_seconds=0.6, outro_seconds=0.6,
                     title_seconds=0.5)
    clip = _make_branded_clip(
        tmp_path / "brand.mp4", seconds=2.0, size="640x480", rate=25,
        channels=2, sample_rate=48000,
    )

    result = polish(_with_intro_clip(ctx, clip))

    assert result.fully_decoded is True
    streams = _probe_streams(Path(result.path))
    video = [stream for stream in streams if stream["codec_type"] == "video"]
    audio = [stream for stream in streams if stream["codec_type"] == "audio"]
    assert len(video) == len(audio) == 1
    assert (int(video[0]["width"]), int(video[0]["height"])) == (320, 240)
    assert int(audio[0]["sample_rate"]) == DRAFT_AUDIO_SAMPLE_RATE
    assert int(audio[0]["channels"]) == DRAFT_AUDIO_CHANNELS
    report = json.loads(Path(result.production_report).read_text(encoding="utf-8"))
    assert report["quality"]["lossy_video_generations"] == 1
    # 6s of segments, the two cards and the clip.
    assert probe(Path(result.path)).duration == pytest.approx(9.2, abs=0.3)


def test_a_missing_intro_clip_is_refused_before_any_cutting_and_names_it(
    project, tmp_path: Path
):
    """In seconds and by name, not after the segments have been cut."""
    missing = tmp_path / "assets" / "artem-intro-opus.mp4"
    ctx, _ = project(["Hook", CLOSING_BEAT])

    with pytest.raises(RuntimeError) as failure:
        polish(_with_intro_clip(ctx, missing))

    message = str(failure.value)
    assert "preflight failed" in message
    assert "polish.intro_clip" in message
    assert str(missing) in message
    assert _delivery_segments(ctx) == [], "preflight ran after the cutting started"
    assert not (ctx.output_dir / "final.mp4").exists()


def test_an_unreadable_intro_clip_is_refused_before_any_cutting_and_names_it(
    project, tmp_path: Path
):
    """A file that is there but is not video. The concat would have accepted it
    as a part and produced a delivery nobody could play."""
    broken = tmp_path / "not-really-a-video.mp4"
    broken.write_text("this is not an mp4\n", encoding="utf-8")
    ctx, _ = project(["Hook", CLOSING_BEAT])

    with pytest.raises(RuntimeError) as failure:
        polish(_with_intro_clip(ctx, broken))

    message = str(failure.value)
    assert "preflight failed" in message
    assert str(broken) in message
    assert _delivery_segments(ctx) == []


def test_a_silent_branded_clip_is_conformed_with_a_silent_track(
    tmp_path: Path, make_silent_clip
):
    """A logo sting with no sound is ordinary artwork. Concatenated as it is, it
    would leave the picture master with less audio than picture from its first
    second, so the conform has to synthesise the silence — the same way a muted
    camera's segment does."""
    clip = make_silent_clip("brand-silent.mp4", seconds=1.0, size="640x360")
    conformed = tmp_path / "conformed.mp4"
    lossless: list[bool] = []

    _conform_intro_clip(
        clip, 1.0, (320, 240), 30.0, 0.03, probe(clip).has_audio, conformed,
        lossless_sink=lossless,
    )

    assert lossless == [True], "the intermediate must not be a lossy generation"
    streams = _probe_streams(conformed)
    video = next(stream for stream in streams if stream["codec_type"] == "video")
    audio = next(stream for stream in streams if stream["codec_type"] == "audio")
    # 16:9 inside a 4:3 delivery frame: scaled down, then padded, never cropped.
    assert (int(video["width"]), int(video["height"])) == (320, 240)
    assert video["r_frame_rate"] == "30/1"
    assert int(audio["sample_rate"]) == DRAFT_AUDIO_SAMPLE_RATE
    assert int(audio["channels"]) == DRAFT_AUDIO_CHANNELS
    assert probe(conformed).duration == pytest.approx(1.0, abs=0.05)


def test_an_intro_clip_is_found_relative_to_the_project(project, tmp_path: Path):
    """`assets/artem-intro-opus.mp4` in a config is a path relative to the project,
    not to whatever directory the CLI happened to be run from."""
    clip = _make_branded_clip(tmp_path / "assets" / "brand.mp4", seconds=1.0)
    ctx, _ = project(["Hook", CLOSING_BEAT], intro_seconds=0.6, outro_seconds=0.6,
                     title_seconds=0.5)

    result = polish(_with_intro_clip(ctx, Path("assets") / clip.name))

    assert Path(result.intro_clip).resolve() == clip.resolve()
    assert result.intro_clip_seconds == pytest.approx(1.0, abs=0.05)


# --- degradation is never called final --------------------------------------


def test_disabled_polish_writes_a_named_fallback_and_no_final(project):
    ctx, draft = project(["Hook", CLOSING_BEAT], enabled=False)

    result = polish(ctx)

    fallback = ctx.output_dir / "preview-fallback.mp4"
    assert Path(result.path) == fallback
    assert fallback.read_bytes() == Path(draft.path).read_bytes()
    assert not (ctx.output_dir / "final.mp4").exists()
    assert result.music_track is None
    assert result.title_count == 0
    assert result.production_report == ""
    assert result.fully_decoded is False
    assert any("not a delivery" in note for note in result.notes)
    assert any("missing:" in note for note in result.notes)


def test_disabled_polish_removes_a_final_left_by_an_earlier_run(project):
    ctx, _ = project(["Hook", CLOSING_BEAT], enabled=False)
    (ctx.output_dir / "final.mp4").write_bytes(b"a delivery from before")
    (ctx.output_dir / "final.srt").write_text("1\n", encoding="utf-8")

    polish(ctx)

    assert not (ctx.output_dir / "final.mp4").exists()
    assert not (ctx.output_dir / "final.srt").exists()


# --- captions ---------------------------------------------------------------


def test_captions_disabled_writes_no_srt_and_clears_a_stale_one(project):
    ctx, _ = project(["Hook", CLOSING_BEAT], captions_enabled=False, contract=False)
    _write_contract(ctx.project_dir, (320, 240), captions=False)
    (ctx.output_dir / "final.srt").write_text("1\nstale\n", encoding="utf-8")

    result = polish(ctx)

    assert not (ctx.output_dir / "final.srt").exists()
    assert result.caption_count == 0
    report = json.loads(Path(result.production_report).read_text(encoding="utf-8"))
    assert report["features"]["captions"] == 0


def test_the_delivered_srt_has_no_overlapping_cues(project):
    ctx, _ = project(["Hook", "Middle", CLOSING_BEAT], caption_words=1)

    result = polish(ctx)

    text = (ctx.output_dir / "final.srt").read_text(encoding="utf-8")
    stamps = re.findall(
        r"(\d\d):(\d\d):(\d\d),(\d\d\d) --> (\d\d):(\d\d):(\d\d),(\d\d\d)", text
    )
    assert stamps, text

    def seconds(parts: tuple[str, ...]) -> float:
        hours, minutes, whole, milli = (int(part) for part in parts)
        return hours * 3600 + minutes * 60 + whole + milli / 1000

    cues = [(seconds(row[:4]), seconds(row[4:])) for row in stamps]
    assert all(end > start for start, end in cues)
    overlaps = [
        (cues[index], cues[index + 1])
        for index in range(len(cues) - 1)
        if cues[index][1] > cues[index + 1][0]
    ]
    assert overlaps == [], f"{len(overlaps)} overlapping cues"
    assert result.caption_count == len(cues)


# --- captions, transitions and ducking as units -----------------------------


def test_clamp_caption_ends_stops_a_cue_at_the_next_cues_start():
    captions = [
        _Caption(start=0.0, end=2.0, text="one"),
        _Caption(start=1.5, end=3.0, text="two"),
        _Caption(start=2.9, end=4.0, text="three"),
    ]

    clamped = clamp_caption_ends(captions)

    assert [(caption.start, caption.end) for caption in clamped] == [
        (0.0, 1.5), (1.5, 2.9), (2.9, 4.0),
    ]


def test_clamp_caption_ends_drops_a_cue_squeezed_below_the_readable_minimum():
    captions = [
        _Caption(start=0.0, end=2.0, text="one"),
        _Caption(start=0.05, end=3.0, text="two"),
        _Caption(start=3.0, end=4.0, text="three"),
    ]

    clamped = clamp_caption_ends(captions)

    assert [caption.text for caption in clamped] == ["two", "three"]


def test_clamp_caption_ends_leaves_non_overlapping_cues_alone():
    captions = [
        _Caption(start=0.0, end=1.0, text="one"),
        _Caption(start=2.0, end=3.0, text="two"),
    ]

    assert clamp_caption_ends(captions) == captions


def test_a_transition_longer_than_the_segment_is_clamped_to_a_third_of_it(
    tmp_path: Path, make_clip
):
    """An unclamped fade would consume the whole segment and deliver black."""
    source = make_clip("clip.mp4", seconds=2.0)
    target = tmp_path / "segment.mp4"

    emitted = _cut_delivery_segment(
        source,
        TimelineClip(src="clip-01", offset=0.0, dur=1.5, start=0.0, beat="Hook"),
        (320, 240), 30.0, 0.03, True,
        transition=10.0, fade_in=True, fade_out=True, dst=target,
    )

    assert emitted == 2
    assert probe(target).duration == pytest.approx(1.5, abs=0.1)
    # The middle of the segment is between the two fades, so it must be at full
    # strength rather than faded to black.
    assert _mean_brightness(target, at=0.75) > 20, "the middle of the segment is black"
    assert _mean_brightness(target, at=0.02) < _mean_brightness(target, at=0.75)


def test_no_fade_is_emitted_when_neither_edge_needs_one(tmp_path: Path, make_clip):
    source = make_clip("clip.mp4", seconds=2.0)

    emitted = _cut_delivery_segment(
        source,
        TimelineClip(src="clip-01", offset=0.0, dur=1.0, start=0.0, beat="Hook"),
        (320, 240), 30.0, 0.03, True,
        transition=0.25, fade_in=False, fade_out=False, dst=tmp_path / "segment.mp4",
    )

    assert emitted == 0


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
