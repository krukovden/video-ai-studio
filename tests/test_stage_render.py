import json
import re
import subprocess
from pathlib import Path

import videoai.stages.s06_render_draft as render_draft_module
from videoai.config import Config
from videoai.core.ffmpeg import probe
from videoai.core.models import ClipInfo, DraftResult, Manifest, Timeline, TimelineClip
from videoai.core.registry import StageContext
from videoai.core.store import ArtifactStore
from videoai.stages.s06_render_draft import (
    DRAFT_AUDIO_CHANNELS,
    DRAFT_AUDIO_CODEC,
    DRAFT_AUDIO_SAMPLE_RATE,
    _audio_filter_chain,
    render_draft,
)


def _probe_streams(path: Path) -> list[dict]:
    """Full ffprobe stream listing (codec, sample rate, channels, ...), which the
    thin `videoai.core.ffmpeg.probe` wrapper doesn't expose."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", str(path)],
        capture_output=True, text=True, check=True,
    )
    return json.loads(result.stdout)["streams"]


def _mean_volume(path: Path) -> float:
    """Mean audio loudness in dB via ffmpeg's volumedetect filter, which writes
    its result to stderr rather than stdout."""
    result = subprocess.run(
        ["ffmpeg", "-i", str(path), "-af", "volumedetect", "-f", "null", "-"],
        capture_output=True, text=True,
    )
    match = re.search(r"mean_volume:\s*(-?[\d.]+) dB", result.stderr)
    if not match:
        raise AssertionError(f"mean_volume not found in ffmpeg stderr:\n{result.stderr}")
    return float(match.group(1))


def _context(tmp_path: Path) -> StageContext:
    for name in ("input", "work", "output"):
        (tmp_path / name).mkdir(exist_ok=True)
    return StageContext(
        project_dir=tmp_path,
        input_dir=tmp_path / "input",
        work_dir=tmp_path / "work",
        output_dir=tmp_path / "output",
        config=Config(),
        store=ArtifactStore(tmp_path / "work"),
    )


def test_draft_is_rendered_from_two_segments(tmp_path: Path, make_clip):
    ctx = _context(tmp_path)
    source = make_clip("a.mp4", seconds=6.0)
    ctx.store.write("01-manifest", Manifest(clips=[
        ClipInfo(clip_id="clip-01", path=str(source), duration=6.0, width=320,
                 height=240, fps=30.0, has_audio=True, proxy_path=str(source)),
    ]), fingerprint="fp")
    ctx.store.write("05-timeline", Timeline(fps=30.0, width=320, height=240, clips=[
        TimelineClip(src="clip-01", offset=0.5, dur=1.5, start=0.0),
        TimelineClip(src="clip-01", offset=3.0, dur=2.0, start=1.5),
    ]), fingerprint="fp")

    result = render_draft(ctx)

    assert isinstance(result, DraftResult)
    assert result.segment_count == 2
    output = Path(result.path)
    assert output.exists()
    assert 3.0 < probe(output).duration < 4.2


def test_draft_output_lands_in_output_directory(tmp_path: Path, make_clip):
    ctx = _context(tmp_path)
    source = make_clip("a.mp4", seconds=4.0)
    ctx.store.write("01-manifest", Manifest(clips=[
        ClipInfo(clip_id="clip-01", path=str(source), duration=4.0, width=320,
                 height=240, fps=30.0, has_audio=True, proxy_path=str(source)),
    ]), fingerprint="fp")
    ctx.store.write("05-timeline", Timeline(fps=30.0, width=320, height=240, clips=[
        TimelineClip(src="clip-01", offset=0.5, dur=1.0, start=0.0),
    ]), fingerprint="fp")

    result = render_draft(ctx)

    assert Path(result.path).parent == ctx.output_dir
    assert Path(result.path).name == "draft.mp4"


def test_empty_timeline_raises(tmp_path: Path, make_clip):
    ctx = _context(tmp_path)
    source = make_clip("a.mp4", seconds=2.0)
    ctx.store.write("01-manifest", Manifest(clips=[
        ClipInfo(clip_id="clip-01", path=str(source), duration=2.0, width=320,
                 height=240, fps=30.0, has_audio=True, proxy_path=str(source)),
    ]), fingerprint="fp")
    ctx.store.write("05-timeline", Timeline(fps=30.0, width=320, height=240, clips=[]),
                    fingerprint="fp")

    try:
        render_draft(ctx)
    except RuntimeError as error:
        assert "empty timeline" in str(error)
    else:
        raise AssertionError("expected RuntimeError")


def test_mixed_audio_and_silent_sources_render_with_one_uniform_audio_stream(
    tmp_path: Path, make_clip, make_silent_clip
):
    """A muted second camera is ordinary for this project's two-camera setups. A
    video-only segment placed next to an audio-bearing one breaks the concat
    demuxer's `-c copy` assumption that every segment has the same stream layout —
    this must not happen regardless of where the silent source lands in the cut."""
    ctx = _context(tmp_path)
    audio_source = make_clip("a.mp4", seconds=3.0)
    silent_source = make_silent_clip("b.mp4", seconds=3.0)
    ctx.store.write("01-manifest", Manifest(clips=[
        ClipInfo(clip_id="clip-01", path=str(audio_source), duration=3.0, width=320,
                 height=240, fps=30.0, has_audio=True, proxy_path=str(audio_source)),
        ClipInfo(clip_id="clip-02", path=str(silent_source), duration=3.0, width=320,
                 height=240, fps=30.0, has_audio=False, proxy_path=str(silent_source)),
    ]), fingerprint="fp")
    ctx.store.write("05-timeline", Timeline(fps=30.0, width=320, height=240, clips=[
        TimelineClip(src="clip-01", offset=0.2, dur=1.0, start=0.0),
        TimelineClip(src="clip-02", offset=0.2, dur=1.0, start=1.0),
    ]), fingerprint="fp")

    result = render_draft(ctx)

    output = Path(result.path)
    streams = _probe_streams(output)
    video_streams = [s for s in streams if s["codec_type"] == "video"]
    audio_streams = [s for s in streams if s["codec_type"] == "audio"]
    assert len(video_streams) == 1
    assert len(audio_streams) == 1
    assert 1.8 < probe(output).duration < 2.6

    segment_dir = ctx.work_dir / "segments"
    segment_files = sorted(segment_dir.glob("seg-*.mp4"))
    assert len(segment_files) == 2
    segment_audio = []
    for segment_file in segment_files:
        audio = [s for s in _probe_streams(segment_file) if s["codec_type"] == "audio"]
        assert len(audio) == 1, f"{segment_file.name} must carry exactly one audio stream"
        segment_audio.append(audio[0])

    # The actual invariant the fix protects: identical codec, sample rate and
    # channel count across every segment, audio-bearing or synthesised-silent.
    assert len({a["codec_name"] for a in segment_audio}) == 1
    assert len({a["sample_rate"] for a in segment_audio}) == 1
    assert len({a["channels"] for a in segment_audio}) == 1


def test_all_silent_sources_still_render_with_an_audio_stream(tmp_path: Path, make_silent_clip):
    ctx = _context(tmp_path)
    source = make_silent_clip("a.mp4", seconds=3.0)
    ctx.store.write("01-manifest", Manifest(clips=[
        ClipInfo(clip_id="clip-01", path=str(source), duration=3.0, width=320,
                 height=240, fps=30.0, has_audio=False, proxy_path=str(source)),
    ]), fingerprint="fp")
    ctx.store.write("05-timeline", Timeline(fps=30.0, width=320, height=240, clips=[
        TimelineClip(src="clip-01", offset=0.2, dur=1.0, start=0.0),
    ]), fingerprint="fp")

    result = render_draft(ctx)

    streams = _probe_streams(Path(result.path))
    audio_streams = [s for s in streams if s["codec_type"] == "audio"]
    assert len(audio_streams) == 1


def test_stereo_48k_source_is_forced_to_the_draft_audio_target(
    tmp_path: Path, make_stereo_clip, make_silent_clip
):
    """This project's real footage is AAC stereo at 48 kHz (from an iPhone), unlike
    `make_clip`'s mono 44100 Hz fixture, which happens to already match the draft's
    audio target and would hide a bug where the audio-bearing branch just inherits
    the source's sample rate/channels instead of being forced to the same values
    the silent branch uses. Without both branches forcing an explicit `-ar`/`-ac`,
    a stereo segment and a synthesised-mono segment would violate the homogeneity
    the concat demuxer's `-c copy` step depends on."""
    ctx = _context(tmp_path)
    stereo_source = make_stereo_clip("a.mp4", seconds=3.0)
    silent_source = make_silent_clip("b.mp4", seconds=3.0)
    ctx.store.write("01-manifest", Manifest(clips=[
        ClipInfo(clip_id="clip-01", path=str(stereo_source), duration=3.0, width=320,
                 height=240, fps=30.0, has_audio=True, proxy_path=str(stereo_source)),
        ClipInfo(clip_id="clip-02", path=str(silent_source), duration=3.0, width=320,
                 height=240, fps=30.0, has_audio=False, proxy_path=str(silent_source)),
    ]), fingerprint="fp")
    ctx.store.write("05-timeline", Timeline(fps=30.0, width=320, height=240, clips=[
        TimelineClip(src="clip-01", offset=0.2, dur=1.0, start=0.0),
        TimelineClip(src="clip-02", offset=0.2, dur=1.0, start=1.0),
    ]), fingerprint="fp")

    result = render_draft(ctx)

    output = Path(result.path)
    streams = _probe_streams(output)
    audio_streams = [s for s in streams if s["codec_type"] == "audio"]
    assert len(audio_streams) == 1
    assert int(audio_streams[0]["sample_rate"]) == DRAFT_AUDIO_SAMPLE_RATE
    assert int(audio_streams[0]["channels"]) == DRAFT_AUDIO_CHANNELS

    # Assert on the SEGMENT files, not just the concatenated output: a mismatch
    # between segments is the actual failure mode the concat demuxer can paper
    # over (or, per the reviewer's report, can silently produce a corrupt output
    # for) — checking only the merged draft can hide it.
    segment_dir = ctx.work_dir / "segments"
    segment_files = sorted(segment_dir.glob("seg-*.mp4"))
    assert len(segment_files) == 2
    segment_audio = []
    for segment_file in segment_files:
        audio = [s for s in _probe_streams(segment_file) if s["codec_type"] == "audio"]
        assert len(audio) == 1, f"{segment_file.name} must carry exactly one audio stream"
        segment_audio.append(audio[0])

    assert {a["codec_name"] for a in segment_audio} == {DRAFT_AUDIO_CODEC}
    assert {int(a["sample_rate"]) for a in segment_audio} == {DRAFT_AUDIO_SAMPLE_RATE}
    assert {int(a["channels"]) for a in segment_audio} == {DRAFT_AUDIO_CHANNELS}


def test_audio_filter_chain_has_no_volume_filter_when_gain_is_zero():
    chain = _audio_filter_chain(fade=0.03, duration=1.5, gain_db=0.0)
    assert "volume" not in chain
    assert "afade=t=in" in chain
    assert "afade=t=out" in chain


def test_audio_filter_chain_adds_volume_filter_for_nonzero_gain():
    chain = _audio_filter_chain(fade=0.03, duration=1.5, gain_db=-6.0)
    assert "volume=-6.0dB" in chain
    # The fades must still apply at both edges alongside the added gain.
    assert "afade=t=in" in chain
    assert "afade=t=out" in chain


def test_render_draft_emits_no_volume_filter_arg_when_clip_gain_is_zero(
    tmp_path: Path, make_clip, monkeypatch
):
    ctx = _context(tmp_path)
    source = make_clip("a.mp4", seconds=3.0)
    ctx.store.write("01-manifest", Manifest(clips=[
        ClipInfo(clip_id="clip-01", path=str(source), duration=3.0, width=320,
                 height=240, fps=30.0, has_audio=True, proxy_path=str(source)),
    ]), fingerprint="fp")
    ctx.store.write("05-timeline", Timeline(fps=30.0, width=320, height=240, clips=[
        TimelineClip(src="clip-01", offset=0.2, dur=1.0, start=0.0, gain_db=0.0),
    ]), fingerprint="fp")

    captured_args: list[list[str]] = []
    original_run_ffmpeg = render_draft_module.run_ffmpeg

    def _capture(args: list[str]) -> None:
        captured_args.append(args)
        original_run_ffmpeg(args)

    monkeypatch.setattr(render_draft_module, "run_ffmpeg", _capture)

    render_draft(ctx)

    af_values = [args[args.index("-af") + 1] for args in captured_args if "-af" in args]
    assert af_values, "expected at least one -af argument to have been passed to ffmpeg"
    assert all("volume" not in value for value in af_values)


def test_render_draft_emits_volume_filter_arg_when_clip_gain_is_nonzero(
    tmp_path: Path, make_clip, monkeypatch
):
    ctx = _context(tmp_path)
    source = make_clip("a.mp4", seconds=3.0)
    ctx.store.write("01-manifest", Manifest(clips=[
        ClipInfo(clip_id="clip-01", path=str(source), duration=3.0, width=320,
                 height=240, fps=30.0, has_audio=True, proxy_path=str(source)),
    ]), fingerprint="fp")
    ctx.store.write("05-timeline", Timeline(fps=30.0, width=320, height=240, clips=[
        TimelineClip(src="clip-01", offset=0.2, dur=1.0, start=0.0, gain_db=-6.0),
    ]), fingerprint="fp")

    captured_args: list[list[str]] = []
    original_run_ffmpeg = render_draft_module.run_ffmpeg

    def _capture(args: list[str]) -> None:
        captured_args.append(args)
        original_run_ffmpeg(args)

    monkeypatch.setattr(render_draft_module, "run_ffmpeg", _capture)

    render_draft(ctx)

    af_values = [args[args.index("-af") + 1] for args in captured_args if "-af" in args]
    assert any("volume=-6.0dB" in value for value in af_values)


def test_negative_gain_produces_a_measurably_quieter_rendered_segment(
    tmp_path: Path, make_clip
):
    ctx = _context(tmp_path)
    source = make_clip("a.mp4", seconds=3.0, tone_hz=440)
    ctx.store.write("01-manifest", Manifest(clips=[
        ClipInfo(clip_id="clip-01", path=str(source), duration=3.0, width=320,
                 height=240, fps=30.0, has_audio=True, proxy_path=str(source)),
    ]), fingerprint="fp")
    ctx.store.write("05-timeline", Timeline(fps=30.0, width=320, height=240, clips=[
        TimelineClip(src="clip-01", offset=0.2, dur=1.0, start=0.0, gain_db=0.0),
        TimelineClip(src="clip-01", offset=0.2, dur=1.0, start=1.0, gain_db=-12.0),
    ]), fingerprint="fp")

    render_draft(ctx)

    segment_dir = ctx.work_dir / "segments"
    segment_files = sorted(segment_dir.glob("seg-*.mp4"))
    assert len(segment_files) == 2
    baseline = _mean_volume(segment_files[0])
    attenuated = _mean_volume(segment_files[1])
    # A comfortable margin below the true ~12 dB drop, to absorb fade/measurement
    # noise without weakening the assertion that gain actually attenuated audio.
    assert attenuated < baseline - 6.0


def test_gain_leaves_audio_codec_sample_rate_and_channels_unchanged(
    tmp_path: Path, make_stereo_clip
):
    """A gain filter must not alter the segment's codec, sample rate or channel
    count: concat's `-c copy` step depends on every segment sharing them."""
    ctx = _context(tmp_path)
    source = make_stereo_clip("a.mp4", seconds=3.0)
    ctx.store.write("01-manifest", Manifest(clips=[
        ClipInfo(clip_id="clip-01", path=str(source), duration=3.0, width=320,
                 height=240, fps=30.0, has_audio=True, proxy_path=str(source)),
    ]), fingerprint="fp")
    ctx.store.write("05-timeline", Timeline(fps=30.0, width=320, height=240, clips=[
        TimelineClip(src="clip-01", offset=0.2, dur=1.0, start=0.0, gain_db=-6.0),
    ]), fingerprint="fp")

    result = render_draft(ctx)

    streams = _probe_streams(Path(result.path))
    audio_streams = [s for s in streams if s["codec_type"] == "audio"]
    assert len(audio_streams) == 1
    assert audio_streams[0]["codec_name"] == DRAFT_AUDIO_CODEC
    assert int(audio_streams[0]["sample_rate"]) == DRAFT_AUDIO_SAMPLE_RATE
    assert int(audio_streams[0]["channels"]) == DRAFT_AUDIO_CHANNELS
