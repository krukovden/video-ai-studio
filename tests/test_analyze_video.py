"""Handing the footage itself to a provider that can watch it.

Everything upstream judges delivery from a transcript and a still. A provider
that reads video gets the spoken spans cut into one reel instead — and a
provider that cannot must keep getting stills, so switching models never
silently changes what the analysis was made from.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from videoai.config import AnalyzeSettings, Config
from videoai.core.models import (
    ClipInfo,
    ClipTranscript,
    Manifest,
    QualityReport,
    Transcript,
    Word,
)
from videoai.core.registry import StageContext
from videoai.core.store import ArtifactStore
from videoai.logic.reel import ReelEntry, ReelSpan
from videoai.stages.s04_analyze import analyze, build_reel, reel_prompt_section


def _clip(path: Path, seconds: float = 8.0) -> Path:
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "lavfi", "-i", f"testsrc=size=320x240:rate=30:duration={seconds}",
         "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
         str(path)],
        check=True,
    )
    return path


def _context(tmp_path: Path, config: Config, source: Path) -> StageContext:
    work = tmp_path / "work"
    store = ArtifactStore(work)
    store.write(
        "01-manifest",
        Manifest(clips=[ClipInfo(
            clip_id="clip-01", path=str(source), proxy_path=str(source), duration=8.0,
            width=320, height=240, fps=30.0, has_audio=True)]),
        fingerprint="m",
    )
    store.write("02-quality", QualityReport(), fingerprint="q")
    store.write(
        "03-transcript",
        Transcript(provider="test", clips=[ClipTranscript(clip_id="clip-01", words=[
            Word(text="hello", start=1.0, end=1.4),
            Word(text="everyone", start=1.5, end=2.0),
            Word(text="look", start=5.0, end=5.3),
            Word(text="here", start=5.4, end=5.8),
        ])]),
        fingerprint="t",
    )
    return StageContext(
        project_dir=tmp_path, input_dir=tmp_path, work_dir=work,
        output_dir=tmp_path / "output", config=config, store=store,
    )


def test_the_reel_holds_only_the_spoken_spans(tmp_path: Path):
    source = _clip(tmp_path / "clip-01.mp4")
    spans = [
        ReelSpan("clip-01", 1.0, 2.0, [ReelEntry("clip-01#001", 0.0, 1.0)]),
        ReelSpan("clip-01", 5.0, 5.8, [ReelEntry("clip-01#002", 1.0, 1.8)]),
    ]
    manifest = Manifest(clips=[ClipInfo(
        clip_id="clip-01", path=str(source), proxy_path=str(source), duration=8.0,
        width=320, height=240, fps=30.0, has_audio=True)])

    reel = build_reel(manifest, spans, tmp_path / "work" / "reel.mp4")

    assert reel.is_file()
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(reel)],
        capture_output=True, text=True, check=True,
    )
    # 1.8 seconds of speech out of an 8-second source, give or take a frame.
    assert abs(float(probe.stdout.strip()) - 1.8) < 0.35


def test_the_prompt_tells_the_model_where_each_phrase_sits():
    spans = [ReelSpan("clip-01", 1.0, 2.0, [ReelEntry("clip-01#001", 0.0, 1.0)])]
    section = reel_prompt_section(spans)
    assert "clip-01#001" in section
    assert "0.00s-1.00s" in section
    # It must be unmistakable that this is a reel, not the original footage.
    assert "reel" in section.lower()


def test_a_video_provider_is_given_the_reel(tmp_path: Path, monkeypatch):
    source = _clip(tmp_path / "clip-01.mp4")
    seen: dict = {}

    class _VideoLLM:
        name = "fake_video"
        reads_video = True

        def complete_json(self, prompt, images, timeout, videos=None):
            seen["videos"] = list(videos or [])
            seen["images"] = list(images)
            seen["prompt"] = prompt
            return {"segments": [
                {"phrase_id": "clip-01#001", "delivery_score": 8, "visual_score": 7},
                {"phrase_id": "clip-01#002", "delivery_score": 6, "visual_score": 6},
            ]}

    monkeypatch.setattr(
        "videoai.stages.s04_analyze.resolve_llm", lambda *a, **k: _VideoLLM()
    )
    config = Config(analyze=AnalyzeSettings(submit_video=True))
    result = analyze(_context(tmp_path, config, source))

    assert len(seen["videos"]) == 1 and seen["videos"][0].is_file()
    # Stills are redundant once the model can watch the footage.
    assert seen["images"] == []
    assert "clip-01#001" in seen["prompt"]
    assert result.segments[0].delivery_score == 8


def test_a_still_only_provider_still_gets_stills(tmp_path: Path, monkeypatch):
    source = _clip(tmp_path / "clip-01.mp4")
    seen: dict = {}

    class _StillsLLM:
        name = "fake_stills"
        reads_video = False

        def complete_json(self, prompt, images, timeout, videos=None):
            seen["videos"] = list(videos or [])
            seen["images"] = list(images)
            return {"segments": [
                {"phrase_id": "clip-01#001", "delivery_score": 5, "visual_score": 5},
                {"phrase_id": "clip-01#002", "delivery_score": 5, "visual_score": 5},
            ]}

    monkeypatch.setattr(
        "videoai.stages.s04_analyze.resolve_llm", lambda *a, **k: _StillsLLM()
    )
    config = Config(analyze=AnalyzeSettings(submit_video=True))
    analyze(_context(tmp_path, config, source))

    assert seen["videos"] == []
    assert seen["images"], "a provider that cannot watch video must still get frames"


def test_submitting_video_can_be_switched_off(tmp_path: Path, monkeypatch):
    """Video is metered; a creator must be able to say no without changing model."""
    source = _clip(tmp_path / "clip-01.mp4")
    seen: dict = {}

    class _VideoLLM:
        name = "fake_video"
        reads_video = True

        def complete_json(self, prompt, images, timeout, videos=None):
            seen["videos"] = list(videos or [])
            return {"segments": [
                {"phrase_id": "clip-01#001", "delivery_score": 5, "visual_score": 5},
                {"phrase_id": "clip-01#002", "delivery_score": 5, "visual_score": 5},
            ]}

    monkeypatch.setattr(
        "videoai.stages.s04_analyze.resolve_llm", lambda *a, **k: _VideoLLM()
    )
    config = Config(analyze=AnalyzeSettings(submit_video=False))
    analyze(_context(tmp_path, config, source))

    assert seen["videos"] == []


def test_the_submitted_seconds_are_recorded(tmp_path: Path, monkeypatch):
    """What was paid for has to be visible afterwards, not inferred from a bill."""
    source = _clip(tmp_path / "clip-01.mp4")

    class _VideoLLM:
        name = "fake_video"
        reads_video = True

        def complete_json(self, prompt, images, timeout, videos=None):
            return {"segments": [
                {"phrase_id": "clip-01#001", "delivery_score": 5, "visual_score": 5},
                {"phrase_id": "clip-01#002", "delivery_score": 5, "visual_score": 5},
            ]}

    monkeypatch.setattr(
        "videoai.stages.s04_analyze.resolve_llm", lambda *a, **k: _VideoLLM()
    )
    result = analyze(
        _context(tmp_path, Config(analyze=AnalyzeSettings(submit_video=True)), source)
    )
    assert result.video_seconds > 0
    assert result.video_seconds < 8.0  # not the whole source
