import json
from pathlib import Path

import pytest

from videoai.config import Config
from videoai.core.models import Manifest, Transcript, Word
from videoai.core.registry import StageContext
from videoai.core.store import ArtifactStore
from videoai.stages.s03_transcribe import derive_speech_spans, transcribe


def _context(tmp_path: Path) -> StageContext:
    for name in ("input", "work", "output"):
        (tmp_path / name).mkdir(exist_ok=True)
    return StageContext(
        project_dir=tmp_path,
        input_dir=tmp_path / "input",
        work_dir=tmp_path / "work",
        output_dir=tmp_path / "output",
        config=Config(providers={"asr": "mock", "llm": "mock"}),
        store=ArtifactStore(tmp_path / "work"),
    )


def _words(*triples: tuple[str, float, float]) -> list[dict]:
    return [{"text": t, "start": s, "end": e} for t, s, e in triples]


def test_derive_speech_spans_splits_on_long_gaps():
    words = [
        Word(text="a", start=0.0, end=0.4),
        Word(text="b", start=0.5, end=0.9),
        Word(text="c", start=2.5, end=2.9),
    ]
    spans = derive_speech_spans(words, gap=0.5)
    assert len(spans) == 2
    assert spans[0].start == 0.0 and spans[0].end == 0.9
    assert spans[1].start == 2.5 and spans[1].end == 2.9


def test_derive_speech_spans_returns_empty_for_no_words():
    assert derive_speech_spans([], gap=0.5) == []


def test_mock_provider_produces_transcript_artifact(tmp_path: Path, make_clip):
    ctx = _context(tmp_path)
    make_clip("a.mp4", seconds=2.0).rename(ctx.input_dir / "a.mp4")
    from videoai.stages.s01_ingest import ingest

    manifest: Manifest = ingest(ctx)
    ctx.store.write("01-manifest", manifest, fingerprint="fp")
    sidecar = Path(manifest.clips[0].audio_path).with_suffix(".words.json")
    sidecar.write_text(
        json.dumps(_words(("Look", 0.2, 0.5), ("here", 0.55, 0.9))), encoding="utf-8"
    )

    result = transcribe(ctx)

    assert isinstance(result, Transcript)
    assert result.provider == "mock"
    clip = result.by_id("clip-01")
    assert [w.text for w in clip.words] == ["Look", "here"]
    assert len(clip.speech_spans) == 1


def test_missing_sidecar_raises_clear_error(tmp_path: Path, make_clip):
    ctx = _context(tmp_path)
    make_clip("a.mp4", seconds=2.0).rename(ctx.input_dir / "a.mp4")
    from videoai.stages.s01_ingest import ingest

    ctx.store.write("01-manifest", ingest(ctx), fingerprint="fp")
    with pytest.raises(FileNotFoundError, match="words.json"):
        transcribe(ctx)


def test_clip_without_audio_yields_empty_transcript(tmp_path: Path):
    ctx = _context(tmp_path)
    import subprocess

    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", "testsrc=size=320x240:rate=30:duration=2",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", str(ctx.input_dir / "a.mp4"),
        ],
        check=True,
    )
    from videoai.stages.s01_ingest import ingest

    ctx.store.write("01-manifest", ingest(ctx), fingerprint="fp")

    result = transcribe(ctx)

    assert result.by_id("clip-01").words == []
