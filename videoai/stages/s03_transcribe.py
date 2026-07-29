"""s03 transcribe: word-level timings plus speech spans derived from word gaps."""
from __future__ import annotations

from pathlib import Path

from videoai.core.models import ClipTranscript, Manifest, SpeechSpan, SyncMap, Transcript, Word
from videoai.core.registry import StageContext, stage
from videoai.providers.base import resolve_asr


def derive_speech_spans(words: list[Word], gap: float) -> list[SpeechSpan]:
    """Contiguous speech regions: split wherever the pause between words exceeds `gap`."""
    if not words:
        return []
    spans: list[SpeechSpan] = []
    start = words[0].start
    previous_end = words[0].end
    for word in words[1:]:
        if word.start - previous_end > gap:
            spans.append(SpeechSpan(start=start, end=previous_end))
            start = word.start
        previous_end = word.end
    spans.append(SpeechSpan(start=start, end=previous_end))
    return spans


@stage(
    id="transcribe",
    produces="03-transcript",
    requires=("01-manifest", "01b-sync"),
    provider_key="asr",
    model=Transcript,
)
def transcribe(ctx: StageContext) -> Transcript:
    manifest = ctx.store.read("01-manifest", Manifest)
    sync_map = ctx.store.read("01b-sync", SyncMap)
    provider = resolve_asr(ctx.config.providers["asr"])
    gap = ctx.config.transcribe.phrase_gap_seconds

    clips: list[ClipTranscript] = []
    for clip in manifest.clips:
        if clip.camera != sync_map.primary_camera or not clip.audio_path:
            clips.append(ClipTranscript(clip_id=clip.clip_id))
            continue
        words = provider.transcribe(Path(clip.audio_path))
        clips.append(
            ClipTranscript(
                clip_id=clip.clip_id,
                words=words,
                speech_spans=derive_speech_spans(words, gap),
            )
        )
    return Transcript(provider=provider.name, clips=clips)
