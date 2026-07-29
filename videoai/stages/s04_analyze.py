"""s04 analyze: score every phrase for delivery, visuals and shorts potential."""
from __future__ import annotations

from pathlib import Path

from videoai.core.ffmpeg import extract_frame
from videoai.core.models import (
    Analysis,
    Manifest,
    PhraseIndex,
    QualityReport,
    SegmentAnalysis,
    TakeGroups,
    Transcript,
)
from videoai.core.project import read_brief
from videoai.core.registry import StageContext, stage
from videoai.logic.phrases import build_phrases, pack_transcript
from videoai.logic.takes import detect_take_groups
from videoai.providers.base import resolve_llm

INSTRUCTIONS = """You are editing a short YouTube video in which a child reviews a toy.

Below is the full transcript of every take, split into phrases. Each phrase has an
id like clip-01#003, a time range and its verbatim text (disfluencies included).

Score every phrase. Reply with one JSON document:

{"segments": [
  {"phrase_id": "clip-01#001", "content": "one short clause describing what happens",
   "delivery_score": 1-10, "visual_score": 1-10, "emotion": "excited|calm|funny|neutral",
   "speaker": "child|helper|both|unclear",
   "is_failed_take": true|false, "shorts_candidate": true|false}
]}

Rules:
- Include every phrase id exactly once. Invent no ids.
- delivery_score rates energy and clarity of speech; visual_score rates how
  interesting the phrase is likely to look on screen.
- is_failed_take is true for restarts, cut-off sentences, obvious mistakes, and
  anything said while the shot is clearly not usable. Most of this footage is
  outtakes, so be willing to mark a lot of it failed.
- speaker: an adult sometimes helps off or on camera. Mark who carries the line.
  "helper" is not automatically bad — a helper question that sets up the child's
  answer earns its place — but the child leads the video.
- shorts_candidate is true only for self-contained, high-energy child moments.
"""


def build_analysis_prompt(
    packed: str, takes: TakeGroups, quality: QualityReport, brief: str
) -> str:
    sections = [INSTRUCTIONS]
    if brief.strip():
        sections.append(f"Creator brief:\n{brief.strip()}")
    if takes.groups:
        lines = [
            f"- {group.group_id}: {', '.join(group.phrase_ids)}" for group in takes.groups
        ]
        sections.append(
            "Phrases that MIGHT be repeated attempts at the same line. This is a "
            "guess made by text similarity, not a fact. Read them: if they really "
            "are attempts at the same line, keep the best one and mark the rest as "
            "failed takes. If they are different things the child said, keep them "
            "all.\n" + "\n".join(lines)
        )
    quality_lines = [
        f"- {clip.clip_id}: blur={clip.blur:.2f} motion={clip.motion:.2f} "
        f"usable={clip.usable} scored={clip.scored}"
        for clip in quality.clips
    ]
    sections.append("Technical quality per clip:\n" + "\n".join(quality_lines))
    sections.append(f"Transcript:\n{packed}")
    return "\n\n".join(sections)


def _keyframes(ctx: StageContext, manifest: Manifest, index: PhraseIndex) -> list[Path]:
    frames_dir = ctx.work_dir / "keyframes"
    paths: list[Path] = []
    for phrase in index.phrases:
        clip = manifest.by_id(phrase.clip_id)
        target = frames_dir / f"{phrase.phrase_id.replace('#', '-')}.jpg"
        if not target.exists():
            source = Path(clip.proxy_path or clip.path)
            if source.exists():
                extract_frame(source, at=(phrase.start + phrase.end) / 2, dst=target)
        if target.exists():
            paths.append(target)
    return paths


@stage(
    id="analyze",
    produces="04-analysis",
    requires=("01-manifest", "02-quality", "03-transcript"),
    provider_key="llm",
    model=Analysis,
)
def analyze(ctx: StageContext) -> Analysis:
    manifest = ctx.store.read("01-manifest", Manifest)
    quality = ctx.store.read("02-quality", QualityReport)
    transcript = ctx.store.read("03-transcript", Transcript)

    settings = ctx.config.transcribe
    index = build_phrases(transcript, settings.phrase_gap_seconds, settings.max_words_per_phrase)
    takes = detect_take_groups(index)
    ctx.store.write("03b-phrases", index, fingerprint="derived")
    ctx.store.write("03c-takes", takes, fingerprint="derived")

    provider = resolve_llm(ctx.config.providers["llm"])
    prompt = build_analysis_prompt(pack_transcript(index), takes, quality, read_brief(ctx.project_dir))
    response = provider.complete_json(
        prompt, _keyframes(ctx, manifest, index), ctx.config.analyze.llm_timeout_seconds
    )

    known = {phrase.phrase_id for phrase in index.phrases}
    scored: dict[str, dict] = {}
    for item in response.get("segments", []):
        phrase_id = item.get("phrase_id")
        if phrase_id not in known:
            raise ValueError(f"model returned an unknown phrase_id: {phrase_id}")
        scored[phrase_id] = item

    segments: list[SegmentAnalysis] = []
    for phrase in index.phrases:
        item = scored.get(phrase.phrase_id, {})
        segments.append(
            SegmentAnalysis(
                phrase_id=phrase.phrase_id,
                clip_id=phrase.clip_id,
                start=phrase.start,
                end=phrase.end,
                text=phrase.text,
                content=item.get("content", ""),
                delivery_score=int(item.get("delivery_score", 5)),
                visual_score=int(item.get("visual_score", 5)),
                emotion=item.get("emotion", "neutral"),
                speaker=item.get("speaker", "unclear"),
                is_failed_take=bool(item.get("is_failed_take", False)),
                take_group=takes.group_of(phrase.phrase_id),
                shorts_candidate=bool(item.get("shorts_candidate", False)),
            )
        )
    return Analysis(provider=provider.name, segments=segments)
