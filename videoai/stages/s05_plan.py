"""s05 plan: the model picks and orders phrases; geometry and validation are ours."""
from __future__ import annotations

from videoai.core.models import (
    Analysis,
    Manifest,
    PlanSection,
    StoryPlan,
    Timeline,
    Transcript,
)
from videoai.core.project import read_brief
from videoai.core.registry import StageContext, stage
from videoai.logic.timeline import build_timeline
from videoai.logic.validate import validate_timeline
from videoai.providers.base import resolve_llm

INSTRUCTIONS = """You are assembling a short YouTube video from a child's toy review.

Below is every usable phrase with its scores. Build the edit by selecting and
ordering phrase ids. Reply with one JSON document:

{"title": "...", "description": "...", "tags": ["..."],
 "sections": [{"name": "Hook", "goal": "why this section exists",
               "phrase_ids": ["clip-01#004"]}]}

Rules:
- Use only phrase ids from the list. Never invent ids and never repeat one.
- Drop phrases marked as failed takes. A take group is a suspicion, not a fact:
  when its members really are attempts at one line, keep only the best delivery;
  when they turn out to be different content, keep what belongs in the story.
- The child is the presenter. Keep helper lines only where they carry the story,
  such as a question the child then answers.
- Most of the source material is unusable outtakes. Selecting a small fraction of
  it is the expected outcome, not a mistake.
- Open with the single strongest moment, then tell the review in a sensible order.
- Aim for the target duration in the brief if one is given.
- title is punchy and under 70 characters; description is two or three sentences.
"""


def _segments_view(analysis: Analysis) -> str:
    lines = []
    for segment in analysis.segments:
        flags = []
        if segment.is_failed_take:
            flags.append("FAILED")
        if segment.take_group:
            flags.append(segment.take_group)
        if segment.shorts_candidate:
            flags.append("shorts")
        if segment.speaker not in {"child", "unclear"}:
            flags.append(segment.speaker)
        suffix = f" [{' '.join(flags)}]" if flags else ""
        lines.append(
            f"{segment.phrase_id} ({segment.end - segment.start:.1f}s, "
            f"delivery={segment.delivery_score}, visual={segment.visual_score}, "
            f"{segment.emotion}){suffix}: {segment.text}"
        )
    return "\n".join(lines)


@stage(
    id="plan",
    produces="05-timeline",
    requires=("01-manifest", "03-transcript", "04-analysis"),
    provider_key="llm",
    model=Timeline,
)
def plan(ctx: StageContext) -> Timeline:
    manifest = ctx.store.read("01-manifest", Manifest)
    transcript = ctx.store.read("03-transcript", Transcript)
    analysis = ctx.store.read("04-analysis", Analysis)

    brief = read_brief(ctx.project_dir)

    provider = resolve_llm(ctx.config.providers["llm"])
    prompt = "\n\n".join([
        INSTRUCTIONS,
        f"Creator brief:\n{brief}" if brief.strip() else "",
        f"Phrases:\n{_segments_view(analysis)}",
    ]).strip()
    response = provider.complete_json(prompt, [], ctx.config.analyze.llm_timeout_seconds)

    story = StoryPlan(
        title=response.get("title", ""),
        description=response.get("description", ""),
        tags=list(response.get("tags", [])),
        sections=[
            PlanSection(
                name=section.get("name", "Section"),
                goal=section.get("goal", ""),
                phrase_ids=list(section.get("phrase_ids", [])),
            )
            for section in response.get("sections", [])
        ],
    )
    ctx.store.write("05a-storyplan", story, fingerprint="derived")

    timeline = build_timeline(
        story,
        analysis,
        manifest,
        padding=ctx.config.transcribe.cut_padding_seconds,
        fps=manifest.clips[0].fps,
    )
    violations = validate_timeline(timeline, manifest, transcript)
    if violations:
        raise ValueError("timeline validation failed:\n" + "\n".join(violations))
    return timeline
