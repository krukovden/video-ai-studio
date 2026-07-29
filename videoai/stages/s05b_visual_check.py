"""s05b visual check: look at the segments the planner actually chose.

Selection upstream is made from TEXT — `analyze` scores phrases, and frames only
ever reach the model for silent inserts. Nothing in the pipeline has ever looked
at what a chosen speech segment shows, so an adult walking through frame, a child
turned away, or a shot of the ceiling all reach the draft unseen. Excluding one
offending phrase does not help: the planner just picks its neighbour, which has
the same thing in frame.

This stage samples frames from inside each selected segment's own span, asks the
model what is in them, and refuses to let the render happen when the answer is
bad — recording the offending phrase ids so the next planning round is not
offered them again.
"""
from __future__ import annotations

from pathlib import Path

from videoai.config import PlanSettings
from videoai.core.ffmpeg import extract_frame
from videoai.core.models import (
    Analysis,
    Manifest,
    RejectedPhrases,
    SegmentVisual,
    Timeline,
    TimelineClip,
    VisualCheck,
)
from videoai.core.registry import StageContext, stage
from videoai.core.store import hash_parts
from videoai.logic.inserts import INSERT_PREFIX
from videoai.providers.base import resolve_llm

# Frames are taken inside the segment's own span, not the source clip's: the cut
# is what ships, and the seconds either side of it are not.
FRAME_FRACTIONS = (0.15, 0.5, 0.85)

# Timeline spans are padded outward from the phrase they were built from, so the
# phrase sits inside the span; the tolerance covers float arithmetic only.
SPAN_EPSILON = 0.01

INSTRUCTIONS = """You are checking frames from segments already chosen for the edit of a
child's toy review. The words are settled; what these shots actually show is not.

Each segment below lists three frames taken from inside that segment: near its start,
its middle and its end. Open each file path with your file tools and look at what is
really in frame.

Reply with one JSON document keyed by segment index:

{"0": {"adult_prominent": true|false, "child_visible": true|false,
       "unusable": true|false, "note": "one short sentence"},
 "1": {...}}

Rules:
- Include every segment index listed below, using its exact number. Invent no indices.
- adult_prominent is true when an adult — or any person who is not the child — takes up
  a large part of the frame, or walks through the shot.
- child_visible is true when the child is actually in frame.
- unusable is true when the shot cannot be used: out of focus, pointing at nothing,
  or obscured.
- note is one short factual sentence saying what the frames show.
- No preamble, no extra keys, no commentary outside the JSON document.
"""


def build_visual_check_prompt(
    timeline: Timeline, frames_by_index: dict[int, list[Path]]
) -> str:
    blocks: list[str] = []
    for index, clip in enumerate(timeline.clips):
        frames = frames_by_index.get(index)
        if not frames:
            continue
        kind = "silent insert" if clip.is_insert else "speech"
        quote = f' "{clip.quote}"' if clip.quote else ""
        listing = "\n".join(str(path) for path in frames)
        blocks.append(
            f"{index} - {clip.src} {clip.offset:.2f}s to {clip.offset + clip.dur:.2f}s "
            f"({kind}){quote}\n{listing}"
        )
    return "\n\n".join([INSTRUCTIONS, "Segments:\n\n" + "\n\n".join(blocks)])


def _segment_frames(
    ctx: StageContext, manifest: Manifest, timeline: Timeline
) -> dict[int, list[Path]]:
    """Three cached frames per timeline segment, sampled inside its own span.

    Keyed by source identity plus timestamp, exactly like the keyframe cache in
    `analyze`: segment indices shift whenever the plan changes, so an index-keyed
    file would serve a frame taken from a different moment of a different clip.
    An existing file is never re-extracted.
    """
    frames_dir = ctx.work_dir / "visual"
    by_index: dict[int, list[Path]] = {}
    for index, clip in enumerate(timeline.clips):
        info = manifest.by_id(clip.src)
        key = info.source_key or hash_parts(info.path)
        source = Path(info.proxy_path or info.path)
        paths: list[Path] = []
        for fraction in FRAME_FRACTIONS:
            at = clip.offset + clip.dur * fraction
            target = frames_dir / f"{key}-{round(at * 1000):08d}.jpg"
            if not target.exists() and source.exists():
                extract_frame(source, at=at, dst=target)
            if target.exists():
                paths.append(target)
        if paths:
            by_index[index] = paths
    return by_index


def _coerce_bool(value: object) -> bool:
    """Parse a flag defensively. Anything that is not recognisably a boolean is
    false: an unreadable value must not invent a finding either way."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return False


def parse_findings(response: dict) -> dict[int, dict]:
    """Findings keyed by segment index, from a reply keyed by segment index.

    A model that wraps its answer in a "visual" object is accepted too; anything
    whose key is not a segment number is ignored rather than guessed at.
    """
    raw = response
    nested = response.get("visual")
    if isinstance(nested, dict):
        raw = nested
    findings: dict[int, dict] = {}
    for key, value in raw.items():
        if not isinstance(value, dict):
            continue
        try:
            index = int(key)
        except (TypeError, ValueError):
            continue
        findings[index] = value
    return findings


def _phrase_id_for(clip: TimelineClip, analysis: Analysis | None) -> str | None:
    """The planner-visible id behind a timeline segment, so it can be withheld.

    An insert is named by its reference; a speech segment is found by locating
    the analysed phrase whose span the cut was built around. Phrases do not
    overlap and padding only ever reaches to the neighbouring word, so at most
    one analysed phrase fits inside a segment's span.
    """
    if clip.is_insert:
        return f"{INSERT_PREFIX}{clip.src}"
    if analysis is None:
        return None
    end = clip.offset + clip.dur
    for segment in analysis.segments:
        if segment.clip_id != clip.src:
            continue
        if segment.start >= clip.offset - SPAN_EPSILON and segment.end <= end + SPAN_EPSILON:
            return segment.phrase_id
    return None


def _reason(entry: SegmentVisual, settings: PlanSettings) -> str | None:
    """Why this segment is refused, or None when it may be rendered."""
    if not entry.checked:
        # Silence is not approval: the model was asked about this segment and did
        # not answer, so nothing is known about what it shows.
        if settings.reject_adult_in_frame or settings.reject_unusable_shots:
            return "the model said nothing about this segment"
        return None
    if settings.reject_adult_in_frame and entry.adult_prominent:
        return "an adult or another person takes up the frame"
    if settings.reject_unusable_shots and entry.unusable:
        return "the shot is unusable"
    return None


def _merge_rejected(ctx: StageContext, phrase_ids: list[str]) -> list[str]:
    """Append to the rejection list without losing earlier rounds' entries.

    Each auto-fix round rejects what the round before it could not see; dropping
    the earlier ids would offer the planner a shot it has already refused.
    """
    known: list[str] = []
    if ctx.store.exists("05c-rejected"):
        known = list(ctx.store.read("05c-rejected", RejectedPhrases).phrase_ids)
    for phrase_id in phrase_ids:
        if phrase_id not in known:
            known.append(phrase_id)
    return known


@stage(
    id="visual_check",
    produces="05b-visual",
    requires=("01-manifest", "05-timeline"),
    provider_key="llm",
    model=VisualCheck,
    config_keys=(
        "analyze.llm_model",
        "plan.reject_adult_in_frame",
        "plan.reject_unusable_shots",
    ),
    prompt=INSTRUCTIONS,
)
def visual_check(ctx: StageContext) -> VisualCheck:
    manifest = ctx.store.read("01-manifest", Manifest)
    timeline = ctx.store.read("05-timeline", Timeline)

    frames_by_index = _segment_frames(ctx, manifest, timeline)
    provider = resolve_llm(ctx.config.providers["llm"], ctx.config.analyze.llm_model)

    findings: dict[int, dict] = {}
    if frames_by_index:
        response = provider.complete_json(
            build_visual_check_prompt(timeline, frames_by_index),
            [frame for frames in frames_by_index.values() for frame in frames],
            ctx.config.analyze.llm_timeout_seconds,
        )
        findings = parse_findings(response)

    segments: list[SegmentVisual] = []
    for index, clip in enumerate(timeline.clips):
        item = findings.get(index)
        if item is None:
            # A segment whose source could not be read was never put in front of
            # the model at all; say so rather than blaming the model's silence.
            missing = "" if index in frames_by_index else "no frames could be extracted"
            segments.append(SegmentVisual(index=index, src=clip.src, note=missing))
            continue
        note = item.get("note", "")
        segments.append(
            SegmentVisual(
                index=index,
                src=clip.src,
                adult_prominent=_coerce_bool(item.get("adult_prominent")),
                child_visible=_coerce_bool(item.get("child_visible")),
                unusable=_coerce_bool(item.get("unusable")),
                note=note.strip() if isinstance(note, str) else "",
                checked=True,
            )
        )
    check = VisualCheck(provider=provider.name, segments=segments)

    offenders = [
        (entry, reason)
        for entry in segments
        if (reason := _reason(entry, ctx.config.plan)) is not None
    ]
    if not offenders:
        return check

    # The artifact only reaches the store when the stage returns, so record the
    # findings that caused the refusal before raising: this file is the record of
    # why the draft was not rendered.
    ctx.store.write("05b-visual", check, fingerprint="derived")

    # Read only to name what was refused in the planner's own vocabulary, and only
    # on the failure path — the check itself judges frames, not text, so the
    # analysis is not one of this stage's inputs and must not be one: it always
    # exists by the time a timeline does.
    analysis = (
        ctx.store.read("04-analysis", Analysis) if ctx.store.exists("04-analysis") else None
    )
    rejected: list[str] = []
    lines = [
        f"visual check rejected {len(offenders)} of {len(segments)} selected segments:"
    ]
    for entry, reason in offenders:
        clip = timeline.clips[entry.index]
        phrase_id = _phrase_id_for(clip, analysis)
        if phrase_id is not None and phrase_id not in rejected:
            rejected.append(phrase_id)
        lines.append(
            f"- #{entry.index} {phrase_id or entry.src} "
            f"{clip.offset:.2f}-{clip.offset + clip.dur:.2f}s in {entry.src}: "
            f"{reason} — {entry.note or 'no note from the model'}"
        )
    if rejected:
        ctx.store.write(
            "05c-rejected",
            RejectedPhrases(phrase_ids=_merge_rejected(ctx, rejected)),
            fingerprint="derived",
        )
        lines.append(
            "Withheld from the next planning round (work/05c-rejected.json): "
            + ", ".join(rejected)
        )
    raise RuntimeError("\n".join(lines))
