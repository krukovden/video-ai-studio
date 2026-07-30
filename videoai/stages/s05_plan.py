"""s05 plan: the model picks and orders phrases; geometry and validation are ours."""
from __future__ import annotations

from videoai.core.models import (
    ClipNotes,
    Analysis,
    Manifest,
    PlanSection,
    RejectedPhrases,
    StoryPlan,
    SyncMap,
    Timeline,
    Transcript,
)
from videoai.core.project import read_brief
from videoai.core.registry import StageContext, stage
from videoai.logic.inserts import INSERT_PREFIX, insert_ref_clip_id, is_insert_ref, resolve_insert_ref
from videoai.logic.action_cuts import avoid_mid_action_cuts
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
- If the brief describes the sequence the creator wants — an arc, a list of beats,
  or must_include items — follow it. The creator knows what happened on the shoot;
  your job is to find the footage for each beat, not to invent a different story.
  Name your sections after their beats.
- Only when the brief gives no structure, choose one yourself: open with the single
  strongest moment, then tell the review in a sensible order. Strongest means most
  engaging for THIS toy, judged from the material you were given, not most
  energetic. For a quiet toy that may be the best reaction or the clearest look at
  the thing itself.
- A beat the brief asks for that has no usable footage is worth reporting: name it
  in the description field rather than silently dropping it.
- The video needs an ending. The last section must be a closing: the presenter
  wrapping up, giving a verdict, saying goodbye, or otherwise signalling that the
  video is over. Ending on a fragment or mid-thought, with no sign-off, is a
  failure even if every beat in the creator's arc was otherwise covered. This does
  not require a specific beat name — the creator's arc still governs the body of
  the video — only that whatever closes it out actually closes it out.
- Closing language is usually one of the last things filmed, so look for it near
  the END of the shoot in recording order (phrase ids and timestamps run in
  recording order within a clip, and clips are numbered in shooting order) rather
  than assuming it sits wherever the arc's last beat happens to fall.
- If no closing material exists anywhere in the footage, do not invent one or end
  arbitrarily: say so plainly in the description field instead.
- Aim for the target duration in the brief if one is given.
- title is punchy and under 70 characters; description is two or three sentences.
- Some phrases the creator has already reviewed and rejected are withheld from
  the list below entirely. If a moment you would expect to see is missing, that
  is why: do not try to reconstruct it from neighbouring phrases or invent a
  replacement for it.

Silent visual inserts:
- Some clips carry no narration at all. They are listed separately below, under
  "Silent visual inserts", with their length, where they sit in the recording
  relative to the narrated phrases, and — where available — a one-sentence
  description of what the clip actually shows. Nobody speaks on them: they are
  what the camera saw, usually shot close up.
- They exist to be cut in where they show what the child is talking about. A
  close-up of the thing happening is often worth more than another line about it,
  so use them. Match an insert to what is being said or shown around it using its
  description, not just its position: a close-up of the syringe filling the toy
  belongs where the child talks about filling it, a close-up of a bubble being
  popped belongs in the popping section, even if that section is not where the
  clip happens to sit chronologically.
- Prefer placing an insert immediately AFTER the line it illustrates rather than
  before it, so the viewer hears what is coming and then sees it happen.
- Place one by putting "insert:<clip_id>" into a section's phrase_ids, at the
  position in the sequence where it should appear — for example
  "phrase_ids": ["clip-01#004", "insert:clip-10", "clip-01#005"].
- Write "insert:<clip_id>@<start>-<end>", with times in seconds measured inside
  that clip, to use only that part of it: "insert:clip-12@4-7".
- Keep inserts short. Two to five seconds is usually right. Trim a long insert to
  its best moment with the ranged form rather than using all of it.
- Use only clip ids from the insert list, and keep each insert's range inside the
  clip's own length.
"""


def _segments_view(analysis: Analysis, excluded: set[str]) -> str:
    lines = []
    for segment in analysis.segments:
        if segment.phrase_id in excluded:
            continue
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


def _global_start(sync: SyncMap, clip_id: str) -> float | None:
    """Where this clip sits on the project-wide timeline, or None when sync never
    placed it (a clip added after the sync map was written)."""
    try:
        return sync.by_id(clip_id).global_start
    except KeyError:
        return None


def _inserts_view(analysis: Analysis, sync: SyncMap, excluded: set[str]) -> str:
    """List every insert clip with its length and its place in recording order.

    "Between these two phrases" is the only handle the planner has on a clip with
    no words: it cannot read the clip, so the neighbouring narration is what tells
    it which moment of the shoot this shot belongs to.
    """
    narrated: list[tuple[float, str]] = []
    for segment in analysis.segments:
        if segment.phrase_id in excluded:
            continue
        origin = _global_start(sync, segment.clip_id)
        if origin is None:
            continue
        narrated.append((origin + segment.start, segment.phrase_id))
    narrated.sort()

    lines: list[str] = []
    for insert in analysis.inserts:
        if f"{INSERT_PREFIX}{insert.clip_id}" in excluded:
            continue
        origin = _global_start(sync, insert.clip_id)
        where = "recording position unknown"
        if origin is not None:
            before = [name for at, name in narrated if at <= origin]
            after = [name for at, name in narrated if at >= origin + insert.duration]
            parts = []
            if before:
                parts.append(f"after {before[-1]}")
            if after:
                parts.append(f"before {after[0]}")
            if parts:
                where = "recorded " + " and ".join(parts)
        shows = f' - shows: "{insert.description}"' if insert.description else ""
        lines.append(
            f"insert:{insert.clip_id} ({insert.duration:.1f}s, "
            f"{insert.speech_density:.2f} words/s) - {where}{shows}"
        )
    return "\n".join(lines)


def _rejected_ids(ctx: StageContext) -> set[str]:
    """Ids the visual gate refused on a previous round, if it has ever run.

    The file is absent until something is rejected, which is the ordinary case;
    an absent file withholds nothing.
    """
    if not ctx.store.exists("05c-rejected"):
        return set()
    return set(ctx.store.read("05c-rejected", RejectedPhrases).phrase_ids)


@stage(
    id="plan",
    produces="05-timeline",
    # "05c-rejected" is written by the downstream visual gate and is deliberately
    # not any stage's declared output: listing it here feeds its content into this
    # stage's fingerprint — so a fresh rejection re-plans — without making the
    # dependency graph circular.
    # "04c-clip-notes" is the description of what each clip contains: listed so a
    # re-description re-plans, because where the cuts fall depends on it.
    requires=(
        "01-manifest", "01b-sync", "03-transcript", "04-analysis", "04c-clip-notes",
        "05c-rejected",
    ),
    provider_key="llm",
    model=Timeline,
    uses_brief=True,
    config_keys=(
        "transcribe.cut_padding_seconds",
        "analyze.llm_model",
        "plan.exclude_phrases",
        "plan.gain_db_by_beat",
    ),
    prompt=INSTRUCTIONS,
)
def plan(ctx: StageContext) -> Timeline:
    manifest = ctx.store.read("01-manifest", Manifest)
    sync_map = ctx.store.read("01b-sync", SyncMap)
    transcript = ctx.store.read("03-transcript", Transcript)
    analysis = ctx.store.read("04-analysis", Analysis)

    brief = read_brief(ctx.project_dir)
    # A visually rejected id is withheld exactly like an excluded one: hidden from
    # the prompt, and refused if it comes back anyway.
    excluded = set(ctx.config.plan.exclude_phrases) | _rejected_ids(ctx)

    inserts_view = _inserts_view(analysis, sync_map, excluded)
    provider = resolve_llm(ctx.config.llm_for("plan"), ctx.config.analyze.llm_model)
    prompt = "\n\n".join([
        INSTRUCTIONS,
        f"Creator brief:\n{brief}" if brief.strip() else "",
        f"Phrases:\n{_segments_view(analysis, excluded)}",
        (
            "Silent visual inserts (no narration on them at all - just what the "
            f"camera saw):\n{inserts_view}"
            if inserts_view
            else ""
        ),
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

    # A hallucinated phrase id is the likeliest failure at this stage; letting
    # `build_timeline` raise a bare KeyError would say nothing about where it
    # came from. An excluded phrase id is checked first and separately: it is
    # known to `analysis` (it was only hidden from the prompt), so it would
    # otherwise pass the "known" check below and slip back into the edit — the
    # id could arrive from a stale cached reply, not just a fresh hallucination.
    known = {segment.phrase_id for segment in analysis.segments}
    for section in story.sections:
        for phrase_id in section.phrase_ids:
            if is_insert_ref(phrase_id):
                if f"{INSERT_PREFIX}{insert_ref_clip_id(phrase_id)}" in excluded:
                    raise RuntimeError(
                        f"planner referenced excluded insert {phrase_id!r} in section "
                        f"{section.name!r}: this clip is excluded and must not appear "
                        "in the edit"
                    )
                # Raises a RuntimeError naming the reference for an unknown clip
                # id, an inverted range, or one reaching past the clip's end.
                resolve_insert_ref(phrase_id, manifest)
                continue
            if phrase_id in excluded:
                raise RuntimeError(
                    f"planner referenced excluded phrase id {phrase_id!r} in section "
                    f"{section.name!r}: this phrase is excluded via plan.exclude_phrases "
                    "and must not appear in the edit"
                )
            if phrase_id not in known:
                raise RuntimeError(
                    f"planner referenced unknown phrase id {phrase_id!r} in section "
                    f"{section.name!r}: no such phrase exists in 04-analysis"
                )

    timeline = build_timeline(
        story,
        analysis,
        manifest,
        transcript,
        padding=ctx.config.transcribe.cut_padding_seconds,
        fps=manifest.clips[0].fps,
        gain_db_by_beat=ctx.config.plan.gain_db_by_beat,
    )
    # The planner cuts where the words end; the picture does not agree. When the
    # footage has been described, move any cut that lands inside an action that
    # has only just begun — the shot ending while a hand is halfway into a glove
    # is the edit losing interest exactly where the viewer did not.
    if ctx.store.exists("04c-clip-notes"):
        timeline = avoid_mid_action_cuts(
            timeline, ctx.store.read("04c-clip-notes", ClipNotes)
        )

    violations = validate_timeline(timeline, manifest, transcript)
    if violations:
        raise ValueError("timeline validation failed:\n" + "\n".join(violations))
    return timeline
