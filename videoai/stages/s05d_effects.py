"""s05d effects: the story decides which cartoon accents the video gets.

This is the stage the hardcoded version did not have. That version put one
video's syringe, its "OH NO!" bubble and its keyword-matched positions into the
renderer, so a second project got a syringe it had never filmed. Here the toy and
the video are already analysed by the time this runs, and the accents are chosen
from a library by what the STORY needs: a pop wants an impact, a squish wants a
sound, a finished build wants a celebration.

The model is given a catalogue of what exists — names, tags and one sentence per
sprite saying what it expresses — and nothing else about how any of it is drawn.
It answers with moments. Everything about pixels lives in the compositor, and
everything about which sprites exist lives in `assets/effects/manifest.yaml`, so
replacing a drawing with a better one changes no code and re-plans nothing.

An empty answer is a success. A calm review needs no effects, and a stage that
could not say "none" would guarantee that every video got some.
"""
from __future__ import annotations

from videoai.core.models import (
    Analysis,
    EffectEvent,
    EffectPlan,
    StoryPlan,
    Timeline,
    Transcript,
)
from videoai.core.project import read_brief
from videoai.core.registry import StageContext, stage
from videoai.logic.effects import (
    GRID_CELLS,
    SPRITE_SCALES,
    EffectLibrary,
    load_library,
    manifest_fingerprint,
)
from videoai.providers.base import resolve_llm

# Times come from a model reading a timeline printed to two decimal places, so an
# event landing a hundredth of a second past the last frame is rounding, not a
# hallucinated moment. Anything further out is refused.
TIME_TOLERANCE = 0.05
# An accent needs room to play. One placed in the last fraction of a second would
# be cut off mid-animation by the outro, so it is pulled back instead.
END_MARGIN = 0.3

INSTRUCTIONS = """You are adding cartoon accents to a finished edit of a child's toy review.

An accent is a small animated sticker composited over the picture for about a
second: a comic starburst, sparkles, a speech bubble. They exist to PUNCTUATE
MOMENTS — a pop, a reveal, a reaction, a joke, a loud squish — the way a comic
book punctuates a panel. They are not decoration and not a layer of business over
the whole video.

Below you are given the sprite library you may draw from, the edit itself with
each segment's beat and quote, the word timings inside each segment, descriptions
of the silent close-up inserts, and the creator's brief.

Reply with one JSON document:

{"events": [
  {"at_seconds": 41.8, "effect_name": "comic_starburst",
   "screen_position": "middle-right", "scale": "medium",
   "text": "", "reason": "the bubble pops right here"}
]}

Rules:
- at_seconds is a time on the EDIT's clock as printed below, where 0.0 is the
  first frame of the first segment. Put the accent on the exact moment, not on the
  segment that contains it.
- effect_name must be one of the library names, spelled exactly.
- screen_position is one of: top-left, top-center, top-right, middle-left,
  center, middle-right, bottom-left, bottom-center, bottom-right.
- scale is small, medium or large.
- NEVER cover the child's face. The child is usually in the middle of the frame
  and usually in the upper half of it, so prefer a corner or an edge cell, and
  use "center" only over a close-up of the toy with nobody in it.
- Never more than one accent at a time. Leave at least two seconds between them.
- Aim for four to eight accents in a three-minute video, fewer in a shorter one.
  A quiet, gentle review may deserve none at all, and answering with an empty list
  is a correct answer — far better than putting a starburst on a moment that has
  no pop in it.
- Choose by what the sprite EXPRESSES against what is happening in the story. Do
  not use the same accent twice in a row.
- Match the accent to the moment's sound and action, not to the words alone: a
  loud squish wants the sound sprite, a satisfying pop wants the impact sprite, a
  clean reveal wants sparkles, the end of a successful build wants celebration.
- text is ONLY for a sprite the library marks as taking text, and is REQUIRED for
  one. Keep it to one to four words in capitals. It must be something the
  presenter actually said or plainly meant at that moment — never put words in a
  child's mouth. If nothing they said fits, use a different accent instead.
- reason is one short clause naming the moment, so a human can check your choice.
- No preamble, no extra keys, no commentary outside the JSON document.
"""


def _timeline_view(timeline: Timeline, transcript: Transcript) -> str:
    """The edit on its own clock: every segment's span, beat, quote and words.

    Word timings are mapped from the source clip into edit time here, because the
    model is asked for a moment and the moment it wants is almost always "as this
    word is said" rather than "somewhere in this segment".
    """
    known = {clip.clip_id for clip in transcript.clips}
    lines: list[str] = []
    at = 0.0
    for clip in timeline.clips:
        kind = "silent insert" if clip.is_insert else "speech"
        head = (
            f"{at:7.2f}-{at + clip.dur:7.2f}s  [{clip.beat or 'no beat'}] "
            f"({kind}, {clip.src})"
        )
        if clip.quote:
            head += f': "{clip.quote}"'
        lines.append(head)
        if not clip.is_insert and clip.src in known:
            words = [
                word
                for word in transcript.by_id(clip.src).words
                if word.start >= clip.offset - 0.01
                and word.end <= clip.offset + clip.dur + 0.01
            ]
            if words:
                timings = "  ".join(
                    f"{word.text}@{at + word.start - clip.offset:.2f}" for word in words
                )
                lines.append(f"         words: {timings}")
        at += clip.dur
    return "\n".join(lines)


def _inserts_view(analysis: Analysis, timeline: Timeline) -> str:
    """What the silent close-ups in this edit show.

    Only the ones that reached the timeline: an insert the planner did not use has
    no moment to punctuate.
    """
    used = {clip.src for clip in timeline.clips if clip.is_insert}
    lines = [
        f"- {insert.clip_id}: {insert.description}"
        for insert in analysis.inserts
        if insert.clip_id in used and insert.description
    ]
    return "\n".join(lines)


def build_effects_prompt(
    library: EffectLibrary,
    timeline: Timeline,
    transcript: Transcript,
    analysis: Analysis,
    story: StoryPlan,
    brief: str,
) -> str:
    inserts = _inserts_view(analysis, timeline)
    sections = [
        INSTRUCTIONS,
        f"Creator brief:\n{brief.strip()}" if brief.strip() else "",
        f"Sprite library:\n{library.catalogue()}",
        (
            f"The video: {story.title}\n{story.description}"
            if (story.title or story.description)
            else ""
        ),
        f"The edit ({timeline.duration:.2f}s total):\n{_timeline_view(timeline, transcript)}",
        (
            "What the silent close-up inserts show (no narration on them at all):\n"
            f"{inserts}"
            if inserts
            else ""
        ),
    ]
    return "\n\n".join(part for part in sections if part).strip()


def _coerce_seconds(value: object, event_index: int) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise RuntimeError(
            f"effect event #{event_index} has an unreadable at_seconds: {value!r}"
        ) from None


def parse_events(
    response: dict, library: EffectLibrary, duration: float, max_events: int
) -> list[EffectEvent]:
    """Validate the model's reply into events, refusing anything unrenderable.

    Every refusal names the offending value. The alternative — dropping a bad
    event quietly — would leave a creator wondering why the accent they can see in
    `05d-effects.json` never appears in the video, or worse, why five of the eight
    they asked for did.
    """
    raw = response.get("events")
    if raw is None:
        # A model that decided the video needs no accents may reasonably reply
        # with an empty document. An empty plan is a valid plan.
        return []
    if not isinstance(raw, list):
        raise RuntimeError(
            f"LLM reply 'events' must be a list, got {type(raw).__name__}"
        )
    if len(raw) > max_events:
        raise RuntimeError(
            f"the model returned {len(raw)} effect events and effects.max_events is "
            f"{max_events}; refusing rather than silently keeping the first "
            f"{max_events} — an over-long list means the model ignored the brief, "
            "not that its first few choices are the good ones"
        )
    events: list[EffectEvent] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise RuntimeError(
                f"effect event #{index} is a {type(item).__name__}, not an object"
            )
        name = item.get("effect_name")
        try:
            sprite = library.get(name) if isinstance(name, str) else None
        except KeyError:
            sprite = None
        if sprite is None:
            raise ValueError(
                f"model returned an unknown effect name: {name!r} (the library has "
                + ", ".join(library.names())
                + ")"
            )
        position = item.get("screen_position")
        if position not in GRID_CELLS:
            raise ValueError(
                f"effect {sprite.name!r} names an unknown screen_position "
                f"{position!r}; the grid is " + ", ".join(GRID_CELLS)
            )
        scale = item.get("scale", "medium")
        if scale not in SPRITE_SCALES:
            raise ValueError(
                f"effect {sprite.name!r} names an unknown scale {scale!r}; use "
                + ", ".join(SPRITE_SCALES)
            )
        at = _coerce_seconds(item.get("at_seconds"), index)
        if at < -TIME_TOLERANCE or at > duration + TIME_TOLERANCE:
            raise ValueError(
                f"effect {sprite.name!r} is placed at {at:.2f}s, outside the "
                f"{duration:.2f}s edit; the model was given the edit's own clock "
                "and this time is not on it"
            )
        at = min(max(0.0, at), max(0.0, duration - END_MARGIN))
        text = item.get("text") or ""
        text = text.strip() if isinstance(text, str) else ""
        if sprite.takes_text and not text:
            raise ValueError(
                f"effect {sprite.name!r} is a text sprite and was given no text; it "
                "is stretched around its words and has no size without them"
            )
        if text and not sprite.takes_text:
            # Only the nine-patch can render words. Dropping the text is the
            # renderable reading of the request; refusing the whole event would
            # throw away a placement that is otherwise fine.
            text = ""
        reason = item.get("reason") or ""
        events.append(
            EffectEvent(
                at_seconds=round(at, 3),
                effect_name=sprite.name,
                screen_position=position,
                scale=scale,
                seconds=sprite.default_seconds,
                text=text,
                reason=reason.strip() if isinstance(reason, str) else "",
            )
        )
    return sorted(events, key=lambda event: event.at_seconds)


def _effects_fingerprint_extras(ctx: StageContext) -> tuple[str, ...]:
    """The library's vocabulary. Adding a sprite changes what may be chosen, so a
    cached plan made without it is stale. Only the manifest is hashed: this stage
    never sees a pixel, so a redrawn PNG must not re-plan."""
    return (f"effects-manifest:{manifest_fingerprint()}",)


@stage(
    id="effects",
    produces="05d-effects",
    # "03-transcript" is here for the word timings the prompt carries: the model is
    # asked for a moment, and a moment is a word, not a segment.
    requires=("03-transcript", "04-analysis", "05-timeline", "05a-storyplan"),
    provider_key="llm",
    model=EffectPlan,
    uses_brief=True,
    config_keys=("effects.enabled", "effects.max_events", "analyze.llm_model"),
    prompt=INSTRUCTIONS,
    fingerprint_extras=_effects_fingerprint_extras,
)
def effects(ctx: StageContext) -> EffectPlan:
    settings = ctx.config.effects
    library = load_library()
    if not settings.enabled or not library.sprites:
        # Disabled, or a library with nothing in it. Either way there is nothing
        # to offer a model and no reason to spend a call finding that out.
        return EffectPlan(provider="", library=str(library.directory))

    timeline = ctx.store.read("05-timeline", Timeline)
    analysis = ctx.store.read("04-analysis", Analysis)
    story = ctx.store.read("05a-storyplan", StoryPlan)
    transcript = ctx.store.read("03-transcript", Transcript)
    if not timeline.clips:
        return EffectPlan(provider="", library=str(library.directory))

    provider = resolve_llm(ctx.config.providers["llm"], ctx.config.analyze.llm_model)
    prompt = build_effects_prompt(
        library, timeline, transcript, analysis, story, read_brief(ctx.project_dir)
    )
    response = provider.complete_json(prompt, [], ctx.config.analyze.llm_timeout_seconds)
    events = parse_events(response, library, timeline.duration, settings.max_events)
    return EffectPlan(
        provider=provider.name, library=str(library.directory), events=events
    )
