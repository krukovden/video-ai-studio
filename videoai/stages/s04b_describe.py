"""s04b describe: watch every source clip once and write down what is in it.

The analyze stage submits a reel of the spoken spans, which is the right thing
for scoring phrases and the wrong thing for knowing what the footage contains:
half the shoot never reaches it, and what it reports is a handful of moments
across the whole edit rather than a record of each video.

This stage is the record. Every clip is watched in full, once, and its
description is kept in `work/04c-clip-notes.json` and written out as
`video-notes.md` in the project folder. Because the cache is keyed by the file's
identity, re-editing costs nothing: a later run reads the notes instead of
sending the footage again, and only genuinely new or replaced clips are ever
described. Deleting the artifact is how you ask for a fresh look.

The events it collects carry a screen position as well as a time. That is what
lets an accent be placed on the thing that is happening instead of in whichever
corner a model guessed.
"""
from __future__ import annotations

from pathlib import Path

from videoai.core.models import ClipEvent, ClipNote, ClipNotes, Manifest
from videoai.core.project import read_brief, resolve_media_path
from videoai.core.registry import StageContext, stage
from videoai.logic.clip_notes import (
    clips_needing_description,
    merge_notes,
    render_notes_markdown,
)
from videoai.providers.base import resolve_llm

# The nine cells the effects stage already speaks in, so a position recorded here
# can be handed straight to an accent without translation.
GRID_CELLS = (
    "top-left", "top-center", "top-right",
    "middle-left", "center", "middle-right",
    "bottom-left", "bottom-center", "bottom-right",
)

INSTRUCTIONS = """You are watching one clip of raw footage and writing down what is in it.

This is a record for the editor, not a review. Nothing is being scored and
nothing is being chosen; the question is only "what happens in this video, and
when".

Reply with one JSON document:

{"summary": "two or three sentences describing what this clip contains",
 "events": [
   {"at": 18.1, "kind": "action|emotion|reaction|speech",
    "what": "one short clause naming what happens",
    "where": "center"}
 ]}

Rules:
- "at" is in seconds from the start of THIS clip.
- List the moments an editor would care about: something opening, something
  being made or destroyed, a reaction, a reveal, a mistake, the start or end of
  an attempt at a take. Be generous — a moment you leave out is one the edit
  cannot use — but do not narrate every second.
- "where" says roughly where on screen it happens, as one of: top-left,
  top-center, top-right, middle-left, center, middle-right, bottom-left,
  bottom-center, bottom-right. This is used to place graphics on the thing that
  is happening, so it matters that it is the location of the ACTION — the hands,
  the object, the face that reacts — and not the middle of the frame by default.
- Describe only what is visible. If the clip is a close-up of an object with
  nobody in it, say so. Do not assume a person is present because the project is
  about one.
- The clip is sampled about once a second, so something shorter than that may
  not be visible. Do not report a moment you had to infer.
- No preamble, no extra keys, no commentary outside the JSON document.
"""


def build_describe_prompt(clip_id: str, duration: float, brief: str) -> str:
    sections = [INSTRUCTIONS]
    if brief.strip():
        sections.append(
            "The creator's brief, for vocabulary only — it does not tell you what "
            f"is in THIS clip:\n{brief.strip()}"
        )
    sections.append(f"This clip is {clip_id}, {duration:.1f} seconds long.")
    return "\n\n".join(sections)


def parse_clip_note(
    response: dict, clip_id: str, source_key: str, duration: float, source_name: str
) -> ClipNote:
    """One clip's description, with anything unusable dropped rather than raised on.

    A note is a record, not a decision: a malformed event costs a line in a log,
    and failing the stage over it would throw away every other clip that was
    described in the same run and already paid for.
    """
    events: list[ClipEvent] = []
    for item in response.get("events") or []:
        if not isinstance(item, dict):
            continue
        what = str(item.get("what", "") or "").strip()
        if not what:
            continue
        try:
            at = float(item.get("at"))
        except (TypeError, ValueError):
            continue
        if not (0.0 <= at <= duration + 1.0):
            # A time outside the clip is a moment in some other video.
            continue
        where = str(item.get("where", "") or "").strip().lower()
        events.append(
            ClipEvent(
                at=round(at, 2),
                kind=str(item.get("kind", "action") or "action").strip().lower(),
                what=what,
                where=where if where in GRID_CELLS else "",
            )
        )
    return ClipNote(
        clip_id=clip_id,
        source_key=source_key,
        duration=duration,
        source_name=source_name,
        summary=str(response.get("summary", "") or "").strip(),
        events=sorted(events, key=lambda event: event.at),
    )


@stage(
    id="describe",
    produces="04c-clip-notes",
    requires=("01-manifest",),
    provider_key="llm",
    model=ClipNotes,
    uses_brief=True,
    config_keys=("describe.enabled", "analyze.llm_model"),
    prompt=INSTRUCTIONS,
)
def describe(ctx: StageContext) -> ClipNotes:
    manifest = ctx.store.read("01-manifest", Manifest)
    if not ctx.config.describe.enabled:
        return ClipNotes()

    # Read back rather than started fresh: this artifact is a cache, and the
    # stage re-running (a changed prompt, a new clip) must not re-bill footage
    # that has already been watched.
    cached = (
        ctx.store.read("04c-clip-notes", ClipNotes)
        if ctx.store.exists("04c-clip-notes")
        else ClipNotes()
    )
    pairs = [(clip.clip_id, clip.source_key or clip.path) for clip in manifest.clips]
    pending = clips_needing_description(cached, pairs)

    if not pending:
        _write_markdown(ctx, cached)
        return cached

    provider = resolve_llm(ctx.config.llm_for("describe"), ctx.config.analyze.llm_model)
    if not getattr(provider, "reads_video", False):
        raise RuntimeError(
            f"describe needs a provider that can watch video; '{provider.name}' is "
            "given stills only. Point it at one in llm_by_stage, or set "
            "describe.enabled: false."
        )

    brief = read_brief(ctx.project_dir)
    fresh: list[ClipNote] = []
    for clip_id in pending:
        clip = manifest.by_id(clip_id)
        source = resolve_media_path(ctx.project_dir, clip.proxy_path or clip.path)
        response = provider.complete_json(
            build_describe_prompt(clip_id, clip.duration, brief),
            [],
            ctx.config.analyze.llm_timeout_seconds,
            videos=[source],
        )
        note = parse_clip_note(
            response,
            clip_id,
            clip.source_key or clip.path,
            clip.duration,
            Path(clip.path).name,
        )
        usage = getattr(provider, "last_usage", None)
        if usage is not None:
            note = note.model_copy(update={
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
            })
        fresh.append(note)
        # Written after every clip, not at the end: describing thirteen videos
        # takes a while, and a failure on the last one must not throw away the
        # twelve that were already paid for.
        merged = merge_notes(cached, fresh).model_copy(update={"provider": provider.name})
        ctx.store.write("04c-clip-notes", merged, fingerprint="incremental")

    merged = merge_notes(cached, fresh).model_copy(update={"provider": provider.name})
    _write_markdown(ctx, merged)
    return merged


def _write_markdown(ctx: StageContext, notes: ClipNotes) -> None:
    """The log the creator actually reads, in the project folder beside the footage."""
    target = ctx.project_dir / "video-notes.md"
    target.write_text(render_notes_markdown(notes), encoding="utf-8")
