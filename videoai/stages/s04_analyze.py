"""s04 analyze: score every phrase for delivery, visuals and shorts potential."""
from __future__ import annotations

from pathlib import Path

from videoai.config import AnalyzeSettings
from videoai.core.ffmpeg import extract_frame
from videoai.core.models import (
    Analysis,
    InsertClip,
    Manifest,
    Phrase,
    PhraseIndex,
    QualityReport,
    SegmentAnalysis,
    TakeGroups,
    Transcript,
)
from videoai.core.project import read_brief
from videoai.core.registry import StageContext, stage
from videoai.core.store import hash_parts
from videoai.logic.inserts import detect_inserts
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
- shorts_candidate is true for a moment that stands on its own without setup and
  would make someone stop scrolling. What that looks like depends on the toy: a
  genuine reaction, a satisfying thing happening on screen, a funny aside, or a
  clear reveal. Do not require action or stunts — many toys are quiet, and a
  calm review can still have standout moments.
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


INSERT_DESCRIPTION_INSTRUCTIONS = """You are looking at silent visual insert clips from a
child's toy review video: shots with no narration on them at all. For each clip below,
three frames are listed, taken near its start, middle and end. Open each file path with
your file tools and look at what is actually happening.

Reply with one JSON document:

{"descriptions": {"clip-09": "one short sentence naming the action", "clip-10": "..."}}

Rules:
- Include every clip id listed below, using its exact id.
- One short sentence per clip, naming the specific physical action (e.g. filling,
  popping, squeezing) so that visually similar clips can be told apart from each other.
- Use the creator's own vocabulary for the toy where the brief below gives one.
- No preamble, no extra keys, no commentary outside the JSON document.
"""


def build_insert_description_prompt(
    inserts: list[InsertClip], frames_by_clip: dict[str, list[Path]], brief: str
) -> str:
    sections = [INSERT_DESCRIPTION_INSTRUCTIONS]
    if brief.strip():
        sections.append(f"Creator brief:\n{brief.strip()}")
    clip_blocks = []
    for insert in inserts:
        frames = frames_by_clip.get(insert.clip_id)
        if not frames:
            continue
        listing = "\n".join(str(path) for path in frames)
        clip_blocks.append(f"{insert.clip_id} ({insert.duration:.1f}s):\n{listing}")
    sections.append("Clips:\n\n" + "\n\n".join(clip_blocks))
    return "\n\n".join(sections)


def _insert_keyframes(
    ctx: StageContext,
    manifest: Manifest,
    inserts: list[InsertClip],
    settings: AnalyzeSettings,
) -> dict[str, list[Path]]:
    """Three cached frames per insert clip, at 20%/50%/80% of its duration.

    Keyed by source identity plus timestamp, exactly like `_keyframes`: an insert
    clip has no phrase to hang a key off, but the same renumbering hazard applies,
    so the key must not depend on the clip's ordinal id either.
    `describe_inserts=False` disables extraction entirely, the same way
    `keyframes_per_phrase == 0` does for phrase keyframes - the two settings are
    independent of each other on purpose.
    """
    if not settings.describe_inserts:
        return {}
    frames_dir = ctx.work_dir / "keyframes"
    by_clip: dict[str, list[Path]] = {}
    for insert in inserts:
        clip = manifest.by_id(insert.clip_id)
        key = clip.source_key or hash_parts(clip.path)
        source = Path(clip.proxy_path or clip.path)
        paths: list[Path] = []
        for fraction in (0.2, 0.5, 0.8):
            at = insert.duration * fraction
            target = frames_dir / f"{key}-{round(at * 1000):08d}.jpg"
            if not target.exists() and source.exists():
                extract_frame(source, at=at, dst=target)
            if target.exists():
                paths.append(target)
        if paths:
            by_clip[insert.clip_id] = paths
    return by_clip


def _describe_inserts(
    ctx: StageContext, manifest: Manifest, inserts: list[InsertClip], brief: str
) -> dict[str, str]:
    """One batched model call describing every insert clip from its keyframes.

    The planner cannot otherwise see what a silent clip shows and has been
    observed placing it in the wrong section. A missing or unparsable reply
    degrades to an empty description per clip rather than failing the stage:
    a worse placement is not the same failure as a broken pipeline.
    """
    if not inserts:
        return {}
    frames_by_clip = _insert_keyframes(ctx, manifest, inserts, ctx.config.analyze)
    if not frames_by_clip:
        return {}
    provider = resolve_llm(ctx.config.providers["llm"], ctx.config.analyze.llm_model)
    prompt = build_insert_description_prompt(inserts, frames_by_clip, brief)
    all_frames = [frame for frames in frames_by_clip.values() for frame in frames]
    try:
        response = provider.complete_json(
            prompt, all_frames, ctx.config.analyze.llm_timeout_seconds
        )
    except Exception:
        return {}
    raw = response.get("descriptions")
    if not isinstance(raw, dict):
        return {}
    descriptions: dict[str, str] = {}
    for insert in inserts:
        value = raw.get(insert.clip_id)
        if isinstance(value, str) and value.strip():
            descriptions[insert.clip_id] = value.strip()
    return descriptions


def _keyframes(
    ctx: StageContext, manifest: Manifest, index: PhraseIndex, settings: AnalyzeSettings
) -> tuple[list[Path], bool]:
    """Extract at most one cached frame per phrase, capped at `max_keyframes`.

    Returns the frame paths plus whether the cap cut the list short, so the
    caller can tell the model it is looking at a sample rather than everything.
    `keyframes_per_phrase == 0` disables extraction entirely.

    Cached frames are keyed by the source clip's identity plus the timestamp, not
    by phrase id: phrase ordinals shift whenever the transcript changes, so a
    phrase-id key would serve a frame taken from a different moment (or, after a
    clip renumbering, from a different clip entirely).
    """
    if settings.keyframes_per_phrase <= 0:
        return [], False
    frames_dir = ctx.work_dir / "keyframes"
    paths: list[Path] = []
    truncated = False
    for phrase in index.phrases:
        if len(paths) >= settings.max_keyframes:
            truncated = True
            break
        clip = manifest.by_id(phrase.clip_id)
        at = (phrase.start + phrase.end) / 2
        key = clip.source_key or hash_parts(clip.path)
        target = frames_dir / f"{key}-{round(at * 1000):08d}.jpg"
        if not target.exists():
            source = Path(clip.proxy_path or clip.path)
            if source.exists():
                extract_frame(source, at=at, dst=target)
        if target.exists():
            paths.append(target)
    return paths, truncated


def _coerce_score(value: object) -> tuple[int, bool]:
    """Parse a 1..10 delivery/visual score defensively.

    Returns (score, was_parsed). A missing, null, or unparseable value falls back
    to the neutral 5 with was_parsed=False, so the caller can mark the segment
    unscored instead of confusing a parse failure with a genuine middling score.
    In-range-but-wrong values (e.g. 15) are clamped rather than rejected.
    """
    if value is None:
        return 5, False
    try:
        score = int(value)
    except (TypeError, ValueError):
        try:
            score = int(float(value))
        except (TypeError, ValueError):
            return 5, False
    return max(1, min(10, score)), True


def _coerce_bool(value: object, default: bool) -> bool:
    """Parse a boolean field defensively.

    Real booleans pass through; "true"/"false" strings are recognised
    case-insensitively (bool("false") would otherwise be True, silently
    inverting is_failed_take); anything else falls back to `default`.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
    return default


def _score_segment(phrase: Phrase, item: dict | None, takes: TakeGroups) -> SegmentAnalysis:
    take_group = takes.group_of(phrase.phrase_id)
    if item is None:
        # The model never answered about this phrase at all: neutral defaults,
        # but scored=False so a genuine middling score can't be confused with silence.
        return SegmentAnalysis(
            phrase_id=phrase.phrase_id,
            clip_id=phrase.clip_id,
            start=phrase.start,
            end=phrase.end,
            text=phrase.text,
            take_group=take_group,
            scored=False,
        )
    delivery_score, delivery_ok = _coerce_score(item.get("delivery_score"))
    visual_score, visual_ok = _coerce_score(item.get("visual_score"))
    return SegmentAnalysis(
        phrase_id=phrase.phrase_id,
        clip_id=phrase.clip_id,
        start=phrase.start,
        end=phrase.end,
        text=phrase.text,
        content=item.get("content", ""),
        delivery_score=delivery_score,
        visual_score=visual_score,
        emotion=item.get("emotion", "neutral"),
        speaker=item.get("speaker", "unclear"),
        is_failed_take=_coerce_bool(item.get("is_failed_take"), False),
        take_group=take_group,
        shorts_candidate=_coerce_bool(item.get("shorts_candidate"), False),
        scored=delivery_ok and visual_ok,
    )


@stage(
    id="analyze",
    produces="04-analysis",
    requires=("01-manifest", "02-quality", "03-transcript"),
    provider_key="llm",
    model=Analysis,
    uses_brief=True,
    config_keys=(
        "transcribe.phrase_gap_seconds",
        "transcribe.max_words_per_phrase",
        "analyze.keyframes_per_phrase",
        "analyze.max_keyframes",
        "analyze.llm_model",
        "analyze.insert_max_words_per_second",
        "analyze.describe_inserts",
    ),
    prompt=INSTRUCTIONS,
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

    brief = read_brief(ctx.project_dir)
    provider = resolve_llm(ctx.config.providers["llm"], ctx.config.analyze.llm_model)
    prompt = build_analysis_prompt(pack_transcript(index), takes, quality, brief)
    frames, truncated = _keyframes(ctx, manifest, index, ctx.config.analyze)
    if truncated:
        prompt += (
            f"\n\nNote: only the first {ctx.config.analyze.max_keyframes} reference frames "
            "are attached (capped); treat them as a sample of the footage, not full coverage."
        )
    response = provider.complete_json(prompt, frames, ctx.config.analyze.llm_timeout_seconds)

    known = {phrase.phrase_id for phrase in index.phrases}

    # A silent failure here (missing/empty/truncated segments) would still return
    # an Analysis full of plausible-looking neutral defaults, which downstream
    # stages would use to make real cut decisions. Fail loudly instead.
    raw_segments = response.get("segments")
    if raw_segments is None:
        raise RuntimeError("LLM reply is missing the 'segments' key")
    if not isinstance(raw_segments, list):
        raise RuntimeError(
            f"LLM reply 'segments' must be a list, got {type(raw_segments).__name__}"
        )
    if not raw_segments and known:
        raise RuntimeError(
            f"LLM reply has an empty segments list but {len(known)} phrases were expected"
        )

    scored: dict[str, dict] = {}
    for item in raw_segments:
        phrase_id = item.get("phrase_id")
        if phrase_id not in known:
            raise ValueError(f"model returned an unknown phrase_id: {phrase_id}")
        scored[phrase_id] = item

    # A reply covering only a fraction of the phrases looks like success (every
    # scored phrase has real values) but silently drops the rest to neutral
    # defaults; a truncated reply is worse than no reply because it hides.
    if known and len(scored) * 2 < len(known):
        raise RuntimeError(
            f"LLM reply answered only {len(scored)} of {len(known)} phrases "
            "(fewer than half) - treating a truncated reply as a failure"
        )

    segments = [_score_segment(phrase, scored.get(phrase.phrase_id), takes) for phrase in index.phrases]
    # Insert candidates are measured, not scored: the model is asked about phrases,
    # and these clips have no phrase to ask about.
    inserts = detect_inserts(
        manifest, transcript, ctx.config.analyze.insert_max_words_per_second
    )
    descriptions = _describe_inserts(ctx, manifest, inserts, brief)
    inserts = [
        insert.model_copy(update={"description": descriptions.get(insert.clip_id, "")})
        for insert in inserts
    ]
    return Analysis(provider=provider.name, segments=segments, inserts=inserts)
