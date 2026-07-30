"""Choosing which seconds of footage a video model is actually shown.

A model that can watch video is billed by the second of footage, and most of a
review's source is silence, fumbling and the seams between takes. Submitting all
of it would pay full price for the parts nobody would ever cut in.

So the reel is the spoken material and nothing else: each phrase's span, padded a
little so a moment does not start mid-syllable, neighbours merged, and the whole
thing capped by a budget the creator sets. Because token cost is duration times
resolution and nothing else, seconds removed here are money saved exactly in
proportion — re-encoding the same footage smaller would save none of it.

Pure arithmetic. The encoding lives in the stage that calls this, so every rule
below is testable without touching ffmpeg.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from videoai.core.models import PhraseIndex

# Two phrases closer together than this are one moment. Cutting them apart pays
# the encoder twice and throws away the beat between them, which is often the
# reaction that made the moment worth keeping.
MERGE_GAP_SECONDS = 0.75


@dataclass(frozen=True)
class ReelEntry:
    """Where one phrase ended up in the assembled reel."""

    phrase_id: str
    reel_start: float
    reel_end: float


@dataclass(frozen=True)
class ReelSpan:
    """One continuous cut taken from one source clip."""

    clip_id: str
    start: float
    end: float
    entries: list[ReelEntry] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


def reel_seconds(spans: list[ReelSpan]) -> float:
    """How much footage this reel submits — the thing that is billed."""
    return round(sum(span.duration for span in spans), 3)


def plan_reel(
    index: PhraseIndex,
    padding: float,
    max_seconds: float | None,
) -> list[ReelSpan]:
    """The spans to cut, in order, with each phrase's position in the result.

    Phrases are taken in the order they appear in the index, merged when they sit
    within `MERGE_GAP_SECONDS` of each other in the same clip, and stopped once
    `max_seconds` of footage has been planned. The budget only ever drops whole
    spans: half a moment is worse than no moment, and a truncated span would end
    mid-sentence.
    """
    if not index.phrases:
        return []

    # 1. Widen each phrase into a span, and merge neighbours within one clip.
    grouped: list[dict] = []
    for phrase in index.phrases:
        start = max(0.0, phrase.start - padding)
        end = phrase.end + padding
        current = grouped[-1] if grouped else None
        if (
            current is not None
            and current["clip_id"] == phrase.clip_id
            and start - current["end"] <= MERGE_GAP_SECONDS
        ):
            current["end"] = max(current["end"], end)
            current["phrases"].append(phrase)
            continue
        grouped.append({
            "clip_id": phrase.clip_id,
            "start": start,
            "end": end,
            "phrases": [phrase],
        })

    # 2. Lay them end to end, recording where each phrase lands, and stop at the
    #    budget. The reel clock advances only for spans that are actually kept.
    spans: list[ReelSpan] = []
    clock = 0.0
    for group in grouped:
        duration = round(group["end"] - group["start"], 3)
        if max_seconds is not None and clock + duration > max_seconds:
            break
        entries: list[ReelEntry] = []
        for phrase in group["phrases"]:
            # Each phrase keeps its own position inside the span it shares, so a
            # merged cut is still addressable phrase by phrase.
            offset = max(0.0, phrase.start - group["start"])
            length = max(0.0, phrase.end - phrase.start)
            entries.append(
                ReelEntry(
                    phrase_id=phrase.phrase_id,
                    reel_start=round(clock + offset, 3),
                    reel_end=round(clock + offset + length, 3),
                )
            )
        spans.append(
            ReelSpan(
                clip_id=group["clip_id"],
                start=round(group["start"], 3),
                end=round(group["end"], 3),
                entries=entries,
            )
        )
        clock = round(clock + duration, 3)

    return spans


def reel_index_lines(spans: list[ReelSpan]) -> list[str]:
    """The mapping the model needs: which phrase is where in the file it watches.

    Without this the model can only describe a moment by its position in the
    reel, and the pipeline scores by phrase id — so this is what makes a reel
    answerable rather than merely watchable.
    """
    return [
        f"{entry.phrase_id}: {entry.reel_start:.2f}s-{entry.reel_end:.2f}s"
        for span in spans
        for entry in span.entries
    ]
