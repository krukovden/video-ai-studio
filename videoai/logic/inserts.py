"""Silent visual inserts: footage the planner can cut in without narration.

The story planner selects material by phrase, so a clip nobody talks over has
nothing to be selected by. On a toy review those are often the most valuable
shots there are — a close-up of the thing happening beats another line about it.
This module finds them and defines the reference grammar the planner uses to
place them: `insert:<clip_id>`, or `insert:<clip_id>@<start>-<end>` to use only
part of the clip.
"""
from __future__ import annotations

from videoai.core.models import InsertClip, Manifest, Transcript

INSERT_PREFIX = "insert:"
RANGE_SEPARATOR = "@"
# Range ends are compared against the source duration with the same tolerance the
# timeline validator uses, so a range naming the exact clip length is not rejected
# by the last few microseconds of container rounding.
BOUNDS_EPSILON = 0.01


def is_insert_ref(reference: str) -> bool:
    return reference.startswith(INSERT_PREFIX)


def speech_density(word_count: int, duration: float) -> float:
    """Words per second. A zero-length clip carries no speech to measure, so it
    reports 0.0 rather than dividing by zero."""
    if duration <= 0:
        return 0.0
    return word_count / duration


def detect_inserts(
    manifest: Manifest, transcript: Transcript, max_words_per_second: float
) -> list[InsertClip]:
    """Clips whose speech density falls below the threshold.

    A clip with no words at all is a candidate whatever the threshold is set to:
    zero density is silence by definition, not a value to be compared.
    """
    words_by_clip = {clip.clip_id: len(clip.words) for clip in transcript.clips}
    inserts: list[InsertClip] = []
    for clip in manifest.clips:
        word_count = words_by_clip.get(clip.clip_id, 0)
        density = speech_density(word_count, clip.duration)
        if word_count == 0 or density < max_words_per_second:
            inserts.append(
                InsertClip(
                    clip_id=clip.clip_id,
                    duration=clip.duration,
                    recorded_at=clip.recorded_at,
                    speech_density=density,
                )
            )
    return inserts


def _parse_time(value: str, reference: str) -> float:
    try:
        return float(value)
    except ValueError:
        raise RuntimeError(
            f"insert reference {reference!r} has an unreadable time range: "
            f"{value!r} is not a number of seconds"
        ) from None


def resolve_insert_ref(reference: str, manifest: Manifest) -> tuple[str, float, float]:
    """Resolve an insert reference into (clip_id, start, end) inside that clip.

    Every failure raises `RuntimeError` naming the offending reference: these
    strings come from a language model, so a bad one has to say what it was.
    """
    if not is_insert_ref(reference):
        raise RuntimeError(f"not an insert reference: {reference!r}")

    body = reference[len(INSERT_PREFIX):]
    clip_id, separator, span = body.partition(RANGE_SEPARATOR)
    try:
        clip = manifest.by_id(clip_id)
    except KeyError:
        raise RuntimeError(
            f"insert reference {reference!r} names an unknown clip id {clip_id!r}: "
            "no such clip exists in the manifest"
        ) from None

    if not separator:
        return clip_id, 0.0, clip.duration

    parts = span.split("-")
    if len(parts) != 2 or not all(part.strip() for part in parts):
        raise RuntimeError(
            f"insert reference {reference!r} has a malformed range: expected "
            f"insert:{clip_id}@<start>-<end> with times in seconds"
        )
    start = _parse_time(parts[0].strip(), reference)
    end = _parse_time(parts[1].strip(), reference)

    if start >= end:
        raise RuntimeError(
            f"insert reference {reference!r} has an inverted range: "
            f"start {start:.2f}s is not before end {end:.2f}s"
        )
    if start < 0 or end > clip.duration + BOUNDS_EPSILON:
        raise RuntimeError(
            f"insert reference {reference!r} is outside {clip_id}, which is "
            f"{clip.duration:.2f}s long"
        )
    return clip_id, start, min(end, clip.duration)
