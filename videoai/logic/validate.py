"""Hard rules every timeline must satisfy before anything is rendered.

These encode the cut-safety knowledge that is expensive to rediscover: never cut
inside a word, keep segments inside their source, keep the timeline contiguous,
and prove each segment really contains the line it claims to.
"""
from __future__ import annotations

import re

from videoai.core.models import ClipInfo, Manifest, Timeline, Transcript

MIN_SEGMENT_SECONDS = 0.3
MIN_SPEECH_SECONDS = 0.25
EPSILON = 0.01
FPS_EPSILON = 0.01


def _proxy_shape(clip: ClipInfo) -> tuple[float, float]:
    """What the clip's proxy will look like, as (aspect ratio, frame rate).

    Proxies scale to a fixed height with an even auto width, so two sources with
    the same aspect ratio produce identically sized proxies however different
    their originals are — and two with different aspect ratios never do.
    """
    aspect = round(clip.width / clip.height, 3) if clip.height else 0.0
    return aspect, round(clip.fps, 2)


def _geometry_violations(timeline: Timeline, known_clips: dict[str, ClipInfo]) -> list[str]:
    """Every source in the timeline must yield the same proxy geometry and frame rate.

    The draft concatenates rendered segments with `-c copy`, which accepts mixed
    geometry without complaint and produces a garbled file: one rotated clip in a
    home shoot is enough. Fail here, at plan time, with the clips named.
    """
    violations: list[str] = []
    reference_id: str | None = None
    reference_shape: tuple[float, float] | None = None
    reported: set[str] = set()

    for clip in timeline.clips:
        source = known_clips.get(clip.src)
        if source is None:
            continue
        shape = _proxy_shape(source)
        if reference_shape is None:
            reference_id, reference_shape = clip.src, shape
            continue
        same_aspect = abs(shape[0] - reference_shape[0]) < 1e-6
        same_fps = abs(shape[1] - reference_shape[1]) <= FPS_EPSILON
        if (same_aspect and same_fps) or clip.src in reported:
            continue
        reported.add(clip.src)
        reference = known_clips[reference_id]
        violations.append(
            f"mixed source geometry: {clip.src} is {source.width}x{source.height} "
            f"@{source.fps:.2f}fps but {reference_id} is "
            f"{reference.width}x{reference.height} @{reference.fps:.2f}fps; "
            "the draft concatenates their proxies with -c copy, which cannot mix them"
        )
    return violations


def _normalise(text: str) -> str:
    """Lowercase, drop punctuation, and collapse whitespace to single spaces.

    Punctuation adjacent to a space (transcript words carry attached punctuation
    by design) must not leave a doubled space behind, or a genuinely spoken quote
    fails the substring check purely on whitespace shape.
    """
    stripped = re.sub(r"[^a-z0-9 ]+", " ", text.lower())
    return re.sub(r"\s+", " ", stripped).strip()


def validate_timeline(
    timeline: Timeline, manifest: Manifest, transcript: Transcript
) -> list[str]:
    violations: list[str] = []
    known_clips = {clip.clip_id: clip for clip in manifest.clips}
    known_transcripts = {clip.clip_id for clip in transcript.clips}

    expected_position = 0.0
    for index, clip in enumerate(timeline.clips):
        label = f"clip {index} ({clip.src} @ {clip.offset:.2f})"

        source = known_clips.get(clip.src)
        if source is None:
            violations.append(f"{label}: unknown source {clip.src}")
            expected_position += clip.dur
            continue

        if clip.offset < 0:
            violations.append(f"{label}: negative offset")
        if clip.dur < MIN_SEGMENT_SECONDS:
            violations.append(
                f"{label}: shorter than {MIN_SEGMENT_SECONDS}s ({clip.dur:.2f}s)"
            )
        if clip.core_dur < MIN_SPEECH_SECONDS:
            violations.append(
                f"{label}: core speech shorter than {MIN_SPEECH_SECONDS}s "
                f"({clip.core_dur:.2f}s)"
            )
        if clip.offset + clip.dur > source.duration + EPSILON:
            violations.append(
                f"{label}: exceeds source duration ({source.duration:.2f}s)"
            )
        if abs(clip.start - expected_position) > EPSILON:
            violations.append(
                f"{label}: timeline is not contiguous, expected start "
                f"{expected_position:.2f} but got {clip.start:.2f}"
            )
        expected_position += clip.dur

        words = transcript.by_id(clip.src).words if clip.src in known_transcripts else []
        cut_in, cut_out = clip.offset, clip.offset + clip.dur
        for word in words:
            if word.start + EPSILON < cut_in < word.end - EPSILON:
                violations.append(f"{label}: cut starts inside word '{word.text}'")
            if word.start + EPSILON < cut_out < word.end - EPSILON:
                violations.append(f"{label}: cut ends inside word '{word.text}'")

        if clip.quote.strip():
            spoken = " ".join(
                word.text for word in words
                if word.start >= cut_in - EPSILON and word.end <= cut_out + EPSILON
            )
            if _normalise(clip.quote) not in _normalise(spoken):
                violations.append(f"{label}: quote not found in segment audio range")

    violations.extend(_geometry_violations(timeline, known_clips))
    return violations
