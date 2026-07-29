"""Hard rules every timeline must satisfy before anything is rendered.

These encode the cut-safety knowledge that is expensive to rediscover: never cut
inside a word, keep segments inside their source, keep the timeline contiguous,
and prove each segment really contains the line it claims to.
"""
from __future__ import annotations

import re

from videoai.core.models import Manifest, Timeline, Transcript

MIN_SEGMENT_SECONDS = 0.3
MIN_SPEECH_SECONDS = 0.25
EPSILON = 0.01


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
    return violations
