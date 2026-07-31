"""Hard rules every timeline must satisfy before anything is rendered.

These encode the cut-safety knowledge that is expensive to rediscover: never cut
inside a word, keep segments inside their source, keep the timeline contiguous,
and prove each segment really contains the line it claims to.
"""
from __future__ import annotations

import re
from pathlib import Path

from videoai.core.ffmpeg import probe
from videoai.core.models import ClipInfo, Manifest, Timeline, Transcript

MIN_SEGMENT_SECONDS = 0.3
MIN_SPEECH_SECONDS = 0.25
EPSILON = 0.01
FPS_EPSILON = 0.01


def _clip_geometry(clip: ClipInfo) -> tuple[int, int, float]:
    """Real pixel dimensions and frame rate of the file the renderer will actually
    use: the proxy once ingest has built one, the original otherwise.

    Proxies are rebuilt independently per source and keyed by their own build
    height, so two sources that share an aspect ratio — or even two proxies of
    one same-aspect pair — can still end up at different pixel sizes. Only
    probing the actual file catches that; inferring shape from the original's
    aspect ratio cannot.
    """
    if clip.proxy_path:
        proxy = Path(clip.proxy_path)
        if proxy.exists():
            info = probe(proxy)
            return info.width, info.height, round(info.fps, 2)
    return clip.width, clip.height, round(clip.fps, 2)


def _geometry_violations(timeline: Timeline, known_clips: dict[str, ClipInfo]) -> list[str]:
    """Every source in the timeline must yield the same real geometry and frame rate.

    The draft concatenates rendered segments with `-c copy`, which accepts mixed
    geometry without complaint and produces a garbled file: one rotated clip in a
    home shoot is enough. Fail here, at plan time, with the clips named.
    """
    violations: list[str] = []
    reference_id: str | None = None
    reference_shape: tuple[int, int, float] | None = None
    reported: set[str] = set()

    for clip in timeline.clips:
        source = known_clips.get(clip.src)
        if source is None:
            continue
        shape = _clip_geometry(source)
        if reference_shape is None:
            reference_id, reference_shape = clip.src, shape
            continue
        same_size = shape[0] == reference_shape[0] and shape[1] == reference_shape[1]
        same_fps = abs(shape[2] - reference_shape[2]) <= FPS_EPSILON
        if (same_size and same_fps) or clip.src in reported:
            continue
        reported.add(clip.src)
        violations.append(
            f"mixed source geometry: {clip.src} is {shape[0]}x{shape[1]} "
            f"@{shape[2]:.2f}fps but {reference_id} is "
            f"{reference_shape[0]}x{reference_shape[1]} @{reference_shape[2]:.2f}fps; "
            "the draft concatenates their proxies with -c copy, which cannot mix them"
        )
    return violations


def ref_violations(timeline: Timeline) -> list[str]:
    """Every clip must carry a ref, and no two may carry the same one.

    A ref is what a creator's decision holds on to across a re-plan: an accent
    says which shot it marks, the running order says which shot goes third. Two
    clips answering to one name makes every such lookup ambiguous, and one clip
    answering to none cannot be spoken about at all.

    A timeline where NO clip has a ref is left alone. That is an artifact written
    before refs existed, and it still renders exactly as it always did — the
    consumers fall back to absolute time. What is refused is the half-way state,
    where some decisions can be carried and others silently cannot.
    """
    refs = [clip.ref for clip in timeline.clips]
    if not any(refs):
        return []
    violations: list[str] = []
    seen: set[str] = set()
    for index, ref in enumerate(refs):
        if not ref:
            violations.append(
                f"clip {index} ({timeline.clips[index].src}) has no ref, but other "
                "clips in this timeline do; a decision about it could never be "
                "carried across a re-plan"
            )
        elif ref in seen:
            violations.append(
                f"clip {index}: duplicate ref {ref!r}; every lookup by ref would be "
                "ambiguous"
            )
        seen.add(ref)
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

        # A silent visual insert is exempt from the two speech rules only: it has
        # no words to cut inside and no quote to re-anchor. Every other rule above
        # — contiguity, source bounds, minimum durations, geometry — still applies.
        if clip.is_insert:
            continue

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

    violations.extend(ref_violations(timeline))
    violations.extend(_geometry_violations(timeline, known_clips))
    return violations
