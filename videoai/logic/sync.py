"""Put every clip on one project timeline.

Two cameras rolling at once produce the same words twice. Only a shared timeline
tells "second angle" apart from "second attempt", so everything downstream that
compares clips depends on this.
"""
from __future__ import annotations

import wave
from pathlib import Path
from statistics import median
from typing import Callable

import numpy as np

from videoai.core.models import ClipInfo, ClipSync, Manifest, SyncMap

MIN_CONFIDENCE = 2.0


def audio_envelope(wav_path: Path, rate: int = 100) -> np.ndarray:
    """Loudness envelope at `rate` Hz. Speech shape survives; pitch does not,
    which is what makes cross-correlation between different microphones work."""
    with wave.open(str(wav_path), "rb") as handle:
        frame_rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())
    samples = np.frombuffer(frames, dtype="<i2").astype(np.float64) / 32768.0
    block = max(1, frame_rate // rate)
    usable = len(samples) - (len(samples) % block)
    if usable <= 0:
        return np.zeros(0)
    return np.abs(samples[:usable]).reshape(-1, block).mean(axis=1)


def estimate_offset(
    reference: np.ndarray,
    other: np.ndarray,
    rate: int = 100,
    max_shift_seconds: float = 600.0,
) -> tuple[float, float]:
    """Seconds to add to `other` to align it with `reference`, plus a confidence
    ratio. Below MIN_CONFIDENCE, treat the answer as unusable.

    Confidence is the best lag's correlation strength over the next-best
    candidate lag found at least a second away, not peak-over-mean. A short,
    bursty reference (a handful of speech onsets in an otherwise silent
    envelope) makes the raw correlation mostly near-zero at the many
    low-overlap lags near either end of the search range; averaging across all
    of them drags the mean down and makes an ordinary, meaningless peak look
    high-confidence by comparison. A genuine alignment produces one dominant,
    unambiguous peak; unrelated audio produces several comparably sized
    spurious ones, so comparing the peak to its strongest rival is what
    actually separates a real sync point from a coincidence.
    """
    if reference.size == 0 or other.size == 0:
        return 0.0, 0.0
    first = reference - reference.mean()
    second = other - other.mean()
    size = 1 << int(np.ceil(np.log2(len(first) + len(second))))
    spectrum = np.fft.rfft(first, size) * np.conj(np.fft.rfft(second, size))
    correlation = np.fft.irfft(spectrum, size)
    correlation = np.concatenate([correlation[-(len(second) - 1):], correlation[: len(first)]])
    lags = np.arange(-(len(second) - 1), len(first))
    limit = int(max_shift_seconds * rate)
    allowed = np.abs(lags) <= limit
    if not allowed.any():
        return 0.0, 0.0
    window = correlation[allowed]
    lag_values = lags[allowed]
    peak = int(np.argmax(window))
    magnitude = float(window[peak])

    baseline = float(np.mean(np.abs(window))) or 1e-9
    far = np.abs(lag_values - lag_values[peak]) > rate  # more than one second from the peak
    runner_up = float(np.max(np.abs(window[far]))) if far.any() else baseline
    confidence = abs(magnitude) / (runner_up or 1e-9)
    return float(lag_values[peak]) / rate, confidence


def choose_primary_camera(
    manifest: Manifest,
    envelope_of: Callable[[ClipInfo], np.ndarray | None],
    override: str | None = None,
) -> str:
    """The camera whose audio is worth transcribing.

    Only one camera carries a real microphone; the other hears the room. Mean
    envelope energy separates them reliably without any configuration.
    """
    cameras = sorted({clip.camera for clip in manifest.clips})
    if not cameras:
        return "main"
    if override:
        if override not in cameras:
            raise ValueError(
                f"config.sync.primary_camera={override!r} is not one of {cameras}"
            )
        return override
    if len(cameras) == 1:
        return cameras[0]

    loudness: dict[str, float] = {}
    for camera in cameras:
        energies = [
            float(envelope.mean())
            for clip in manifest.clips
            if clip.camera == camera
            and (envelope := envelope_of(clip)) is not None
            and envelope.size
        ]
        if energies:
            loudness[camera] = sum(energies) / len(energies)
    if not loudness:
        return cameras[0]
    return max(loudness, key=loudness.__getitem__)


def build_sync_map(
    manifest: Manifest,
    envelope_of: Callable[[ClipInfo], np.ndarray | None],
    primary_camera: str | None = None,
) -> SyncMap:
    cameras: dict[str, list[ClipInfo]] = {}
    for clip in manifest.clips:
        cameras.setdefault(clip.camera, []).append(clip)

    placements: dict[str, tuple[float, str]] = {}
    for camera, clips in cameras.items():
        if all(clip.recorded_at is not None for clip in clips):
            for clip in clips:
                placements[clip.clip_id] = (float(clip.recorded_at), "metadata")
        else:
            cursor = 0.0
            for clip in clips:
                placements[clip.clip_id] = (cursor, "sequential")
                cursor += clip.duration

    corrections: dict[str, float] = {name: 0.0 for name in cameras}
    methods: dict[str, str] = {}
    names = sorted(cameras)
    if len(names) > 1:
        anchor = names[0]
        anchor_envelopes = {clip.clip_id: envelope_of(clip) for clip in cameras[anchor]}
        for name in names[1:]:
            offsets: list[float] = []
            for clip in cameras[name]:
                other = envelope_of(clip)
                if other is None:
                    continue
                for anchor_clip in cameras[anchor]:
                    reference = anchor_envelopes.get(anchor_clip.clip_id)
                    if reference is None:
                        continue
                    coarse_gap = placements[clip.clip_id][0] - placements[anchor_clip.clip_id][0]
                    if abs(coarse_gap) > max(anchor_clip.duration, clip.duration) + 60.0:
                        continue
                    shift, confidence = estimate_offset(reference, other)
                    if confidence >= MIN_CONFIDENCE:
                        offsets.append(
                            placements[anchor_clip.clip_id][0] + shift - placements[clip.clip_id][0]
                        )
            if offsets:
                corrections[name] = median(offsets)
                methods[name] = "audio"

    origin = min(start for start, _ in placements.values())
    synced: list[ClipSync] = []
    for clip in manifest.clips:
        start, method = placements[clip.clip_id]
        synced.append(
            ClipSync(
                clip_id=clip.clip_id,
                camera=clip.camera,
                global_start=start + corrections[clip.camera] - origin,
                method=methods.get(clip.camera, method),
                confidence=1.0 if methods.get(clip.camera) == "audio" else 0.0,
            )
        )
    return SyncMap(
        clips=synced,
        primary_camera=primary_camera
        or choose_primary_camera(manifest, envelope_of),
    )
