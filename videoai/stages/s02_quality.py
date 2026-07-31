"""s02 quality gate: deterministic technical scoring before any model sees the footage."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from videoai.core.models import ClipQuality, Manifest, QualityReport
from videoai.core.registry import StageContext, stage

SAMPLE_COUNT = 12
# Laplacian variance of a comfortably sharp frame. Measured against the first real
# project (56 clips, iPhone 4K HEVC, scored at 720p proxy): mean Laplacian variance
# ranged 276-1493, giving blur scores 0.000-0.447 — nowhere near the 0.95 unusable
# cutoff below, so no real footage is falsely rejected. The cutoff is deliberately
# conservative, tuned to reject only genuine defocus. The variance is read off the
# proxy's pixels, so it also depends on which encoder built that proxy — which is
# why ingest names the encoder in the proxy's filename rather than reusing one
# encoder's file for another's run.
BLUR_REFERENCE = 500.0
BLACK_LUMA_THRESHOLD = 12.0


def _sample_frames(path: Path, duration: float, count: int) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    frames: list[np.ndarray] = []
    try:
        for index in range(count):
            position = duration * (index + 0.5) / count
            capture.set(cv2.CAP_PROP_POS_MSEC, position * 1000.0)
            ok, frame = capture.read()
            if ok and frame is not None:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    finally:
        capture.release()
    return frames


def _motion(frames: list[np.ndarray]) -> float:
    """Mean absolute luma difference between frames sampled seconds apart.

    This measures how much the picture changes over the course of the clip
    (scene/content change, pans, zooms) — not camera stability. Consecutive
    sampled frames are `duration/SAMPLE_COUNT` seconds apart, far coarser than
    the ~1/30s spacing needed to detect high-frequency handheld jitter.
    """
    differences = [
        float(np.mean(np.abs(frames[i].astype(np.int16) - frames[i - 1].astype(np.int16))))
        for i in range(1, len(frames))
    ]
    return min(1.0, float(np.mean(differences)) / 64.0) if differences else 0.0


def _score_clip(clip_id: str, path: Path, duration: float) -> ClipQuality:
    frames = _sample_frames(path, duration, SAMPLE_COUNT)
    if not frames:
        # Could not decode any frame: unknown, not proven bad. Leave the numeric
        # scores neutral and let `scored=False` tell downstream consumers this
        # clip was never actually measured, so they don't treat it as unusable
        # footage when it might just be an unreadable proxy.
        return ClipQuality(
            clip_id=clip_id, blur=0.0, motion=0.0, black_ratio=0.0, usable=False, scored=False
        )

    sharpness = [float(cv2.Laplacian(frame, cv2.CV_64F).var()) for frame in frames]
    blur = 1.0 - min(1.0, float(np.mean(sharpness)) / BLUR_REFERENCE)

    motion = _motion(frames)

    black_ratio = sum(1 for frame in frames if float(np.mean(frame)) < BLACK_LUMA_THRESHOLD) / len(frames)
    usable = black_ratio < 0.5 and blur < 0.95
    return ClipQuality(
        clip_id=clip_id, blur=blur, motion=motion, black_ratio=black_ratio, usable=usable
    )


@stage(id="quality", produces="02-quality", requires=("01-manifest",), model=QualityReport)
def quality(ctx: StageContext) -> QualityReport:
    manifest = ctx.store.read("01-manifest", Manifest)
    return QualityReport(
        clips=[
            _score_clip(clip.clip_id, Path(clip.proxy_path or clip.path), clip.duration)
            for clip in manifest.clips
        ]
    )
