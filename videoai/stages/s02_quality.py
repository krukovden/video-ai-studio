"""s02 quality gate: deterministic technical scoring before any model sees the footage."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from videoai.core.models import ClipQuality, Manifest, QualityReport
from videoai.core.registry import StageContext, stage

SAMPLE_COUNT = 12
BLUR_REFERENCE = 500.0  # Laplacian variance of a comfortably sharp frame
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


def _score_clip(clip_id: str, path: Path, duration: float) -> ClipQuality:
    frames = _sample_frames(path, duration, SAMPLE_COUNT)
    if not frames:
        return ClipQuality(clip_id=clip_id, blur=1.0, shake=0.0, black_ratio=1.0, usable=False)

    sharpness = [float(cv2.Laplacian(frame, cv2.CV_64F).var()) for frame in frames]
    blur = 1.0 - min(1.0, float(np.mean(sharpness)) / BLUR_REFERENCE)

    differences = [
        float(np.mean(np.abs(frames[i].astype(np.int16) - frames[i - 1].astype(np.int16))))
        for i in range(1, len(frames))
    ]
    shake = min(1.0, float(np.mean(differences)) / 64.0) if differences else 0.0

    black_ratio = sum(1 for frame in frames if float(np.mean(frame)) < BLACK_LUMA_THRESHOLD) / len(frames)
    usable = black_ratio < 0.5 and blur < 0.95
    return ClipQuality(
        clip_id=clip_id, blur=blur, shake=shake, black_ratio=black_ratio, usable=usable
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
