"""s02b sync: one timeline for every camera."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from videoai.core.models import ClipInfo, Manifest, SyncMap
from videoai.core.registry import StageContext, stage
from videoai.logic.sync import audio_envelope, build_sync_map, choose_primary_camera


@stage(
    id="sync",
    produces="01b-sync",
    requires=("01-manifest",),
    model=SyncMap,
    config_keys=("sync.primary_camera",),
)
def sync(ctx: StageContext) -> SyncMap:
    manifest = ctx.store.read("01-manifest", Manifest)
    cache: dict[str, np.ndarray | None] = {}

    def envelope_of(clip: ClipInfo) -> np.ndarray | None:
        if clip.clip_id not in cache:
            envelope: np.ndarray | None = None
            if clip.audio_path and Path(clip.audio_path).exists():
                computed = audio_envelope(Path(clip.audio_path))
                envelope = computed if computed.size else None
            cache[clip.clip_id] = envelope
        return cache[clip.clip_id]

    return build_sync_map(
        manifest,
        envelope_of,
        primary_camera=choose_primary_camera(
            manifest, envelope_of, ctx.config.sync.primary_camera
        ),
    )
