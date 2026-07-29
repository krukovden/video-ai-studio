"""Artifact models. Every stage input and output is defined here."""
from __future__ import annotations

from pydantic import BaseModel, Field


class ClipInfo(BaseModel):
    clip_id: str
    path: str
    duration: float
    width: int
    height: int
    fps: float
    has_audio: bool
    audio_path: str | None = None
    proxy_path: str | None = None


class Manifest(BaseModel):
    clips: list[ClipInfo] = Field(default_factory=list)

    def by_id(self, clip_id: str) -> ClipInfo:
        for clip in self.clips:
            if clip.clip_id == clip_id:
                return clip
        raise KeyError(f"unknown clip_id: {clip_id}")
