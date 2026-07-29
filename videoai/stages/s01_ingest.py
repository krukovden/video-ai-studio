"""s01 ingest: index the source clips, normalise audio, build proxies.

Derived media is expensive (44 minutes of 4K HEVC in the first real project), so
anything already on disk is reused; deleting `work/media` forces a rebuild.
Reuse is keyed by the SOURCE's identity, never by the positional `clip-NN` id:
adding a clip that sorts earlier renumbers every id after it, and a positional
cache key would then serve one clip's audio and proxy under another clip's name.
"""
from __future__ import annotations

import re
from pathlib import Path

from videoai.core.ffmpeg import extract_audio, make_proxy, probe
from videoai.core.models import ClipInfo, Manifest
from videoai.core.project import list_camera_clips, resolve_clip_dir
from videoai.core.registry import StageContext, stage
from videoai.core.store import source_key

_UNSAFE_STEM = re.compile(r"[^A-Za-z0-9_-]+")


def derived_stem(source: Path) -> str:
    """Filename stem for everything derived from `source`.

    The readable part is only a label; the digest is what makes the name unique
    and stable across renumbering.
    """
    label = _UNSAFE_STEM.sub("_", source.stem)[:40] or "clip"
    return f"{label}-{source_key(source)}"


def _proxy_filename(stem: str, height: int) -> str:
    """Proxy filenames must fold in the build height: unlike the audio, a proxy's
    content depends on it, so a height change has to produce a different name or
    reuse would silently keep serving the old resolution."""
    return f"{stem}-proxy-{height}p.mp4"


@stage(id="ingest", produces="01-manifest", requires=(), model=Manifest, config_keys=("render.draft_height",))
def ingest(ctx: StageContext) -> Manifest:
    clip_dir = resolve_clip_dir(ctx.input_dir)
    cameras = list_camera_clips(clip_dir)
    if not any(sources for sources in cameras.values()):
        raise RuntimeError(f"no video files found in {clip_dir}")

    media_dir = ctx.work_dir / "media"
    clips: list[ClipInfo] = []
    index = 0
    for camera in sorted(cameras):
        for source in cameras[camera]:
            index += 1
            clip_id = f"clip-{index:02d}"
            info = probe(source)
            stem = derived_stem(source)

            audio_path: Path | None = None
            if info.has_audio:
                audio_path = media_dir / f"{stem}.wav"
                if not audio_path.exists():
                    extract_audio(source, audio_path)

            proxy_path = media_dir / _proxy_filename(stem, ctx.config.render.draft_height)
            if not proxy_path.exists():
                make_proxy(source, proxy_path, height=ctx.config.render.draft_height)

            clips.append(
                ClipInfo(
                    clip_id=clip_id,
                    path=str(source),
                    duration=info.duration,
                    width=info.width,
                    height=info.height,
                    fps=info.fps,
                    has_audio=info.has_audio,
                    camera=camera,
                    recorded_at=info.created_at,
                    source_key=source_key(source),
                    audio_path=str(audio_path) if audio_path else None,
                    proxy_path=str(proxy_path),
                )
            )
    return Manifest(clips=clips)
