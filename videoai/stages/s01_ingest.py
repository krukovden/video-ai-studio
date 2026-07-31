"""s01 ingest: index the source clips, normalise audio, build proxies.

Derived media is expensive (44 minutes of 4K HEVC in the first real project), so
anything already on disk is reused; deleting `work/media` forces a rebuild.
Reuse is keyed by the SOURCE's identity, never by the positional `clip-NN` id:
adding a clip that sorts earlier renumbers every id after it, and a positional
cache key would then serve one clip's audio and proxy under another clip's name.
That identity is the file's content and its place in the project (see
`source_key`), so footage that was rsynced, restored or synced back down — same
frames, brand new mtimes — keeps every proxy it already has.
"""
from __future__ import annotations

import re
from pathlib import Path

from videoai.core.ffmpeg import extract_audio, make_proxy, probe, proxy_encoder
from videoai.core.models import ClipInfo, Manifest
from videoai.core.project import list_camera_clips, resolve_clip_dir
from videoai.core.registry import StageContext, stage
from videoai.core.store import source_key

_UNSAFE_STEM = re.compile(r"[^A-Za-z0-9_-]+")


def derived_stem(source: Path, root: Path | None = None) -> str:
    """Filename stem for everything derived from `source`.

    The readable part is only a label; the digest is what makes the name unique
    and stable across renumbering.
    """
    label = _UNSAFE_STEM.sub("_", source.stem)[:40] or "clip"
    return f"{label}-{source_key(source, root)}"


def _proxy_filename(stem: str, height: int, encoder: str) -> str:
    """Proxy filenames fold in everything that changes a proxy's pixels.

    The height is the obvious one: unlike the audio, a proxy's content depends on
    it, so a height change has to produce a different name or reuse would
    silently keep serving the old resolution. The encoder is the unobvious one
    and matters as much, because nothing configures it — it is chosen by a live
    capability probe, so a run started while the media engine was busy writes
    libx264 pixels into the file a later VideoToolbox run would reuse. The two
    do not measure alike: s02's Laplacian blur score moves, that number reaches
    the analyze prompt, and the edit moves with it.
    """
    return f"{stem}-proxy-{height}p-{encoder}.mp4"


@stage(id="ingest", produces="01-manifest", requires=(), model=Manifest, config_keys=("render.draft_height",))
def ingest(ctx: StageContext) -> Manifest:
    clip_dir = resolve_clip_dir(ctx.input_dir)
    cameras = list_camera_clips(clip_dir)
    if not any(sources for sources in cameras.values()):
        raise RuntimeError(f"no video files found in {clip_dir}")

    media_dir = ctx.work_dir / "media"
    # Probed once for the whole run so the name a proxy is looked up under and the
    # encoder that would write it cannot disagree half way through the shoot.
    encoder = proxy_encoder()
    clips: list[ClipInfo] = []
    index = 0
    for camera in sorted(cameras):
        for source in cameras[camera]:
            index += 1
            clip_id = f"clip-{index:02d}"
            info = probe(source)
            stem = derived_stem(source, ctx.project_dir)

            audio_path: Path | None = None
            if info.has_audio:
                audio_path = media_dir / f"{stem}.wav"
                if not audio_path.exists():
                    extract_audio(source, audio_path)

            proxy_path = media_dir / _proxy_filename(
                stem, ctx.config.render.draft_height, encoder
            )
            if not proxy_path.exists():
                make_proxy(
                    source, proxy_path, height=ctx.config.render.draft_height,
                    encoder=encoder,
                )

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
                    source_key=source_key(source, ctx.project_dir),
                    audio_path=str(audio_path) if audio_path else None,
                    proxy_path=str(proxy_path),
                    proxy_encoder=encoder,
                )
            )
    return Manifest(clips=clips)
