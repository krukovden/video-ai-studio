"""s01 ingest: index the source clips, normalise audio, build proxies.

Derived media is expensive (44 minutes of 4K HEVC in the first real project), so
anything already on disk is reused; deleting `work/media` forces a rebuild.
"""
from __future__ import annotations

from pathlib import Path

from videoai.core.ffmpeg import extract_audio, list_video_files, make_proxy, probe
from videoai.core.models import ClipInfo, Manifest
from videoai.core.project import resolve_clip_dir
from videoai.core.registry import StageContext, stage


@stage(id="ingest", produces="01-manifest", requires=(), model=Manifest)
def ingest(ctx: StageContext) -> Manifest:
    clip_dir = resolve_clip_dir(ctx.input_dir)
    sources = list_video_files(clip_dir)
    if not sources:
        raise RuntimeError(f"no video files found in {clip_dir}")

    media_dir = ctx.work_dir / "media"
    clips: list[ClipInfo] = []
    for index, source in enumerate(sources, start=1):
        clip_id = f"clip-{index:02d}"
        info = probe(source)

        audio_path: Path | None = None
        if info.has_audio:
            audio_path = media_dir / f"{clip_id}.wav"
            if not audio_path.exists():
                extract_audio(source, audio_path)

        proxy_path = media_dir / f"{clip_id}-proxy.mp4"
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
                audio_path=str(audio_path) if audio_path else None,
                proxy_path=str(proxy_path),
            )
        )
    return Manifest(clips=clips)
