"""Choosing the music bed and crediting it.

The library is a fixed folder of royalty-free tracks, so selection is a lookup,
not a search: the project's declared style picks a track when it names one we
have, and otherwise the project's own name picks one deterministically. The same
project must always get the same bed — a final render that arrives with different
music each time is not a finished video, it is a slot machine.
"""
from __future__ import annotations

from pathlib import Path

from videoai.core.store import hash_parts

MUSIC_SUFFIXES = {".mp3", ".m4a", ".wav", ".aac", ".flac", ".ogg"}

# Style from project.yaml to the track that fits it. Only styles this library
# actually has a match for are listed; anything else falls through to the
# deterministic pick rather than being forced into an approximate mood.
STYLE_TRACKS = {
    "playful": "bensound-funday.mp3",
    "fun": "bensound-funday.mp3",
    "happy": "bensound-funday.mp3",
    "energetic": "bensound-energy.mp3",
    "upbeat": "bensound-energy.mp3",
    "calm": "bensound-slowlife.mp3",
    "gentle": "bensound-slowlife.mp3",
    "relaxed": "bensound-slowlife.mp3",
    "nostalgic": "bensound-mychildhood.mp3",
    "dreamy": "bensound-floatinggarden.mp3",
    "inspiring": "bensound-inspire.mp3",
}


def list_tracks(music_dir: Path) -> list[Path]:
    """Playable tracks directly inside `music_dir`, sorted; empty when absent."""
    if not music_dir.is_dir():
        return []
    return sorted(
        path for path in music_dir.iterdir()
        if path.is_file()
        and not path.name.startswith(".")
        and path.suffix.lower() in MUSIC_SUFFIXES
    )


def select_track(tracks: list[Path], style: str, project_name: str) -> Path | None:
    """The bed for this project: by style when the library has that track, else
    by a stable digest of the project name."""
    if not tracks:
        return None
    # Sorted here rather than trusting the caller: the fallback below indexes into
    # this list, so a directory listing that came back in a different order would
    # otherwise change the bed without anything about the project changing.
    ordered = sorted(tracks, key=lambda path: path.name)
    wanted = STYLE_TRACKS.get(style.strip().lower())
    if wanted:
        for track in ordered:
            if track.name.lower() == wanted:
                return track
    # `hash_parts` rather than `hash()`: the built-in is salted per process, so it
    # would hand the same project a different track on every run.
    index = int(hash_parts(project_name), 16) % len(ordered)
    return ordered[index]


def track_title(track: Path) -> str:
    stem = track.stem
    prefix = "bensound-"
    if stem.lower().startswith(prefix):
        stem = stem[len(prefix):]
    return stem.replace("_", " ").strip().title()


def attribution_line(track: Path) -> str:
    """The credit Bensound's free licence requires wherever the video is published."""
    return (
        f'Music: "{track_title(track)}" ({track.name}) by Bensound — '
        "https://www.bensound.com. Bensound's free licence requires this credit to "
        "appear with the video wherever it is published."
    )
