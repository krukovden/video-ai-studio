"""Measuring where on screen something is happening.

A model asked where an event was can only answer from memory of a video it
skimmed at one frame a second, and on real footage it answered "middle-right"
for a moment whose subject sat left of centre. Where the picture actually
changes is not a judgement — it is a subtraction, and it costs nothing.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from videoai.logic.motion import GRID_CELLS, busiest_cell, cell_centre, change_onset


def _clip_with_moving_square(path: Path, x: str, y: str, seconds: float = 3.0) -> Path:
    """A still grey frame with one small square that appears halfway through."""
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "lavfi", "-i", f"color=c=gray:s=600x300:r=30:d={seconds}",
         "-vf", f"drawbox=x={x}:y={y}:w=60:h=60:color=red@1:t=fill:"
                f"enable='gte(t,{seconds / 2})'",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)],
        check=True,
    )
    return path


def test_it_finds_the_corner_something_appears_in(tmp_path: Path):
    # 600x300 frame: a square at x=480,y=220 is bottom-right.
    clip = _clip_with_moving_square(tmp_path / "br.mp4", x="480", y="220")
    assert busiest_cell(clip, at=1.5) == "bottom-right"


def test_it_finds_the_opposite_corner_too(tmp_path: Path):
    clip = _clip_with_moving_square(tmp_path / "tl.mp4", x="20", y="20")
    assert busiest_cell(clip, at=1.5) == "top-left"


def test_it_finds_the_middle(tmp_path: Path):
    clip = _clip_with_moving_square(tmp_path / "c.mp4", x="270", y="120")
    assert busiest_cell(clip, at=1.5) == "center"


def test_a_still_shot_yields_nothing_rather_than_a_guess(tmp_path: Path):
    """Nothing moving means there is nothing to point at, and picking a cell
    anyway is how a graphic ends up in a random corner."""
    clip = tmp_path / "still.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "lavfi", "-i", "color=c=gray:s=600x300:r=30:d=2",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(clip)],
        check=True,
    )
    assert busiest_cell(clip, at=1.0) is None


def test_a_missing_file_is_an_error_rather_than_a_still_shot(tmp_path: Path):
    """None means "nothing moved". A file that cannot be opened is a broken
    project, and answering None for it is how a missing proxy read as a still
    shot and every accent quietly kept the position a model had guessed."""
    with pytest.raises(OSError, match="gone.mp4"):
        busiest_cell(tmp_path / "gone.mp4", at=1.0)


# --- Locating a described moment: a model says roughly when, the footage says
# exactly when, and only the second may set a cut ---


def test_it_finds_the_frame_the_action_actually_starts_on(tmp_path: Path):
    clip = _clip_with_moving_square(tmp_path / "onset.mp4", x="480", y="220", seconds=4.0)
    # The square appears at 2.0s; the model reported it a little late.
    assert abs(change_onset(clip, at=2.3) - 2.0) <= 0.15


def test_two_guesses_at_the_same_action_land_on_the_same_moment(tmp_path: Path):
    """The reason this exists. A duration computed from a model's own number
    changes whenever the model rounds itself differently, and a changed duration
    is a changed timeline hash and a full re-render."""
    clip = _clip_with_moving_square(tmp_path / "same.mp4", x="480", y="220", seconds=4.0)
    assert change_onset(clip, at=2.1) == change_onset(clip, at=2.4)


def test_a_still_stretch_yields_nothing_to_snap_to(tmp_path: Path):
    clip = tmp_path / "still.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "lavfi", "-i", "color=c=gray:s=600x300:r=30:d=4",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(clip)],
        check=True,
    )
    assert change_onset(clip, at=2.0) is None


def test_locating_in_a_missing_file_is_an_error(tmp_path: Path):
    with pytest.raises(OSError, match="gone.mp4"):
        change_onset(tmp_path / "gone.mp4", at=1.0)


def test_every_cell_has_a_centre_inside_the_frame():
    for cell in GRID_CELLS:
        x, y = cell_centre(cell, 1920, 1080)
        assert 0 < x < 1920 and 0 < y < 1080


def test_the_centre_of_each_cell_is_where_it_should_be():
    assert cell_centre("center", 900, 900) == (450, 450)
    assert cell_centre("top-left", 900, 900) == (150, 150)
    assert cell_centre("bottom-right", 900, 900) == (750, 750)


def test_an_unknown_cell_falls_back_to_the_middle():
    assert cell_centre("nowhere", 900, 900) == (450, 450)
