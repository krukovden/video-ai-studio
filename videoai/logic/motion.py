"""Where on screen something is happening, measured rather than asked.

A model that watched a clip can say what happened and roughly when. Asked *where*
it happened it answers from memory of a video it saw at one frame a second, and
on this project's own footage it put "middle-right" on a moment whose subject sat
left of centre. A graphic placed on that answer lands in a random corner, which
is exactly the complaint that produced this module.

Where the picture changes is not a judgement. It is a subtraction between two
frames, it is exact, and it costs nothing — so the model is asked what happens
and this works out where to point at it.
"""
from __future__ import annotations

from pathlib import Path

# The nine cells the effects stage and the clip notes both already speak in.
GRID_CELLS = (
    "top-left", "top-center", "top-right",
    "middle-left", "center", "middle-right",
    "bottom-left", "bottom-center", "bottom-right",
)

# How far either side of the moment to look. An event is a change, so it needs
# two frames separated by enough time for the change to have happened.
WINDOW_SECONDS = 0.5

# Below this mean absolute difference nothing really moved, and naming a cell
# anyway would be the guess this module exists to replace.
MIN_CHANGE = 1.5


def cell_centre(cell: str, width: int, height: int) -> tuple[int, int]:
    """The middle of a named cell, in pixels."""
    try:
        index = GRID_CELLS.index(cell)
    except ValueError:
        index = GRID_CELLS.index("center")
    column, row = index % 3, index // 3
    return (
        int(width * (column * 2 + 1) / 6),
        int(height * (row * 2 + 1) / 6),
    )


def busiest_cell(video_path: Path | str, at: float) -> str | None:
    """Which of the nine cells changed most around `at`, or None if nothing did.

    Returns None rather than a best guess when the shot is still: an accent with
    nowhere to go is better dropped than dropped somewhere arbitrary.
    """
    import cv2
    import numpy as np

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return None
    try:
        frames = []
        for moment in (max(0.0, at - WINDOW_SECONDS / 2), at + WINDOW_SECONDS / 2):
            capture.set(cv2.CAP_PROP_POS_MSEC, moment * 1000.0)
            ok, frame = capture.read()
            if not ok:
                return None
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.int16))
    finally:
        capture.release()

    if frames[0].shape != frames[1].shape:
        return None
    difference = np.abs(frames[1] - frames[0])
    if float(difference.mean()) < MIN_CHANGE / 3:
        return None

    height, width = difference.shape
    best_cell, best_score = None, 0.0
    for index, cell in enumerate(GRID_CELLS):
        column, row = index % 3, index // 3
        patch = difference[
            row * height // 3:(row + 1) * height // 3,
            column * width // 3:(column + 1) * width // 3,
        ]
        score = float(patch.mean())
        if score > best_score:
            best_cell, best_score = cell, score
    return best_cell if best_score >= MIN_CHANGE else None
