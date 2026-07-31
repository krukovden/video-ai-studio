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

# The same question asked of the whole frame rather than of one ninth of it:
# below this, the picture is holding still and there is nothing to point at or to
# locate.
FRAME_CHANGE_FLOOR = MIN_CHANGE / 3

# How far either side of a reported moment to look for the change it names. A
# model describing a clip it watched at roughly a frame a second can be most of a
# second out; search much wider and this starts finding a different action.
ONSET_SEARCH_SECONDS = 0.75

# How finely that window is sampled. Coarser than a frame on purpose: an action
# takes several frames to become visible, and a grid this size is what makes two
# runs pointing at the same action resolve to the same answer.
ONSET_STEP_SECONDS = 0.1


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


def _open(video_path: Path | str):
    """The capture, or an error naming the file.

    A file that cannot be opened is a broken project, not a still shot. Both used
    to come back as None here, so a missing proxy read as "nothing is moving" and
    every accent quietly kept the position a model had guessed — which is the one
    outcome this module exists to prevent.
    """
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise OSError(f"cannot open video for motion measurement: {video_path}")
    return capture


def busiest_cell(video_path: Path | str, at: float) -> str | None:
    """Which of the nine cells changed most around `at`, or None if nothing did.

    Returns None rather than a best guess when the shot is still: an accent with
    nowhere to go is better dropped than dropped somewhere arbitrary.
    """
    import cv2
    import numpy as np

    capture = _open(video_path)
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
    if float(difference.mean()) < FRAME_CHANGE_FLOOR:
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


def change_onset(
    video_path: Path | str,
    at: float,
    search: float = ONSET_SEARCH_SECONDS,
    step: float = ONSET_STEP_SECONDS,
) -> float | None:
    """When the picture near `at` really starts changing, or None if it never does.

    A model can say what happened in a clip; the second it says it happened at is
    a number it wrote down, and anything that turns such a number into a cut has
    handed the edit's geometry to the model. This turns the number back into a
    question about the footage and answers it by subtraction.

    Sampled on an ABSOLUTE grid of multiples of `step`, not on one relative to
    `at`, so the answer depends on the footage rather than on the guess that
    pointed at it: two runs reporting the same action a quarter of a second apart
    compare the same frames and come back with the same onset. That is the whole
    point — the moment this returns goes on to set a clip's duration, and a
    duration must not move because a model rounded itself differently.

    The onset rather than the peak: the first interval whose change is at least
    half the strongest in the window. A pop is at its loudest several frames after
    it begins, and cutting to the loudest frame is already inside the action.
    """
    import cv2
    import numpy as np

    first = max(0.0, at - search)
    last = at + search
    grid = [
        index * step
        for index in range(int(first / step + 0.999), int(last / step) + 1)
    ]
    if len(grid) < 2:
        return None

    capture = _open(video_path)
    try:
        samples: list[tuple[float, object]] = []
        for moment in grid:
            capture.set(cv2.CAP_PROP_POS_MSEC, moment * 1000.0)
            ok, frame = capture.read()
            if not ok:
                # Past the end of this clip. What was read before it still
                # measures the window that exists.
                break
            samples.append(
                (moment, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.int16))
            )
    finally:
        capture.release()

    changes = [
        (samples[index][0], float(np.abs(samples[index + 1][1] - samples[index][1]).mean()))
        for index in range(len(samples) - 1)
        if samples[index][1].shape == samples[index + 1][1].shape
    ]
    if not changes:
        return None
    peak = max(score for _, score in changes)
    if peak < FRAME_CHANGE_FLOOR:
        return None
    threshold = max(FRAME_CHANGE_FLOOR, peak / 2)
    for moment, score in changes:
        if score >= threshold:
            return round(moment, 3)
    return None
