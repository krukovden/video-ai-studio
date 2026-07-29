"""Text for the final render: a real font through ffmpeg when the build has one,
a rasterised strip otherwise.

`drawtext` is the right way to burn a title in, but it only exists in ffmpeg
builds linked against libfreetype, and Homebrew's macOS bottle currently is not
(`ffmpeg -filters | grep drawtext` comes back empty on this machine). A title
card is not worth failing a render over, so text is also renderable to an RGBA
PNG here and overlaid as an ordinary image. Both routes hand the rest of the
filter graph the same thing — one RGBA stream to fade and overlay — so the
polish stage only branches on how that stream is produced.
"""
from __future__ import annotations

import re
import subprocess
from functools import cache
from pathlib import Path

# Fonts shipped with macOS, most preferred first. `.ttf` before `.ttc`: a
# collection needs a face index that drawtext cannot be told about.
FONT_CANDIDATES = (
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Verdana Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/Library/Fonts/Arial Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
)


@cache
def resolve_font() -> Path | None:
    """The preferred macOS font, or None when none of them is installed.

    None is not an error: `drawtext` falls back to its own built-in default when
    it is given no `fontfile`, and a missing font must never stop a render.
    """
    for candidate in FONT_CANDIDATES:
        path = Path(candidate)
        if path.is_file():
            return path
    return None


@cache
def drawtext_available() -> bool:
    """Whether this ffmpeg build has the `drawtext` filter (needs libfreetype)."""
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-filters"], capture_output=True, text=True
    )
    if result.returncode != 0:
        return False
    return re.search(r"^\s*\S+\s+drawtext\s", result.stdout, re.MULTILINE) is not None


def escape_filter_path(path: Path | str) -> str:
    """Escape a path for use as a filter option value inside a filter graph."""
    text = str(path)
    for char in ("\\", "'", ":", ",", "[", "]", ";"):
        text = text.replace(char, "\\" + char)
    return text


def drawtext_filter(
    text_file: Path, font_size: int, box_alpha: float, y_expression: str
) -> str:
    """A `drawtext` invocation reading its text from a file.

    The text comes from a file rather than the graph so a title containing a
    colon, comma or apostrophe — all filter-graph syntax — needs no escaping, and
    `expansion=none` stops `%{...}` in a title being read as an expression. The
    box is drawn by drawtext itself so that `alpha` fades the plate and the text
    together; a separately drawn box would stay solid while the text faded.
    """
    font = resolve_font()
    parts = [
        f"textfile={escape_filter_path(text_file)}",
        "expansion=none",
        "fontcolor=white",
        f"fontsize={font_size}",
        "borderw=2",
        "bordercolor=black@0.7",
        "x=(w-text_w)/2",
        f"y={y_expression}",
    ]
    if font is not None:
        parts.insert(0, f"fontfile={escape_filter_path(font)}")
    if box_alpha > 0:
        parts += ["box=1", f"boxcolor=black@{box_alpha}", "boxborderw=24"]
    return "drawtext=" + ":".join(parts)


def _wrap(text: str, max_chars: int) -> list[str]:
    lines: list[str] = []
    current = ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        if current and len(candidate) > max_chars:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def wrap_text(text: str, width: int, font_size: int, max_lines: int = 2) -> str:
    """`text` broken into at most `max_lines` lines that roughly fit `width`.

    `drawtext` does not wrap, and it cannot be asked how wide a string will be,
    so the estimate below (a little over half the point size per character for a
    proportional face) is the only handle we have on a long title.
    """
    max_chars = max(4, int(width / max(1.0, font_size * 0.55)))
    lines = _wrap(text, max_chars)
    if len(lines) > max_lines:
        lines = lines[: max_lines - 1] + [" ".join(lines[max_lines - 1:])]
    return "\n".join(lines)


def render_text_image(
    dst: Path,
    text: str,
    width: int,
    height: int,
    *,
    plate_alpha: float = 0.0,
    max_lines: int = 3,
) -> None:
    """Write an RGBA strip of `width`x`height` holding `text`, sized to fit.

    Used when the ffmpeg build has no `drawtext`. OpenCV is already a dependency
    of this project and its Hershey faces are the only text rasteriser available
    without one, so the strip is drawn here and overlaid as an image.
    """
    import cv2
    import numpy as np

    dst.parent.mkdir(parents=True, exist_ok=True)
    image = np.zeros((height, width, 4), dtype=np.uint8)
    if plate_alpha > 0:
        image[:, :, :3] = (24, 20, 16)
        image[:, :, 3] = int(round(255 * plate_alpha))

    font = cv2.FONT_HERSHEY_DUPLEX
    padding = max(8, width // 30)
    available_width = width - 2 * padding
    available_height = height - padding

    def _lines_at(scale: float) -> tuple[list[str], int]:
        (unit_width, unit_height), _ = cv2.getTextSize("M", font, scale, 2)
        per_line = max(4, int(available_width / max(1.0, unit_width * 0.62)))
        return _wrap(text, per_line), max(1, int(unit_height * 1.9))

    smallest = 0.25
    chosen: tuple[list[str], float] | None = None
    for candidate in [step / 20 for step in range(60, int(smallest * 20) - 1, -1)]:
        candidate_lines, line_height = _lines_at(candidate)
        if len(candidate_lines) > max_lines:
            continue
        widest = max(
            cv2.getTextSize(line, font, candidate, 2)[0][0] for line in candidate_lines
        )
        if widest <= available_width and line_height * len(candidate_lines) <= available_height:
            chosen = (candidate_lines, candidate)
            break
    if chosen is None:
        # Nothing fits: rather than draw a line that runs off both edges, take the
        # smallest size and keep only the lines the strip has room for.
        candidate_lines, line_height = _lines_at(smallest)
        chosen = (candidate_lines[: max(1, available_height // line_height)], smallest)

    lines, scale = chosen
    thickness = max(1, int(round(scale * 1.6)))
    (_, glyph_height), _ = cv2.getTextSize("Mg", font, scale, thickness)
    line_height = int(glyph_height * 1.9)
    total = line_height * len(lines)
    top = (height - total) // 2 + glyph_height
    for index, line in enumerate(lines):
        (line_width, _), _ = cv2.getTextSize(line, font, scale, thickness)
        origin = ((width - line_width) // 2, top + index * line_height)
        # Dark stroke first, white on top: legible over a busy frame even where
        # the plate is transparent.
        cv2.putText(image, line, origin, font, scale, (0, 0, 0, 255),
                    thickness + 3, cv2.LINE_AA)
        cv2.putText(image, line, origin, font, scale, (255, 255, 255, 255),
                    thickness, cv2.LINE_AA)
    if not cv2.imwrite(str(dst), image):
        raise RuntimeError(f"could not write text image: {dst}")
