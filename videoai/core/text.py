"""Text for the final render: a real typeface rasterised to an RGBA strip.

Every title, caption and card in this pipeline is text over picture, and the
cheapest visible improvement available to it is not another effect — it is text
that does not look like a 1980s plotter. OpenCV's Hershey faces are stroked
vector outlines with no kerning, no weight and no rounded terminals; they read as
CAD annotations sitting on a child's video.

So text is drawn here with Pillow and a real font file. The strip comes out as
RGBA — a semi-transparent plate with hard glyphs on it — which is what both
consumers expect: the alpha graphics track composites it onto the picture, and the
intro/outro cards composite it onto their background.

Font resolution is a chain, not a requirement. macOS ships the rounded face this
project prefers; Linux CI ships neither it nor Arial, and Pillow's own bundled
face is the last resort. None of those paths may raise: a missing font is a
slightly plainer title, never a failed delivery.
"""
from __future__ import annotations

from functools import cache
from pathlib import Path

# Most preferred first. Rounded before Arial Bold because a friendlier face suits
# a kids' channel; Arial Bold before Helvetica because a `.ttc` collection needs a
# face index. The last two are what a Debian-based CI image usually has, so font
# resolution behaves the same way on a machine that has never seen macOS.
FONT_CANDIDATES = (
    "/System/Library/Fonts/Supplemental/Arial Rounded Bold.ttf",
    "/Library/Fonts/Arial Rounded Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/Library/Fonts/Arial Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
)

# Text is white with a dark outline so it stays legible over a busy frame even
# where the plate behind it is fully transparent.
TEXT_COLOUR = (255, 255, 255, 255)
STROKE_COLOUR = (0, 0, 0, 255)
PLATE_COLOUR = (24, 20, 16)

_MINIMUM_FONT_SIZE = 8
# A reference string carrying both an accented cap and descenders, so a line's
# height is the face's real ascender-to-descender span rather than the height of
# whichever glyphs this particular title happens to use.
_METRIC_SAMPLE = "ÁÇgjpqy"
_LINE_SPACING = 1.22


@cache
def resolve_font() -> Path | None:
    """The first installed font from `FONT_CANDIDATES`, or None when there is none.

    None is not an error: `load_font` falls back to the face Pillow bundles, so a
    machine with no system fonts still renders readable text.
    """
    for candidate in FONT_CANDIDATES:
        path = Path(candidate)
        if path.is_file():
            return path
    return None


@cache
def load_font(size: int):
    """A Pillow font at `size` points, from the best face this machine has.

    Cached because fitting a string to a strip tries a range of sizes, and the
    delivery render fits one strip per caption.
    """
    from PIL import ImageFont

    size = max(_MINIMUM_FONT_SIZE, int(size))
    path = resolve_font()
    if path is not None:
        try:
            return ImageFont.truetype(str(path), size=size)
        except OSError:
            # An installed but unreadable or unsupported face (a broken `.ttc`
            # index, a font the build's FreeType cannot parse) must not be fatal.
            pass
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        # Pillow before 10.1 has an unscalable default face. Still text.
        return ImageFont.load_default()


def _measure(font, text: str) -> tuple[int, int]:
    left, top, right, bottom = font.getbbox(text)
    return right - left, bottom - top


def _line_height(font) -> int:
    return max(1, int(_measure(font, _METRIC_SAMPLE)[1] * _LINE_SPACING))


def wrap_to_width(text: str, font, max_width: int) -> list[str]:
    """`text` broken into lines that fit `max_width` at this font size.

    Measured with the font rather than estimated from the point size: the old
    character-count estimate had to be pessimistic to be safe, which shrank every
    title that contained a few wide letters.
    """
    lines: list[str] = []
    current = ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        if current and font.getlength(candidate) > max_width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines or [""]


def fit_text(
    text: str, width: int, height: int, max_lines: int
) -> tuple[list[str], object, int]:
    """The largest font size at which `text` fits `width`x`height` in `max_lines`.

    Returned as the wrapped lines, the font, and the line height to advance by.
    Found by bisection rather than by stepping down a point at a time: a 1080p
    title strip is 150px tall, and trying every size for every caption of a
    three-minute video is thousands of font loads for the same answer.
    """
    largest = max(_MINIMUM_FONT_SIZE, int(height * 0.9))

    def fits(size: int) -> tuple[list[str], int] | None:
        font = load_font(size)
        lines = wrap_to_width(text, font, width)
        if len(lines) > max_lines:
            return None
        line_height = _line_height(font)
        if line_height * len(lines) > height:
            return None
        if any(font.getlength(line) > width for line in lines):
            return None
        return lines, line_height

    best: tuple[int, list[str], int] | None = None
    low, high = _MINIMUM_FONT_SIZE, largest
    while low <= high:
        middle = (low + high) // 2
        outcome = fits(middle)
        if outcome is None:
            high = middle - 1
        else:
            best = (middle, *outcome)
            low = middle + 1
    if best is None:
        # Nothing fits, which for a strip this small means a very long single
        # word. Take the smallest size and keep the lines there is room for
        # rather than drawing off both edges.
        font = load_font(_MINIMUM_FONT_SIZE)
        line_height = _line_height(font)
        lines = wrap_to_width(text, font, width)
        return lines[: max(1, height // line_height)], font, line_height
    size, lines, line_height = best
    return lines, load_font(size), line_height


def render_text_image(
    dst: Path,
    text: str,
    width: int,
    height: int,
    *,
    plate_alpha: float = 0.0,
    max_lines: int = 3,
) -> None:
    """Write an RGBA strip of `width`x`height` holding `text`, sized to fit."""
    from PIL import Image, ImageDraw

    dst.parent.mkdir(parents=True, exist_ok=True)
    plate = (*PLATE_COLOUR, int(round(255 * max(0.0, min(1.0, plate_alpha)))))
    image = Image.new("RGBA", (max(1, width), max(1, height)), plate)
    draw = ImageDraw.Draw(image)

    padding = max(8, width // 30)
    lines, font, line_height = fit_text(
        text.strip(), max(1, width - 2 * padding), max(1, height - padding), max_lines
    )
    stroke = max(2, int(getattr(font, "size", 16) / 12))
    top = (height - line_height * len(lines)) // 2
    for index, line in enumerate(lines):
        draw.text(
            ((width - font.getlength(line)) / 2, top + index * line_height),
            line,
            font=font,
            fill=TEXT_COLOUR,
            stroke_width=stroke,
            stroke_fill=STROKE_COLOUR,
        )
    image.save(dst)
