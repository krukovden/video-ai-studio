"""Draw the starter sprite library with Pillow, so effects work with no API calls.

Every sprite here is a flat comic drawing: bold black outline, a small palette of
saturated fills, no gradients and no text. That is deliberate — it is the look a
kids' channel wants, it is what a generated replacement will be prompted for (see
`docs/EFFECTS-LIBRARY.md`), and it survives being scaled to a quarter of its size
over busy 4K footage, which a soft or detailed drawing does not.

Everything is drawn `_SUPERSAMPLE` times larger than it ships and reduced with a
Lanczos filter at the end. Pillow's own polygon and arc drawing is aliased, and a
hard-edged shape with a jagged outline composited over real footage looks like a
decoding fault rather than a cartoon.

Nothing in this module is imported by the pipeline: it produces files that the
pipeline then reads. Re-run it with `videoai seed-effects` after changing a
drawing, and commit the PNGs it writes.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from math import cos, pi, sin
from pathlib import Path

import yaml

SIZE = 512
_SUPERSAMPLE = 4

BLACK = (24, 22, 28, 255)
WHITE = (255, 255, 255, 255)
YELLOW = (255, 209, 61, 255)
ORANGE = (255, 122, 41, 255)
RED = (240, 74, 74, 255)
PINK = (255, 122, 176, 255)
CYAN = (77, 208, 240, 255)
BLUE = (60, 120, 230, 255)
GREEN = (86, 205, 128, 255)
PURPLE = (158, 116, 235, 255)

CONFETTI_COLOURS = (YELLOW, ORANGE, RED, PINK, CYAN, GREEN, PURPLE, BLUE)

# The bubble's geometry, in supersampled-canvas fractions of `SIZE`. Kept here
# rather than inline because the manifest's nine-patch insets are derived from the
# very same numbers, and a bubble whose insets disagree with its drawing stretches
# its own outline.
BUBBLE_BODY = (8, 8, 504, 376)
BUBBLE_RADIUS = 64
BUBBLE_STROKE = 14
BUBBLE_TAIL = ((110, 348), (150, 478), (208, 356))
# The centre band must be pure interior, so it has to stop above the bottom
# rounded corners (which start at body bottom minus the radius).
BUBBLE_CENTRE_BOTTOM = 300
BUBBLE_INSETS = {
    # Wide enough to contain the whole tail, so the tail never stretches.
    "left": 224,
    "top": 96,
    "right": 96,
    "bottom": SIZE - BUBBLE_CENTRE_BOTTOM,
}


@dataclass(frozen=True)
class _Canvas:
    """A supersampled drawing surface plus the scale factor to its shipped size."""

    image: object
    draw: object
    scale: int

    def at(self, *values: float) -> tuple[int, ...]:
        return tuple(round(value * self.scale) for value in values)


def _canvas(width: int = SIZE, height: int = SIZE) -> _Canvas:
    from PIL import Image, ImageDraw

    image = Image.new(
        "RGBA", (width * _SUPERSAMPLE, height * _SUPERSAMPLE), (0, 0, 0, 0)
    )
    return _Canvas(image=image, draw=ImageDraw.Draw(image), scale=_SUPERSAMPLE)


def _reduce(canvas: _Canvas, width: int = SIZE, height: int = SIZE):
    from PIL import Image

    return canvas.image.resize((width, height), Image.LANCZOS)


def _star_points(
    centre: tuple[float, float],
    outer: float,
    inner: float,
    spikes: int,
    phase: float = 0.0,
) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for index in range(spikes * 2):
        angle = phase + index * pi / spikes
        radius = outer if index % 2 == 0 else inner
        points.append((centre[0] + cos(angle) * radius, centre[1] + sin(angle) * radius))
    return points


def _grow(
    points: list[tuple[float, float]], centre: tuple[float, float], amount: float
) -> list[tuple[float, float]]:
    """`points` pushed `amount` pixels away from `centre`.

    How the bold outline is drawn: the black pass is the same polygon grown by the
    stroke width. Pillow's `polygon(outline=..., width=...)` centres the stroke on
    the path and mitres spikes badly at this weight.
    """
    grown: list[tuple[float, float]] = []
    for x, y in points:
        dx, dy = x - centre[0], y - centre[1]
        length = (dx * dx + dy * dy) ** 0.5 or 1.0
        grown.append((x + dx / length * amount, y + dy / length * amount))
    return grown


def _outlined_polygon(
    canvas: _Canvas,
    points: list[tuple[float, float]],
    centre: tuple[float, float],
    fill: tuple[int, int, int, int],
    stroke: float,
) -> None:
    scaled_centre = (centre[0] * canvas.scale, centre[1] * canvas.scale)
    scaled = [(x * canvas.scale, y * canvas.scale) for x, y in points]
    canvas.draw.polygon(
        _grow(scaled, scaled_centre, stroke * canvas.scale), fill=BLACK
    )
    canvas.draw.polygon(scaled, fill=fill)


def draw_comic_starburst():
    """A jagged impact star: yellow spikes, orange core, white spark."""
    canvas = _canvas()
    centre = (SIZE / 2, SIZE / 2)
    outer = _star_points(centre, 246, 118, 13, phase=-pi / 2)
    _outlined_polygon(canvas, outer, centre, YELLOW, 9)
    inner = _star_points(centre, 138, 62, 13, phase=-pi / 2 + pi / 13)
    _outlined_polygon(canvas, inner, centre, ORANGE, 6)
    spark = _star_points(centre, 62, 24, 4, phase=-pi / 2)
    canvas.draw.polygon(
        [(x * canvas.scale, y * canvas.scale) for x, y in spark], fill=WHITE
    )
    return _reduce(canvas)


def draw_speed_lines():
    """A whoosh: tapered streaks fanning out behind something that just moved.

    Directional rather than radial. An even ring of spokes around a clear centre
    is what a loading spinner looks like; speed reads from streaks that all point
    the same way, blunt at the trailing edge and tapering to a point ahead.
    """
    canvas = _canvas()
    # (y, length, thickness, colour) — the long fast ones in the middle of the fan.
    streaks = (
        (108.0, 250.0, 15.0, WHITE),
        (150.0, 330.0, 21.0, CYAN),
        (196.0, 420.0, 27.0, CYAN),
        (246.0, 470.0, 31.0, BLUE),
        (298.0, 410.0, 27.0, CYAN),
        (346.0, 322.0, 21.0, CYAN),
        (390.0, 242.0, 15.0, WHITE),
    )
    tip = 486.0
    stroke = 8.5

    def streak(y: float, length: float, thickness: float, pad: float):
        """The streak grown by `pad` on every edge.

        Built explicitly rather than by `_grow`, which pushes points away from a
        centroid: on a shape twenty times longer than it is thick that scales the
        length and leaves the long edges with almost no outline at all.
        """
        tail = tip - length
        half = thickness / 2 + pad
        return [
            (tip + pad, y),
            (tail + thickness * 0.9 - pad, y - half),
            (tail - pad, y - half),
            (tail - pad, y + half),
            (tail + thickness * 0.9 - pad, y + half),
        ]

    for y, length, thickness, colour in streaks:
        canvas.draw.polygon(
            [coordinate for point in streak(y, length, thickness, stroke)
             for coordinate in canvas.at(*point)],
            fill=BLACK,
        )
        canvas.draw.polygon(
            [coordinate for point in streak(y, length, thickness, 0.0)
             for coordinate in canvas.at(*point)],
            fill=colour,
        )
    return _reduce(canvas)


def draw_sparkle_stars():
    """Three four-pointed sparkles of different sizes, plus two small dots."""
    canvas = _canvas()
    sparkles = (
        ((188, 190), 148.0, WHITE),
        ((350, 306), 96.0, YELLOW),
        ((232, 388), 66.0, CYAN),
    )
    for centre, outer, fill in sparkles:
        points = _star_points(centre, outer, outer * 0.20, 4, phase=-pi / 2)
        _outlined_polygon(canvas, points, centre, fill, 8.5)
    for centre, radius in (((392, 156), 20.0), ((136, 336), 14.0)):
        box = canvas.at(
            centre[0] - radius - 6, centre[1] - radius - 6,
            centre[0] + radius + 6, centre[1] + radius + 6,
        )
        canvas.draw.ellipse(box, fill=BLACK)
        inner = canvas.at(
            centre[0] - radius, centre[1] - radius,
            centre[0] + radius, centre[1] + radius,
        )
        canvas.draw.ellipse(inner, fill=WHITE)
    return _reduce(canvas)


def draw_sound_rings():
    """Concentric rings radiating outwards, fading as they go: a loud noise."""
    canvas = _canvas()
    centre = SIZE / 2
    # Arcs, not closed circles: three concentric rings drawn all the way round is a
    # dartboard. Broken on both sides so the shape emits sound symmetrically and
    # works wherever on the frame it is placed.
    rings = ((112.0, 27.0, 255), (176.0, 23.0, 210), (238.0, 19.0, 155))
    spans = ((-52.0, 52.0), (128.0, 232.0))
    for radius, weight, alpha in rings:
        # Pillow strokes an arc inwards from its box, so each pass names the box
        # its own outer edge sits on.
        def arc(outer: float, width: float, colour: tuple[int, int, int, int]) -> None:
            box = canvas.at(centre - outer, centre - outer, centre + outer, centre + outer)
            for start, end in spans:
                canvas.draw.arc(
                    box, start=start, end=end,
                    fill=(colour[0], colour[1], colour[2], alpha),
                    width=round(width * canvas.scale),
                )

        arc(radius + weight / 2 + 7, weight + 14, BLACK)
        arc(radius + weight / 2, weight, CYAN)
    # A solid dot at the source, so the arcs have something to come from.
    canvas.draw.ellipse(canvas.at(centre - 46, centre - 46, centre + 46, centre + 46), fill=BLACK)
    canvas.draw.ellipse(canvas.at(centre - 33, centre - 33, centre + 33, centre + 33), fill=YELLOW)
    return _reduce(canvas)


def draw_confetti_burst():
    """A burst of flat comic confetti: squares, circles and streamers.

    Positions come from a seeded generator so the committed PNG is reproducible;
    an unseeded scatter would produce a different file on every run and make the
    library's fingerprint meaningless.
    """
    canvas = _canvas()
    rng = random.Random(20260729)
    # Thrown upwards from just below the bottom edge and spread over the whole
    # canvas: a burst measured from the middle piles up in the centre instead.
    centre = (SIZE / 2, SIZE * 1.02)
    for _ in range(46):
        angle = rng.uniform(-pi * 0.93, -pi * 0.07)
        distance = 320.0 * rng.random() ** 0.42
        x = centre[0] + cos(angle) * distance * 1.12
        y = centre[1] + sin(angle) * distance * 1.05
        colour = CONFETTI_COLOURS[rng.randrange(len(CONFETTI_COLOURS))]
        kind = rng.random()
        size = rng.uniform(16, 31)
        if kind < 0.45:
            spin = rng.uniform(0, pi)
            corners = [
                (
                    x + cos(spin + step * pi / 2) * size,
                    y + sin(spin + step * pi / 2) * size * 0.62,
                )
                for step in range(4)
            ]
            _outlined_polygon(canvas, corners, (x, y), colour, 4.5)
        elif kind < 0.78:
            box_outer = canvas.at(
                x - size - 4.5, y - size - 4.5, x + size + 4.5, y + size + 4.5
            )
            canvas.draw.ellipse(box_outer, fill=BLACK)
            canvas.draw.ellipse(canvas.at(x - size, y - size, x + size, y + size), fill=colour)
        else:
            spin = rng.uniform(0, pi)
            length = size * 2.4
            band = [
                (x - cos(spin) * length, y - sin(spin) * length),
                (x + cos(spin) * length * 0.4, y + sin(spin) * length * 0.4 - size * 0.7),
                (x + cos(spin) * length, y + sin(spin) * length),
                (x - cos(spin) * length * 0.4, y - sin(spin) * length * 0.4 + size * 0.7),
            ]
            _outlined_polygon(canvas, band, (x, y), colour, 4.5)
    return _reduce(canvas)


def _bubble_shape(canvas: _Canvas, inset: float, fill: tuple[int, int, int, int]) -> None:
    """The body and the tail as one silhouette.

    Drawn as two overlapping filled shapes at the same colour: the union gives a
    continuous outline on the black pass and a continuous interior on the white
    pass, which is far easier to keep clean than building one path for a rounded
    rectangle with a tail on it.
    """
    left, top, right, bottom = BUBBLE_BODY
    canvas.draw.rounded_rectangle(
        canvas.at(left + inset, top + inset, right - inset, bottom - inset),
        radius=round((BUBBLE_RADIUS - inset) * canvas.scale),
        fill=fill,
    )
    (ax, ay), (bx, by), (cx, cy) = BUBBLE_TAIL
    # The tail inset moves its tip up and its shoulders inwards, so the stroke
    # keeps an even weight round the point.
    canvas.draw.polygon(
        [
            *canvas.at(ax + inset * 0.9, ay),
            *canvas.at(bx, by - inset * 2.0),
            *canvas.at(cx - inset * 0.9, cy),
        ],
        fill=fill,
    )


def draw_speech_bubble():
    """A nine-patch speech bubble: white, bold outline, tail at the bottom left."""
    canvas = _canvas()
    _bubble_shape(canvas, 0.0, BLACK)
    _bubble_shape(canvas, BUBBLE_STROKE, WHITE)
    return _reduce(canvas)


# name -> (drawing function, manifest entry without `file`)
SEEDS: tuple[tuple[str, object, dict], ...] = (
    (
        "comic_starburst",
        draw_comic_starburst,
        {
            "tags": ["impact", "pop", "hit", "surprise", "reveal"],
            "expresses": "something popped, burst or landed hard, right now",
            "anchor": "center",
            "default_seconds": 0.7,
            "animation": "pop-in",
        },
    ),
    (
        "speed_lines",
        draw_speed_lines,
        {
            "tags": ["speed", "fast", "motion", "rush", "zoom"],
            "expresses": "this is happening fast, or something whipped past",
            "anchor": "center",
            "default_seconds": 0.6,
            "animation": "pulse",
        },
    ),
    (
        "sparkle_stars",
        draw_sparkle_stars,
        {
            "tags": ["sparkle", "shiny", "clean", "magic", "pretty", "reveal"],
            "expresses": "a clean, shiny or magical reveal — the nice-looking moment",
            "anchor": "center",
            "default_seconds": 1.0,
            "animation": "pulse",
        },
    ),
    (
        "sound_rings",
        draw_sound_rings,
        {
            "tags": ["sound", "loud", "squish", "squeak", "noise", "listen"],
            "expresses": "a noise the viewer should notice: a squish, a squeak, a bang",
            "anchor": "center",
            "default_seconds": 0.9,
            "animation": "pulse",
        },
    ),
    (
        "confetti_burst",
        draw_confetti_burst,
        {
            "tags": ["celebration", "success", "yay", "finale", "win"],
            "expresses": "it worked, it is finished, or it is worth celebrating",
            "anchor": "center",
            "default_seconds": 1.4,
            "animation": "drift-up",
        },
    ),
    (
        "speech_bubble",
        draw_speech_bubble,
        {
            "tags": ["reaction", "uh-oh", "joke", "voice", "comment", "oops"],
            "expresses": (
                "a short reaction in words, in a comic bubble with a tail — only for "
                "something the presenter actually said or plainly meant"
            ),
            "anchor": "bottom",
            "default_seconds": 1.6,
            "animation": "shake",
            "nine_patch": {
                "left": BUBBLE_INSETS["left"],
                "top": BUBBLE_INSETS["top"],
                "right": BUBBLE_INSETS["right"],
                "bottom": BUBBLE_INSETS["bottom"],
                # Room for the outline, plus 12px of air, on every side. The bottom
                # also has to clear the body's own bottom edge and the tail below it.
                "text_left": BUBBLE_STROKE + 20,
                "text_top": BUBBLE_STROKE + 20,
                "text_right": BUBBLE_STROKE + 20,
                "text_bottom": SIZE - BUBBLE_BODY[3] + BUBBLE_STROKE + 20,
                "max_lines": 3,
            },
        },
    ),
)

MANIFEST_HEADER = """\
# The cartoon-effects sprite library.
#
# Every entry is one RGBA PNG in this directory plus the metadata the pipeline
# needs to choose it, place it and animate it:
#
#   name             what the placement stage calls it; the LLM must use it exactly
#   file             the RGBA PNG, in this directory
#   tags             what it expresses, as words a story can be matched against
#   expresses        one sentence shown to the model instead of the picture
#   anchor           which point of the sprite lands on the grid cell's point
#                    (center | top | bottom | left | right)
#   default_seconds  how long one event of it runs
#   animation        a built-in motion the compositor implements:
#                    pop-in | pulse | shake | drift-up | none
#   nine_patch       present only for a sprite stretched around dynamic text
#
# THE DRAWINGS ARE SEEDS, NOT THE POINT. Everything here was drawn procedurally
# with Pillow (`videoai/logic/effect_seeds.py`, re-runnable with
# `uv run videoai seed-effects`) so that the effects stage works with zero API
# calls and zero subscriptions. ANY entry can be replaced by a better PNG — for
# example one generated with OpenAI's image model — by dropping a file with the
# SAME NAME into this directory. Nothing else changes: the manifest, the
# placement stage and the compositor never look at the pixels. Replacing a file
# re-renders the delivery (the library's content is in the polish stage's
# fingerprint) and does not re-plan the effects.
#
# docs/EFFECTS-LIBRARY.md has the prompt template, the cut-out steps and the
# size and transparency requirements.
#
# A replacement must be: RGBA with a genuinely transparent background, square-ish
# and at least 512px on its long edge, and free of text (text belongs in the
# nine-patch bubble, where the pipeline renders it in the video's own typeface).
"""


def seed_library(directory: Path) -> Path:
    """Redraw every seeded sprite and rewrite the manifest around it.

    Entries the seeds do not own are carried across untouched. The library ships
    with artwork this repository did not draw — the badge set — and those entries
    are not reproducible from `SEEDS`, so a plain overwrite would silently delete
    most of the vocabulary the effects stage is offered. Redrawing a seed must
    never cost the library everything beside it.
    """
    directory.mkdir(parents=True, exist_ok=True)
    entries: list[dict] = []
    for name, drawing, metadata in SEEDS:
        file = f"{name}.png"
        drawing().save(directory / file)
        entries.append({"name": name, "file": file, **metadata})

    path = directory / "manifest.yaml"
    seeded = {name for name, _, _ in SEEDS}
    if path.is_file():
        existing = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        entries.extend(
            entry
            for entry in existing.get("sprites") or []
            if isinstance(entry, dict) and entry.get("name") not in seeded
        )

    body = yaml.safe_dump({"sprites": entries}, sort_keys=False, allow_unicode=True)
    path.write_text(f"{MANIFEST_HEADER}\n{body}", encoding="utf-8")
    return path
