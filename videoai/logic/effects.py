"""The cartoon-effects library: what can be shown, where it goes, how it moves.

Effects are seasoning, not structure. This module owns the vocabulary the LLM is
offered and the geometry the compositor obeys, and nothing in it knows anything
about a particular video: an effect is chosen from a library by what it
*expresses*, placed by a grid cell, and animated by one of a small set of
built-in motions. That is what separates this from the version that hardcoded one
video's syringe into the renderer.

Three things live here because all three are shared by the placement stage (which
must only offer what exists) and the compositor (which must only render what was
offered):

- the manifest: a named, tagged sprite with an anchor, a duration and a motion;
- the nine-cell placement grid, inset by a safe-area margin;
- the animation curves, as pure functions of normalised progress.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import cos, pi, sin
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from videoai.core.store import hash_file, hash_parts

MANIFEST_FILENAME = "manifest.yaml"

# The nine cells an effect may be placed in. Named rather than numbered because
# the model writes them, and "bottom-left" cannot be off by one the way "7" can.
GRID_CELLS = (
    "top-left", "top-center", "top-right",
    "middle-left", "center", "middle-right",
    "bottom-left", "bottom-center", "bottom-right",
)

# How far in from the frame edge the grid lives. The same title-safe reasoning as
# `lower_third_y`: a sprite half off the edge of an overscanning TV reads as a
# rendering fault rather than as a comic accent.
SAFE_MARGIN = 0.07

# A sprite's drawn height as a fraction of the delivery frame's height.
SPRITE_SCALES = {"small": 0.14, "medium": 0.22, "large": 0.32}
# A speech bubble is sized by its text instead, so its scale caps the width it may
# grow to as a fraction of the frame's width.
BUBBLE_WIDTH_FRACTIONS = {"small": 0.26, "medium": 0.36, "large": 0.48}

ANIMATIONS = ("pop-in", "pulse", "shake", "drift-up", "none")
# Where a sprite's visual weight sits, i.e. which point of it is put on the cell's
# anchor point. A speech bubble's tail points down, so "bottom" puts the bubble
# above the cell and the tail on it.
ANCHORS = ("center", "top", "bottom", "left", "right")


class NinePatch(BaseModel):
    """Fixed corner insets, in pixels of the sprite's own file.

    A nine-patch is stretched to fit dynamic text: the four corners keep their
    pixels, the top and bottom edges stretch horizontally, the left and right
    edges stretch vertically, and the centre stretches both ways. Every inset
    therefore has to be wide enough to contain everything that must not distort —
    for the shipped speech bubble that means the left inset contains the whole
    tail, and the bottom inset starts above the rounded bottom corners.
    """

    model_config = ConfigDict(frozen=True)

    left: int
    top: int
    right: int
    bottom: int
    # Where text may be drawn, as insets from the edges of the *stretched* image.
    # Measured from the rendered edges rather than from the interior box because
    # only the corner bands keep a fixed size, so the interior's position is only
    # knowable relative to them.
    text_left: int
    text_top: int
    text_right: int
    text_bottom: int
    max_lines: int = 3

    @property
    def minimum_size(self) -> tuple[int, int]:
        return self.left + self.right, self.top + self.bottom


class Sprite(BaseModel):
    """One entry of `assets/effects/manifest.yaml`."""

    model_config = ConfigDict(frozen=True)

    name: str
    file: str
    # What the sprite expresses, in the words the model is asked to match against.
    tags: tuple[str, ...] = ()
    expresses: str = ""
    anchor: str = "center"
    default_seconds: float = 0.8
    animation: str = "none"
    nine_patch: NinePatch | None = None

    @field_validator("animation")
    @classmethod
    def _known_animation(cls, value: str) -> str:
        if value not in ANIMATIONS:
            raise ValueError(
                f"unknown animation {value!r}; the compositor implements "
                + ", ".join(ANIMATIONS)
            )
        return value

    @field_validator("anchor")
    @classmethod
    def _known_anchor(cls, value: str) -> str:
        if value not in ANCHORS:
            raise ValueError(
                f"unknown anchor {value!r}; known anchors are " + ", ".join(ANCHORS)
            )
        return value

    @property
    def takes_text(self) -> bool:
        """True for a nine-patch: its size comes from the text it is given, so a
        nine-patch without text has no size and nothing to say."""
        return self.nine_patch is not None


class EffectLibrary(BaseModel):
    model_config = ConfigDict(frozen=True)

    directory: Path
    sprites: tuple[Sprite, ...] = Field(default_factory=tuple)

    def names(self) -> tuple[str, ...]:
        return tuple(sprite.name for sprite in self.sprites)

    def get(self, name: str) -> Sprite:
        for sprite in self.sprites:
            if sprite.name == name:
                return sprite
        raise KeyError(name)

    def path_of(self, sprite: Sprite) -> Path:
        return self.directory / sprite.file

    def catalogue(self) -> str:
        """The library as the placement prompt sees it: name, what it expresses,
        its tags and how long it naturally runs."""
        lines: list[str] = []
        for sprite in self.sprites:
            tags = ", ".join(sprite.tags)
            text = " — takes short text" if sprite.takes_text else ""
            lines.append(
                f"- {sprite.name} [{tags}] {sprite.default_seconds:.1f}s: "
                f"{sprite.expresses}{text}"
            )
        return "\n".join(lines)


def default_library_dir() -> Path:
    """`assets/effects` beside the repository's own contract file.

    Resolved from the package rather than from the working directory so the
    library is found whichever project the CLI was pointed at.
    """
    return Path(__file__).resolve().parents[2] / "assets" / "effects"


def manifest_path(directory: Path | None = None) -> Path:
    return (directory or default_library_dir()) / MANIFEST_FILENAME


def load_library(directory: Path | None = None) -> EffectLibrary:
    """Read the manifest. A missing library is empty, not an error: the effects
    stage then has nothing to offer and produces no events, which is a valid
    outcome for the same reason a calm video needs no effects."""
    directory = directory or default_library_dir()
    path = manifest_path(directory)
    if not path.is_file():
        return EffectLibrary(directory=directory)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    entries = raw.get("sprites") or []
    if not isinstance(entries, list):
        raise RuntimeError(f"{path}: 'sprites' must be a list of entries")
    sprites = tuple(Sprite(**entry) for entry in entries)
    names = [sprite.name for sprite in sprites]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise RuntimeError(f"{path}: duplicate sprite names: {', '.join(duplicates)}")
    library = EffectLibrary(directory=directory, sprites=sprites)
    for sprite in sprites:
        file = library.path_of(sprite)
        if not file.is_file():
            raise RuntimeError(
                f"{path}: sprite {sprite.name!r} names {sprite.file}, which is not in "
                f"{directory}"
            )
    return library


def manifest_fingerprint(directory: Path | None = None) -> str:
    """A digest of the placement vocabulary: which sprites exist and what they say
    they express. This is all the placement stage reads, so replacing a PNG with a
    better drawing of the same thing must not re-plan the effects."""
    path = manifest_path(directory)
    return hash_parts(path.read_text(encoding="utf-8") if path.is_file() else "")


def library_fingerprint(directory: Path | None = None) -> str:
    """A digest of the manifest *and* every sprite file's content.

    The compositor draws the pixels, so swapping in a generated PNG under the same
    name has to re-render the delivery even though nothing about the placement
    changed.
    """
    directory = directory or default_library_dir()
    path = manifest_path(directory)
    parts = [path.read_text(encoding="utf-8") if path.is_file() else ""]
    for file in sorted(directory.glob("*.png")) if directory.is_dir() else []:
        parts.append(f"{file.name}:{hash_file(file)}")
    return hash_parts(*parts)


@dataclass(frozen=True)
class SpriteTransform:
    """One frame of a built-in motion, in units of the sprite's own size."""

    scale: float = 1.0
    dx: float = 0.0
    dy: float = 0.0
    rotation: float = 0.0
    alpha: float = 1.0


def _fade_envelope(progress: float) -> float:
    """Up fast, down a little slower, so an accent never appears or vanishes on a
    single frame. Every animation is multiplied by this."""
    return max(0.0, min(1.0, progress / 0.10, (1.0 - progress) / 0.16))


def animation_transform(animation: str, progress: float) -> SpriteTransform:
    """The transform for `animation` at `progress` through its duration.

    Pure arithmetic on purpose: the compositor renders thousands of frames and
    every one of them has to be reproducible from the event alone.
    """
    progress = max(0.0, min(1.0, progress))
    alpha = _fade_envelope(progress)
    if animation == "pop-in":
        # Overshoot, then settle: a sprite that arrives at exactly its final size
        # reads as a cut, not as a pop.
        if progress < 0.32:
            scale = 1.18 * (1.0 - (1.0 - progress / 0.32) ** 3)
        elif progress < 0.55:
            step = (progress - 0.32) / 0.23
            scale = 1.18 - 0.18 * step * step * (3.0 - 2.0 * step)
        else:
            scale = 1.0
        return SpriteTransform(scale=scale, alpha=alpha)
    if animation == "pulse":
        return SpriteTransform(
            scale=1.0 + 0.09 * sin(2.0 * pi * 1.5 * progress), alpha=alpha
        )
    if animation == "shake":
        # Decaying so the last frames are still rather than frozen mid-wobble.
        return SpriteTransform(
            rotation=7.0 * sin(2.0 * pi * 4.0 * progress) * (1.0 - progress),
            alpha=alpha,
        )
    if animation == "drift-up":
        return SpriteTransform(
            scale=1.0 + 0.10 * progress,
            dy=-0.34 * progress,
            alpha=alpha * (1.0 - 0.7 * progress),
        )
    return SpriteTransform(alpha=alpha)


def _cell_parts(cell: str) -> tuple[str, str]:
    if cell == "center":
        return "middle", "center"
    row, _, column = cell.partition("-")
    return row, column


def cell_anchor_point(cell: str, frame: tuple[int, int]) -> tuple[int, int]:
    """The point inside the safe area that `cell` names."""
    if cell not in GRID_CELLS:
        raise ValueError(f"unknown screen position: {cell!r}")
    width, height = frame
    margin_x = width * SAFE_MARGIN
    margin_y = height * SAFE_MARGIN
    usable_width = width - 2 * margin_x
    usable_height = height - 2 * margin_y
    row, column = _cell_parts(cell)
    fractions = {"left": 1 / 6, "center": 0.5, "right": 5 / 6}
    rows = {"top": 1 / 6, "middle": 0.5, "bottom": 5 / 6}
    return (
        round(margin_x + usable_width * fractions[column]),
        round(margin_y + usable_height * rows[row]),
    )


def place_sprite(
    size: tuple[int, int], cell: str, anchor: str, frame: tuple[int, int]
) -> tuple[int, int]:
    """The top-left pixel at which a sprite of `size` goes for this cell.

    The anchor decides which point of the sprite lands on the cell's point; the
    result is then pulled back inside the safe area so an accent placed in a
    corner cell cannot hang off the frame. A sprite too big for the safe area is
    centred in it rather than clamped to one edge.
    """
    width, height = frame
    sprite_width, sprite_height = size
    anchor_x, anchor_y = cell_anchor_point(cell, frame)
    offsets = {
        "center": (sprite_width / 2, sprite_height / 2),
        "top": (sprite_width / 2, 0.0),
        "bottom": (sprite_width / 2, float(sprite_height)),
        "left": (0.0, sprite_height / 2),
        "right": (float(sprite_width), sprite_height / 2),
    }
    offset_x, offset_y = offsets[anchor]
    x = anchor_x - offset_x
    y = anchor_y - offset_y
    margin_x = width * SAFE_MARGIN
    margin_y = height * SAFE_MARGIN
    if sprite_width <= width - 2 * margin_x:
        x = min(max(x, margin_x), width - margin_x - sprite_width)
    else:
        x = (width - sprite_width) / 2
    if sprite_height <= height - 2 * margin_y:
        y = min(max(y, margin_y), height - margin_y - sprite_height)
    else:
        y = (height - sprite_height) / 2
    return round(x), round(y)


def sprite_target_height(scale: str, frame_height: int) -> int:
    if scale not in SPRITE_SCALES:
        raise ValueError(f"unknown scale: {scale!r}")
    return max(8, round(frame_height * SPRITE_SCALES[scale]))


def bubble_max_width(scale: str, frame_width: int) -> int:
    if scale not in BUBBLE_WIDTH_FRACTIONS:
        raise ValueError(f"unknown scale: {scale!r}")
    return max(64, round(frame_width * BUBBLE_WIDTH_FRACTIONS[scale]))


def nine_patch_resize(image, spec: NinePatch, size: tuple[int, int]):
    """Stretch `image` to `size` keeping its four corners pixel-for-pixel.

    Takes and returns a Pillow RGBA image. The edges are resized along one axis
    only and the centre along both, which is what keeps a bold outline the same
    weight all the way round a bubble that grew to fit three lines of text.
    """
    from PIL import Image

    minimum_width, minimum_height = spec.minimum_size
    target_width = max(minimum_width, size[0])
    target_height = max(minimum_height, size[1])
    source_width, source_height = image.size
    left, top = spec.left, spec.top
    right, bottom = spec.right, spec.bottom

    middle_source_width = max(1, source_width - left - right)
    middle_source_height = max(1, source_height - top - bottom)
    middle_width = target_width - left - right
    middle_height = target_height - top - bottom

    out = Image.new("RGBA", (target_width, target_height), (0, 0, 0, 0))
    columns = (
        (0, left, 0, left),
        (left, middle_source_width, left, middle_width),
        (source_width - right, right, target_width - right, right),
    )
    rows = (
        (0, top, 0, top),
        (top, middle_source_height, top, middle_height),
        (source_height - bottom, bottom, target_height - bottom, bottom),
    )
    for source_x, source_w, dest_x, dest_w in columns:
        for source_y, source_h, dest_y, dest_h in rows:
            if source_w <= 0 or source_h <= 0 or dest_w <= 0 or dest_h <= 0:
                continue
            tile = image.crop((source_x, source_y, source_x + source_w, source_y + source_h))
            if (dest_w, dest_h) != (source_w, source_h):
                tile = tile.resize((dest_w, dest_h), Image.NEAREST)
            out.paste(tile, (dest_x, dest_y))
    return out


_SMALLEST_BUBBLE_TYPE = 12


@lru_cache(maxsize=64)
def _bubble_font(size: int):
    from videoai.core.text import load_font

    return load_font(size)


def _hard_wrap(
    lines: list[str], font, content_width: int, max_lines: int
) -> list[str]:
    """Break any line still wider than `content_width` at a character boundary.

    The last resort under word wrapping and type-size reduction, and the only one
    that can bound the result: an unbreakable string has no size at which it fits.
    """
    broken: list[str] = []
    for line in lines:
        if font.getlength(line) <= content_width:
            broken.append(line)
            continue
        current = ""
        for character in line:
            if current and font.getlength(current + character) > content_width:
                broken.append(current)
                current = character
            else:
                current += character
        if current:
            broken.append(current)
    return broken[:max_lines] or [""]


def render_speech_bubble(
    sprite_path: Path,
    spec: NinePatch,
    text: str,
    max_width: int,
    dst: Path,
) -> tuple[int, int]:
    """Write an RGBA bubble stretched around `text`; return its size.

    The type size is a fixed fraction of `max_width` and shrinks only when the
    text will not fit in `spec.max_lines`, so two bubbles in one video are drawn
    with the same lettering however different their words are.
    """
    from PIL import Image, ImageDraw

    from videoai.core.text import (
        STROKE_COLOUR,
        line_height,
        wrap_to_width,
    )

    source = Image.open(sprite_path).convert("RGBA")
    content_width = max(16, max_width - spec.text_left - spec.text_right)
    size = max(_SMALLEST_BUBBLE_TYPE, round(max_width * 0.115))
    while True:
        font = _bubble_font(size)
        lines = wrap_to_width(text.strip(), font, content_width)
        too_wide = any(font.getlength(line) > content_width for line in lines)
        if (len(lines) <= spec.max_lines and not too_wide) or size <= _SMALLEST_BUBBLE_TYPE:
            break
        size = max(_SMALLEST_BUBBLE_TYPE, int(size * 0.88))
    # `wrap_to_width` can only break on spaces, so at the smallest type size a
    # single very long word is still wider than the box. Broken by character here
    # rather than allowed to grow the bubble past the width it was given: the
    # placement grid budgeted that width, and a bubble that ignores it would run
    # off the frame.
    lines = _hard_wrap(lines, font, content_width, spec.max_lines)
    step = line_height(font)
    text_width = max(1, round(max(font.getlength(line) for line in lines)))
    text_height = max(1, step * len(lines))

    bubble = nine_patch_resize(
        source,
        spec,
        (
            text_width + spec.text_left + spec.text_right,
            text_height + spec.text_top + spec.text_bottom,
        ),
    )
    draw = ImageDraw.Draw(bubble)
    box_left = spec.text_left
    box_width = bubble.width - spec.text_left - spec.text_right
    top = spec.text_top
    for index, line in enumerate(lines):
        draw.text(
            (box_left + (box_width - font.getlength(line)) / 2, top + index * step),
            line,
            font=font,
            # Ink, not the graphics track's white-on-dark lettering: this text is
            # inside a white bubble.
            fill=STROKE_COLOUR,
        )
    dst.parent.mkdir(parents=True, exist_ok=True)
    bubble.save(dst)
    return bubble.size
