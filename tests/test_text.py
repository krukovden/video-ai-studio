"""Rasterising a title to an RGBA strip with a real typeface."""
from pathlib import Path

import numpy as np
import pytest

from videoai.core import text as text_module
from videoai.core.text import (
    fit_text,
    load_font,
    render_text_image,
    resolve_font,
    wrap_to_width,
)


@pytest.fixture(autouse=True)
def clear_font_cache():
    """`resolve_font` and `load_font` are cached for the delivery render's sake, so
    a test that changes the candidate list has to start from an empty cache."""
    resolve_font.cache_clear()
    load_font.cache_clear()
    yield
    resolve_font.cache_clear()
    load_font.cache_clear()


def _strip(dst: Path, text: str = "Popping Time", **kwargs) -> np.ndarray:
    from PIL import Image

    render_text_image(dst, text, 640, 96, **kwargs)
    return np.array(Image.open(dst))


def test_a_rendered_strip_is_rgba_of_the_requested_size(tmp_path: Path):
    strip = _strip(tmp_path / "title.png")

    assert strip.shape == (96, 640, 4)
    assert strip.dtype == np.uint8


def test_the_glyphs_are_really_drawn_and_really_opaque(tmp_path: Path):
    """A strip whose text failed to draw is still a valid PNG of the right size, so
    the assertion has to be about ink: opaque white pixels with dark ones around
    them, and enough of both to be a word rather than a speck."""
    strip = _strip(tmp_path / "title.png")

    white = ((strip[:, :, :3] > 240).all(axis=2) & (strip[:, :, 3] == 255)).sum()
    outline = ((strip[:, :, :3] < 40).all(axis=2) & (strip[:, :, 3] == 255)).sum()

    assert white > 200, "no opaque white glyph pixels: the text did not draw"
    assert outline > 100, "no dark outline: the text would vanish over pale footage"


def test_a_transparent_plate_leaves_the_background_alpha_at_zero(tmp_path: Path):
    strip = _strip(tmp_path / "title.png", plate_alpha=0.0)

    assert strip[:, :, 3].min() == 0, "a plate-less strip must not be a grey box"
    assert strip[:, :, 3].max() == 255


def test_the_plate_alpha_is_honoured(tmp_path: Path):
    strip = _strip(tmp_path / "title.png", plate_alpha=0.62)

    corner = strip[0, 0]
    assert corner[3] == pytest.approx(158, abs=1)
    assert tuple(corner[:3]) == (24, 20, 16)


def test_text_is_wrapped_to_the_line_limit_and_still_fits(tmp_path: Path):
    from PIL import Image

    long_title = "A Very Long Toy Review Title That Cannot Possibly Fit On One Line"
    render_text_image(tmp_path / "title.png", long_title, 640, 200, max_lines=2)
    strip = np.array(Image.open(tmp_path / "title.png"))

    lines, font, line_height = fit_text(long_title, 620, 190, 2)
    assert len(lines) <= 2
    assert all(font.getlength(line) <= 620 for line in lines)
    assert line_height * len(lines) <= 190
    # Nothing was drawn outside the strip, so the ink is inside the picture.
    assert strip.shape == (200, 640, 4)
    assert strip[:, :, 3].max() == 255


def test_a_short_title_is_set_larger_than_a_long_one():
    _, small_font, _ = fit_text(
        "Every Single Word Of This Title Has To Fit Somewhere", 600, 120, 2
    )
    _, big_font, _ = fit_text("Pop", 600, 120, 2)

    assert big_font.size > small_font.size


def test_wrapping_is_measured_rather_than_estimated():
    font = load_font(40)
    wide = wrap_to_width("WWWWW WWWWW WWWWW", font, 200)
    narrow = wrap_to_width("iiiii iiiii iiiii", font, 200)

    assert len(wide) > len(narrow), (
        "wide and narrow strings of the same length wrapped identically, so the "
        "wrap is counting characters instead of measuring them"
    )


def test_font_resolution_falls_back_cleanly_when_no_font_is_installed(
    tmp_path: Path, monkeypatch
):
    """Linux CI has none of the macOS faces. A missing font is a plainer title, not
    a failed render, so nothing in this path may raise."""
    monkeypatch.setattr(
        text_module, "FONT_CANDIDATES", ("/nowhere/NoSuchFont.ttf",)
    )
    resolve_font.cache_clear()
    load_font.cache_clear()

    assert resolve_font() is None

    font = load_font(28)
    assert font is not None
    assert font.getlength("Popping Time") > 0

    strip = _strip(tmp_path / "title.png")
    assert strip[:, :, 3].max() == 255, "no ink was drawn with the fallback face"


def test_an_unreadable_font_file_falls_back_instead_of_raising(
    tmp_path: Path, monkeypatch
):
    broken = tmp_path / "Broken.ttf"
    broken.write_bytes(b"this is not a font")
    monkeypatch.setattr(text_module, "FONT_CANDIDATES", (str(broken),))
    resolve_font.cache_clear()
    load_font.cache_clear()

    assert resolve_font() == broken
    assert load_font(24).getlength("Pop") > 0


def test_the_preferred_face_is_the_rounded_one_when_it_is_installed():
    """The face choice is a deliberate editorial decision: rounded reads friendlier
    on a kids' channel than Arial Bold does."""
    assert text_module.FONT_CANDIDATES[0].endswith("Arial Rounded Bold.ttf")
    installed = resolve_font()
    if installed is not None and Path(text_module.FONT_CANDIDATES[0]).is_file():
        assert installed == Path(text_module.FONT_CANDIDATES[0])
