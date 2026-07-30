"""A credit travels with the video only when the art that needs it is in it.

Twemoji is CC BY 4.0, so a video using one of its badges owes a credit. A video
that used none owes nothing — and printing the notice anyway is how a licence
file becomes boilerplate nobody reads.
"""
from __future__ import annotations

from pathlib import Path

from videoai.logic.effects import load_library
from videoai.stages.s08_polish import sprite_attributions, write_attribution

LIBRARY = Path("/Users/denyskriukov/Documents/TestDocAI/VideoAI/assets/effects")


def test_a_badge_that_needs_a_credit_produces_one():
    lines = sprite_attributions(["badge_boom"], load_library(LIBRARY))
    assert len(lines) == 1
    assert "Twemoji" in lines[0] and "CC BY 4.0" in lines[0]


def test_art_that_needs_no_credit_produces_none():
    assert sprite_attributions(["comic_starburst"], load_library(LIBRARY)) == []


def test_the_same_credit_is_not_repeated_for_every_badge():
    lines = sprite_attributions(
        ["badge_boom", "badge_eyes", "badge_star"], load_library(LIBRARY)
    )
    assert len(lines) == 1


def test_an_unknown_sprite_name_is_ignored_rather_than_raising():
    assert sprite_attributions(["nothing_like_this"], load_library(LIBRARY)) == []


def test_a_credit_is_written_once_however_often_the_render_repeats(tmp_path: Path):
    line = "Emoji graphics by Twemoji, CC BY 4.0."
    assert write_attribution(tmp_path, line) is True
    assert write_attribution(tmp_path, line) is False
    assert (tmp_path / "metadata.md").read_text(encoding="utf-8").count("Twemoji") == 1
