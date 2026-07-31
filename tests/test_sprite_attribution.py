"""A credit travels with the video only when the art that needs it is in it.

The badge art carries a licence — Fluent Emoji is MIT and wants its copyright
notice kept — so a video using one of those badges owes a credit. A video that
used none owes nothing, and printing the notice anyway is how a licence file
becomes boilerplate nobody reads.

The library under test is resolved from the package, never written out as a
path. A literal one here pointed at a different checkout of this repository for
long enough to matter: these tests kept passing against that other tree's art
while the art beside them changed source and licence entirely, which is the
exact failure a licence test exists to prevent.
"""
from __future__ import annotations

from pathlib import Path

from videoai.logic.effects import default_library_dir, load_library
from videoai.stages.s08_polish import sprite_attributions, write_attribution

LIBRARY = default_library_dir()


def test_a_badge_that_needs_a_credit_produces_one():
    lines = sprite_attributions(["badge_boom"], load_library(LIBRARY))
    assert len(lines) == 1
    assert "Fluent Emoji" in lines[0] and "MIT" in lines[0]


def test_the_credit_names_the_source_the_manifest_actually_declares():
    """Guards the hazard above: the assertion is derived from the library on
    disk, so art swapped for a differently licensed source fails here rather
    than shipping under the wrong notice."""
    library = load_library(LIBRARY)
    declared = {sprite.attribution for sprite in library.sprites if sprite.attribution}
    assert declared, "the library declares no attribution at all"
    for line in sprite_attributions(sorted(library.names()), library):
        assert line in declared


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
    line = "Emoji graphics from Microsoft Fluent Emoji, MIT licence."
    assert write_attribution(tmp_path, line) is True
    assert write_attribution(tmp_path, line) is False
    assert (tmp_path / "metadata.md").read_text(encoding="utf-8").count("Fluent") == 1
