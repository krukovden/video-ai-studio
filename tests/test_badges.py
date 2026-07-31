"""The badge library: the files, the manifest, and the credits that bind them.

`videoai fetch-badges` runs once and then nobody looks at it again, so every way
it can quietly go wrong is checked here against what is actually on disk. Three
of these have already caught something real: a sprite whose drawing was a solid
green square with no transparency, a manifest naming a file that was never
written, and two badges the placement model had no way to choose between.

Nothing here reaches the network. The library is a checked-in artifact; a test
that re-downloaded it would be testing jsDelivr.
"""
from __future__ import annotations

from PIL import Image

from videoai.logic.badges import (
    BADGES,
    BADGE_SIZE,
    FLUENT_CREDIT,
    TWEMOJI_CREDIT,
    manifest_entry,
    source_url,
)
from videoai.logic.effects import default_library_dir, load_library

LIBRARY = default_library_dir()


def _badge_sprites():
    return {sprite.name: sprite for sprite in load_library(LIBRARY).sprites}


def test_every_badge_has_a_file_and_the_manifest_names_it():
    sprites = _badge_sprites()
    for badge in BADGES:
        assert badge.name in sprites, f"{badge.name} is missing from the manifest"
        path = LIBRARY / sprites[badge.name].file
        assert path.is_file(), f"{badge.name} names {path.name}, which is not there"


def test_the_manifest_says_what_the_spec_says():
    sprites = _badge_sprites()
    for badge in BADGES:
        entry = manifest_entry(badge)
        sprite = sprites[badge.name]
        assert sprite.expresses == entry["expresses"]
        assert list(sprite.tags) == entry["tags"]
        assert sprite.animation == entry["animation"]
        assert sprite.default_seconds == entry["default_seconds"]
        assert sprite.attribution == entry["attribution"]


def test_no_badge_is_left_in_the_library_after_being_dropped_from_the_spec():
    wanted = {badge.name for badge in BADGES}
    for path in LIBRARY.glob("badge_*.png"):
        assert path.stem in wanted, f"{path.name} is no longer in BADGES"


def test_every_badge_is_transparent_and_large_enough_to_never_be_upscaled():
    for badge in BADGES:
        with Image.open(LIBRARY / f"{badge.name}.png") as image:
            image.load()
            assert image.mode == "RGBA", f"{badge.name} has no alpha channel"
            assert min(image.size) >= BADGE_SIZE, f"{badge.name} is only {image.size}"
            alpha = image.getchannel("A")
            low, high = alpha.getextrema()
            # A drawing flattened onto a white background still has an alpha
            # channel; what it does not have is a transparent pixel in it.
            assert low == 0, f"{badge.name} is matted, not transparent"
            assert high == 255, f"{badge.name} has no fully opaque pixel"
            assert not any(
                image.getpixel(corner)[3]
                for corner in ((0, 0), (image.width - 1, 0),
                               (0, image.height - 1),
                               (image.width - 1, image.height - 1))
            ), f"{badge.name} reaches into its corners, so it is probably a filled square"


def test_no_two_badges_express_the_same_thing():
    """The one property the placement model actually depends on.

    `expresses` is the whole interface between this library and the model: two
    badges that read alike leave it choosing by novelty, which is how a laughing
    face ends up on a printer finishing a part.
    """
    seen: dict[str, str] = {}
    for badge in BADGES:
        clash = seen.get(badge.expresses)
        assert clash is None, f"{badge.name} expresses the same as {clash}"
        seen[badge.expresses] = badge.name


def test_names_and_codepoints_are_unique():
    # The fetch command resolves a badge by its codepoint, so a duplicate would
    # silently give two badges the same drawing.
    assert len({badge.name for badge in BADGES}) == len(BADGES)
    assert len({badge.code for badge in BADGES}) == len(BADGES)


def test_the_credit_follows_the_source_the_badge_declares():
    for badge in BADGES:
        expected = FLUENT_CREDIT if badge.fluent else TWEMOJI_CREDIT
        assert manifest_entry(badge)["attribution"] == expected
    assert "Microsoft Corporation" in FLUENT_CREDIT and "MIT" in FLUENT_CREDIT
    assert "CC BY 4.0" in TWEMOJI_CREDIT


def test_the_mit_notice_ships_beside_the_sprites_it_covers():
    # MIT asks for the notice to travel with copies of the work, which a credit
    # line in a video description does not do on its own.
    notice = (LIBRARY / "LICENSE-fluent-emoji.txt").read_text(encoding="utf-8")
    assert "Copyright (c) Microsoft Corporation." in notice
    assert "Permission is hereby granted" in notice


def test_a_fluent_badge_resolves_to_its_asset_and_a_toned_one_to_the_default_tone():
    boom = next(badge for badge in BADGES if badge.name == "badge_boom")
    assert source_url(boom).endswith("/assets/Collision/Color/collision_color.svg")
    clap = next(badge for badge in BADGES if badge.name == "badge_clap")
    assert source_url(clap).endswith(
        "/assets/Clapping%20hands/Default/Color/clapping_hands_color_default.svg"
    )


def test_a_badge_fluent_does_not_cover_falls_back_to_twemoji():
    spare = BADGES[0].__class__(
        "badge_spare", "1f9ff", "", ("nazar",), "an amulet Fluent has never drawn"
    )
    assert source_url(spare).endswith("/svg/1f9ff.svg")
    assert manifest_entry(spare)["attribution"] == TWEMOJI_CREDIT
