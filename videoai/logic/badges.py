"""Building the badge library from free, licence-checked sources.

The sprites this pipeline shipped with were drawn procedurally, which kept it
free of licence questions and also kept it looking like something drawn by a
program. A kids' review wants accents that read instantly — a burst, a pair of
eyes, a party popper — and the honest free source for those is Twemoji.

Two things are true about Twemoji and both are handled here rather than left to
be discovered later. It is CC BY 4.0, so a credit has to travel with any video
that uses one, and the credit is attached to the sprite so it appears only when
such a badge is actually composited. And it ships only 72x72 PNGs and SVG, which
is too small for a 1080p overlay, so the SVG is rasterised at a usable size.

Kenney's particle pack is CC0 and needs no credit, but it is soft greyscale VFX
rather than badges: useful underneath a badge, useless as one. It is not fetched
here for that reason — this module is about the things a viewer reads.
"""
from __future__ import annotations

from dataclasses import dataclass

TWEMOJI_SVG = "https://cdn.jsdelivr.net/gh/jdecked/twemoji@latest/assets/svg/{code}.svg"

# CC BY 4.0. Worded for a video description, which is the equivalent of the
# README mention the project asks for.
TWEMOJI_CREDIT = (
    "Emoji graphics by Twemoji (https://github.com/jdecked/twemoji), "
    "licensed CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/)."
)

# Rendered big enough for the largest scale the compositor uses (0.32 of a 1080p
# frame is 346px) with room to spare, so an accent is never upscaled.
BADGE_SIZE = 512


@dataclass(frozen=True)
class BadgeSpec:
    """One badge to fetch: what it is, and what a model should match it against."""

    name: str
    code: str
    tags: tuple[str, ...]
    expresses: str
    animation: str = "pop-in"
    seconds: float = 0.9


# Chosen for what a toy review actually needs: a handful of physical events and a
# wider range of reactions, since most moments in a review are somebody's face
# rather than something exploding. Deliberately small — a list of eighty badges
# would make the placement model pick by novelty instead of by meaning.
BADGES: tuple[BadgeSpec, ...] = (
    # Things that happen
    BadgeSpec("badge_boom", "1f4a5", ("impact", "pop", "burst", "hit", "explode"),
              "something burst, popped or landed hard, right now"),
    BadgeSpec("badge_sparkles", "2728", ("sparkle", "clean", "reveal", "shiny", "magic"),
              "a clean reveal, or something turning out beautifully", "pulse"),
    BadgeSpec("badge_party", "1f389", ("celebrate", "finished", "success", "done", "win"),
              "a finished build or a moment worth celebrating", "drift-up", 1.4),
    BadgeSpec("badge_confetti", "1f38a", ("celebrate", "party", "success", "ending"),
              "a cheerful sign-off or a small triumph", "drift-up", 1.4),
    BadgeSpec("badge_splash", "1f4a6", ("liquid", "squirt", "splash", "wet", "squeeze"),
              "liquid squirting, splashing or being squeezed out"),
    BadgeSpec("badge_fire", "1f525", ("cool", "intense", "hot", "impressive"),
              "this bit is genuinely impressive or intense", "pulse"),
    BadgeSpec("badge_star", "2b50", ("great", "favourite", "best", "highlight"),
              "the best moment of a section, or a clear favourite", "pop-in"),
    # How somebody reacted
    BadgeSpec("badge_eyes", "1f440", ("look", "watch", "attention", "closer", "notice"),
              "look closely at this — attention on the thing itself", "pulse"),
    BadgeSpec("badge_shock", "1f631", ("shock", "surprise", "scared", "sudden"),
              "a genuine shock or a startled reaction"),
    BadgeSpec("badge_mindblown", "1f92f", ("amazed", "wow", "unexpected", "disbelief"),
              "amazement — something worked far better than expected"),
    BadgeSpec("badge_laugh", "1f602", ("funny", "laugh", "joke", "silly"),
              "a genuinely funny moment or an aside that lands"),
    BadgeSpec("badge_love", "1f60d", ("love", "delight", "favourite", "adore"),
              "delight — the presenter clearly loves this"),
    BadgeSpec("badge_clap", "1f44f", ("well done", "applause", "achievement"),
              "an achievement worth applauding", "shake"),
    BadgeSpec("badge_thumbsup", "1f44d", ("approve", "good", "verdict", "recommend"),
              "approval, or a positive verdict", "pop-in"),
    BadgeSpec("badge_question", "2753", ("curious", "unsure", "what", "mystery"),
              "curiosity or genuine uncertainty about what happens next", "shake"),
    BadgeSpec("badge_yuck", "1f92e", ("gross", "yuck", "disgust", "messy"),
              "something satisfyingly gross or messy"),
)


def manifest_entry(badge: BadgeSpec) -> dict:
    """The manifest record for one badge, credit included.

    The attribution lives on the sprite rather than in a global list so a video
    that used no Twemoji badge carries no Twemoji credit — a credit for something
    that is not in the video is noise, and noise in a licence notice is how real
    obligations get ignored.
    """
    return {
        "name": badge.name,
        "file": f"{badge.name}.png",
        "tags": list(badge.tags),
        "expresses": badge.expresses,
        "anchor": "center",
        "default_seconds": badge.seconds,
        "animation": badge.animation,
        "attribution": TWEMOJI_CREDIT,
    }


def rasterise_badge(code: str, size: int = BADGE_SIZE) -> bytes:
    """One Twemoji SVG as a PNG with a real alpha channel.

    Twemoji ships 72x72 PNGs, which would be upscaled fivefold onto a 1080p
    frame; the SVG is the only source at a usable size. macOS has no rasteriser
    that preserves transparency (QuickLook flattens onto an opaque background,
    `sips` cannot read SVG at all), so this needs cairo.
    """
    import cairosvg

    return cairosvg.svg2png(
        url=TWEMOJI_SVG.format(code=code), output_width=size, output_height=size
    )
