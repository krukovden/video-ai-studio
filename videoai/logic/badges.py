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

    # Objects and actions. These exist so the library is not tied to one kind of
    # video: a Lego build, a drum kit and a 3D print each need something concrete
    # to point at, and a channel that only owns reaction faces ends up putting a
    # laughing emoji on a printer finishing a part.
    BadgeSpec("badge_target", "1f3af", ("accurate", "hit", "aim", "nailed it", "precise"),
              "hitting exactly what was aimed at"),
    BadgeSpec("badge_brick", "1f9f1", ("build", "blocks", "lego", "assemble", "piece"),
              "building or assembling something out of parts"),
    BadgeSpec("badge_gear", "2699", ("mechanism", "how it works", "machine", "moving parts"),
              "the mechanism itself — how the thing actually works", "pulse"),
    BadgeSpec("badge_tools", "1f6e0", ("fix", "build", "tools", "assembly", "repair"),
              "building, fixing or assembling with tools"),
    BadgeSpec("badge_package", "1f4e6", ("unbox", "parcel", "delivery", "box", "arrive"),
              "a package or box, especially one about to be opened"),
    BadgeSpec("badge_gift", "1f381", ("present", "gift", "surprise", "birthday"),
              "a present, or the moment before a surprise is revealed"),
    BadgeSpec("badge_paint", "1f3a8", ("colour", "paint", "art", "mixing", "palette"),
              "colour being chosen, mixed or applied"),
    BadgeSpec("badge_rainbow", "1f308", ("colourful", "many colours", "bright", "variety"),
              "lots of colours at once", "drift-up", 1.2),
    BadgeSpec("badge_bolt", "26a1", ("power", "energy", "fast", "zap", "sudden"),
              "sudden power or speed", "shake"),
    BadgeSpec("badge_rocket", "1f680", ("fast", "launch", "takeoff", "speed", "go"),
              "something taking off or going very fast", "drift-up", 1.2),
    BadgeSpec("badge_timer", "23f1", ("time", "wait", "how long", "speed run", "countdown"),
              "how long something takes, or a wait beginning", "pulse"),

    # A verdict. A review earns these, and without them the library can express
    # excitement but never judgement.
    BadgeSpec("badge_trophy", "1f3c6", ("win", "best", "champion", "achievement"),
              "the win — the thing finally worked, or came first"),
    BadgeSpec("badge_perfect", "1f4af", ("perfect", "full marks", "exactly", "nailed"),
              "as good as it could possibly have gone"),
    # U+2714, the bare tick — NOT U+2705, which Twemoji draws as a filled green
    # square and which would paste a solid rectangle over the footage. Caught by
    # the library's own transparency test at 99% opaque.
    BadgeSpec("badge_tick", "2714", ("correct", "worked", "done", "yes", "success"),
              "it worked, or the step is complete"),
    BadgeSpec("badge_cross", "274c", ("failed", "no", "wrong", "broke", "nope"),
              "it did not work — an honest failure, not a joke"),
    BadgeSpec("badge_ok", "1f44c", ("nice", "good", "neat", "well done"),
              "neatly done, a small approving beat", "pop-in"),
    BadgeSpec("badge_crown", "1f451", ("best", "favourite", "top", "winner"),
              "the favourite of the whole video"),

    # More of the range a face actually has. Shock and delight were covered;
    # hesitation, mild surprise and embarrassment were not, and most of a review
    # lives in those.
    BadgeSpec("badge_thinking", "1f914", ("thinking", "unsure", "considering", "hmm"),
              "weighing something up, or not yet convinced", "pulse"),
    BadgeSpec("badge_surprise", "1f62e", ("surprise", "oh", "unexpected", "didn't expect"),
              "mild surprise — smaller than a shock"),
    BadgeSpec("badge_awkward", "1f605", ("oops", "close call", "awkward", "nearly"),
              "an awkward moment or a near miss, taken in good humour"),
    BadgeSpec("badge_starstruck", "1f929", ("impressed", "amazing", "wow", "love it"),
              "genuinely impressed by what just happened"),
    BadgeSpec("badge_cheer", "1f64c", ("celebrate", "yes", "hooray", "relief"),
              "a small cheer — relief or triumph", "drift-up", 1.2),
    BadgeSpec("badge_wave", "1f44b", ("hello", "goodbye", "intro", "outro", "greeting"),
              "a hello at the start or a goodbye at the end", "shake", 1.2),
    BadgeSpec("badge_cool", "1f60e", ("cool", "confident", "smooth", "easy"),
              "playing it cool, or making something look easy"),

    # Asked for by name. These are the faces and objects people reach for most,
    # and a library without them keeps forcing a near-miss.
    #
    # `badge_gun` is worth being explicit about: Twemoji draws U+1F52B as a green
    # WATER PISTOL, not a firearm — every major vendor redesigned it that way. For
    # a channel reviewing toy blasters that is exactly the right picture, and it
    # carries none of the risk a realistic weapon would on a kids' video.
    BadgeSpec("badge_gun", "1f52b", ("blaster", "shoot", "toy gun", "water pistol", "fire"),
              "a toy blaster or water pistol being used"),
    BadgeSpec("badge_glasses", "1f453", ("glasses", "look", "inspect", "read", "detail"),
              "looking closely at detail, or putting glasses on"),
    BadgeSpec("badge_sunglasses", "1f576", ("cool", "style", "confident", "shades"),
              "style, or a deliberately cool moment", "pop-in"),
    BadgeSpec("badge_smile", "1f604", ("happy", "smile", "pleased", "fun", "enjoy"),
              "plain, uncomplicated enjoyment"),
    BadgeSpec("badge_grin", "1f601", ("grin", "pleased", "cheeky", "delighted"),
              "a wide, slightly cheeky grin"),
    BadgeSpec("badge_blush", "1f60a", ("shy", "pleased", "warm", "sweet"),
              "quiet pleasure, or a shy moment"),
    BadgeSpec("badge_cry", "1f622", ("sad", "cry", "disappointed", "aww"),
              "disappointment, or a sad beat played gently"),
    BadgeSpec("badge_sob", "1f62d", ("sob", "very sad", "gutted", "dramatic"),
              "dramatic, over-the-top upset — the funny kind"),
    BadgeSpec("badge_pleading", "1f97a", ("please", "hopeful", "aww", "want"),
              "hoping for something, or asking nicely"),
    BadgeSpec("badge_heart", "2764", ("love", "favourite", "adore", "heart"),
              "plain affection for the thing on screen", "pulse"),
    BadgeSpec("badge_pray", "1f64f", ("please", "fingers crossed", "hope", "thanks"),
              "hoping it works, or a thank-you", "pulse"),
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
