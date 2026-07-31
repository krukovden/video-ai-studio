"""Building the badge library from free, licence-checked sources.

The sprites this pipeline shipped with were drawn procedurally, which kept it
free of licence questions and also kept it looking like something drawn by a
program. The first fix was Twemoji, which reads instantly but is flat 2D vector
work: correct shapes, no light on them. Next to a modern thumbnail they look
weak, which is exactly the note the creator gave.

Microsoft's Fluent Emoji is the upgrade, and it is MIT rather than CC BY, so the
obligation is a retained copyright notice instead of a licence-linked credit.
Which *style* of it to take was the real decision:

- the "3D" style is a Blender render and looks superb, but it ships as PNG only
  and every one of them is 256x256. A large accent is drawn 0.32 of a 1080p
  frame high, i.e. 346px, so every big badge would be an upscale of the source
  art. Verified against the repository, not assumed: all of them are 256.
- the "Color" style is the same artwork as vector, gradients and shading
  included, and being vector it rasterises to whatever size the compositor can
  ask for. Rendered side by side with the 3D PNG it is all but identical.

So Color at 512 is the source: the modern look, at a size nothing ever has to
upscale. Twemoji stays as the per-badge fallback because a set that covers
1,285 emoji will eventually miss one this library wants, and because a badge
whose Fluent drawing does not survive rasterising is better served by a flat
picture than by no picture.

The two sources have different obligations, so the credit is chosen per badge
from the source that badge actually declares, and travels on the sprite rather
than in a global list — a video that used no Fluent badge should carry no
Microsoft copyright notice.

Kenney's particle pack is CC0 and needs no credit, but it is soft greyscale VFX
rather than badges: useful underneath a badge, useless as one. It is not fetched
here for that reason — this module is about the things a viewer reads.
"""
from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import quote

# Fluent's own path shape, and there are two of them: anything a human hand
# appears in is filed under a skin tone, and "Default" is the yellow one.
FLUENT_COLOR = (
    "https://cdn.jsdelivr.net/gh/microsoft/fluentui-emoji@main/"
    "assets/{folder}/Color/{stem}_color.svg"
)
FLUENT_COLOR_TONED = (
    "https://cdn.jsdelivr.net/gh/microsoft/fluentui-emoji@main/"
    "assets/{folder}/Default/Color/{stem}_color_default.svg"
)
TWEMOJI_SVG = "https://cdn.jsdelivr.net/gh/jdecked/twemoji@latest/assets/svg/{code}.svg"

# MIT. It asks for the copyright notice and the permission notice to travel with
# copies, so the notice itself is shipped next to the sprites as
# LICENSE-fluent-emoji.txt and this line — the one a video description can carry
# — names the holder and links the full text.
FLUENT_CREDIT = (
    "Emoji graphics from Microsoft Fluent Emoji "
    "(https://github.com/microsoft/fluentui-emoji), Copyright (c) Microsoft "
    "Corporation, used under the MIT licence "
    "(https://github.com/microsoft/fluentui-emoji/blob/main/LICENSE)."
)

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
    # The Fluent asset folder, empty when Fluent has nothing usable and the badge
    # falls back to Twemoji. Declared per badge rather than discovered at fetch
    # time because the licence notice is chosen from it: silently swapping source
    # would attach the wrong copyright to the sprite.
    fluent: str
    tags: tuple[str, ...]
    expresses: str
    animation: str = "pop-in"
    seconds: float = 0.9
    toned: bool = False


# Chosen for what a toy review actually needs, and no wider. Every entry has to
# earn its place by expressing something no other entry does: two badges a model
# cannot tell apart are worse than one, because then it picks by novelty. That is
# why the wording of `expresses` is fussed over — it is the whole interface
# between this list and the placement stage.
BADGES: tuple[BadgeSpec, ...] = (
    # Things that physically happen. A review is mostly talking, so these are the
    # moments where the video has an event to point at.
    BadgeSpec("badge_boom", "1f4a5", "Collision",
              ("impact", "pop", "burst", "hit", "explode"),
              "something burst, popped or landed hard, right now", "zoom-punch"),
    BadgeSpec("badge_sparkles", "2728", "Sparkles",
              ("sparkle", "clean", "reveal", "shiny", "magic"),
              "a clean reveal, or something turning out beautifully", "pulse"),
    BadgeSpec("badge_splash", "1f4a6", "Sweat droplets",
              ("liquid", "squirt", "splash", "wet", "squeeze"),
              "liquid squirting, splashing or being squeezed out", "squash"),
    BadgeSpec("badge_spin", "1f300", "Cyclone",
              ("spin", "whirl", "rotate", "top", "twirl"),
              "something spinning or whirling on the spot", "spin-in"),
    BadgeSpec("badge_bolt", "26a1", "High voltage",
              ("power", "energy", "zap", "switch on", "electric"),
              "a jolt of power — the thing switching on or firing up", "shake"),
    BadgeSpec("badge_rocket", "1f680", "Rocket",
              ("fast", "launch", "takeoff", "speed", "go"),
              "something taking off or travelling very fast", "drift-up", 1.2),
    BadgeSpec("badge_snail", "1f40c", "Snail",
              ("slow", "crawl", "sluggish", "takes ages"),
              "far slower than expected — a crawl", "swing", 1.2),
    BadgeSpec("badge_music", "1f3b6", "Musical notes",
              ("sound", "noise", "tune", "beep", "plays music"),
              "the noise or tune the thing itself makes", "bounce", 1.2),
    BadgeSpec("badge_fire", "1f525", "Fire",
              ("cool", "intense", "hot", "impressive"),
              "this bit is genuinely impressive or intense", "pulse"),

    # Celebration and judgement. A review earns these, and without them the
    # library can express excitement but never a verdict.
    BadgeSpec("badge_party", "1f389", "Party popper",
              ("celebrate", "finished", "milestone", "done", "built"),
              "a build finished, or a milestone reached", "drift-up", 1.4),
    BadgeSpec("badge_confetti", "1f38a", "Confetti ball",
              ("sign-off", "ending", "goodbye", "party"),
              "the cheerful sign-off at the very end", "drift-up", 1.4),
    BadgeSpec("badge_cheer", "1f64c", "Raising hands",
              ("relief", "at last", "hooray", "it worked"),
              "relief that it finally worked", "drift-up", 1.2, toned=True),
    BadgeSpec("badge_trophy", "1f3c6", "Trophy",
              ("win", "champion", "beat it", "achievement"),
              "the win — the thing beat the challenge it was set"),
    BadgeSpec("badge_crown", "1f451", "Crown",
              ("best", "top pick", "winner", "overall"),
              "the overall pick of everything in the video", "bounce", 1.2),
    BadgeSpec("badge_star", "2b50", "Star",
              ("great", "highlight", "standout", "moment"),
              "the standout moment of this section", "pop-in"),
    BadgeSpec("badge_perfect", "1f4af", "Hundred points",
              ("perfect", "full marks", "exactly", "nailed"),
              "as good as it could possibly have gone"),
    BadgeSpec("badge_target", "1f3af", "Bullseye",
              ("accurate", "aim", "nailed it", "precise", "on target"),
              "hitting exactly what was aimed at"),
    # The bare tick, NOT the check-mark BUTTON: every vendor draws that one as a
    # filled green square, which pastes a solid rectangle over the footage. The
    # library's own transparency test catches it at 99% opaque, and did once.
    BadgeSpec("badge_tick", "2714", "Check mark",
              ("correct", "worked", "done", "yes", "success"),
              "it worked, or the step is complete"),
    BadgeSpec("badge_cross", "274c", "Cross mark",
              ("failed", "no", "wrong", "broke", "nope"),
              "it did not work — an honest failure, not a joke"),
    BadgeSpec("badge_thumbsup", "1f44d", "Thumbs up",
              ("approve", "recommend", "verdict", "buy it"),
              "a clear positive verdict on the whole thing", toned=True),
    BadgeSpec("badge_thumbsdown", "1f44e", "Thumbs down",
              ("reject", "not worth it", "avoid", "poor"),
              "a negative verdict, given plainly", toned=True),
    BadgeSpec("badge_ok", "1f44c", "Ok hand",
              ("nice", "neat", "tidy", "in passing"),
              "a small 'that is neat' said in passing", "pop-in", toned=True),
    BadgeSpec("badge_warning", "26a0", "Warning",
              ("careful", "caution", "watch out", "risk", "sharp"),
              "a caution worth flagging before somebody copies this", "shake"),

    # How somebody reacted. Most moments in a review are a face rather than an
    # event, so this is the widest group — but each shade of it is a different
    # size of feeling, not a synonym.
    BadgeSpec("badge_eyes", "1f440", "Eyes",
              ("look", "watch", "attention", "closer", "notice"),
              "look at this — attention on the thing itself", "pulse"),
    BadgeSpec("badge_shock", "1f631", "Face screaming in fear",
              ("shock", "scared", "sudden", "aaah"),
              "a genuine shock or a startled reaction", "shake"),
    BadgeSpec("badge_surprise", "1f62e", "Face with open mouth",
              ("surprise", "oh", "unexpected", "didn't expect"),
              "mild surprise — smaller than a shock"),
    BadgeSpec("badge_mindblown", "1f92f", "Exploding head",
              ("disbelief", "no way", "unexpected", "wow"),
              "disbelief that it actually worked", "zoom-punch"),
    BadgeSpec("badge_starstruck", "1f929", "Star-struck",
              ("impressed", "amazing", "beautiful", "admire"),
              "star-struck admiration for how good it looks"),
    BadgeSpec("badge_laugh", "1f602", "Face with tears of joy",
              ("funny", "laugh", "joke", "silly"),
              "a genuinely funny moment or an aside that lands"),
    BadgeSpec("badge_rofl", "1f923", "Rolling on the floor laughing",
              ("hilarious", "howling", "helpless", "cannot stop"),
              "helpless laughter — funnier than anyone expected", "swing", 1.2),
    BadgeSpec("badge_love", "1f60d", "Smiling face with heart-eyes",
              ("love", "delight", "adore", "want one"),
              "delight — the presenter clearly loves this"),
    BadgeSpec("badge_heart", "2764", "Red heart",
              ("love", "affection", "sweet", "heart"),
              "warmth towards the thing itself, with no face involved", "pulse"),
    BadgeSpec("badge_smile", "1f604", "Grinning face with smiling eyes",
              ("happy", "smile", "pleased", "fun", "enjoy"),
              "plain, uncomplicated enjoyment"),
    BadgeSpec("badge_grin", "1f601", "Beaming face with smiling eyes",
              ("grin", "cheeky", "mischief", "delighted"),
              "a wide, slightly cheeky grin"),
    BadgeSpec("badge_blush", "1f60a", "Smiling face with smiling eyes",
              ("shy", "quiet", "warm", "sweet"),
              "quiet pleasure, or a shy moment"),
    BadgeSpec("badge_cool", "1f60e", "Smiling face with sunglasses",
              ("cool", "confident", "smooth", "easy"),
              "playing it cool, or making something look easy"),
    BadgeSpec("badge_thinking", "1f914", "Thinking face",
              ("thinking", "considering", "hmm", "not convinced"),
              "weighing something up, or not yet convinced", "pulse"),
    BadgeSpec("badge_question", "2753", "Red question mark",
              ("curious", "what is it", "mystery", "guess"),
              "an open question the video has not answered yet", "shake"),
    BadgeSpec("badge_awkward", "1f605", "Grinning face with sweat",
              ("oops", "close call", "awkward", "nearly"),
              "a near miss or awkward moment, taken in good humour"),
    BadgeSpec("badge_angry", "1f621", "Pouting face",
              ("annoyed", "frustrating", "fiddly", "grr"),
              "frustration — this part is infuriating to deal with", "shake"),
    BadgeSpec("badge_yuck", "1f92e", "Face vomiting",
              ("gross", "yuck", "disgust", "messy"),
              "something satisfyingly gross or messy"),
    BadgeSpec("badge_cry", "1f622", "Crying face",
              ("sad", "disappointed", "aww", "shame"),
              "disappointment, played gently"),
    BadgeSpec("badge_sob", "1f62d", "Loudly crying face",
              ("sob", "gutted", "dramatic", "devastated"),
              "over-the-top upset — the funny kind"),
    BadgeSpec("badge_pleading", "1f97a", "Pleading face",
              ("please", "want", "hopeful", "aww"),
              "wanting something badly, or asking nicely"),
    BadgeSpec("badge_pray", "1f64f", "Folded hands",
              ("fingers crossed", "hope", "moment of truth"),
              "fingers crossed, right before the moment of truth",
              "pulse", toned=True),
    BadgeSpec("badge_clap", "1f44f", "Clapping hands",
              ("applause", "well done", "bravo"),
              "an achievement worth applauding", "shake", toned=True),
    BadgeSpec("badge_wave", "1f44b", "Waving hand",
              ("hello", "goodbye", "intro", "outro", "greeting"),
              "a hello at the start or a goodbye at the end", "swing", 1.2,
              toned=True),
    BadgeSpec("badge_strong", "1f4aa", "Flexed biceps",
              ("tough", "strong", "durable", "survived", "sturdy"),
              "strength or toughness being tested", "squash", toned=True),

    # Objects and actions. These exist so the library is not tied to one kind of
    # video: a Lego build, a drum kit and a 3D print each need something concrete
    # to point at, and a channel that only owns reaction faces ends up putting a
    # laughing emoji on a printer finishing a part.
    BadgeSpec("badge_package", "1f4e6", "Package",
              ("unbox", "parcel", "delivery", "box", "arrive"),
              "a package or box, especially one about to be opened"),
    BadgeSpec("badge_scissors", "2702", "Scissors",
              ("cut", "open it", "snip", "packaging", "free it"),
              "cutting something open or cutting a part free", "shake"),
    BadgeSpec("badge_gift", "1f381", "Wrapped gift",
              ("present", "gift", "surprise", "birthday"),
              "a present, or the moment before a surprise is revealed"),
    BadgeSpec("badge_brick", "1f9f1", "Brick",
              ("blocks", "lego", "stack", "click together"),
              "stacking or clicking blocks together"),
    BadgeSpec("badge_puzzle", "1f9e9", "Puzzle piece",
              ("fits", "slots in", "last piece", "jigsaw"),
              "a piece finally fitting where it belongs", "pop-in"),
    BadgeSpec("badge_magnet", "1f9f2", "Magnet",
              ("magnetic", "snaps together", "sticks", "attract"),
              "parts snapping together magnetically", "zoom-punch"),
    BadgeSpec("badge_tools", "1f6e0", "Hammer and wrench",
              ("fix", "assembly", "tools", "repair", "screwdriver"),
              "building, fixing or assembling with tools"),
    BadgeSpec("badge_gear", "2699", "Gear",
              ("mechanism", "how it works", "machine", "moving parts"),
              "the mechanism itself — how the thing actually works", "spin-in"),
    BadgeSpec("badge_battery", "1f50b", "Battery",
              ("batteries", "power", "charge", "aa", "runs out"),
              "batteries: needed, going in, or running out"),
    BadgeSpec("badge_bulb", "1f4a1", "Light bulb",
              ("idea", "it clicks", "realise", "tip", "trick"),
              "an idea, or the moment something clicks", "pop-in"),
    BadgeSpec("badge_paint", "1f3a8", "Artist palette",
              ("colour", "paint", "art", "mixing", "palette"),
              "colour being chosen, mixed or applied"),
    BadgeSpec("badge_rainbow", "1f308", "Rainbow",
              ("colourful", "many colours", "bright", "variety"),
              "lots of colours at once", "drift-up", 1.2),
    BadgeSpec("badge_ruler", "1f4cf", "Straight ruler",
              ("size", "how big", "measure", "dimensions", "scale"),
              "how big the thing actually is"),
    BadgeSpec("badge_scale", "2696", "Balance scale",
              ("compare", "versus", "side by side", "which one"),
              "two things compared against each other"),
    BadgeSpec("badge_price", "1f4b0", "Money bag",
              ("price", "cost", "worth it", "value", "cheap"),
              "what it costs, and whether that is fair", "bounce"),
    BadgeSpec("badge_timer", "23f1", "Stopwatch",
              ("time", "how long", "speed run", "countdown"),
              "how long something takes, or a wait beginning", "pulse"),
    BadgeSpec("badge_zzz", "1f4a4", "Zzz",
              ("boring", "dull", "nothing happening", "slow bit"),
              "the dull stretch, signposted honestly", "drift-up", 1.2),
    BadgeSpec("badge_glasses", "1f453", "Glasses",
              ("small print", "label", "spec", "read it"),
              "the small print — a label or a spec being read"),
    BadgeSpec("badge_sunglasses", "1f576", "Sunglasses",
              ("style", "shades", "looks good", "swagger"),
              "style, or a deliberately cool-looking object", "pop-in"),
    # Worth being explicit about: U+1F52B is drawn as a green WATER PISTOL by
    # every major vendor, Fluent included, not as a firearm. For a channel
    # reviewing toy blasters that is exactly the right picture, and it carries
    # none of the risk a realistic weapon would on a kids' video.
    BadgeSpec("badge_gun", "1f52b", "Water pistol",
              ("blaster", "shoot", "toy gun", "water pistol", "squirt"),
              "a toy blaster or water pistol being used"),
)


def source_url(badge: BadgeSpec) -> str:
    """Where this badge's drawing is fetched from.

    Fluent names its files after the asset folder, lowercased with spaces turned
    into underscores and hyphens left alone, so the folder is the only thing that
    has to be written down per badge.
    """
    if not badge.fluent:
        return TWEMOJI_SVG.format(code=badge.code)
    template = FLUENT_COLOR_TONED if badge.toned else FLUENT_COLOR
    return template.format(
        folder=quote(badge.fluent), stem=badge.fluent.lower().replace(" ", "_")
    )


def manifest_entry(badge: BadgeSpec) -> dict:
    """The manifest record for one badge, credit included.

    The attribution lives on the sprite rather than in a global list so a video
    that used no Twemoji badge carries no Twemoji credit — a credit for something
    that is not in the video is noise, and noise in a licence notice is how real
    obligations get ignored. It follows the badge's declared source for the same
    reason: crediting the wrong project is worse than crediting nobody.
    """
    return {
        "name": badge.name,
        "file": f"{badge.name}.png",
        "tags": list(badge.tags),
        "expresses": badge.expresses,
        "anchor": "center",
        "default_seconds": badge.seconds,
        "animation": badge.animation,
        "attribution": FLUENT_CREDIT if badge.fluent else TWEMOJI_CREDIT,
    }


_BY_CODE = {badge.code: badge for badge in BADGES}


def rasterise_badge(code: str, size: int = BADGE_SIZE) -> bytes:
    """One badge's artwork as a PNG with a real alpha channel.

    Keyed by codepoint because that is all the fetch command hands over, and the
    codepoint is the badge's identity in both source sets. Both sources are
    vector: Fluent ships its 3D style as 256px PNG only, which a large accent
    would upscale, and Twemoji's PNGs are 72px. macOS has no rasteriser that
    preserves transparency (QuickLook flattens onto an opaque background, `sips`
    cannot read SVG at all), so this needs cairo.
    """
    import cairosvg

    badge = _BY_CODE.get(code)
    url = source_url(badge) if badge else TWEMOJI_SVG.format(code=code)
    return cairosvg.svg2png(url=url, output_width=size, output_height=size)
