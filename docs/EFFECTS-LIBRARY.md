# The cartoon-effects sprite library

`assets/effects/` holds the sprites the pipeline can composite over a delivery,
plus `manifest.yaml`, which is the only place the pipeline learns what exists.

Everything shipped there was drawn procedurally with Pillow so the effects stage
works out of the box with **zero API calls and zero subscriptions**. Those
drawings are seeds, not the point. Any one of them can be replaced by a better
picture — including one generated with OpenAI's image model — and the swap needs
no code change at all.

This document is how to do that swap.


## How an effect actually reaches the screen

Three parts, deliberately separate:

1. **The library** (`assets/effects/manifest.yaml` + the PNGs). Names, tags, an
   anchor, a duration and one of five built-in animations.
2. **The `effects` stage** (`videoai/stages/s05d_effects.py`, artifact
   `05d-effects`). It sends the model the manifest's *words* — names, tags, one
   sentence per sprite saying what it expresses — together with the edit, its word
   timings, the silent inserts' descriptions and the brief. It gets back moments.
   It never sees a pixel.
3. **The compositor** (in `videoai/stages/s08_polish.py`). It loads the PNG, runs
   the animation curve, places it by grid cell inside the safe area and
   alpha-blends it into the same finite graphics track the titles and captions use.

So: **the model chooses by the manifest's words, the renderer draws by the file.**
A better drawing under the same name improves every video and re-plans nothing.


## Replacing a sprite with a generated PNG

### 1. Generate it

The prompt template, which matches the flat comic look the library was seeded
with:

```
flat cartoon comic-style <thing>, bold black outline, transparent background, no text
```

Fill `<thing>` with what the manifest entry says it expresses, for example:

| Entry | `<thing>` |
| --- | --- |
| `comic_starburst` | `impact starburst, jagged yellow and orange spikes` |
| `speed_lines` | `speed lines whoosh streaks pointing right` |
| `sparkle_stars` | `three four-pointed sparkle stars, white and yellow` |
| `sound_rings` | `sound waves radiating from a point, concentric arcs` |
| `confetti_burst` | `confetti burst, bright paper pieces flying upward` |
| `speech_bubble` | `empty rounded speech bubble with a tail at the bottom left` |

With the OpenAI images API, request a transparent background explicitly:

```bash
curl https://api.openai.com/v1/images/generations \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-image-1",
    "prompt": "flat cartoon comic-style impact starburst, jagged yellow and orange spikes, bold black outline, transparent background, no text",
    "size": "1024x1024",
    "background": "transparent",
    "output_format": "png",
    "n": 1
  }' | jq -r '.data[0].b64_json' | base64 --decode > starburst-raw.png
```

Keep `no text` in the prompt even for the bubble. Text is rendered by the
pipeline, in the video's own typeface, at the size the words need — a generated
picture of lettering would be the wrong words in the wrong face.

### 2. Cut it out

A generated PNG usually still needs work before it is a sprite:

- **The background must be genuinely transparent**, not white and not a
  checkerboard the model drew. Check it:
  ```bash
  uv run python -c "
  from PIL import Image
  im = Image.open('starburst-raw.png').convert('RGBA')
  print(im.size, 'corner alpha:', [im.getpixel(p)[3] for p in
        [(0,0),(im.width-1,0),(0,im.height-1),(im.width-1,im.height-1)]])"
  ```
  Four zeros means it is cut out. Anything else needs the background removed —
  Preview's Instant Alpha on macOS, or `rembg i in.png out.png`.
- **Trim the transparent margin.** The compositor scales a sprite by its *height*,
  so a drawing floating in a large empty canvas comes out small.
  ```bash
  uv run python -c "
  from PIL import Image
  im = Image.open('starburst-raw.png').convert('RGBA')
  im.crop(im.getbbox()).save('starburst-trimmed.png')"
  ```
- **Long edge at least 512px.** It will be drawn at roughly a fifth to a third of
  a 1080p frame's height, and it is only ever scaled down.
- **No text, no drop shadow, no gradient background.** All three read as a
  screenshot pasted onto the video rather than as a drawn accent.

### 3. Drop it in

Save it into `assets/effects/` under the **same file name** the manifest already
names, and commit it:

```bash
cp starburst-trimmed.png assets/effects/comic_starburst.png
git add assets/effects/comic_starburst.png
```

That is the whole change. The next `videoai run` re-renders the delivery (the
library's file contents are in the polish stage's fingerprint) and does **not**
re-run the effects stage (only the manifest's text is in that one's fingerprint,
and it did not change).


## Adding a new effect

Add a PNG and a manifest entry:

```yaml
- name: thought_cloud
  file: thought_cloud.png
  tags: [thinking, confused, wondering]
  expresses: the presenter is puzzled or wondering what happens next
  anchor: bottom
  default_seconds: 1.4
  animation: pulse
```

| Field | Meaning |
| --- | --- |
| `name` | what the model must spell exactly; an unknown name is refused by name |
| `file` | the RGBA PNG, in this directory |
| `tags` | the words the story is matched against |
| `expresses` | one sentence, shown to the model instead of the picture |
| `anchor` | which point of the sprite lands on the grid cell's point: `center`, `top`, `bottom`, `left`, `right` |
| `default_seconds` | how long one event runs |
| `animation` | `pop-in`, `pulse`, `shake`, `drift-up` or `none` |
| `nine_patch` | only for a sprite stretched around dynamic text (see below) |

Adding an entry changes the manifest, so the effects stage re-plans on the next
run and may start using it. Nothing else changes.

The five animations are implemented as per-frame transforms in the compositor
(`animation_transform` in `videoai/logic/effects.py`):

- **pop-in** — overshoots to 118% and settles. For impacts.
- **pulse** — sinusoidal scale, ±9%, one and a half cycles. For sparkles, sound.
- **shake** — decaying rotation jitter, ±7°. For reactions and bubbles.
- **drift-up** — rises 34% of its own height while fading out. For confetti.
- **none** — still, with the shared fade in and out.

Every animation carries the same fade envelope, so nothing ever appears or
disappears on a single frame.


## The nine-patch speech bubble

`speech_bubble` is the one entry whose size comes from its content. The model
supplies short text; the compositor lays that text out, works out the box it needs
and stretches the PNG around it as a nine-patch: **the four corners keep their
pixels**, the top and bottom edges stretch horizontally, the left and right edges
stretch vertically, and the centre stretches both ways. That is what keeps a bold
outline the same weight around a bubble that grew from one line to three.

The insets say where those bands are, in pixels of the PNG itself:

```yaml
nine_patch:
  left: 224      # wide enough to contain the whole tail
  top: 96
  right: 96
  bottom: 212    # starts above the rounded bottom corners
  text_left: 34  # where text may go, from the edges of the STRETCHED image
  text_top: 34
  text_right: 34
  text_bottom: 170
  max_lines: 3
```

Two constraints that are easy to get wrong when replacing this file:

- **Everything that must not distort has to be inside a corner band.** The
  shipped bubble's tail is in the bottom-left corner, which is why `left` is 224
  and not 96 — a tail in the bottom *edge* band would stretch with the bubble.
- **The centre and the edge bands must be uniform along the axis they stretch.**
  The centre has to be pure interior, so `bottom` starts above the rounded bottom
  corners; if the corner curve reached into the right-hand band, stretching that
  band vertically would smear the curve.

To check a replacement, render a short and a long bubble and compare their
corners — the tests do exactly this
(`tests/test_effects.py::test_nine_patch_stretching_keeps_the_corners_undistorted`).


## Redrawing the seeds

The procedural drawings live in `videoai/logic/effect_seeds.py`. After editing
one:

```bash
uv run videoai seed-effects            # rewrites assets/effects/, no network
git add assets/effects
```

`videoai seed-effects --dir <path>` writes the library somewhere else, which is
what the tests use so they never touch the committed one.


## Turning effects off

```yaml
effects:
  enabled: false
```

No events are planned and none are composited. The delivery is otherwise
identical: effects are not one of `production-contract.yaml`'s required features,
because they are seasoning and not structure. `effects.max_events` (8 by default)
is a hard ceiling — a model that returns more is refused rather than truncated,
because an over-long list means it ignored the brief, not that its first eight
choices are the good ones.
