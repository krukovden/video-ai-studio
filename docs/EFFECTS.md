# Effects

What it would take to add visual effects to this pipeline's output, which
routes are actually permitted on footage of a child, and what to build first.

Every price and policy claim below carries a source link. Anything I derived
by arithmetic, or estimated rather than measured, is marked as such. Prices
were checked on 29 July 2026 and will move; the policy positions move more
slowly but they do move, and two of the services named here have already
announced model sunsets inside the next eight weeks.

The creator asked for three things: an exaggerated pop when the child presses
a bubble, colour correction of the real footage, and a moment where the child
is cut out and placed in a different reality. This document answers all three,
and adds two routes the creator raised separately — generated clips made from
the product photo, and animated text — because both turn out to matter more
than the first three do.


## The recommendation, in one paragraph

Build the bubble-pop effect first, locally, with ffmpeg, on the silent
close-up inserts. It is the one the creator asked for, it lands on the only
footage in this project that contains no person at all, it costs nothing, it
carries no policy risk, and the pipeline already knows the exact second it
should happen. Do the "no AI at all" package in the same pass — a punch-in, a
two-frame flash, a short speed ramp and a sound effect on the pop — because
for a macro close-up of a toy that *is* the effect, and a generative model
would produce a worse version of it at a hundred times the price. Do colour
grading second, as a single measured constant per clip rather than an
adaptive filter, and be prepared to ship it switched off. Treat the
"different reality" background swap as the last thing, do it locally with
Apple's Vision framework, and design it as a partial effect (blur, darken,
push the real background out of focus) rather than a full replacement,
because a locally matted child on a bright new background will look cheap and
a locally matted child against a softened version of his own room will not.
Do not send footage of the child to any hosted generative service; the reason
is not that they would all refuse — most would not — but that the terms you
would be accepting over a child's face are worse than the effect is worth.


## The fact that changes everything

The pipeline's own visual gate already recorded which shots contain the
child. In `assets/1.Toy_Pimple_Popping/work/05b-visual.json`, nineteen of the
twenty-two selected segments carry `child_visible: true`. The remaining three
carry `child_visible: false` and `adult_prominent: false`, and the notes the
model wrote for them read:

- index 5 (`clip-12`): "Close-up insert of a gloved hand with syringe over the paint-covered squishy toy, no adult present."
- index 12 (`clip-10`): "Close-up insert of the paint-covered squishy toy and syringe on the table, no person prominent."
- index 18 (`clip-11`): "Close-up insert of gloved hands manipulating the bubbled toy, no adult present."

Those three are the `Analysis.inserts` — silent close-ups the planner cut in
because nobody was narrating over them. They are also, precisely, where the
bubble-pop effect belongs. `clip-11`'s description from the analyze stage is
"Close-up of a paint-smeared hand pressing on the toy and popping one of its
bubbles" — the creator's requested moment, already identified, already placed
in the timeline, and containing no face.

This splits the problem in two, and the two halves have completely different
risk profiles. The pop effect and any product-shot work happen on frames with
no identifiable person in them. Only the background swap necessarily involves
the child, because the child is the subject being swapped.


### Where the policy line actually falls

The widely repeated claim that "AI video services refuse content involving
minors" is mostly wrong as stated. Almost every service's minors clause is
written around *exploitation* — sexualisation, grooming, CSAM — not around
depiction. What actually blocks people is either an automated classifier that
fires on faces regardless of what the written policy says (Google, Adobe), or
one genuinely categorical upload ban (Pika).

| Service | What the rule keys on | Hand + toy, no face | Child's face in frame |
|---|---|---|---|
| Runway | consent for "another person", plus a stricter threshold when a child is depicted | Clears the child rule; arguably still inside the consent clause | Permitted in writing, but see below |
| Google Veo | the `personGeneration` parameter, adult/child axis | Clean — no person, parameter never engages | Blocked in practice |
| OpenAI Sora 2 | faces, regardless of age or consent | Clean | Blocked outright |
| Pika | age of the depicted subject, absolutely | Probably clean, thinnest margin | Banned |
| Kling | portrait, personality and personal-data rights | Clean | Consent-based, permitted |
| Luma | consent and identifiability | Clean | Consent-based, permitted |
| MiniMax / Hailuo | "Portrait" = facial features or likeness | Clean, clearest text | Consent-based, permitted |
| fal.ai | consent, likeness, biometric data | Clean | Consent-based, plus platform classifiers |
| Replicate | almost nothing at platform level | Clean | Not restricted; per-model licences apply |
| Bria, VEED (matting) | generic unlawful-content clauses only | Clean | Not addressed at all |

The evidence for each row:

Runway's [usage policy](https://runway.com/safety/usage-policy) (updated 6
March 2026) says under Children's Safety, after listing CSAM, grooming and
abuse: "We also apply a stricter standard for potential harmful and
inappropriate content (as outlined below) when a child is depicted." That
sentence presupposes children *may* be depicted; what changes is the
threshold on the other categories. Separately it prohibits "Use of an image,
video, or audio of another person without their permission" — which has no
identifiability floor, so a gloved hand is textually still footage of another
person, and the clause is discharged by having permission rather than by
removing the face. The frequently quoted line about "characters based on the
face or voice of a person under the age of 18" is real but is scoped in the
source document to Runway's Characters and Game Worlds products, not to
Gen-4.5, Aleph 2.0 or video-to-video.

Google is the one that genuinely blocks. The [responsible-AI guidance for
Veo](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/video/responsible-ai-and-usage-guidelines)
documents safety codes 58061214 and 17301594 under category "Child":
"Rejects requests to generate content depicting children if
`personGeneration` isn't set to `allow_all` or if the project isn't on the
allowlist for this feature." And [the Veo API
docs](https://ai.google.dev/gemini-api/docs/veo) show that for every current
model on the image-to-video path, `allow_adult` is the only permitted value —
`allow_all` is unavailable. In the EU, UK, Switzerland and MENA it is
`allow_adult` only for all paths. Developers have been publicly requesting
allowlist access for children's-content channels through mid-2026 with no
visible grants ([example
thread](https://discuss.ai.google.dev/t/request-allowlist-access-for-veo-3-1-person-generation-image-to-video-minors-project-little-learning-lab/174587)).

OpenAI is blunter and does not use age as the axis at all. The [video
generation
guide](https://developers.openai.com/api/docs/guides/video-generation) states
"Input images with faces of humans are currently rejected" and "Character
uploads that depict human likeness are blocked by default." Note also that
the Sora 2 API is scheduled for removal on 24 September 2026 with no
recommended replacement ([deprecations
page](https://platform.openai.com/docs/deprecations)), so it is not a
foundation to build on regardless.

Pika is the only categorical ban. Its [acceptable use
policy](https://pika.art/acceptable-use-policy) says, in bold in the source:
"You must not upload any images that depict or appear to depict individuals
under the age of 18." There is no parental-consent carve-out. Parental
permission does not cure it. Pika also has no first-party API for its
video-to-video features, so it is doubly unusable here.

Adobe Firefly deserves a mention because it is the service most likely to be
reached for by accident. An Adobe community manager wrote in a
[2026 thread](https://community.adobe.com/questions-404/p-not-sure-why-my-images-violate-guidelines-image-depicts-children-1475167)
that "Firefly does not allow video generation that includes children or
minors in most models." Adobe's *local* tools — After Effects Roto Brush, and
the Object Matte Tool added in AE 26.2 — run outside that filter and are
fine.


## The routes, ranked

Costs are anchored to this project: a 181.95-second draft of 22 segments, of
which 16.2 seconds are the three silent close-up inserts. Sources are 3840x2160
at 30 fps; the draft renders from proxies.

| Route | Cost for this video | Policy risk, child in frame | Policy risk, hand + toy only | Expected quality |
|---|---|---|---|---|
| ffmpeg-only effects: punch-in, flash, speed ramp, sound, alpha particle overlay | $0 | None — nothing leaves the machine | None | High. This is what a human editor would cut. |
| ffmpeg colour grading from a measured constant | $0 | None | None | Modest but real. Easy to make worse. |
| Local matting, Apple Vision + ffmpeg composite | $0 | None | n/a | Mixed. Good on silhouette, fails on hair. Honest verdict below. |
| Pillow-rendered animated titles piped to ffmpeg | $0 | None | None | High for a kids' channel. Currently the pipeline's weakest visual element. |
| Hosted matting: Bria video background removal | $0.60 for the whole draft (derived: 182 s x $0.0033) | Terms silent on minors; moderation is opt-in and has no minors category | None | Better edges than Vision. Uploads the child. |
| Replicate + RobustVideoMatting | ~$0.05–0.12 per 60 s clip (estimate) | AUP silent on minors; inputs auto-deleted after 1 hour via API | None | Comparable to Bria. GPL-3.0 model. |
| Runway `aleph2` video-to-video | $1.40 per 5 s; $50.95 for the whole draft (derived: 28 credits/s at $0.01) | Permitted in writing; stricter threshold; perpetual training licence over inputs | Low | Strongest hosted video-to-video available. |
| Kling 3.0 Omni video input | $0.84 per 5 s at 1080p (derived: $0.168/s) | Consent-based, permitted; only vendor with a contractual no-train commitment on the API tier | Low | Good. Singapore-hosted. |
| Luma Ray 3.2 `video_edit` | $2.16 per 5 s at 1080p | Permitted, but the worst input terms of the set | Low | Good. |
| Generated clips from the product photo | $0.13–0.60 per 5 s clip; budget 4–8 attempts | No person in the input at all | n/a | Unreliable on this specific subject. See below. |
| Google Veo, OpenAI Sora, Pika, Adobe Firefly cloud | — | Refused or banned | Clean but pointless | n/a |

Price sources: [Runway API pricing](https://docs.dev.runwayml.com/guides/pricing)
("Credits can be purchased for $0.01 per credit"; `aleph2` at 28 credits per
second with a 56-credit minimum; `gen4.5` at 12; `gen4_turbo` at 5);
[Kling API pricing](https://kling.ai/document-api/pricing/base/video.md);
[Luma pricing](https://docs.agents.lumalabs.ai/guides/pricing/index.md);
[Bria pricing](https://bria.ai/pricing) ("Video Remove Background ...
$0.0033/sec"); [Replicate pricing](https://replicate.com/pricing) and the
[RobustVideoMatting model page](https://replicate.com/arielreplicate/robust_video_matting);
[Seedance on Replicate](https://replicate.com/bytedance/seedance-2.0).

Two deadlines worth noting: Runway's `gen4_aleph` and `gen3a_turbo` sunset on
30 July 2026 and `veo3` on 4 August 2026 (see the deprecation banner on
Runway's pricing page), so any Runway work targets `aleph2` and `gen4.5`.

### Is Runway usable here?

Yes on the close-up inserts, where there is no face. Yes in writing on the
footage containing the child, and no user reports were found of Runway
refusing benign footage of a child — the suspension wave visible in its
community through 2026 is attributed by users and by Runway support to
scripting and account-sharing detection, not content. But three things make
it the wrong choice for the child-bearing footage in this project.

First, routing. Runway's API has no background-removal, inpainting,
object-removal or colour-grade endpoint. Those are web-app "Apps", and
Runway's help centre states plainly: "Is there an API available for Apps? No,
Apps cannot be accessed via the API"
([source](https://help.runwayml.com/hc/en-us/articles/45570040112531-Creating-with-Apps)).
The legacy Remove Background tool was retired into the Backdrop app in April
2026. So the two operations the creator asked for that Runway is famous for
cannot be automated at all; only `aleph2` video-to-video, `act_two` and the
upscalers are API-reachable.

Second, the failure mode is expensive and one-way. Runway moderates inputs as
well as outputs, and its [task-failures
documentation](https://docs.dev.runwayml.com/errors/task-failures) says
"Unlike other failures, credits are not refunded for `SAFETY.INPUT.*`
failures" and "You should not retry these generations." Its help centre adds
that moderation "cannot be disabled for an account, project, or topic by the
support team" and that repeatedly attempting a blocked input "may result in
account suspension"
([source](https://help.runwayml.com/hc/en-us/articles/21745792516371-Why-is-my-input-getting-content-moderated)).
An automated pipeline that retries on failure is exactly the shape that turns
a false positive into a ban.

Third, and decisively, the terms. Runway's [terms of
use](https://runway.com/terms-of-use) §4.4 grants Runway "a non-exclusive,
irrevocable, perpetual, worldwide, royalty-free ... license to use any Inputs
and Outputs" and states that inputs "may be used by the Company to train and
improve its AI models". Its privacy policy separately treats "scans of faces
that you submit" as biometric data. Uploading a child's face under a
perpetual, irrevocable training licence is a decision about the child, not
about the video, and it is not one this pipeline should make automatically.

That third objection applies to nearly every hosted service, in varying
degrees. Luma's free tier goes further still, granting Luma the right to
"publicly display, publicly perform ... and distribute Input"
([terms](https://lumalabs.ai/legal/terms-of-service) §4.2). Viggle's terms
grant an irrevocable, perpetual licence to train on your upload *and* to
"provide and make available your User Content to other users"
([§5(b)](https://viggle.ai/terms-of-use)) and should be avoided outright.
Kling is the one contrary case: its [API paid-service
terms](https://kling.ai/document-api/guides/protocols/paid-service.md) §5.1
say "We will not use, or authorize any person or entity to use, your data to
train, retrain, or otherwise improve the Services" — but that applies to the
API tier only, not the consumer app.


## The bubble pop: a specific proposal

The technique is not generative. It is the one a human editor would use: a
punch-in that snaps on the frame the bubble bursts, a two-frame white flash
under it, a short slow-motion ramp out of it, an alpha-channel particle burst
composited on top, and a sound effect. Five elements, all local, all free,
all in ffmpeg 8.1.1 as installed on this machine (verified: `overlay`,
`blend`, `curves`, `eq`, `setpts`, `zoompan`, `lut3d`, `colorlevels`,
`colortemperature`, `normalize`, `minterpolate` and `deflicker` are all
present; VP9 and ProRes decode are available).

### Timing it from data the pipeline already has

There are two different timing problems here, and conflating them is the main
way this goes wrong.

**Spoken "pop" moments.** The transcript carries word-level timings from
parakeet. Mapping a word to its position in the rendered draft is arithmetic
on artifacts that already exist: for a timeline clip `c` and a word `w` in
clip `c.src` with `c.offset <= w.start < c.offset + c.dur`, the word lands at
`c.start + (w.start - c.offset)` in the draft. Running that over
`03-transcript.json` and `05-timeline.json` for this project yields six
utterances, every one of them inside the "Pop Time" beat:

| Draft time | Source | Word | Timeline clip |
|---|---|---|---|
| 157.86 | clip-08 @ 445.20 | "popping" | 15 |
| 158.42 | clip-08 @ 445.76 | "popping." | 15 |
| 164.88 | clip-08 @ 303.28 | "pop" | 16 |
| 166.56 | clip-08 @ 304.96 | "pop" | 16 |
| 169.66 | clip-08 @ 309.12 | "pop" | 17 |
| 175.68 | clip-08 @ 144.76 | "popped" | 19 |

That is genuinely useful, and it is free. But it must be said clearly: **a
word timestamp marks when the child says "pop", not when a bubble visually
bursts.** The two are typically a beat apart and sometimes a sentence apart —
"that's how you pop them" is an explanation, not an event. Hanging a flash
frame on the word would look wrong. The spoken timings are the right anchor
for *text* — a "POP!" callout, a counter — and the wrong anchor for a
physical effect.

**Actual visual pops.** These live in the silent inserts, which carry no
words at all, so the transcript cannot help. Three sources of truth are
available in descending order of cheapness:

1. *Audio transient detection.* All three insert clips have audio
   (`has_audio: true` in `01-manifest.json`). A bubble bursting is a sharp
   broadband onset against near-silence. A first pass with ffmpeg's
   `silencedetect` or `astats` over the insert's own audio, or a short-window
   RMS derivative in numpy, will find it to within a frame or two. This costs
   nothing and needs no model.
2. *Frame differencing.* A pop is a fast, spatially localised luminance and
   shape change in an otherwise slow macro shot. Mean absolute frame
   difference over the insert, peak-picked, corroborates the audio onset.
   OpenCV is already a dependency.
3. *Ask the model.* The analyze stage already sends keyframes to Claude and
   already writes one-sentence descriptions of each insert. Extending
   `InsertClip` with an optional list of event timestamps, and the prompt with
   a request for them, is a small change that costs $0 under the existing
   Claude subscription — the same basis on which `analyze`, `plan` and
   `visual_check` already cost $0.

The right design is to use (1) as the detector and (3) as the corroborator,
and to write the result into a new artifact rather than inferring it at
render time.

### Where the stage goes, and the drift trap

An effects stage belongs between `render_draft` and `polish`, producing a new
artifact — call it `07a-effects` — that records, per timeline clip index, a
list of effect instances with a local offset and a type. `polish` then
composites them in its existing single filter-graph invocation.

The trap is that draft time is not timeline time. `s08_polish.py` already
documents this and already solves it: rendered segments are whole numbers of
frames, so each is a frame or so longer than the timeline asked for, and the
error accumulates across twenty-odd cuts — "enough to put a dissolve on the
wrong side of a cut". Its `segment_starts()` function measures the actual
segment files on disk and falls back to stretching the timeline onto the
draft's measured length. An effects stage must use the same function rather
than recomputing starts from `TimelineClip.start`, or the flash will land on
the wrong frame. And `polish` shifts everything again by `intro_offset` and
by one transition per dissolve — the existing lower-third placement code is
the worked example to copy.

### The five elements, concretely

A punch-in is a scale-and-crop, not `zoompan`. Measurement on this machine
showed `zoompan` produces uneven per-frame width deltas (8,8,8,7,8,8,9,7...)
while `scale` with `eval=frame` followed by `crop` produces uniform ones —
and `zoompan`'s `s=` parameter silently defaults to `hd720`, which would
downscale a 1080p frame without saying so. Note also that `crop`'s `w` and
`h` are evaluated once at initialisation; only `x` and `y` are per-frame, so
the widely cited "scale up then animate the crop" recipe does not work.

A flash should be gated on frame number, not time: `between(t,...)` with
endpoints inclusive gives three frames where two were intended. Use
`eq=brightness=0.9:enable='between(n,150,151)'`.

A particle burst needs a real alpha channel. Two warnings. First, ffmpeg's
native VP9 decoder **silently drops alpha** — the stream reports `yuv420p`
and composites as an opaque black box; the container's `alpha_mode=1` tag is
the real signal, and decoding requires `-c:v libvpx-vp9` placed *before* the
input. ProRes 4444 round-trips reliably (and is reported back as
`yuva444p12le`, so do not string-match on `p10le`). Second, if the element
is black-background stock with no alpha and you screen-blend it, force RGB
around the blend or you get a severe magenta cast: screening the U and V
planes, where 128 is neutral, yields roughly 192. Measured on a test frame,
`blend=screen` on `yuv420p` moved UAVG from 124 to 187 and VAVG from 128 to
194; forcing `gbrp` first left both correct.

The element itself can be generated rather than licensed, which sidesteps the
stock-licence question entirely. This produces a clean expanding ring with
real alpha:

```
ffmpeg -f lavfi -i "color=c=black:s=600x600:r=30:d=0.5" -vf \
 "format=rgba,geq=r='255':g='220':b='60':a='255*exp(-pow((hypot(X-300,Y-300)-(T*900))/40,2))*(1-T*2)',format=yuva420p" \
 -c:v prores_ks -profile:v 4444 -pix_fmt yuva444p10le pop.mov
```

Note that `drawbox` does not write the alpha plane, so drawing onto a
transparent RGBA source leaves alpha at zero and the element invisible; `geq`
with an explicit `a=` expression is the way to author alpha procedurally. If
stock is preferred instead, [Pexels' licence](https://www.pexels.com/license/)
is verified as free for commercial and monetised use with no attribution
required. Licences for ProductionCrate, ActionVFX, Videvo, Pixabay and Mixkit
could not be verified — every attempt returned 403 or hid the terms behind
JavaScript — so those need checking by hand before use.

Overlaying at a timestamp has one non-obvious trap: `enable=` gates only the
compositing, not the element's own playback, so by the time the window opens
the element has already finished. Delay the element with
`setpts=PTS-STARTPTS+3/TB`, `tpad`, or `-itsoffset`, and pass
`eof_action=pass` or the last frame freezes on screen forever. Also pass
`format=auto` on `overlay`: it defaults to `yuv420`, which chroma-subsamples
the particles into mush.

The sound effect is the cheapest and most effective element of the five and
needs no new machinery — `polish` already mixes and ducks audio.


## Colour grading: worth doing, but not adaptively

Yes, worth doing. No, not with an adaptive filter.

ffmpeg has a genuine automatic white balance in `colorcorrect` with
`analyze=average|minmax|median`, and an auto-levels filter in `normalize`.
Both re-decide on every frame, and that is exactly the problem: their job is
to remove variation, and some of the variation is the thing you filmed.
Measured on a clip with a deliberate brightness ramp — the shape of a bright
toy entering frame — the standard deviation of average luma across frames was
41.06 unfiltered, and 8.86 with `normalize` at defaults. The filter destroyed
78 percent of the intended lighting change. `normalize`'s defaults are
actively hostile: `smoothing` defaults to 0, meaning maximum temporal flicker,
and `independence` defaults to 1, meaning fully per-channel correction, which
is where colour casts come from.

Clipping tells the same story. Fraction of pixels pinned at 0 or 255 on a
test clip: 0.08 percent high in the original, 2.00 percent with
`colorcorrect=analyze=median`, 4.68 percent with `analyze=minmax`, and 4.94
percent with a full-strength LUT — sixty times the original. The same LUT at
40 percent strength brought it back to 0.09 percent.

So the pattern that works is: measure once in Python, apply a constant. Sample
every sixtieth frame with `signalstats`, take the median of `YMIN`, `YMAX`,
`UAVG` and `VAVG` so one bright toy does not skew the result, derive a fixed
`colorbalance` and `eq` from the offsets from neutral (128 for U and V), and
apply those constants for the whole clip. Stretch levels to 16–235 rather
than 0–255 to keep headroom. This is stable, has no pumping, no flicker, and
is cheap enough to run in the same pass as everything else.

If a LUT is used, run it at partial strength through a `split`/`blend` pair
rather than at full strength, and never apply a log-to-Rec709 conversion LUT
to phone footage that is already Rec709 — that is the 0.08-to-4.94 percent
failure above, in miniature.

One more honest caveat. This footage is iPhone footage, which arrives already
graded by a very good pipeline. The realistic upside of automatic grading
here is small: correcting a mild indoor colour cast and lifting a slightly
crushed black point. The realistic downside is making twenty-two segments
inconsistent with each other because each was measured separately. Grade per
*source clip*, not per segment — this project's 22 segments come from only
six source clips — and provide a config switch that turns the whole thing off,
defaulted to off until it has been eyeballed on a real render.

On disclosure: YouTube's [altered or synthetic content
policy](https://support.google.com/youtube/answer/14328491) explicitly exempts
"Color adjustment or lighting filters" and "Special effects filters, like
adding background blur or vintage effects" from its disclosure requirement.
Grading and the pop effect are both clearly inside that exemption.


## The "different reality" moment

This is the one that necessarily involves the child, and it is the one to do
last.

Do it locally. Apple's Vision framework is the pragmatic choice: it ships with
macOS, is GPU-accelerated, has no licence problem, and never touches the
network — which for footage of a child is the entire point. Measured on an M1
Pro at 1080p, `VNGeneratePersonSegmentationRequest` at `accurate` quality
returns a 2016x1512 mask in 57.2 ms per frame (17.5 fps); an end-to-end
pipeline of ffmpeg decode, Vision, mask cleanup and re-encode ran at 12.4 fps,
so one minute of 30 fps footage costs about 2.4 minutes. That is fine for
offline batch work.

Two measured gotchas. The raw mask is not solid — Vision returns
confidence-like values inside the subject, and composited raw the child comes
out semi-transparent with background bleeding through his torso. Interior
solidity measured 0.744 raw and 0.933 after a levels remap of
`clip((a-0.35)/0.30, 0, 1)`; that remap is mandatory. And
`VNGeneratePersonInstanceMaskRequest`, despite returning a full-resolution
float mask, is *worse* for compositing — interior solidity 0.372, values
ranging from -0.17 to 1.28 from upsampling overshoot, and a visible halo.

The honest quality verdict, from actually looking at composites rather than
reading model cards: silhouette and clothing are genuinely good and would pass
at 1080p playback speed. Hair fails — the mask cuts a smooth blob through
curls, background bleeds through, and every flyaway strand is gone. Fingers at
the frame edge come out partly eaten. Per-frame segmentation flickers along
the boundary, worst on hair and fast-moving arms, and an exponential moving
average on the mask (`0.6*new + 0.4*prev`) is needed to calm it. Edge fringing
is real but strongly background-dependent: over a saturated magenta background
it was glaring; over a dark navy background the same matte looked convincing.

That last observation is the most useful practical lever in this whole
document. **Choose a replacement background close in luminance and hue to the
original and most of the matte's sins disappear. Choose a bright saturated
cartoon background and it will look cheap.** And the version that actually
ships is the partial one: blur, darken, desaturate or push the real background
out of focus rather than replacing it. That hides essentially every failure
mode, still reads as a deliberate effect, and — per the YouTube policy quoted
above — is explicitly exempt from disclosure as a "special effects filter,
like adding background blur".

If a hosted matting service is ever wanted for better edges, the two worth
knowing about are [Bria](https://docs.bria.ai/video-editing.md) at $0.0033 per
second, whose terms are silent on minors and whose moderation is opt-in with
no minors category at all, and Replicate running RobustVideoMatting, whose
[acceptable use policy](https://replicate.com/acceptable-use-policy) is
likewise silent on minors and whose [API deletes inputs and outputs after one
hour](https://replicate.com/docs/topics/predictions/data-retention) — note
that this deletion applies to the API only, and that data from the web
interface "is kept indefinitely". Both are non-generative: they return a
matte, not invented pixels, which is why neither has a reason to run a
person-detecting classifier over the frames. If the local route's edges prove
unacceptable, these are the correct next step and not Runway.

The model-licence picture matters if this ever moves off Vision. The current
state of the art in video matting, MatAnyone and MatAnyone 2, is under the
NTU S-Lab licence, which permits non-commercial use only — unusable for a
monetised channel. RobustVideoMatting is GPL-3.0 (fine for producing video;
it only bites if the pipeline itself is distributed) and its last release note
is from November 2021. SAM 2 is Apache-2.0 but its MPS support is
[broken](https://github.com/facebookresearch/sam2/issues/687) and its masks
are segmentation-grade, not matting-grade — its correct role is producing a
first-frame mask, not processing a clip. BiRefNet and BEN2 are MIT and are the
permissive fallbacks.


## The two routes that were not asked about but should outrank the ones that were

### Animated text

The pipeline's titles are currently its weakest visual element, and the cause
is not ffmpeg. `videoai/core/text.py` falls back to OpenCV's
`cv2.FONT_HERSHEY_DUPLEX` when `drawtext` is unavailable — and `drawtext` *is*
unavailable, because this machine's Homebrew ffmpeg 8.1.1 is built without
libfreetype, exactly as the README notes. Hershey faces are single-stroke
vector plotter fonts from the 1960s: no real typeface, no kerning, no variable
weight, and "bold" is just a thicker stroke. So the fallback path is what
actually runs on every render.

Pillow is not in `pyproject.toml` and is not installed (verified:
`ModuleNotFoundError: No module named 'PIL'`). Pillow ships its own bundled
FreeType and can load system TTFs directly, so adding one dependency replaces
plotter-font titles with real typography without rebuilding ffmpeg. Going
further and piping Pillow-generated RGBA frames into ffmpeg as `rawvideo`,
rather than writing PNGs, was benchmarked at 108.7 fps for a 1080p30 animated
title composited and hardware-encoded — 3.6 times realtime. That unlocks
per-frame animation: eased slide-ins, pop-ins, counters, and word-by-word
highlighting driven by the word-level timings the pipeline already has. For a
kids' channel that is worth more than any generative effect in this document,
and it costs a dependency and half a day.

[Remotion](https://www.remotion.dev) is the credible step beyond that, and it
is free here: its [licence](https://raw.githubusercontent.com/remotion-dev/remotion/main/LICENSE.md)
grants free commercial use to "an individual" or "a for-profit organization
with up to 3 employees", with no watermark and no feature gate. The correct
integration is to render a transparent ProRes 4444 overlay and composite it
with the `overlay` filter already in use — never to render the whole video in
it. The cost is a Node toolchain and React components inside a Python project,
which is a real and permanent maintenance burden. Do Pillow first; reach for
Remotion only if something specifically needs flexbox layout, per-glyph
springs, blurs or masking.

### Generated clips from the product photo

The creator asked about stitching short generated clips in at the beginning,
middle and end, made from the clean product photo at
`assets/1.Toy_Pimple_Popping/description/OIP.jpeg`. That photo contains no
people, so it sidesteps the minors question entirely, and the cost is trivial:
Seedance via Replicate at about $0.13 per silent 5-second 720p clip,
Runway `gen4_turbo` at $0.25 and `gen4.5` at $0.60 for the same length. Even
twenty attempts across three beats comes in under $8.

Cost is not the constraint. Three other things are.

The subject is close to the worst case for current models. A squishy toy that
deforms and recovers under pressure is a conservation-of-mass problem, and
[VideoPhy-2](https://arxiv.org/abs/2503.06800) (UCLA, ICLR 2026, human
annotated) finds the best models reach only about 22 percent joint semantic
and physical-commonsense adherence on its hard subset, with mass and momentum
conservation among the named failures. Hand-object interaction is a documented
distortion trigger, and on-product text and markings are the single most
reliably broken element. A slow push-in with no hands and no text in frame is
the models' happy path and would likely be usable in perhaps one generation in
two to four (estimate); AI-generated squishing, or AI hands touching the toy,
is the intersection of the two worst failure modes and should not be
attempted.

The prompt itself is a moderation risk that has nothing to do with children.
"Pimple popping" vocabulary — pus, extraction, squeezing, lesions — can hit
gore and medical classifiers even when the input is unambiguously a plastic
toy. Describe it mechanically: soft silicone novelty toy, coloured gel,
textured surface, macro product shot, studio lighting, slow push-in. And note
that Runway charges full credits for generations its moderation rejects.

The third objection is the real one. YouTube's disclosure policy requires
disclosure when content "Generates a realistic scene that didn't actually
occur", and none of the exemptions reach a synthesised shot of the actual
product cut in so that it reads as filmed. Since
[May 2026](https://blog.youtube/news-and-events/improving-ai-labels-viewers-creators/)
YouTube has been applying labels automatically where its systems detect
significant photorealistic AI use, and the label sits directly below the
player. Fifteen seconds of gloss would attach a permanent AI badge to the
whole video. For a channel that is almost certainly Made for Kids, in a year
when [200-plus organisations publicly asked YouTube to prohibit AI-generated
Made-for-Kids videos](https://fortune.com/2026/04/01/ai-slop-200-organizations-letter-youtube-google/),
that is a bad trade — and the inference that a self-applied AI label could
affect distribution inside the Kids app is exactly that, an inference, but it
is the highest-stakes unknown in this document.

The alternative gets most of the benefit with none of the risk: shoot the
beauty shot for real. A phone on a cheap turntable, a sheet of paper and a
lamp produce a genuine five-second push-in of the actual toy, with no
disclosure and no policy surface. The pipeline already prefers real
close-ups — `videoai/logic/inserts.py` exists precisely to find them.


## What not to attempt

**Do not send footage containing the child to any hosted generative service.**
Not because it would be refused — Runway, Kling, Luma and MiniMax would all
most likely accept it — but because the terms are worse than the effect. Every
consumer-tier hosted service in this survey except Kling's API tier takes a
perpetual or irrevocable licence over uploaded inputs for model training, and
several classify submitted faces as biometric data. That is a decision about a
child's likeness, and an automated pipeline should not be the thing that makes
it.

**Do not build on Google Veo or OpenAI Sora for anything involving people.**
Veo's `personGeneration` parameter has no value permitting minors on the
image-to-video path on any current model, and none at all in the EU, UK,
Switzerland and MENA; the allowlist has no public process. Sora rejects input
images containing human faces outright, and its API is scheduled for removal
on 24 September 2026.

**Do not use Pika.** Its acceptable use policy bans uploading images depicting
anyone under 18, with no consent carve-out, and its video-to-video features
have no API anyway.

**Do not build an automated retry loop against a moderated endpoint.**
Runway's documentation is explicit that safety-input failures are not refunded
and should not be retried, and that repeated attempts risk suspension. A
pipeline that treats a moderation refusal like a network error will
eventually get the account banned.

**Do not use adaptive colour filters.** `normalize` at defaults measurably
destroys deliberate lighting changes and flickers; auto white balance
re-decides every frame. Measure once, apply a constant.

**Do not attempt a full photorealistic background replacement of the child.**
Local matting cannot do hair, hosted matting means uploading the child, and a
photorealistic swap is the one variant that triggers YouTube's disclosure
requirement. A partial background treatment achieves the same beat, looks
better, and is explicitly exempt.

**Do not generate AI footage of hands squishing the toy.** It is the
intersection of the two most reliably broken things current video models do.

**Do not reach for generative video before fixing the titles.** The pipeline
currently renders every on-screen word in a 1960s plotter font because one
dependency is missing. That is visible in every frame of every video, and it
costs one line in `pyproject.toml`.


## Sequenced plan

1. **Add Pillow and replace the Hershey text renderer.** One dependency, a few
   hours, visible in every video. Highest return in this document.
2. **Build the effects stage on the silent inserts.** Audio-transient
   detection to find the pop, then punch-in, two-frame flash, speed ramp,
   procedurally generated alpha burst and a sound effect. $0, no policy
   surface, and it is the creator's actual request.
3. **Extend the same stage to spoken "pop" moments as text**, using the
   word-level timings mapped through `segment_starts()`. A "POP!" callout on
   the word is correct; a physical effect on the word is not.
4. **Add measured, constant colour grading per source clip**, defaulted off
   until it has been reviewed on a real render.
5. **Only then, the background moment** — locally, with Vision, as a partial
   treatment against a tonally matched background, on one shot, as a single
   deliberate beat rather than a repeated effect.
6. **Shoot a real product beauty shot** instead of generating one.

Steps 1 through 5 cost nothing but time and touch no third party. That is not
a coincidence: on this footage, for this audience, the effects that would
actually improve the video are the ones a human editor would cut by hand, and
the pipeline already knows where every one of them goes.
