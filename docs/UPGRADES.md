# Upgrades

What each stage runs on today, what could replace it, what that would actually
change about the finished video, and roughly what it would cost. Every price
in this document is anchored to one real project: 13 clips, 28.8 minutes of
combined source footage, edited down to a 199-second (about 3.3-minute) draft
(`assets/1.Toy_Pimple_Popping/`). Where a price depends on a provider's public
rate card rather than something measured directly against this project, that
is said explicitly and marked as an estimate — treat it as an order of
magnitude, not a quote.

## What the current, free stack costs

| Stage | Runs on today | Cost per video |
|---|---|---|
| ingest | ffmpeg / ffprobe (local) | $0 |
| quality | OpenCV, local | $0 |
| sync | numpy cross-correlation, local | $0 |
| transcribe | parakeet-mlx, local (Apple Silicon GPU) | $0 |
| analyze | Claude Sonnet via the Claude Code CLI (subscription) | $0 |
| plan | Claude Sonnet via the Claude Code CLI (subscription) | $0 |
| visual_check | Claude Sonnet via the Claude Code CLI (subscription) | $0 |
| render_draft | ffmpeg (local) | $0 |

Total marginal cost per video today: **$0**, aside from the Claude
subscription itself and electricity. Every external call in the pipeline goes
through the Claude Code CLI under that subscription rather than a metered API
key, which is why "analyze", "plan" and "visual_check" all show $0 rather than
a per-call price — the same three stages would cost real money per call if
pointed at the Anthropic API directly instead.

This corrects the older `docs/state/STACK.md`, which listed seven stages and
omitted `visual_check` entirely; the pipeline has run eight stages, including
the visual gate, since it was added.

## Analyze: reading a transcript versus watching the video

This is the highest-value upgrade available, and it is not close.

Today, `analyze` never watches anything. It reads the packed transcript —
phrases only, in text — and, if `analyze.keyframes_per_phrase` is on, one
still frame per phrase at that phrase's midpoint. That is enough to judge
*what was said* and, roughly, *what a single instant looked like*. It cannot
judge delivery the way a person watching the clip would: comic timing, a
building reaction, the specific way a kid's voice catches when something
actually works — none of that is present in a transcript, and a single still
frame per phrase cannot show motion at all. For a loud toy with an obvious
payoff (something bursts, something falls over) text-only scoring is a
reasonable proxy for what's fun to watch. For a *quiet* toy — and the toy
review this pipeline was built and tested against is exactly that, a
squeeze-and-pop stress toy with no explosions in it — the best moments are
almost entirely a matter of delivery and small on-screen motion: a bubble
actually finishing a pop, a kid's face doing something the words don't
capture, the satisfying moment lasting exactly as long as it should. None of
that survives being turned into a transcript, so a text-only judge is
structurally blind to the thing that would actually make the video good.

Gemini's video understanding reads video natively — real frames sampled
across the whole clip, plus the actual audio track, not a transcript of it —
so it can judge energy, timing and on-screen payoff directly, the same way a
human editor scrubbing the footage would, instead of inferring all of it from
words on a page. This would change what `analyze`'s `delivery_score`,
`emotion` and `shorts_candidate` fields actually mean: today they are a text
model's guess at how something sounded from a transcript of it; with Gemini
they would be a judgment made from watching and hearing the moment itself.
Since `plan` selects and orders the edit entirely from `analyze`'s scores,
a better analysis pays off twice: once directly, and again through every
downstream editorial decision that reads it.

**Cost, estimated.** Gemini's public pricing processes video at roughly 260
tokens per second of footage (video frames plus the audio track combined),
and Gemini 2.5 Pro is priced at $1.25 per million input tokens and $10 per
million output tokens for prompts under 200K tokens. Submitting this
project's full 28.8 minutes (1,728 seconds) of source once comes to roughly
450,000 input tokens, or about $0.55 of input cost; a segment-scoring reply
covering ~200 phrases is a few thousand to perhaps 20,000 output tokens,
another $0.05–$0.20. That puts a full-source Gemini analyze pass at roughly
**$0.50–$1.50 per video** — call it about a dollar. The cheaper Flash tier
(around $0.15 per million input tokens) would land closer to $0.10–$0.20, at
some cost in judgment quality. Either way this is cheap relative to what it
buys, and cheaper still if only the clips that already passed the `quality`
gate are submitted rather than everything. Building this means adding a
`providers/llm_gemini.py` implementing the same `LLMProvider` protocol every
other model call already uses, and pointing `providers.llm` at it — the rest
of the pipeline (the prompts, the artifact shapes, the caching) does not need
to change.

## Speech recognition: parakeet-mlx versus a paid API

`transcribe` runs parakeet-mlx locally, for two reasons: it costs nothing, and
its output is verbatim — every stutter, restart and false start stays in the
transcript, because that disfluency is exactly what `detect_take_groups` and
`analyze`'s `is_failed_take` scoring use to tell a repeated attempt at a line
from a finished one. A cleaner ASR that "fixes up" speech the way dictation
software does would quietly delete the signal this pipeline depends on.

On the actual child speech in this project's own footage (spot-checked
directly against `work/03-transcript.json` for the real project, not a
synthetic benchmark), parakeet's output reads as clean and coherent — informal
contractions and kid vocabulary ("gonna", "y'all", "wanna") came through
correctly in the samples reviewed, with no obvious misrecognitions. That is an
anecdotal read of one project, not a measured word-error rate, and it is worth
noting honestly that the `Word` model carries a `confidence` field the
provider never actually populates — every word in this project's transcript
reports `confidence: 1.0`, meaning there is currently no signal anywhere in
the pipeline that would flag a *specific* word as likely wrong. A paid ASR
API's main advantage here would not be accuracy on the evidence available (this
model already did fine on this material) so much as genuine per-word
confidence you could act on, plus not depending on Apple Silicon or local GPU
memory.

**Cost, estimated.** AssemblyAI's async transcription is priced per hour of
audio (around $0.15/hour for its standard tier, more for its higher-accuracy
tier), billed per second. Transcribing this project's 28.8 minutes of primary-
camera audio comes to a few cents — well under $0.15 per video either way.
Cost is not the reason to consider this upgrade; it is close to free either
way. Switching means adding `providers/asr_assemblyai.py` behind the existing
`ASRProvider` protocol and setting `providers.asr: assemblyai`.

## B-roll: what exists today and what generating it would add

It is worth being precise about what "no b-roll" actually means here, because
something adjacent to b-roll already works well. The `analyze` stage already
finds every clip too quiet to carry a spoken phrase (`insert_max_words_per_second`,
the same test that catches a genuinely silent clip) and offers it to `plan` as
a placeable shot — described by a keyframe-reading model call so the planner
can match it to what's being said, not just to where it sits in the
recording. On this project's footage that mechanism correctly picked out
three close-up clips (a bubble-covered toy, a hand pressing bubbles, a syringe
being pushed) and cut all three into the final draft. That is your own
footage, used well; it is not generated.

What is genuinely missing is anything that does not exist in the source
footage at all: a shot the creator never filmed. Paid image-to-video
generation (Kling, Runway, Google's Veo, and similar, typically reached
through an aggregator like fal.ai) can animate a still photo of the actual toy
into a few seconds of motion — useful for an establishing shot, a beat the
brief's `must_include` calls for but nobody filmed, or a cutaway when the real
footage of a moment is unusable.

**Cost, estimated.** Per-clip pricing across these providers runs roughly
**$0.30–$1.50 for a 5-second clip**, depending on resolution and whether audio
is generated too — Kling's own published rate for a 5-second 1080p clip with
generated audio is about $0.70; video-only and lower resolution options run
cheaper. A draft that used two or three generated inserts would add
roughly $1–$4 per video. This is the upgrade with the weakest case for this
project specifically, precisely because the creator already shot strong
silent close-ups that the pipeline already knows how to use — generated b-roll
matters far more for a channel whose real footage has gaps than for one that
already has good payoff shots on hand.

## Music and sound: missing entirely

There is no music stage. `render_draft` produces exactly the cut dialogue and
whatever ambient audio was on the original clips, with fades at cut points and
an optional per-beat volume adjustment (`plan.gain_db_by_beat`) — nothing is
generated or licensed and laid under the edit. A stock library
(`assets/Music/`, already present in this repository) exists but nothing in
the pipeline reads it. Two paths from here: a licensed stock track picked and
trimmed to the edit's length (effectively free beyond what's already in the
repository, but generic and not written to fit this edit's specific beats),
or a generated score sized to the exact cut. ElevenLabs Music is one current
option for the latter, priced by generated minute — **roughly $0.30–$0.80 per
minute** by public reporting, which has been moving as pricing tiers change,
so treat this range loosely. For a roughly 3.3-minute draft, a generated score
would run **about $1–$3 per video**.

## Rendering: captions, graphics, shorts

None of these exist today, and none of them need a paid API — the data they
would need is already sitting in the pipeline's own artifacts. Word-level
timing for every spoken segment is already in `03-transcript.json` and
`05-timeline.json`; burning that in as captions is an `ffmpeg` subtitle filter
away, not a new data source. The same is true of a simple title card or lower
third. `SegmentAnalysis.shorts_candidate` is already computed by `analyze` for
every phrase and currently goes nowhere — nothing in `render_draft` reads it
or produces a second, shorts-format cut from the phrases it flags. All three
are implementation work in `videoai/stages/`, not a subscription: **$0 in
marginal cost**, and arguably the highest-value *free* work left on the table,
since delivery-quality analysis (above) is the highest-value *paid* one.

## Two paid configurations, compared

| | Current (free) | Configuration A: better analysis | Configuration B: full paid stack |
|---|---|---|---|
| analyze | Claude, text + stills | Gemini, native video | Gemini, native video |
| transcribe | parakeet-mlx, local | parakeet-mlx, local | AssemblyAI |
| b-roll | creator's own silent clips only | creator's own silent clips only | + 3–5 generated clips |
| music | none | none | one generated score |
| **Cost per video (estimated)** | **$0** | **≈ $0.50–$1.50** | **≈ $5–$10** |

Configuration A changes what the edit *contains*: the same kind of draft as
today, but built from scores that reflect what the footage actually sounds
and looks like in motion, rather than a text summary of it. That is the
change most likely to be visible in the finished video — a better choice of
best take, a shorts candidate that is actually the funniest three seconds
rather than the one with the most exclamation marks in its transcript.

Configuration B changes what the edit *is*: a video with generated cutaways
the creator never filmed and a custom score under it, on top of everything
Configuration A already improves. It costs meaningfully more per video and
adds two new kinds of failure to watch for (a generated clip that doesn't
match the real toy's colour or shape, a score that doesn't fit the cut's
timing) for a smaller marginal improvement than Configuration A alone,
because this project's real footage already supplies its own close-up payoff
shots and its own room tone.

## What I would buy first

Configuration A, and specifically just the `analyze` upgrade on its own,
before anything else on this list. It is the only upgrade that touches the
thing a quiet toy review actually lives or dies on — whether the edit found
the moments that were genuinely satisfying to watch, not just the ones that
were easy to describe in a transcript — and it costs about a dollar a video to
find out. Transcription is already close to free and, on the one project
checked directly, already accurate; there is little to buy there. Generated
b-roll and music both cost more per video than better analysis and address a
gap (missing footage, no score) that this particular project does not
strongly have, since the creator already shot usable close-ups and a stock
library already sits unused in the repository. Captions, graphics and a shorts
cut are worth building next, but as engineering, not spend — the data for all
three already exists in the pipeline's own artifacts.
