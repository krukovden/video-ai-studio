# Upgrade steps

This is the one file to open when deciding what to spend time or money on next.
Every number here is anchored to the one project that has actually been run
through the pipeline: 13 clips, 28.8 minutes of combined source footage
(`assets/1.Toy_Pimple_Popping/`), transcribed into 2,171 words, edited down to a
199-second draft and a 184-second (about 3-minute) finished video. Today that
finished video costs **$0 per video** in marginal spend: transcription, the
technical quality gate, camera sync and rendering all run locally with free
tools, and the three points where a language model is involved — `analyze`,
`plan`, `visual_check` — run through the Claude Code CLI under a Claude
subscription rather than a metered API key, so they cost nothing per call
either.

Each section below is a single, independent upgrade. Do one, stop, and judge
the result before moving to the next — nothing here depends on anything else
in this document. Every config line quoted below was checked directly against
`videoai/config.py` and `config.yaml` as they stand on `feat/polish`; where a
line does not exist yet because the feature isn't built, that is said
explicitly rather than implied.

**Buy this first: Gemini for analysis (Upgrade 1).** It is the only upgrade on
this list that touches the thing a quiet toy review actually lives or dies
on — whether the edit found the moments that were genuinely satisfying to
watch — and it costs about a dollar a video to find out. Read the free
changes in the next section first, though: they cost nothing and some of them
you already own but haven't turned on.

## Do this first, because it's free

Four things already exist in the pipeline and cost nothing to use — they are
just not switched on, or not yet populated with anything specific to this
project. Try all of these before spending a cent.

| Lever | Where | Default today | What it does |
|---|---|---|---|
| Auto-fix loop | CLI flag `--auto-fix N` | off (`--auto-fix 0`) | When the visual gate rejects a shot (an adult in frame, an unusable shot), automatically re-plans without it, up to N times, instead of stopping and making you decide |
| Exclusion list | `config.yaml` → `plan.exclude_phrases: []` | empty | Permanently bans a specific phrase id from ever being offered to the planner again, once you've watched a draft and know a moment is never usable |
| Per-beat volume | `config.yaml` → `plan.gain_db_by_beat: {}` | empty | Turns one story section up or down in the mix, e.g. `{Popping: -6}` to quiet a loud section by 6 dB |
| Insert descriptions | `videoai/config.py` → `AnalyzeSettings.describe_inserts` | **already on** (`true`) | Every silent close-up clip gets a one-sentence description from its keyframes so the planner can place it by what it shows, not just by chronology |

The first three are real, unused levers. The fourth is not actually a lever
you need to flip — it's already true by default and is exactly what let the
pipeline place this project's three silent close-ups (the bubble-covered toy,
the hand pressing bubbles, the syringe) correctly in the final cut. It's
listed here so you know it exists and don't spend money re-solving a problem
that's already handled.

**How to use the exclusion list and per-beat volume, concretely.** After
watching a draft, if a specific shot must never appear again — the README
documents a real example from this project, `clip-04#002`, an adult walking
in front of the camera — open `config.yaml` and add it:

```yaml
plan:
  exclude_phrases:
    - "clip-04#002  # an adult walks in front of the camera during this shot"
  gain_db_by_beat:
    Popping: -6
```

Then re-run: `uv run videoai run assets/1.Toy_Pimple_Popping --stage plan --force`.
**How you know it worked:** the excluded phrase no longer appears in
`work/05-timeline.json`, and `uv run python inspect-result.py <project>`
no longer lists it in the cut list. **Undo:** delete the line from
`config.yaml` and re-run the same command; the phrase becomes eligible again
(unless the visual gate has also rejected it independently — check
`work/05c-rejected.json`).

**How to use `--auto-fix`, concretely.** Instead of:

```bash
uv run videoai run assets/1.Toy_Pimple_Popping
```

run:

```bash
uv run videoai run assets/1.Toy_Pimple_Popping --auto-fix 2
```

**How you know it worked:** the run that used to stop with "visual check
rejected N of M selected segments" now continues past that point on its own,
and the final rejection list — if any survives after N rounds — is still
readable at `work/05c-rejected.json`. **Undo:** just omit the flag next time;
it has no config-file state of its own.

## Upgrade 1 — Gemini for analysis (the big one)

**What's wrong today.** `analyze` never watches this footage. It reads the
packed transcript, phrase by phrase, and — because `analyze.keyframes_per_phrase`
defaults to `1` — looks at exactly one still JPEG per phrase, sampled at that
phrase's midpoint (`videoai/stages/s04_analyze.py`, `_keyframes`). That is
enough to judge what was said and, roughly, what one instant looked like. It
cannot judge delivery, timing, or motion, because a still frame has none of
those and a transcript captures none of them either. For a toy with an
obvious bang or crash, text-only scoring is a workable stand-in for what's fun
to watch. This project's toy is the opposite case: a quiet squeeze-and-pop
stress toy whose best moments are a bubble finishing a pop just right, or the
particular way a kid's voice catches when something actually works. Nothing
in a transcript says "the bubble popped satisfyingly" — the words spoken over
that moment might be identical to the words spoken over a pop that fizzled.
A judge that only ever reads text and looks at one frozen instant per phrase
is structurally blind to the exact thing this video needed to get right.

**What changes if you do it.** Gemini's video understanding reads the actual
proxy video files — real frames sampled continuously, plus the actual audio
track — rather than a transcript of them. `analyze`'s three model-written
fields (`delivery_score`, `emotion`, `shorts_candidate`) stop being a text
model's guess at how a moment sounded and become a judgment made from
watching and hearing it, the way a human editor scrubbing the footage would.
Because `plan` selects and orders the whole edit from `analyze`'s scores, a
better analysis pays off twice — once directly, and again through every
downstream decision that reads it. Concretely, on this project you'd expect
the "best pop" moments and the `shorts_candidate` picks to shift toward
whichever attempt actually looked and sounded best in motion, not whichever
transcript line had the most exclamation marks. This is a real, structural
improvement, not a marginal one — but exactly which specific cuts change is
something you'd only know by running it, since it depends on what the
current transcript-only scoring already got right by luck.

**What it costs.** Gemini's public pricing processes video at roughly 260
tokens per second of footage (frames plus audio combined). Submitting this
project's full 28.8 minutes once is about 450,000 input tokens. At Gemini 2.5
Pro's published rate ($1.25 / million input tokens, $10 / million output
tokens for prompts under 200K tokens), that's roughly $0.55 of input cost
plus $0.05–$0.20 for a several-thousand-token scored reply — call it **$0.50–
$1.50 per video, about a dollar**. The cheaper Flash tier (around $0.15 /
million input tokens) would land near **$0.10–$0.20 per video** at some cost
in judgment quality. All of these are estimates from public rate cards, not
measured invoices.

| | Per video (estimate) | Per month at 4 videos (estimate) |
|---|---|---|
| Gemini 2.5 Pro | $0.50–$1.50 | $2–$6 |
| Gemini 2.5 Flash | $0.10–$0.20 | $0.40–$0.80 |

**What you have to do.**

1. Create a Google AI Studio account at ai.google.dev, enable billing, and
   generate an API key. `.env.example` already has a line reserved for it:
   ```
   GEMINI_API_KEY=
   ```
   Copy your `.env` (`cp .env.example .env` if you haven't already) and fill
   that line in.
2. This provider doesn't exist yet — `videoai/providers/` today holds only
   `asr_parakeet.py`/`asr_mock.py` (speech) and `llm_claude_cli.py`/`llm_mock.py`
   (language model). Someone has to write `videoai/providers/llm_gemini.py`
   implementing the same `LLMProvider` protocol every model call already goes
   through (`videoai/providers/base.py`):
   ```python
   class LLMProvider(Protocol):
       name: str
       def complete_json(self, prompt: str, images: list[Path], timeout: int) -> dict: ...
   ```
   and adding one branch to `resolve_llm` in that same file, next to the
   existing `mock`/`claude_cli` branches, so `resolve_llm("gemini", ...)`
   returns it.
3. **The part worth knowing before you ask for this to be built:** just adding
   the provider isn't enough on its own to get the benefit described above.
   `s04_analyze.py` currently only ever extracts one still JPEG per phrase
   (`_keyframes`) — pointing a new provider at that same single-frame input
   would just be a different model looking at the same still picture, not
   Gemini "watching the video." To get the real benefit, `analyze`'s
   media-gathering also needs to change so the whole clip (or the audio-plus-
   frames Gemini needs) is handed to the model, not one JPEG per phrase. This
   is genuinely two pieces of work, not one: the provider file, and a change
   to what `analyze` sends it.
4. `providers.llm` is one shared setting read by three stages — `analyze`,
   `plan`, and `visual_check` all call `resolve_llm(ctx.config.providers["llm"], ...)`.
   Changing this one line switches all three to Gemini, not just `analyze`.
   That's probably fine for `visual_check` (it already only ever looks at
   images, so a different model judging the same three frames is a smaller
   change) and likely fine for `plan` (it reasons over already-scored text,
   no video involved) — but if you ever want Gemini on `analyze` only while
   keeping `plan`/`visual_check` on Claude, that needs a second provider slot
   added to `config.py`, which does not exist today.
5. Once the provider exists, the actual switch is one line in `config.yaml`:
   ```yaml
   providers:
     llm: gemini      # was: claude_cli
   ```
6. Re-run: `uv run videoai run assets/1.Toy_Pimple_Popping --stage analyze --force`.

**How you know it worked.** `work/04-analysis.json`'s `provider` field reads
`"gemini"` instead of `"claude_cli"`. Compare `delivery_score`/`shorts_candidate`
against the same fields in the version you have today (or a copy saved before
switching) — a real change in judgment should show up as different scores on
at least some phrases, not an identical file with a different provider name.

**Undo.** Set `providers.llm` back to `claude_cli` and re-run
`--stage analyze --force`. Nothing about the rest of the pipeline depends on
which provider produced `04-analysis.json` — the artifact shape is identical
either way.

## Upgrade 2 — Paid speech recognition

**Be honest about this one: there isn't much to buy.** `transcribe` runs
parakeet-mlx locally, for free, on the Apple Silicon GPU, and on this
project's real footage it already produced 2,171 words that read as clean and
coherent on direct inspection of `work/03-transcript.json` — informal kid
speech ("gonna", "y'all", "wanna") came through correctly in every sample
checked, with no obvious misrecognitions. That's an anecdotal read of one
project's transcript, not a measured word-error rate, but it's the only
evidence available and it doesn't show a problem paying money would fix.

**What is actually missing** is not accuracy, it's per-word confidence. The
`Word` model has a `confidence` field, but parakeet-mlx never populates it —
every single word in this project's real transcript reports `confidence: 1.0`.
That means there is currently no signal anywhere in the pipeline that could
flag "this specific word is probably wrong," even though the mechanism to use
such a signal doesn't exist downstream either. A paid ASR's case here is
genuine per-word confidence you could act on, plus not depending on Apple
Silicon/local GPU memory — not better transcripts on the evidence you have.

**When this would be worth paying for:** if you shoot on a device other than
a Mac with Apple Silicon, if you ever hit words that matter (a product name,
a number) getting silently misheard and want a confidence score to catch it
automatically, or if GPU memory becomes a bottleneck on longer footage. **When
it would not:** for a project like this one, where a spot check already shows
clean output and the switch's own honest selling point is a field
(`confidence`) that nothing downstream reads yet anyway.

**What it costs.** AssemblyAI's async transcription is priced per hour of
audio, around $0.15/hour on its standard tier. This project's 28.8 minutes of
primary-camera audio comes to a few cents.

| | Per video (estimate) | Per month at 4 videos (estimate) |
|---|---|---|
| AssemblyAI, standard tier | under $0.15 | under $0.60 |

**What you have to do.**

1. Create an AssemblyAI account at assemblyai.com and copy the API key from
   its dashboard. `.env.example` already reserves the line:
   ```
   ASSEMBLYAI_API_KEY=
   ```
2. Write `videoai/providers/asr_assemblyai.py` implementing the existing
   `ASRProvider` protocol (`videoai/providers/base.py`):
   ```python
   class ASRProvider(Protocol):
       name: str
       def transcribe(self, audio_path: Path) -> list[Word]: ...
   ```
   and add a branch to `resolve_asr` in that file, next to `parakeet`/`mock`.
   Note that whatever replaces parakeet must keep transcripts **verbatim** —
   `detect_take_groups` and `analyze`'s `is_failed_take` scoring both depend on
   disfluencies (restarts, stutters, false starts) surviving into the
   transcript. A "cleaned-up" dictation-style ASR would quietly delete the
   exact signal those two things need.
3. Flip the config line:
   ```yaml
   providers:
     asr: assemblyai   # was: parakeet
   ```
4. Re-run: `uv run videoai run assets/1.Toy_Pimple_Popping --stage transcribe --force`.

**How you know it worked.** `work/03-transcript.json`'s `provider` field
reads `"assemblyai"`, and — if the new provider populates it — `Word.confidence`
values are no longer uniformly `1.0`.

**Undo.** Set `providers.asr` back to `parakeet` and re-run
`--stage transcribe --force`. Everything downstream re-derives from whichever
transcript is current; there's no state to clean up.

## Upgrade 3 — Generated B-roll from a photo of the toy

**What already works, so this isn't starting from zero.** `analyze` already
finds every clip too quiet to carry spoken narration
(`insert_max_words_per_second`, default `0.5` words/second — a genuinely
silent clip always qualifies) and offers it to `plan` as a placeable shot,
described from its own keyframes so the planner can match it to what's being
said. On this project's real footage that mechanism correctly picked out
three close-up clips — the bubble-covered toy, a hand pressing bubbles, the
syringe being pushed — and all three made it into the final draft. That is
real footage, used well. It is not generated, and it is better than anything
generated would be, because it's an actual close-up of the actual toy in
actual light.

**What's genuinely missing** is a shot that was never filmed at all: an
establishing shot, a `must_include` beat nobody caught on camera, or a cutaway
for a moment whose real footage the visual gate rejected. Paid image-to-video
generation (Kling, Runway, Veo, usually reached through an aggregator like
fal.ai) can animate a still photo of the toy into a few seconds of motion for
exactly that gap. **The hard constraint that already shapes this project:**
every one of these generators refuses to animate content involving children,
so the only thing that can ever be generated here is the toy itself, alone,
never the child. That rules out generating anything with the reviewer in it —
reactions, hands, faces — which is most of what makes this video work in the
first place.

**Rank this honestly last of the paid options.** It costs more per video than
better analysis, and it's solving a problem — missing footage — that this
particular project doesn't strongly have: the creator already shot usable,
better-than-generated close-ups, and the visual gate already knows how to
place them.

**What it costs.** Per-clip pricing runs roughly $0.30–$1.50 for a 5-second
clip depending on resolution and generated audio — Kling's own published rate
for a 5-second 1080p clip with audio is about $0.70. Two or three inserts add:

| | Per video (estimate) | Per month at 4 videos (estimate) |
|---|---|---|
| 2–3 generated 5s clips | $1–$4 | $4–$16 |

**What you have to do.** There is no pipeline stage for this today — it is a
manual, outside-the-pipeline step, not a config flip:

1. Create an account with a provider (fal.ai is the simplest on-ramp; it
   fronts Kling, Runway and others behind one API/billing relationship).
2. Take or crop a clean still photo of the toy — from this project, one of
   the images already sitting in `assets/1.Toy_Pimple_Popping/description/`
   would work, or a fresh frame extracted from a proxy.
3. Generate the clip through the provider's web UI or API, download the
   result as an `.mp4`.
4. Drop the file into the project's `video/` folder like any other clip (a
   flat folder of clips is treated as one camera named `main`, so this just
   works), then re-run the pipeline from `ingest` onward so it's indexed,
   quality-scored, and offered to `plan` as a silent insert the same way the
   real close-ups already are:
   ```bash
   uv run videoai run assets/1.Toy_Pimple_Popping --force
   ```
   You may want to write a one-sentence description directly into
   `project.yaml`'s `notes` so the planner favors placing it where you intend,
   since a generated clip has no dialogue or context of its own to place it by.

**How you know it worked.** `work/01-manifest.json` lists the new clip; `uv
run python inspect-result.py <project>` shows it in the cut list if the
planner chose to use it (it may not — nothing forces a generated clip into
the edit).

**Undo.** Delete the generated file from `video/` and re-run; the clip
disappears from every downstream artifact on the next run because `ingest`
re-indexes from the folder's actual contents.

## Upgrade 4 — Music and sound

**What exists today.** `polish` already adds a music bed: one track picked
from the royalty-free library already sitting in `assets/Music/` (Bensound),
looped or trimmed to length, faded at both ends, and ducked under speech with
an ffmpeg `sidechaincompress` so it's never competing with the child talking
(`polish.music_gain_db: -22.0`, `polish.music_duck_db: -12.0` in
`config.yaml`). Track selection follows the project's `style` field in
`project.yaml` when it names a track the library has, and otherwise a stable
digest of the project's own name, so the same project always gets the same
bed on every re-render. Bensound's free licence requires a credit, and the
pipeline already appends that line to `output/metadata.md` automatically.
This is a real, working feature — the gap is that the library is one small
folder of generic tracks, none written to fit this specific edit's beats.

**Two paths from here.**

*A paid stock library* (Artlist, Epidemic Sound, and similar) gives you a much
larger, better-curated pool of tracks to point `polish.music_track` at, at a
flat monthly subscription rather than a per-video cost — typically in the
**$10–$15/month** range as a rough estimate for an individual creator tier,
independent of how many videos you make that month. This buys better taste,
not a track written for this edit's specific beats.

*A generated score*, sized to the exact cut, is the other path. ElevenLabs
Music is one current option, priced by generated minute at roughly **$0.30–
$0.80 per minute** by public reporting (this has been moving as pricing tiers
change, so treat it loosely). For this project's roughly 3-minute video, that's:

| | Per video (estimate) | Per month at 4 videos (estimate) |
|---|---|---|
| Stock library subscription | $0 marginal (flat ~$10–$15/month) | ~$10–$15/month flat |
| Generated score, ~3 min/video | $1–$3 | $4–$12 |

**What you have to do — stock library.**

1. Subscribe, download tracks licensed for your use, and drop the files into
   `assets/Music/` (or a project-specific folder).
2. Either let automatic selection pick among them, or name one explicitly:
   ```yaml
   polish:
     music_track: your-new-track.mp3
   ```
3. Re-run: `uv run videoai run assets/1.Toy_Pimple_Popping --stage polish --force`.

**What you have to do — generated score.**

1. Create an account with the generation service, describe the mood/length
   you want (matching this project's cut length, `output/final.mp4`'s
   duration), generate and download the track.
2. Same two steps as above: drop it in `assets/Music/` (or wherever
   `polish.music_dir` points), and either point `polish.music_track` at it
   directly or let the automatic picker find it.
3. Re-run `--stage polish --force`.

**How you know it worked.** `work/08-final.json`'s `music_track` field names
the new file, and `output/final.mp4`'s music bed audibly changes.
`output/metadata.md` gains a new attribution line if the new track requires
one under its licence — check the licence yourself; the pipeline's
auto-attribution logic (`videoai/logic/music.py`) is written specifically for
Bensound's credit wording and won't know a different library's requirements.

**Undo.** Clear `polish.music_track` (or delete the added files) and re-run
`--stage polish --force`; the automatic picker falls back to whatever's left
in `assets/Music/`.

## Upgrade 5 — Captions, motion graphics and vertical shorts

None of these need a paid API. This section is different from the four above:
it's a list of engineering work, not a subscription, because the data every
one of these would need already exists inside the pipeline's own artifacts.
This is here so you know what to ask for next, and roughly how big each ask
is.

**Burned-in captions.** Word-level timing for every spoken segment already
sits in `work/03-transcript.json` and `work/05-timeline.json`. Burning that in
as on-screen captions is an `ffmpeg` subtitle filter reading data that's
already there — no new data source, no new stage dependency. Of the three
items here, this is the smallest piece of work: closer in size to adding one
more filter to `polish`'s existing single ffmpeg invocation than to writing a
new stage.

**Motion graphics beyond what exists.** `polish` already produces a title card
and a lower third per story section (`polish.intro_seconds`, `polish.title_seconds`).
Anything beyond that — animated lower thirds, on-screen counters, callouts
tied to specific moments — is real ffmpeg filter-graph work with no existing
scaffolding to extend, since the current overlay system
(`_TextOverlay`/`_overlay_inputs` in `videoai/stages/s08_polish.py`) is built
specifically for the two things it does today. This is the least-bounded item
on this list — size depends entirely on what you actually want built.

**A vertical shorts cut.** `analyze` already computes `shorts_candidate` for
every phrase — it's in `work/04-analysis.json` right now — and today it goes
nowhere: `plan` only shows it to the planner as a flag in the phrase listing
(`s05_plan.py`, `_segments_view`), and nothing produces a second, shorts-format
export from the phrases it flags. Building this means a new stage that reads
`04-analysis` for the flagged phrases, `05-timeline`/`01-manifest` for the
actual footage, and crops/re-renders a vertical cut — comparable in scope to
the existing `export_edit` stage (a self-contained stage plus a pure-logic
module, in the same size range as that stage's own ~130 lines plus tests).

All three: **$0 marginal cost per video**, and arguably the highest-value free
work left on the table precisely because Upgrade 1 (better analysis) is the
highest-value *paid* one — the two are not in competition with each other.

## The full picture

| | Free (today) | Configuration A: better analysis | Configuration B: full paid stack |
|---|---|---|---|
| analyze | Claude, text + one still per phrase | Gemini, native video + audio | Gemini, native video + audio |
| transcribe | parakeet-mlx, local | parakeet-mlx, local | AssemblyAI |
| b-roll | creator's own silent close-ups only | creator's own silent close-ups only | + 2–3 generated clips of the toy |
| music | one Bensound track, ducked | one Bensound track, ducked | one generated score |
| **Cost per video (estimate)** | **$0** | **≈ $0.50–$1.50** | **≈ $5–$10** |
| **Cost per month at 4 videos (estimate)** | **$0** | **≈ $2–$6** | **≈ $20–$40** |

Configuration A changes what the edit **contains**: the same kind of draft as
today, built from scores that reflect what the footage actually sounds and
looks like in motion instead of a text summary of it. That's the change most
likely to be visible on screen — a better choice of best take, a shorts
candidate that's actually the funniest three seconds rather than the one with
the most exclamation marks in its transcript.

Configuration B changes what the edit **is**: cutaways the creator never
filmed and a custom score under everything Configuration A already improves.
It costs several times more per video and introduces two new kinds of failure
to check for by eye — a generated clip whose toy doesn't quite match the real
one's color or shape, a score that doesn't sit right against the cut's own
timing — for a smaller marginal improvement than Configuration A alone, since
this project's real footage already supplies its own close-up payoff shots
and its own room tone.

**Buy Configuration A's `analyze` upgrade first, on its own, before anything
else here.** It's the only upgrade that touches whether the edit found the
moments that were genuinely satisfying to watch, and it costs about a dollar
a video to find out. Transcription is already close to free and, on the one
project checked directly, already accurate — there's little to buy there.
Generated b-roll and a generated score both cost more per video than better
analysis and solve a gap (missing footage, no custom score) this project
doesn't strongly have. Captions, graphics and a shorts cut are worth building
next, but as engineering to ask for, not money to spend.
