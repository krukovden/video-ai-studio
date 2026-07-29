# Handing the cut draft to a human editor

Written July 2026. This is a research and recommendation document, not a
description of shipped behaviour: nothing described under "what to add" exists
in the pipeline yet. Every pricing, version, and feature claim below has a
source link; where none exists in the public record and the statement rests on
reading this project's own code or on a judgement call, it is marked
"estimate" or "judgement", not "verified".

## 1. What the pipeline already has to hand over

Read directly from `videoai/core/models.py`, `videoai/logic/timeline.py`,
`videoai/stages/s06_render_draft.py`, and the real artifact at
`assets/1.Toy_Pimple_Popping/work/05-timeline.json`.

The central artifact is `Timeline` (`videoai/core/models.py:226`): a flat list
of `TimelineClip` records, each with

- `src` — a `clip_id` (e.g. `clip-04`), resolved against `Manifest` to a real
  file path (`videoai/core/models.py:7`, e.g.
  `assets/1.Toy_Pimple_Popping/video/IMG_5665.MOV`),
- `offset` / `dur` — the in-point and duration in the **source** clip's own
  timebase,
- `start` — the clip's position on the assembled timeline,
- `beat` — the story section it belongs to (e.g. "The Reveal", "Filling It
  Up"), taken from `StoryPlan.sections[].name`,
- `quote` — the verbatim words spoken in that segment (empty for inserts),
- `reason` — a one-line human-readable justification the planner wrote,
- `core_dur` — the unpadded, "true" speech length inside `dur` (the rest is
  padding into silence),
- `gain_db` — a per-beat audio gain already decided by the plan stage,
- `is_insert` — true for a silent B-roll cutaway with no speech.

`Manifest` (`videoai/core/models.py:25`) supplies, per source clip: `path` (the
original 4K file), `proxy_path` (a rendered 540p H.264 proxy — confirmed by
`ffprobe` on this project's cache: original is HEVC 4K/2160p 30fps, `yuv420p`,
AAC 48 kHz stereo; the proxy is H.264 960×540), `duration`, `width`/`height`,
`fps`, and `has_audio`. `Transcript` (`videoai/core/models.py:72`) gives
word-level timing (`Word.start`/`end`, confidence) per clip, which is what
`build_timeline()` uses to pad cuts into silence without ever cutting inside a
word (`videoai/logic/timeline.py:20-33`).

`render_draft` (`videoai/stages/s06_render_draft.py`) does not use any of this
structurally: it re-encodes every segment from the *proxy* file to H.264/AAC
mono 44.1 kHz, applies short audio fades and any `gain_db`, and concatenates
with `-c copy` into `output/draft.mp4` (720p, per `config.yaml`
`render.draft_height`). The draft is a disposable review file, not a source of
truth — the editable truth is `Timeline` + `Manifest` + `Transcript`, all
plain JSON, all already keyed by real file paths and timecodes. That is what
any handoff format should be built from, not from `draft.mp4`.

Two things the models track that most interchange formats have no field for:
`beat` (story structure) and `gain_db` (a per-beat mixing decision). Both are
addressed in section 5.

## 2. Is free DaVinci Resolve actually sufficient?

Yes, with high confidence, for everything the creator described: nudging a
cut, moving an insert, repositioning music, fixing a title, adjusting audio
levels.

**Resolution/format ceiling.** DaVinci Resolve (free) as of version 21
(current as of June–July 2026) "includes the same high-quality processing as
Studio" and can ingest unlimited-resolution media, but "limits project
mastering and output to Ultra HD resolution or lower" and to a single
processing GPU on Windows/Linux — [Blackmagic Design's own pricing
page](https://www.blackmagicdesign.com/products/davinciresolve/price). This
project's source is exactly Ultra HD (3840×2160, HEVC, 8-bit 4:2:0), so it
sits precisely at the free ceiling, not above it. The single-GPU limit is
irrelevant on a Mac, which has one GPU per chip regardless. There is no frame
rate or bit-depth surprise here: 30 fps, 8-bit, no HDR.

**What free Resolve actually gates (Studio-exclusive, per the same official
comparison page and [Blackmagic's compare
page](https://www.blackmagicdesign.com/products/davinciresolve/compare)):**
temporal/AI spatial noise reduction, motion blur, the camera tracker, voice
isolation, surround/immersive audio mixing, most Neural Engine AI tools
(Magic Mask, some auto-tracking), additional Resolve FX/Fairlight FX beyond a
baseline set, film grain, optical blur, advanced HDR grading tools, and
multi-GPU/remote rendering. None of these are needed to trim a cut, slide an
insert, retime a beat, retitle a card, or ride a fader. Basic titling (the
Text+ / Fusion titling tools), Fairlight audio (levels, fades, mixing), and
the Edit/Cut page trim tools are all in the free build.

**Import formats in free Resolve, verified:**

- **OTIO** — supported natively in Resolve since 18.5, with no Studio gate
  reported: import is via File → Import Timeline, both `.otio` (references
  external media) and `.otioz` (bundles media). Source quality here is mixed
  (mirrors of the official manual plus community write-ups agree on this;
  no single canonical Blackmagic page was found stating "OTIO import is
  free-tier" in so many words, so treat this specific "not Studio-gated"
  claim as high-confidence but not a verbatim-quoted certainty).
  [Resolve manual mirror on OTIO
  import](https://www.steakunderwater.com/VFXPedia/__man/Resolve18-6/DaVinciResolve18_Manual_files/part1411.htm).
- **FCPXML** — imports via File → Import Timeline in the Edit page in the
  free build; Resolve has historically lagged Apple's schema versions (one
  report: Resolve 18 added support up to FCPXML 1.10, while Final Cut Pro
  itself is at **1.14** as of its January 2026 release —
  [Apple's Final Cut Pro release
  notes](https://support.apple.com/en-us/102825)). Practical effect: export
  the oldest FCPXML version your writer supports, since older is safer for
  Resolve compatibility.
- **CMX3600 EDL** — imports in the free build via the same Import Timeline
  dialog; this is the most conservative, least-likely-to-fail path since EDL
  is a small, old, extremely well-supported format, at the cost of carrying
  almost no metadata (see the format table below).
- **AAF** — this is the one genuinely unclear item. Multiple sources describe
  AAF import/export on Resolve's Fairlight page as present in the free build
  for audio round-tripping to Pro Tools, but the search results were not
  conclusive on whether the Edit-page AAF *video* timeline import has ever
  been Studio-gated, and there are scattered forum reports of AAF import
  regressions in specific point releases unrelated to licensing tier. AAF is
  not the recommended interchange format here regardless (see section 3), so
  this ambiguity does not affect the recommendation, but it should not be
  asserted as definitely-free without hands-on verification.

**Conclusion:** the creator's belief that they need paid Resolve is very
likely wrong for this use case. Free Resolve is recommended as the primary
editor (section 4). For reference, DaVinci Resolve Studio is a one-time
$295 purchase, no subscription — [Blackmagic Design's price
page](https://www.blackmagicdesign.com/products/davinciresolve/price) — so
even if a specific Studio-only feature turns out to be wanted later (e.g.
noise reduction on a noisy phone-mic take), it is a bounded, one-time cost,
not evidence the free tier is unusable today.

## 3. What should the pipeline emit, and what does it cost to produce?

| | OpenTimelineIO (OTIO) | CMX3600 EDL | FCPXML |
|---|---|---|---|
| What it is | Open, JSON-based timeline interchange from the Academy Software Foundation | Decades-old plain-text edit list from CMX systems | Apple's XML timeline format, current schema version 1.14 |
| Per-clip in/out points | Yes, frame-accurate, native concept (`SourceRange`) | Yes, but reel/tape-name based; awkward with file-based sources, one line per event | Yes, native, with rational-time precision |
| Multiple tracks | Yes, first-class (`Stack` of `Track`, video and audio) | No — one video + limited audio channels, no nesting | Yes, multiple video/audio lanes, including nested "compound clips" |
| Story-beat labels / notes | Yes — `Marker` objects with name, color, and an arbitrary metadata dict; also generic `metadata` dicts on any object | No native concept; only comment lines some tools half-support | Yes — `<marker>` elements, plus `note` attributes on some elements |
| Captions/music as separate tracks (for later plans) | Yes, cleanly — a subtitle/caption track and a music track are just more `Track` objects in the same `Stack` | No — EDL has no reliable way to carry a caption or music track distinct from picture | Yes — separate lanes; Final Cut/Resolve both understand FCPXML `<caption>` and audio lanes |
| Accepted by editors evaluated here | Resolve (native, free), Kdenlive (native, rewritten in C++, current) | Resolve (free), Shotcut (export only, no import found), OpenShot (import) | Resolve (free), OpenShot (legacy FCP7 XML variant); no native Kdenlive/Shotcut import found |

Sources: OTIO adapter and license facts from the [OpenTimelineIO GitHub
repository](https://github.com/AcademySoftwareFoundation/OpenTimelineIO) and
its [cmx3600
adapter](https://github.com/OpenTimelineIO/otio-cmx3600-adapter); Resolve OTIO
support from the manual mirror cited above; Kdenlive's OTIO rewrite from
[Kdenlive's own "State of Kdenlive 2026"
post](https://kdenlive.org/news/2026/state-2026/) ("rewrote our
OpenTimelineIO import and export function using the C++ library"); Shotcut's
format support from its [official features
page](https://mltframework.github.io/shotcut_web/features/) (lists EDL
*export* only, no AAF/FCPXML/OTIO); OpenShot's EDL/XML(FCP7) support from its
[official user
guide](https://www.openshot.org/static/files/user-guide/import_export.html);
FCPXML current version 1.14 from [Apple's Final Cut Pro release
notes](https://support.apple.com/en-us/102825).

**Recommendation on format: OTIO as primary, EDL as a safety-net fallback.**
OTIO is the only one of the three that simultaneously (a) survives in both
leading free candidate editors, (b) carries multi-track structure cleanly for
the future caption/music/graphics tracks, and (c) has a first-party,
Apache-2.0-licensed, actively maintained Python library that already models
almost exactly what `Timeline`/`TimelineClip` need. EDL is kept as a fallback
specifically because it needs no library beyond OTIO's own built-in cmx_3600
adapter (or a plain-text writer of a few dozen lines) and is the format least
likely to ever fail to import somewhere — its cost is that beat labels,
gain, and future caption/music tracks cannot ride along in it at all.

**What producing OTIO costs.** The `opentimelineio` PyPI package (Apache
License 2.0 — verified from the [repository's
LICENSE.txt](https://raw.githubusercontent.com/AcademySoftwareFoundation/OpenTimelineIO/main/LICENSE.txt))
gives a Python object model (`Timeline`, `Track`, `Clip`, `ExternalReference`,
`Marker`, `RationalTime`/`TimeRange`) that maps almost one-to-one onto this
project's own `Timeline`/`TimelineClip`: one `Clip` per `TimelineClip`, a
`SourceRange` built from `offset`/`dur` at the clip's native fps, a `Marker`
per clip carrying `beat`, `quote`, `reason` as metadata, and an
`ExternalReference` pointing at the manifest's `path`. Writing an `.otio` file
is `otio.adapters.write_to_file(timeline, path)` — no new file format to
design by hand. The built-in `cmx_3600` adapter (also Apache-2.0, part of the
core package) can emit the EDL fallback from the same in-memory `Timeline`
object essentially for free once the OTIO object is built. A dedicated
FCPXML writer (`otio-fcpx-xml-adapter`, Apache-2.0) exists too, but its last
release was 2023 and it is explicitly "contrib" (community, not core)
quality; a `steele-fcpxml` package surfaced in this research but is GPLv3,
alpha-quality, and only weeks old as of this writing — not recommended as a
dependency in a project without a copyleft license itself. FCPXML is
therefore not proposed as a stage output now; OTIO/EDL cover the near-term
need and both target editors already speak OTIO natively.

## 4. Editors, ranked

| Editor | Cost | Imports | 4K HEVC iPhone + proxies on Apple Silicon | Stability | Learning curve for a technical non-editor |
|---|---|---|---|---|---|
| **DaVinci Resolve (free)** | $0 ([official price page](https://www.blackmagicdesign.com/products/davinciresolve/price)) | OTIO (native), FCPXML, EDL, and likely AAF (see caveat above) | Native Apple Silicon build; handles 4K HEVC directly and can also generate its own "Optimized Media"/proxies internally | Mature, professional-grade, widely deployed; occasional point-release regressions reported but generally stable | Moderate — a full color/audio/VFX suite with a lot of surface area, but the Cut/Edit page alone (trim, retime, titles, basic audio) is learnable in a session |
| **Kdenlive** (current: v26.07, per [KDE's release tags](https://kdenlive.org)) | $0, GPLv3 | Native OTIO import/export (rewritten in C++ per the project's own 2026 state-of-project post); no confirmed native AAF/FCPXML | Native Apple Silicon builds since v25.12 ([Kdenlive installation docs](https://docs.kdenlive.org/en/getting_started/installation.html)); built on MLT/ffmpeg so it decodes HEVC, and it explicitly supports building proxy clips for smoother 4K editing — estimate/moderate confidence on Apple Silicon-specific hardware-decode performance, no authoritative benchmark found for this exact case | Historically less polished than Resolve/Premiere but has visibly matured; still a smaller QA budget than a commercial NLE | Simple, focused NLE UI — arguably gentler than Resolve for someone who just wants to trim and place clips |
| **Shotcut** | $0, GPLv3 | MLT XML native; EDL **export only** — no EDL/AAF/FCPXML/OTIO *import* found in its own [features page](https://mltframework.github.io/shotcut_web/features/) | Universal macOS binary (Intel + Apple Silicon), FFmpeg-based decode; multiple secondary sources describe it as not fully using Metal/GPU acceleration on Mac and sluggish on 4K/complex timelines — moderate confidence, sourcing quality here was mixed | Generally usable, less battle-tested than Resolve | Simple, timeline-first UI, easy to pick up |
| **OpenShot** | $0, GPLv3 | Imports EDL and legacy FCP7 XML per its [own user guide](https://www.openshot.org/static/files/user-guide/import_export.html); an open GitHub issue (#3163) shows FCPXML (modern) import is requested, not present | Multiple independent sources describe OpenShot as slow and prone to choppy playback on 4K on Mac even with proxies — moderate confidence | Least stable of this group in community reporting (crashes on longer/heavier projects are a recurring complaint in forums) — moderate confidence | Easiest UI of the group, but the performance and stability issues make it a poor fit for a real 4K project |
| **Blender VSE** | $0, GPL | No native FCPXML import (a third-party add-on, `tin2tin/fcpxml_import`, does it); EDL support is import-light/export-oriented per Blender's own tools ([`tin2tin/Export_EDL`](https://github.com/tin2tin/Export_EDL)); no OTIO import found | Runs natively on Apple Silicon, has a proxy system in its Video Sequence Editor, but it is a 3D application with an editor bolted on, not an NLE | Stable as a Blender feature, but the VSE is a secondary citizen inside a much larger, unrelated application | Steep — the whole surrounding application is a 3D content-creation tool; not a reasonable ask for someone who wants to "just open the cut and nudge it" |

Ranking and why: **DaVinci Resolve (free) is the clear first choice** — it is
the only one of the five with native, current-version support for the exact
format this pipeline should emit (OTIO), it is the only one confirmed to also
accept FCPXML and EDL as fallbacks, it is Apple Silicon native, and its
learning curve is bounded by only touching the Edit/Cut and Fairlight pages.
**Kdenlive is the practical fallback** if the creator ever wants to avoid
Blackmagic's ecosystem entirely: it now speaks OTIO natively too, is
Apple-Silicon native, and is simpler to learn than Resolve, at the cost of a
smaller professional user base and no confirmed FCPXML/EDL/AAF import.
Shotcut, OpenShot and Blender VSE are not recommended as the primary path:
Shotcut and OpenShot were not confirmed to import OTIO at all (the whole point
of this design), and OpenShot in particular has repeated, if not fully
authoritative, reports of instability and poor 4K performance; Blender VSE
is a capable editor but asks the creator to learn Blender to change a cut.

## 5. Concrete recommendation

**Primary path:** free DaVinci Resolve 21, fed an `.otio` file.
**Fallback path:** Kdenlive (also OTIO), or a plain `.edl` for any editor
that chokes on the OTIO file.

**What the pipeline should add.** A new stage, e.g. `export_edit`
(`videoai/stages/s07_export_edit.py`), producing a new artifact — call it
`07-handoff` — with `requires=("01-manifest", "05-timeline")`. It does not
need `06-draft`: it reads exactly the same two artifacts `render_draft`
already reads, so it can run in parallel with (or instead of) rendering the
MP4, and adding it does not disturb the existing eight-stage pipeline or its
caching. Its body would:

1. Build an in-memory `opentimelineio.schema.Timeline` from `Timeline` +
   `Manifest`: one video `Track` (plus a parallel audio `Track` if picture and
   sound are ever split later), one `Clip` per `TimelineClip`, each `Clip`'s
   `media_reference` an `ExternalReference` to the manifest's original file
   `path` (not `proxy_path` — see media-reference discussion below), and its
   `source_range` built from `offset`/`dur` converted to `RationalTime` at the
   clip's own `fps` from `Manifest`, not the timeline's nominal `fps` (they
   currently agree at 30 fps in this project, but the code should not assume
   that).
2. Attach a `Marker` to each `Clip` carrying `beat`, `quote`, and `reason` as
   marker metadata/comment text, so they show up as visible, readable markers
   once imported.
3. Write two files into `output/`: `edit.otio` (via
   `otio.adapters.write_to_file`) and `edit.edl` (via the same call with the
   built-in `cmx_3600` adapter) as the fallback.
4. Emit `DraftResult`-style artifact recording both output paths and a count,
   mirroring how `render_draft` already reports its own output.

**Size estimate (planner-usable):** this is comparable in scope to the
existing `render_draft` stage plus `logic/timeline.py` combined — those are
128 and 123 lines respectively — plus a pure-logic module (e.g.
`logic/interchange.py`) doing the `Timeline` → `otio.Timeline` conversion, so
that the mapping logic is unit-testable without ffmpeg or real media, the
same way `logic/timeline.py` is tested today. `render_draft`'s own test file
is 349 lines and `timeline.py`'s is 410; a new stage plus converter should
land in a similar band — call it one medium-sized stage, a new dependency
(`opentimelineio`, Apache-2.0), and a test file of comparable size. This is
an estimate, not a measured figure — it depends on how much time goes into
getting marker/metadata formatting right for Resolve specifically.

**What could go wrong (concrete risks for the planner):**

- Frame-rate mismatches between the timeline's nominal fps and an individual
  clip's actual fps would silently misplace cuts by fractions of a frame if
  the wrong rate is used for the conversion — this is exactly the kind of bug
  `logic/timeline.py`'s existing tests already guard against for the render
  path, and the new converter needs the same discipline.
- OTIO's marker/metadata conventions are not identical across every importer;
  Resolve is expected to show marker names and comments, but the exact
  mapping (color, note text length, whether `reason` and `quote` should be
  one marker or two) is a judgement call that should be checked by hand
  against a real Resolve import before being called done.
- A source clip with `has_audio: false` (there are some in this project's
  manifest, e.g. silent inserts) should simply have a video-only `Clip` in
  OTIO — no need to reproduce `render_draft`'s synthesized-silence trick,
  which exists only to satisfy ffmpeg's concat demuxer, not because the
  interchange format needs it.

**How the handoff should reference media.** Point at the **original 4K
files**, not the 540p analysis proxies. The proxies exist so earlier stages
(quality, sync, transcribe, analyze) run fast; they are not good enough
picture quality to finish a video in, and Resolve/Kdenlive both have their own
built-in optimized-media/proxy systems the creator can turn on locally if
playback of the real 4K HEVC is too slow on their machine — handing off a
540p proxy as the "real" media would silently cap the final output's quality
at 540p unless the creator remembers to manually relink every clip back to
the original, which defeats the purpose of a smooth handoff.

Paths should be written as **absolute** paths (resolved from
`ctx.project_dir` at export time), not the relative, run-cwd-dependent paths
currently stored in the manifest (e.g.
`assets/1.Toy_Pimple_Popping/video/IMG_5665.MOV`, which is only valid from
whatever directory the pipeline happened to be run from). Every interchange
format here (OTIO, EDL, FCPXML) ultimately resolves media by a file path or
URL, and none of the three candidate editors are known to do project-relative
path resolution in a way this pipeline could rely on.

**What happens when the creator moves the project folder:** the media links
break, exactly as they would with any NLE project. This is not solved by
format choice — it is inherent to referencing files by path rather than
copying them into a project bundle. The mitigation is procedural, not
technical: keep the project folder in place for the duration of the manual
touch-up pass, and if it must move, use the target editor's own relink
workflow (Resolve: right-click → "Relink Selected Clips" and point it at the
new folder, matching by filename; Kdenlive has an equivalent "Fix clip path"
dialog). `.otioz` (the OTIO variant that bundles media rather than
referencing it) was considered and rejected for this project's typical
15–20 minutes of 4K footage — bundling would multiply the handoff file's size
by the size of every referenced source clip, defeating the purpose of a
lightweight interchange file, for a problem (folder gets moved) that is rare
and has a normal manual fix.

## 6. What will NOT survive the handoff — will have to be redone by hand

- **`gain_db` (per-beat audio gain).** The plan stage already decides a
  loudness adjustment per story beat (`config.yaml`'s
  `plan.gain_db_by_beat`), and `render_draft` applies it as an ffmpeg
  `volume` filter. No interchange format considered here has a standard,
  cross-editor concept of "apply this many dB to this clip" that Resolve or
  Kdenlive would import as an actual fader move — at best it can ride along
  as marker text ("gain: -6dB") for the creator to read and apply by hand on
  the Fairlight/audio mixer. This is a real gap, not a formatting detail.
- **The audio fades `render_draft` applies at every cut.** These exist to
  avoid clicks in the disposable draft render; they are not part of
  `Timeline` itself and are not carried by any handoff file. The editor will
  need its own crossfades/fades if the creator wants clean audio at each cut
  in the version they finish by hand — most NLEs make this a one-keystroke
  operation, but it is not automatic.
- **`core_dur` (the unpadded "true" speech span inside each padded cut).**
  This is internal reasoning data (it is what let the planner distinguish
  "the actual words" from "the silence padding around them"); no interchange
  format has a field for "this sub-range of the clip is the important part."
  It can be exposed as marker text at best.
- **The visual gate's rejections (`work/05c-rejected.json`) and quality
  scores (`work/02-quality.json`).** These already shaped which segments made
  it into `Timeline` — they are not separately carried forward, and do not
  need to be: by the time a segment is in `Timeline` it already passed those
  gates. But if the creator swaps in a different take from the same
  `take_group` by hand in the editor, none of that scoring travels with the
  substitution — it is a fresh decision.
- **Everything from the "later plans" (burned-in captions, motion graphics,
  music, vertical shorts) — because none of it is implemented yet.** This is
  worth stating plainly so the creator is not surprised twice: today's
  handoff is picture-and-sync-audio only. When those features land, the
  recommendation in this document (route them onto separate OTIO tracks
  rather than baking them into the picture) is what would let them still be
  hand-editable later; if they are instead implemented by burning captions
  or graphics directly into the rendered frame (the common shortcut), that
  choice would make them permanently non-editable in a downstream NLE no
  matter what interchange format is used, and this document's premise (a
  human can still nudge things afterward) would quietly stop being true for
  that layer.
- **Color and exposure consistency across clips/cameras.** The pipeline does
  no color matching; every clip arrives in the editor exactly as shot. This
  was true before any handoff format existed and remains true after — worth
  naming so it is not mistaken for something the export step "should have"
  carried.
- **Transitions.** The current pipeline only produces hard cuts (confirmed by
  reading every clip in `05-timeline.json` and `build_timeline()` — there is
  no field for a transition type or duration between clips). If the creator
  adds a cross-dissolve in the editor, re-running the pipeline and
  re-exporting would not know that decision existed and would overwrite it
  with a fresh hard-cut timeline — this is a general "the pipeline output is
  regenerated, not merged" property, not specific to any one format, but it
  is worth the creator knowing before they invest editing time downstream of
  a file that a future pipeline run could silently replace.

## Sources

- [Blackmagic Design — DaVinci Resolve pricing and version
  page](https://www.blackmagicdesign.com/products/davinciresolve/price)
- [Blackmagic Design — DaVinci Resolve free vs Studio
  comparison](https://www.blackmagicdesign.com/products/davinciresolve/compare)
- [DaVinci Resolve manual mirror — importing OTIO project
  files](https://www.steakunderwater.com/VFXPedia/__man/Resolve18-6/DaVinciResolve18_Manual_files/part1411.htm)
- [Apple — Final Cut Pro release notes (FCPXML 1.14, January
  2026)](https://support.apple.com/en-us/102825)
- [OpenTimelineIO — GitHub
  repository](https://github.com/AcademySoftwareFoundation/OpenTimelineIO)
  and its
  [LICENSE.txt](https://raw.githubusercontent.com/AcademySoftwareFoundation/OpenTimelineIO/main/LICENSE.txt)
  (Apache License 2.0)
- [otio-cmx3600-adapter — GitHub
  repository](https://github.com/OpenTimelineIO/otio-cmx3600-adapter)
  (Apache-2.0, EDL read/write)
- [otio-fcpx-xml-adapter — GitHub
  repository](https://github.com/OpenTimelineIO/otio-fcpx-xml-adapter)
  (Apache-2.0, confirmed via GitHub's license API)
- [pycmx — CMX3600 EDL parser, MIT
  license](https://github.com/iluvcapra/pycmx)
- [steele-fcpxml on PyPI](https://pypi.org/project/steele-fcpxml/) (GPLv3,
  alpha, released June 2026 — not recommended as a dependency)
- [Kdenlive — "State of Kdenlive 2026"](https://kdenlive.org/news/2026/state-2026/)
- [Kdenlive — installation docs (Apple Silicon
  support)](https://docs.kdenlive.org/en/getting_started/installation.html)
- [Shotcut — official features
  page](https://mltframework.github.io/shotcut_web/features/)
- [OpenShot — official import/export
  documentation](https://www.openshot.org/static/files/user-guide/import_export.html)
- [OpenShot — open GitHub issue requesting modern FCPXML
  support](https://github.com/OpenShot/openshot-qt/issues/3163)
- [Blender add-on for FCPXML import into the
  VSE](https://github.com/tin2tin/fcpxml_import)
- [Blender EDL export tool for the
  VSE](https://github.com/tin2tin/Export_EDL)
