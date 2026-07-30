# VideoAI Production Contract

This file is the single source of operating instructions for every coding agent
and every LLM provider working in this repository. `AGENTS.md` and `CLAUDE.md`
are symbolic links to this file. Do not duplicate these rules elsewhere.

## Definition of done

A task is not complete because code was written, a command started, or a
fallback file exists. It is complete only when the requested outcome has been
produced, verified, and handed to the creator. When a run fails or deadlocks,
diagnose and fix the cause, retain reusable artifacts, rerun the failed phase,
and continue until the contract passes or an external decision is genuinely
required.

Never label a degraded preview as `final.mp4`. A required feature that cannot be
applied is a production failure. A deliberately degraded artifact must be named
`preview-fallback.mp4` — the name `failure_policy.fallback_output_name` gives —
and its missing features must be recorded in the stage artifact. A failed
validation also removes `final.srt` and `production-report.json`, so `output/`
never describes a delivery that is not there.

## Provider-independent pipeline

Claude and Codex are interchangeable editorial providers. They may score
phrases, describe visuals, choose material, and propose story structure. They
must not control stage ordering, timestamps, rendering rules, approval, or the
definition of a valid final. Those are deterministic Python responsibilities.

Every production run follows this order:

1. Ingest original media and build disposable proxies.
2. Technical quality analysis and multi-camera synchronization.
3. Word-timed transcription.
4. Editorial analysis, story plan, and visual safety check.
5. Render a review draft from the exact current timeline.
6. Obtain creator approval bound to timeline, draft, and effective config
   (`polish.require_approval`, which the `videoai produce` workflow uses).
7. Delivery preflight: check the contract against the configured delivery height
   and the source's display aspect, the music library, the closing beat, and the
   free disk space — before a single frame is cut.
8. Build delivery media from original sources through lossless intermediates.
9. Apply intro, outro, section transitions, section titles, captions, music, and
   speech ducking using finite, independently testable render passes.
10. Decode the entire result, measure every required feature, and validate.
11. Write the YouTube-ready package and a machine-readable production report.

## Required delivery features

The machine-readable authority is `production-contract.yaml`. Every rule in that
file is measured against the file that was actually written; a rule this
repository cannot check does not belong in it. By default a finished video must
have:

- original-source 1920x1080 delivery, with the path each segment was really cut
  from recorded and checked against the manifest's disposable proxies;
- one lossy video generation at most;
- intro and outro;
- section titles, one per story-section change, named by that section's beat and
  rendered only as lower thirds in the bottom safe area;
- at least one section transition, counted as fade filters actually emitted
  rather than as boundaries the planner marked;
- word-timed captions as a viewer-controlled sidecar by default, no cue
  overlapping the next;
- background music whose bed measurably drops at least 3 dB under speech,
  measured by comparing the bed's speech-window RMS with and without the
  sidechain compressor;
- a closing story beat;
- a full successful decode (`ffmpeg -v error -f null -`) and a production report.

If any required feature is missing, stop with a clear diagnostic and do not
write or retain a file called `final.mp4`.

## Editing and verification rules

- Preserve creator changes and unrelated worktree changes.
- Keep intermediates under `work/`; they are disposable and resumable.
- Write outputs atomically whenever practical.
- Use finite media inputs and explicit durations. Never depend on an unbounded
  loop inside a large final filter graph.
- Split complex renders into phases so a failure does not repeat source cutting.
- Run focused tests while iterating and the complete test suite before handoff.
- For a real-video request, inspect representative frames and fully decode the
  final before showing it.
- Record which features were requested and actually applied.
- Editing `production-contract.yaml` re-runs delivery: the resolved file's content
  is hashed into the polish stage's fingerprint.

## Standard creator workflow

```bash
videoai produce PROJECT --config config.yaml
open PROJECT/output/draft.mp4
videoai approve PROJECT --config config.yaml
videoai produce PROJECT --config config.yaml
```

The first `produce` run stops after the review draft. The second refuses stale
approval, renders the delivery, validates it, and creates `final.mp4` only after
the production contract passes.
