# Current stack

| Stage | Artifact | Provider now | Cost per video |
|---|---|---|---|
| ingest | 01-manifest | ffmpeg (local) | $0 |
| quality | 02-quality | OpenCV (local) | $0 |
| sync | 01b-sync | numpy cross-correlation (local) | $0 |
| transcribe | 03-transcript | parakeet-mlx (local) | $0 |
| analyze | 04-analysis | Claude Code CLI (Max subscription) | $0 |
| plan | 05-timeline | Claude Code CLI (Max subscription) | $0 |
| render_draft | 06-draft | ffmpeg (local) | $0 |

Seven stages, run in this order. `sync` groups clips by camera and places them
on one shared timeline using each clip's `creation_time` metadata, refined by
audio cross-correlation between cameras; it also names the camera whose audio
gets transcribed (the one with the real microphone). It runs entirely locally
and costs $0.

Switching a provider: edit `config.yaml`, re-run `videoai run <project>`. Only the
affected stage and everything downstream of it re-runs; earlier artifacts are reused.

Editing the creator's brief (`project.yaml`, `notes.md`, `description/`) only
re-runs `analyze` and `plan` (and `render_draft`, which depends on `plan`'s
output) — `ingest`, `quality`, `sync` and `transcribe` don't read the brief, so
their cached artifacts are reused untouched.

Mock providers (`asr: mock`, `llm: mock`) run the whole pipeline offline and are what
the test suite uses.
