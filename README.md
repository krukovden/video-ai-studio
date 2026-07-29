# VideoAI

Automated editing pipeline for a kid's toy-review channel: a folder of raw clips
becomes a cut draft video, with every intermediate decision stored as a readable
JSON artifact.

## Requirements

- macOS on Apple Silicon
- ffmpeg 8.x (`brew install ffmpeg`)
- Python 3.13 via uv (`brew install uv`)
- Claude Code CLI, authenticated

## Setup

```bash
uv venv --python 3.13
uv sync
cp .env.example .env   # then fill in the keys you have
```

## Project layout

A project is a folder. The pipeline adapts to how it is organized:

```
projects/my-review/
  video/               # or input/ — either name works
    cam-a/              # per-camera subfolder (optional: a flat folder of
    cam-b/               # clips is treated as a single camera called "main")
  project.yaml          # the brief: title, arc, must_include, avoid, notes...
  notes.md               # free-form notes, optional
  description/            # longer brief material, optional — .md, .txt, .docx
```

Clips may also sit directly in the project folder with no `video/`/`input/`
subfolder at all. Everything the pipeline needs to know about is either video
files or text: it never touches anything else in the project folder.

`project.yaml` is the only file worth writing by hand. See
[`docs/project-yaml-example.yaml`](docs/project-yaml-example.yaml) for a
filled-in brief, including an `arc` (the sequence of beats the creator actually
shot) that the planner follows instead of inventing its own structure, and a
`must_include` list of moments that must survive the edit.

Running the pipeline creates two more folders next to the brief:

- `work/` — every stage's artifact (JSON) plus caches (proxies, keyframes).
  Safe to delete; everything rebuilds from the source clips.
- `output/` — the final rendered files, including `draft.mp4`.

## Use

```bash
uv run videoai run projects/my-review
open projects/my-review/output/draft.mp4
```

`videoai run` only re-runs what actually changed. Editing `project.yaml`,
`notes.md` or `description/` re-runs `analyze` and `plan` (and the render that
depends on the plan) but leaves `ingest`, `quality`, `sync` and `transcribe`
alone — they don't read the brief, so their cached artifacts are reused as-is.
Editing a clip, or adding/removing one, invalidates from `ingest` onward.

Useful commands:

```bash
uv run videoai stages                              # pipeline order
uv run videoai run projects/my-review --stage plan --force   # re-run one stage
uv run videoai config                              # effective configuration
```

## Pipeline

Seven stages run in order, each reading and writing artifacts under `work/`:

| Stage | Artifact | What it does |
|---|---|---|
| ingest | `01-manifest` | Probes every clip (ffprobe), normalizes audio loudness, builds a proxy |
| quality | `02-quality` | Flags unusable footage (blur, motion) before any model sees it |
| sync | `01b-sync` | Groups clips by camera, places every camera on one shared timeline, picks the camera to transcribe |
| transcribe | `03-transcript` | Verbatim, word-level speech-to-text |
| analyze | `04-analysis` | Scores every phrase for delivery, visual interest and shorts potential |
| plan | `05-timeline` | Selects and orders phrases into sections, following the brief's arc when one is given |
| render_draft | `06-draft` | Cuts the draft video with ffmpeg, with audio fades at every cut |

See [`docs/state/STACK.md`](docs/state/STACK.md) for what runs each stage today
and [`docs/state/UPGRADES.md`](docs/state/UPGRADES.md) for paid providers that
raise quality at each stage.

## Layout

- `videoai/stages/` — one file per stage; each reads and writes artifacts only
- `videoai/providers/` — swappable implementations (local, subscription, paid)
- `videoai/logic/` — pure functions: phrases, take detection, sync, timeline, validation
- `projects/<name>/work/` — artifacts and cache; safe to delete, everything rebuilds
- `docs/state/` — what the stack is now and how to upgrade it
