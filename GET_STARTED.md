# VideoAI Studio

An automated AI-powered video production pipeline that transforms raw camera footage into YouTube-ready review videos.

---

## 🎬 What It Does

1. **Ingests Raw Video:** Takes unedited camera clips (`.MOV`, `.mp4`).
2. **Analyzes Footage:** Multi-camera sync, technical quality check, and GPU-accelerated local transcription.
3. **Builds Story & Timeline:** AI story planner cuts key moments, removes fluff, and arranges story sections.
4. **Adds Dynamic Animations:** Composites cartoon badges, starbursts, and reaction stickers onto key beats.
5. **Human Approval:** Interactive web UI (`videoai edit`) lets you toggle shots, drag, scale, rotate, or swap cartoon badges.
6. **Renders Final Video:** Lossless multi-pass render with lower-third section titles, captions, background music ducking, and full decode validation.

---

## 🚀 How to Run

### 1. First Time Run (Human Developer)

```bash
# 1. Check system & auto-create .env file
uv run videoai doctor --fix

# 2. Build review draft video
uv run videoai produce projects/my-review

# 3. Open interactive web editor to adjust shots & badges
uv run videoai edit projects/my-review

# 4. Approve & render final delivery video
uv run videoai approve projects/my-review
uv run videoai produce projects/my-review
```

### 2. Running with an AI Agent / LLM (Copilot CLI, Claude, Cursor, Codex)

Copy & paste this prompt to your AI agent:

```text
You are working in VideoAI Studio repository. Follow these steps:
1. Run `videoai doctor --fix` to validate dependencies (ffmpeg, uv, cairo) and prepare .env file.
2. Check if raw video clips exist in <project>/video/ (or <project>/).
3. Ensure <project>/description/brief.md exists or create it.
4. Run `videoai produce <project>` to build the review draft.
5. Guide the user to open output/draft.mp4 or launch `videoai edit <project>` for interactive storyboard approval.
```

---

## 💻 Machine Requirements

* **Operating System:** macOS on Apple Silicon (M1/M2/M3/M4) — local ASR runs on Apple GPU/Neural Engine via MLX.
* **Python:** Python 3.13 (`uv` manages Python versions automatically).
* **System CLI Tools:**
  * `ffmpeg` 8.x — video decoding, cutting, audio ducking, and rendering.
  * `uv` — Python package manager.
  * `cairo` — rasterizes vector SVG badges with transparency.
* **LLM Subscriptions / API Keys:**
  * Subscription CLI (Default): Claude Code CLI (`claude`) or Codex CLI (`codex`).
  * Optional API Key: `GEMINI_API_KEY` in `.env` for Gemini multimodal video analysis.

---

## 🩺 Getting Started & Auto-Fix (`videoai doctor`)

Run the health check command to validate system configuration and auto-prepare `.env`:

```bash
uv run videoai doctor --fix
```

### What `videoai doctor` checks & fixes:
* **System Tools:** Checks if `ffmpeg`, `uv`, `git`, and `cairo` are installed.
* **Environment File:** Automatically copies `.env.example` -> `.env` if missing.
* **Fix Guidance:** If `ffmpeg` or `cairo` is missing, it provides the exact Homebrew command:
  ```bash
  brew install ffmpeg uv cairo
  ```
* **Project Preflight:** Running `videoai produce <folder>` auto-creates missing `video/` and `description/brief.md` folders and tells you where to place raw video clips.

---

## ⚙️ Pipeline Phases & Tools Used

| Phase | Stage ID | Description | Tools & Tech |
| :--- | :--- | :--- | :--- |
| **1. Ingest & Sync** | `01-manifest`, `01b-sync` | Scans raw video clips, builds proxies, syncs multi-camera audio. | `ffmpeg`, OpenCV |
| **2. Quality Check** | `02-quality` | Detects dark frames, out-of-focus shots, and severe audio clipping. | `ffmpeg`, OpenCV |
| **3. Transcribe** | `03-transcript` | Word-timed speech recognition directly from audio. | `parakeet-mlx` (GPU local ASR) |
| **4. Story Analysis** | `04-analysis`, `05a-storyplan` | Extracts key quotes, identifies toy beats, builds story structure. | Claude CLI / Codex CLI / Gemini |
| **5. Cut & Plan** | `05-timeline`, `05b-visual` | Cuts segments into a tight timeline and verifies visual safety. | OpenTimelineIO, OpenCV |
| **6. Cartoon Effects** | `05d-effects` | Selects reaction badges (sparkles, starbursts, speech bubbles). | Cartoon Sprite Library |
| **7. Draft Render** | `06-draft` | Renders fast review draft video (`draft.mp4`). | `ffmpeg` |
| **8. Web Approval** | `edit` / `approve` | Interactive browser UI to reorder shots, toggle clips, drag/scale badges. | Local Python Server + Web UI |
| **9. Final Polish** | `08-final` | Lossless render: lower-thirds, captions, music ducking, decode check. | `ffmpeg`, `cairosvg`, PyYAML |
