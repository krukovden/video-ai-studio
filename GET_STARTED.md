# Get Started with VideoAI Studio

Welcome to **VideoAI Studio** — an automated end-to-end pipeline for turning raw footage into YouTube-ready review videos with AI story planning, lower third section titles, captions, background music ducking, and custom cartoon badges.

---

## ⚡ Quick Start Checklist (For Users & AI Agents)

When downloading or setting up this project for the first time, follow these steps or instruct your Coding Agent / Copilot CLI to run them:

### Step 1: Install System Tools (Homebrew)

```bash
brew install ffmpeg uv cairo
```

### Step 2: Set Up Python Virtual Environment

```bash
# Create Python 3.13 virtual environment
uv venv --python 3.13

# Activate virtual environment
source .venv/bin/activate

# Sync all dependencies
uv sync
```

---

## 🩺 Automated Environment Health Check

To verify that your machine is configured correctly, run the built-in system doctor:

```bash
videoai doctor --fix
```

The doctor command validates:
- [x] **Operating System:** macOS Apple Silicon (arm64) for GPU-accelerated local ASR (`parakeet-mlx`).
- [x] **Python Version:** Python >= 3.13.
- [x] **CLI Tools:** `ffmpeg`, `uv`, `git`, and optional LLM CLIs (`claude`, `codex`).
- [x] **Rasterization Library:** `cairosvg` / `cairo`.
- [x] **Environment File:** Auto-creates `.env` from `.env.example` if missing.

---

## 🔑 Setting Up Secrets & `.env` File

Running `videoai doctor --fix` automatically generates a `.env` file from `.env.example`:

```bash
cp .env.example .env
```

### `.env` File Template:

```env
# Optional: OpenAI API Key (if using direct OpenAI API)
OPENAI_API_KEY=

# Optional: Anthropic API Key (if using direct Anthropic API)
ANTHROPIC_API_KEY=

# Optional: Google Gemini API Key (if using Gemini multimodal video analysis)
GEMINI_API_KEY=your_gemini_api_key_here
```

*Note: By default, VideoAI uses subscription-backed CLIs (`claude_cli` or `codex_cli`), so an API key is not required for standard editorial planning!*

---

## 🎬 Running Your First Video Production

### 1. Generate Review Draft

```bash
videoai produce <path/to/project_folder> --config config.yaml
```

### 2. Interactive Storyboard & Badge Editing

To review video shots, toggle clips on/off, drag badges, scale/rotate badges, or swap cartoon sprites:

```bash
videoai edit <path/to/project_folder> --config config.yaml
```

Click **Save** on the interactive web page. Your storyboard decisions will be saved directly to `description/Animation Details/animation_details.json`.

### 3. Render Final Contract-Validated Video

```bash
videoai produce <path/to/project_folder> --config config.yaml
```

The completed delivery video will be saved at `<path/to/project_folder>/output/final.mp4`.

---

## 🧪 Verification & Testing

Verify that all unit tests pass on your machine:

```bash
pytest
```
