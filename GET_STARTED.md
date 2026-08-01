# Get Started with VideoAI Studio

Welcome to **VideoAI Studio** — an automated end-to-end pipeline for turning raw footage into YouTube-ready review videos with AI story planning, lower third section titles, captions, background music ducking, and custom cartoon badges.

---

## 🛠️ Scenario 1: First-Time Setup & Production (Human Developer)

### Step 1: Clone Repository & Install System Dependencies
```bash
git clone https://github.com/krukovden/video-ai-studio.git
cd video-ai-studio

# Install system dependencies via Homebrew
brew install ffmpeg uv cairo
```

### Step 2: Validate Environment & Auto-Prepare `.env` File
Run the built-in system doctor:
```bash
uv run videoai doctor --fix
```

The doctor command automatically checks:
- [x] **Operating System:** macOS Apple Silicon (`arm64`) for GPU-accelerated local ASR (`parakeet-mlx`).
- [x] **Python Version:** Python >= 3.13 (`uv` handles Python installation).
- [x] **CLI Tools:** `ffmpeg`, `uv`, `git`, and optional LLM CLIs (`claude`, `codex`).
- [x] **Rasterization Library:** `cairosvg` / `cairo`.
- [x] **Environment File:** Auto-creates `.env` from `.env.example` if missing.

### Step 3: Run Your First Video Production
```bash
uv run videoai produce projects/my-review
```
*If `projects/my-review` is empty or missing, `videoai produce` will automatically create `projects/my-review/video/` and `description/brief.md` and give clear guidance on where to copy your raw video clips (`.MOV`, `.mp4`).*

### Step 4: Interactive Storyboard & Final Delivery
```bash
# 1. Review draft video
open projects/my-review/output/draft.mp4

# 2. Open interactive web editor to adjust running order, disable shots, drag/scale/rotate badges
uv run videoai edit projects/my-review

# 3. Approve timeline and render final video
uv run videoai approve projects/my-review
uv run videoai produce projects/my-review
```

Your contract-validated final video is ready at `projects/my-review/output/final.mp4`!

---

## 🤖 Scenario 2: Starting with an AI Agent / LLM (Copilot CLI, Claude, Cursor, Codex)

When starting a session with an AI Coding Agent, provide this prompt:

```text
You are working in VideoAI Studio repository. Please follow these onboarding steps:
1. Run `videoai doctor --fix` to validate system dependencies (ffmpeg, uv, cairo) and auto-prepare .env file.
2. Check if the project folder exists and has raw video clips in <project>/video/ (or <project>/).
3. Ensure <project>/description/brief.md exists or create it.
4. Run `videoai produce <project>` to build the review draft.
5. Provide the user with the link to open output/draft.mp4 or launch `videoai edit <project>` for interactive storyboard approval.
```

---

## 🔑 Setting Up Secrets & `.env` File

Running `videoai doctor --fix` automatically generates `.env` from `.env.example`:

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

## 💡 Troubleshooting & Guidance

| Issue | Cause | Solution |
| :--- | :--- | :--- |
| `Missing system tool: ffmpeg` | `ffmpeg` is not installed | Run `brew install ffmpeg` |
| `No video files found in project` | Raw `.MOV`/`.mp4` clips are missing | Place raw videos inside `<project>/video/` or `<project>/` |
| `cairosvg missing` | Homebrew `cairo` library missing | Run `brew install cairo && uv sync` |
| `Python < 3.13` | Environment Python version too old | Run `uv venv --python 3.13 && uv sync` |

