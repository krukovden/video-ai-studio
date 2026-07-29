# Upgrade map

Each row is a stage whose quality can be raised by paying for a better provider.
Prices are per video for a typical project (15–20 minutes of source footage).

| Stage | Now | Paid upgrade | Price | What improves | How to switch |
|---|---|---|---|---|---|
| analyze | Claude reads a transcript + a sample of keyframes | Gemini API watches the video natively | $0.30–1.00 | The free path only reads text and looks at still frames — it cannot hear delivery or see what is actually satisfying to watch. That matters most for a quiet toy, where the best moments are a tone of voice or a small on-screen motion, not a shout. Gemini watches the real video and hears it, so best-take and shorts-candidate judgement stops relying on text alone. | Add `GEMINI_API_KEY` to `.env`, implement `providers/llm_gemini.py`, set `providers.llm: gemini` |
| transcribe | parakeet-mlx locally | AssemblyAI | ~$0.04 | Fewer errors on child speech, better punctuation | Add `ASSEMBLYAI_API_KEY` to `.env`, implement `providers/asr_assemblyai.py`, set `providers.asr: assemblyai` |
| b-roll | not implemented yet (Plan 3) | Kling / Hailuo / Veo via fal.ai | $0.30–3.00 per clip | Photoreal shots of the actual toy, animated from a photo | Plan 3 |
| music | local library (Plan 3) | ElevenLabs Music | ~$0.60 | Custom-length score that fits the edit | Plan 3 |

This table is ordered by value: `analyze` is the first upgrade worth making,
because it is the stage where the free path is weakest. `plan` reads the
scores `analyze` produces, so a better analysis pays off twice — once
directly, once through every downstream decision.

Always compare before adopting: run the stage with both providers and diff the
artifacts (`work/04-analysis.json` versus a copy) before switching permanently.
