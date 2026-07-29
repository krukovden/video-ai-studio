"""Pipeline configuration: provider selection per stage plus tunables."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, model_validator

PROVIDER_KEYS = ("asr", "llm")


class TranscribeSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    phrase_gap_seconds: float = 0.5
    max_words_per_phrase: int = 30
    cut_padding_seconds: float = 0.15
    chunk_duration_seconds: float = 120.0
    overlap_duration_seconds: float = 15.0


class AnalyzeSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    keyframes_per_phrase: int = 1
    max_keyframes: int = 60
    llm_model: str = "sonnet"
    llm_timeout_seconds: int = 600
    # Below this many words per second a clip counts as a silent visual insert:
    # footage with no narration to select it by, offered to the planner as a shot
    # rather than as a line.
    insert_max_words_per_second: float = 0.5
    # Describe each insert clip from its keyframes so the planner can place it by
    # what it shows rather than guessing from chronology alone. Independent of
    # `keyframes_per_phrase`: turning phrase keyframes off must not silently turn
    # this off too, since insert descriptions are the whole point of an insert.
    describe_inserts: bool = True


class RenderSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    draft_height: int = 720
    draft_crf: int = 23
    audio_fade_seconds: float = 0.03


class SyncSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    primary_camera: str | None = None


class PlanSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    # Phrase ids the creator has explicitly banned from the edit — a "never use
    # that" moment named after reviewing a draft, not something the planner can
    # be argued out of via the brief.
    exclude_phrases: list[str] = Field(default_factory=list)
    # Story-section name -> gain in dB, applied to every clip in that beat by
    # `build_timeline`. Unknown beat names are ignored silently.
    gain_db_by_beat: dict[str, float] = Field(default_factory=dict)
    # The visual gate: refuse a draft in which a selected segment shows an adult
    # filling or crossing the frame, or a shot that cannot be used at all. Both
    # are on by default because the failure they catch (a draft that opens on an
    # adult's head) is the one the creator has to watch the whole draft to find.
    reject_adult_in_frame: bool = True
    reject_unusable_shots: bool = True


class Config(BaseModel):
    model_config = ConfigDict(frozen=True)

    providers: dict[str, str] = Field(
        default_factory=lambda: {"asr": "parakeet", "llm": "claude_cli"}
    )
    transcribe: TranscribeSettings = TranscribeSettings()
    analyze: AnalyzeSettings = AnalyzeSettings()
    render: RenderSettings = RenderSettings()
    sync: SyncSettings = SyncSettings()
    plan: PlanSettings = PlanSettings()

    @model_validator(mode="after")
    def _check_provider_keys(self) -> "Config":
        for key in self.providers:
            if key not in PROVIDER_KEYS:
                raise ValueError(f"unknown provider key: {key}")
        return self


def load_config(path: Path | None = None) -> Config:
    """Load config.yaml if present, merged over defaults. Also loads .env."""
    load_dotenv()
    if path is None or not path.exists():
        return Config()
    raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    defaults = Config()
    providers = {**defaults.providers, **(raw.get("providers") or {})}
    return Config(
        providers=providers,
        transcribe=TranscribeSettings(**(raw.get("transcribe") or {})),
        analyze=AnalyzeSettings(**(raw.get("analyze") or {})),
        render=RenderSettings(**(raw.get("render") or {})),
        sync=SyncSettings(**(raw.get("sync") or {})),
        plan=PlanSettings(**(raw.get("plan") or {})),
    )
