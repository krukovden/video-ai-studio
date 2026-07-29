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


class AnalyzeSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    keyframes_per_phrase: int = 1
    max_keyframes: int = 60
    llm_model: str = "sonnet"
    llm_timeout_seconds: int = 600


class RenderSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    draft_height: int = 720
    draft_crf: int = 23
    audio_fade_seconds: float = 0.03


class SyncSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    primary_camera: str | None = None


class Config(BaseModel):
    model_config = ConfigDict(frozen=True)

    providers: dict[str, str] = Field(
        default_factory=lambda: {"asr": "parakeet", "llm": "claude_cli"}
    )
    transcribe: TranscribeSettings = TranscribeSettings()
    analyze: AnalyzeSettings = AnalyzeSettings()
    render: RenderSettings = RenderSettings()
    sync: SyncSettings = SyncSettings()

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
    )
