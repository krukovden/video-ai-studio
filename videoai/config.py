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
    # Hand the footage itself to a provider that can watch it, instead of a
    # transcript and a still per phrase. Only takes effect when the stage's
    # provider reads video — a still-only model keeps getting stills, so
    # switching model never silently changes what the analysis was made from.
    #
    # This is the only metered thing in the pipeline, so it is a switch of its
    # own rather than something implied by the provider: a creator must be able
    # to say "not this run" without changing model.
    submit_video: bool = True
    # A moment does not start on the first syllable. Padding either side of a
    # phrase buys the run-up and the reaction, which is usually where the
    # delivery actually is.
    video_padding_seconds: float = 0.6
    # The budget, in seconds of footage submitted. Video is billed by the
    # second — roughly 100 input tokens per second at low media resolution — so
    # this is the one setting that decides what a run costs. 900s of speech is
    # about 90k tokens: a few cents on a Flash model.
    max_video_seconds: float = 900.0


class DescribeSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    # Watch every source clip once and keep what it contains. The result is a
    # cache keyed by file identity, so this is paid for once per clip ever, not
    # once per run.
    #
    # Off by default because it needs a provider that can watch video, and the
    # pipeline's default provider cannot: switching it on without also pointing
    # `describe` at such a provider is an error rather than a quiet no-op.
    enabled: bool = False


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


class EffectsSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    # The cartoon-effects stage: one LLM call that reads the story and picks which
    # sprites from `assets/effects/` punctuate which moments. Off means no events
    # are planned and none are composited; the delivery is otherwise identical,
    # because effects are not a required production feature.
    enabled: bool = True
    # A hard ceiling on the accents one video may carry. The prompt asks for four
    # to eight in three minutes; this is what stops a model that got carried away,
    # and it refuses rather than silently truncating.
    max_events: int = 8


class PolishSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    # False does not produce a delivery: it copies the review draft out under the
    # contract's fallback name. There is one renderer and the contract always
    # applies, so there is no setting that turns enforcement off.
    enabled: bool = True
    # When enabled, `videoai approve PROJECT` must be run after reviewing the
    # current draft. Approval is tied to the exact timeline, so any re-plan
    # requires another review.
    require_approval: bool = False
    intro_seconds: float = 2.5
    outro_seconds: float = 2.5
    outro_text: str = "Thanks for watching!"
    title_seconds: float = 2.0
    # Always generate final.srt when captions are enabled. Burning every spoken
    # word into the picture is optional; YouTube can use the sidecar as a
    # viewer-controlled caption track.
    captions_enabled: bool = True
    burn_captions: bool = False
    caption_words: int = 4
    # Where the music bed sits before ducking, and how much further it drops while
    # the child is talking. Both are attenuations: the bed is never the loudest
    # thing in the mix.
    music_gain_db: float = -22.0
    music_duck_db: float = -12.0
    transition_frames: int = 8
    # An explicit filename inside `music_dir`, which overrides the automatic pick.
    music_track: str | None = None
    music_dir: str = "assets/Music"
    # Delivery, as opposed to review. The draft is cut from the 540p proxies so a
    # human can check edit decisions in seconds; the final is cut from the
    # original files and these three settings govern what comes out of that.
    # 1080 because a clean downscale from 4K looks excellent and keeps the file
    # manageable; 2160 delivers the source resolution untouched. The frame keeps
    # the source's aspect — only the height is named here.
    output_height: int = 1080
    # Visually near-lossless, against the draft's 26.
    output_crf: int = 18
    # VideoToolbox when the build has it, libx264 when it does not.
    hardware_encode: bool = True


class Config(BaseModel):
    model_config = ConfigDict(frozen=True)

    providers: dict[str, str] = Field(
        default_factory=lambda: {"asr": "parakeet", "llm": "claude_cli"}
    )
    transcribe: TranscribeSettings = TranscribeSettings()
    analyze: AnalyzeSettings = AnalyzeSettings()
    describe: DescribeSettings = DescribeSettings()
    render: RenderSettings = RenderSettings()
    sync: SyncSettings = SyncSettings()
    plan: PlanSettings = PlanSettings()
    effects: EffectsSettings = EffectsSettings()
    polish: PolishSettings = PolishSettings()
    # Archive output/'s deliverables into a timestamped subfolder after a run
    # that actually executed something, so a re-run's overwrite never erases the
    # only copy of a previous final.mp4. Bookkeeping, not pipeline content: no
    # stage's fingerprint depends on this setting.
    output_snapshots: bool = True
    # A different model for one stage, without touching the others. Models are
    # interchangeable editorial voices and they are not equally worth paying for
    # at every step: watching the footage is worth a metered call, naming a
    # section title is not. Keyed by stage id, e.g. {"analyze": "gemini_api"}.
    llm_by_stage: dict[str, str] = Field(default_factory=dict)

    def llm_for(self, stage_id: str) -> str:
        """The provider this stage should call: its override, or the default."""
        return self.llm_by_stage.get(stage_id) or self.providers.get("llm", "")

    @model_validator(mode="after")
    def _check_provider_keys(self) -> "Config":
        for key in self.providers:
            if key not in PROVIDER_KEYS:
                raise ValueError(f"unknown provider key: {key}")
        return self

    @model_validator(mode="after")
    def _check_stage_overrides(self) -> "Config":
        """Refuse an override naming a stage that does not exist or does not call
        a model. Either is a typo, and a typo here is silent: the stage would
        keep the default while the creator believed they had switched it."""
        if not self.llm_by_stage:
            return self
        # Imported here, not at module scope: the registry imports config.
        from videoai.core.registry import REGISTRY

        import videoai.stages  # noqa: F401  (registers every stage)

        for stage_id in self.llm_by_stage:
            spec = REGISTRY.get(stage_id)
            if spec is None:
                known = ", ".join(sorted(s.id for s in REGISTRY.values() if s.provider_key == "llm"))
                raise ValueError(
                    f"llm_by_stage names an unknown stage: {stage_id} (model-calling "
                    f"stages are: {known})"
                )
            if spec.provider_key != "llm":
                raise ValueError(
                    f"llm_by_stage names '{stage_id}', which does not call a model"
                )
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
        describe=DescribeSettings(**(raw.get("describe") or {})),
        render=RenderSettings(**(raw.get("render") or {})),
        sync=SyncSettings(**(raw.get("sync") or {})),
        plan=PlanSettings(**(raw.get("plan") or {})),
        effects=EffectsSettings(**(raw.get("effects") or {})),
        polish=PolishSettings(**(raw.get("polish") or {})),
        output_snapshots=raw.get("output_snapshots", defaults.output_snapshots),
        llm_by_stage=raw.get("llm_by_stage") or {},
    )
