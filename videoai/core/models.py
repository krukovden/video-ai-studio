"""Artifact models. Every stage input and output is defined here."""
from __future__ import annotations

from pydantic import BaseModel, Field


class ClipInfo(BaseModel):
    clip_id: str
    path: str
    duration: float
    width: int
    height: int
    fps: float
    has_audio: bool
    camera: str = "main"
    recorded_at: float | None = None
    # Digest of the source file's identity (path, size, mtime). `clip_id` is
    # positional and shifts when clips are added; this does not, so every piece
    # of derived media is keyed by it.
    source_key: str = ""
    audio_path: str | None = None
    proxy_path: str | None = None


class Manifest(BaseModel):
    clips: list[ClipInfo] = Field(default_factory=list)

    def by_id(self, clip_id: str) -> ClipInfo:
        for clip in self.clips:
            if clip.clip_id == clip_id:
                return clip
        raise KeyError(f"unknown clip_id: {clip_id}")


class ClipQuality(BaseModel):
    clip_id: str
    blur: float
    motion: float
    black_ratio: float
    usable: bool
    scored: bool = True


class QualityReport(BaseModel):
    clips: list[ClipQuality] = Field(default_factory=list)

    def by_id(self, clip_id: str) -> ClipQuality:
        for clip in self.clips:
            if clip.clip_id == clip_id:
                return clip
        raise KeyError(f"unknown clip_id: {clip_id}")


class Word(BaseModel):
    text: str
    start: float
    end: float
    confidence: float = 1.0


class SpeechSpan(BaseModel):
    start: float
    end: float


class ClipTranscript(BaseModel):
    clip_id: str
    words: list[Word] = Field(default_factory=list)
    speech_spans: list[SpeechSpan] = Field(default_factory=list)


class Transcript(BaseModel):
    provider: str
    clips: list[ClipTranscript] = Field(default_factory=list)

    def by_id(self, clip_id: str) -> ClipTranscript:
        for clip in self.clips:
            if clip.clip_id == clip_id:
                return clip
        raise KeyError(f"unknown clip_id: {clip_id}")


class Phrase(BaseModel):
    phrase_id: str
    clip_id: str
    start: float
    end: float
    text: str
    word_start: int
    word_end: int


class PhraseIndex(BaseModel):
    phrases: list[Phrase] = Field(default_factory=list)

    def by_id(self, phrase_id: str) -> Phrase:
        for phrase in self.phrases:
            if phrase.phrase_id == phrase_id:
                return phrase
        raise KeyError(f"unknown phrase_id: {phrase_id}")


class ClipSync(BaseModel):
    clip_id: str
    camera: str
    global_start: float
    method: str
    confidence: float = 0.0


class SyncMap(BaseModel):
    clips: list[ClipSync] = Field(default_factory=list)
    primary_camera: str = "main"

    def clips_of(self, camera: str) -> list[ClipSync]:
        return [clip for clip in self.clips if clip.camera == camera]

    def by_id(self, clip_id: str) -> ClipSync:
        for clip in self.clips:
            if clip.clip_id == clip_id:
                return clip
        raise KeyError(f"unknown clip_id: {clip_id}")

    def overlaps(self, clip_a: str, clip_b: str, manifest: "Manifest") -> bool:
        """True when both clips were recording at the same moment — different
        angles of one performance rather than two attempts at it."""
        first, second = self.by_id(clip_a), self.by_id(clip_b)
        if first.camera == second.camera:
            return False
        first_end = first.global_start + manifest.by_id(clip_a).duration
        second_end = second.global_start + manifest.by_id(clip_b).duration
        return first.global_start < second_end and second.global_start < first_end


class TakeGroup(BaseModel):
    group_id: str
    phrase_ids: list[str] = Field(default_factory=list)


class TakeGroups(BaseModel):
    groups: list[TakeGroup] = Field(default_factory=list)

    def group_of(self, phrase_id: str) -> str | None:
        for group in self.groups:
            if phrase_id in group.phrase_ids:
                return group.group_id
        return None


class SegmentAnalysis(BaseModel):
    phrase_id: str
    clip_id: str
    start: float
    end: float
    text: str
    content: str = ""
    delivery_score: int = 5
    visual_score: int = 5
    emotion: str = "neutral"
    speaker: str = "unclear"
    is_failed_take: bool = False
    take_group: str | None = None
    shorts_candidate: bool = False
    scored: bool = True


class InsertClip(BaseModel):
    """A clip nobody narrates: a silent visual shot the planner may cut in.

    Selection downstream is phrase-based, so a clip carrying no speech has no
    phrase to be chosen by and could never reach the timeline on its own.
    """

    clip_id: str
    duration: float
    recorded_at: float | None = None
    speech_density: float = 0.0
    # One short sentence naming what the clip physically shows, from the LLM
    # looking at its keyframes. Empty when description generation is disabled
    # or the model's reply did not cover this clip; a missing description
    # degrades placement in the plan stage, it does not block anything.
    description: str = ""


class ClipEvent(BaseModel):
    """One moment inside a source clip, on that clip's own clock."""

    at: float
    kind: str = "action"
    what: str = ""
    # Roughly where on screen it happens, as one of the nine grid cells the
    # effects stage already speaks in. Without it an accent has nothing to aim
    # at and lands wherever the model guessed.
    where: str = ""


class ClipNote(BaseModel):
    """What one source clip contains, described once and kept.

    Keyed by `source_key` rather than `clip_id`: ids are positional and shift
    whenever a clip is added, while the key identifies the file itself. That is
    what makes this reusable — renumbering costs nothing, and only genuinely new
    or replaced footage is ever sent to be watched again.
    """

    clip_id: str
    source_key: str
    duration: float
    source_name: str = ""
    summary: str = ""
    events: list[ClipEvent] = Field(default_factory=list)
    # What describing this one clip cost, so the log can total it honestly.
    input_tokens: int = 0
    output_tokens: int = 0


class ClipNotes(BaseModel):
    """Every clip's description, as a cache that only ever grows."""

    provider: str = ""
    notes: list[ClipNote] = Field(default_factory=list)

    def by_key(self, source_key: str) -> ClipNote:
        for note in self.notes:
            if note.source_key == source_key:
                return note
        raise KeyError(f"no note for source_key: {source_key}")

    def keys(self) -> set[str]:
        return {note.source_key for note in self.notes}


class Observation(BaseModel):
    """One thing a model actually saw happen, put back on the source clock.

    This is the only place a model's own timestamps enter the pipeline, and they
    enter as evidence rather than as instructions: an observation says what
    happened and when it was seen, and every stage that uses one still decides
    for itself whether to act on it. It is the difference between a model reading
    a printed timeline and guessing when a thing burst, and a model watching the
    thing burst.

    `reel_at` is kept alongside `at` so a finding can be checked against the file
    the model was actually shown.
    """

    clip_id: str
    at: float
    reel_at: float
    # action | emotion | reaction. A coarse split, so a stage can ask for the
    # physical moments without wading through every flicker of expression.
    kind: str = "action"
    what: str = ""


class Analysis(BaseModel):
    provider: str
    segments: list[SegmentAnalysis] = Field(default_factory=list)
    inserts: list[InsertClip] = Field(default_factory=list)
    # What the model saw while watching, in source-clip time. Empty when the
    # scores were made from a transcript and stills.
    observations: list[Observation] = Field(default_factory=list)
    # Seconds of footage actually submitted to a video-capable model, and zero
    # when the scores were made from the transcript and stills. Video is the one
    # metered input in this pipeline and it is billed by the second, so what was
    # paid for is recorded rather than left to be inferred from a bill.
    video_seconds: float = 0.0
    # What the provider reported spending, when it reports anything. The Gemini
    # API exposes no balance and no quota header, so a reply's own accounting is
    # the only honest measure of a run's cost — and it is worth keeping, because
    # nothing else will tell you afterwards.
    input_tokens: int = 0
    output_tokens: int = 0
    video_tokens: int = 0

    def by_phrase(self, phrase_id: str) -> SegmentAnalysis:
        for segment in self.segments:
            if segment.phrase_id == phrase_id:
                return segment
        raise KeyError(f"unknown phrase_id: {phrase_id}")


class PlanSection(BaseModel):
    name: str
    goal: str = ""
    phrase_ids: list[str] = Field(default_factory=list)


class StoryPlan(BaseModel):
    sections: list[PlanSection] = Field(default_factory=list)
    title: str = ""
    description: str = ""
    tags: list[str] = Field(default_factory=list)


class TimelineClip(BaseModel):
    src: str
    offset: float
    dur: float
    start: float
    quote: str = ""
    reason: str = ""
    beat: str = ""
    angles: list[str] = Field(default_factory=list)
    core_dur: float = 0.0
    gain_db: float = 0.0
    # A silent visual insert rather than a line of speech. Rules that only make
    # sense against words (word-boundary cuts, quote anchoring) do not apply to it.
    is_insert: bool = False


class Timeline(BaseModel):
    fps: float
    width: int
    height: int
    clips: list[TimelineClip] = Field(default_factory=list)

    @property
    def duration(self) -> float:
        return sum(clip.dur for clip in self.clips)


class SegmentVisual(BaseModel):
    """What the frames of one already-selected timeline segment actually show.

    Every flag defaults to the pessimistic reading: nothing here may be inferred
    from silence. `checked=False` means the model said nothing about this
    segment, which is not the same as it saying the segment is fine.
    """

    index: int
    src: str
    adult_prominent: bool = False
    child_visible: bool = False
    unusable: bool = False
    note: str = ""
    checked: bool = False


class VisualCheck(BaseModel):
    provider: str
    segments: list[SegmentVisual] = Field(default_factory=list)


class RejectedPhrases(BaseModel):
    """Phrase ids (and `insert:<clip_id>` references) the visual gate refused.

    Written by the visual check and read back by the plan stage, which hides
    them from the planner exactly like `plan.exclude_phrases`. The list only
    ever grows: a shot rejected for what it shows is rejected for good, and a
    later planning round must not be offered it again.
    """

    phrase_ids: list[str] = Field(default_factory=list)


class EffectEvent(BaseModel):
    """One cartoon accent, placed on the timeline's clock by the story.

    `at_seconds` is measured on the TIMELINE, not on the delivery: the model that
    chose it was shown the edit, and the delivery adds an intro card and drifts by
    a frame per cut. The compositor maps it forward through the measured segment
    starts, exactly as it maps captions.
    """

    at_seconds: float
    effect_name: str
    screen_position: str
    scale: str = "medium"
    seconds: float = 0.0
    # Speech-bubble entries only: the words to render inside the nine-patch.
    text: str = ""
    reason: str = ""


class EffectPlan(BaseModel):
    """Which accents to show, and where. An empty list is a valid answer: a calm
    review needs no effects, and inventing some would be worse than none."""

    provider: str = ""
    library: str = ""
    events: list[EffectEvent] = Field(default_factory=list)


class DraftResult(BaseModel):
    path: str
    duration: float
    segment_count: int
    # The exact edit this review file represents. Approval must never bind a
    # stale draft to a newer plan merely because both files happen to exist.
    timeline_hash: str = ""


class Approval(BaseModel):
    """Explicit creator approval bound to one exact timeline."""

    timeline_hash: str
    draft_hash: str = ""
    config_hash: str = ""
    approved_at: str


class FinalResult(BaseModel):
    """The watchable cut plus an exact record of the applied production layers.

    `production-contract.yaml` is validated against these measurements before the
    file is allowed to keep the name `final.mp4`; a render that fails it, or a
    deliberately degraded one, is named by the contract's fallback instead and
    says here what it is missing.
    """

    path: str
    duration: float
    # The delivered frame, and how long it took to build. Both are recorded
    # because the final is cut from the originals rather than from the draft:
    # the resolution is the thing that used to be silently wrong, and the cost of
    # getting it right is what a creator needs to know before re-rendering.
    width: int = 0
    height: int = 0
    render_seconds: float = 0.0
    intro: bool = False
    intro_title: str = ""
    title_count: int = 0
    transition_count: int = 0
    caption_count: int = 0
    burned_in_captions: bool = False
    # Cartoon accents actually composited, and one line each naming what and when.
    # Seasoning, not structure: the production contract does not require any, and a
    # delivery with none is as valid as a delivery with eight.
    effect_count: int = 0
    effects: list[str] = Field(default_factory=list)
    outro: bool = False
    music_ducking: bool = False
    # How far the music bed measurably dropped under speech, in dB. Measured from
    # the delivered mix rather than assumed from the requested setting.
    music_ducking_db: float = 0.0
    fully_decoded: bool = False
    production_report: str = ""
    music_track: str | None = None
    # Bensound's free licence requires the credit to travel with the video, so it
    # is recorded here as well as written to output/metadata.md.
    music_attribution: str = ""
    notes: list[str] = Field(default_factory=list)


class ExportResult(BaseModel):
    """What `export_edit` wrote for a human editor to open in DaVinci Resolve
    (or any other OTIO/EDL-capable NLE)."""

    otio_path: str
    edl_path: str
    clip_count: int
    # Absolute source paths (from `ClipInfo.path`) that did not exist on disk at
    # export time. The interchange files are still written with these references
    # — Resolve's own "Relink Selected Clips" is the normal fix — this list just
    # tells the creator up front which files it will ask about.
    missing_media: list[str] = Field(default_factory=list)
