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


class Analysis(BaseModel):
    provider: str
    segments: list[SegmentAnalysis] = Field(default_factory=list)
    inserts: list[InsertClip] = Field(default_factory=list)

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

    Legacy preview mode may degrade individual elements. Strict production mode
    validates the required fields and removes an invalid `final.mp4`.
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
    outro: bool = False
    music_ducking: bool = False
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
