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


class DraftResult(BaseModel):
    path: str
    duration: float
    segment_count: int
