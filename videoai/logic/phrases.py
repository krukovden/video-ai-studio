"""Words to phrases, and phrases to the compact view handed to the LLM.

Phrases are the unit every later stage reasons about: they are short enough to
score individually and long enough to carry meaning, and their boundaries sit in
silence, which makes them safe cut points.
"""
from __future__ import annotations

from videoai.core.models import Phrase, PhraseIndex, Transcript, Word


def _flush(
    clip_id: str, ordinal: int, words: list[Word], word_start: int
) -> Phrase:
    return Phrase(
        phrase_id=f"{clip_id}#{ordinal:03d}",
        clip_id=clip_id,
        start=words[0].start,
        end=words[-1].end,
        text=" ".join(word.text for word in words),
        word_start=word_start,
        word_end=word_start + len(words),
    )


def build_phrases(transcript: Transcript, gap: float, max_words: int) -> PhraseIndex:
    phrases: list[Phrase] = []
    for clip in transcript.clips:
        ordinal = 0
        buffer: list[Word] = []
        buffer_start = 0
        for index, word in enumerate(clip.words):
            if buffer:
                too_long = len(buffer) >= max_words
                long_pause = word.start - buffer[-1].end > gap
                if too_long or long_pause:
                    ordinal += 1
                    phrases.append(_flush(clip.clip_id, ordinal, buffer, buffer_start))
                    buffer = []
            if not buffer:
                buffer_start = index
            buffer.append(word)
        if buffer:
            ordinal += 1
            phrases.append(_flush(clip.clip_id, ordinal, buffer, buffer_start))
    return PhraseIndex(phrases=phrases)


def pack_transcript(index: PhraseIndex) -> str:
    """Compact, id-addressable transcript. Roughly a tenth of raw JSON in tokens."""
    lines: list[str] = []
    current_clip: str | None = None
    for phrase in index.phrases:
        if phrase.clip_id != current_clip:
            current_clip = phrase.clip_id
            lines.append(f"\n## {current_clip}")
        lines.append(
            f"[{phrase.phrase_id}] {phrase.start:.2f}-{phrase.end:.2f} {phrase.text}"
        )
    return "\n".join(lines).strip()
