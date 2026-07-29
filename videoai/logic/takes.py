"""Detect repeated attempts at the same line.

Recall is deterministic and cheap here; precision is the model's job later. The
detector only proposes candidate groups — which attempt is best is an editorial
judgement made during analysis.
"""
from __future__ import annotations

import re

from rapidfuzz import fuzz

from videoai.core.models import PhraseIndex, TakeGroup, TakeGroups

MIN_WORDS = 2
# Short phrases carry little signal (few tokens means a single word swap moves
# the score a lot), so they need near-identity rather than the configured
# threshold to avoid false positives.
SHORT_PHRASE_WORDS = 4
SHORT_PHRASE_SIMILARITY = 90


def _normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", text.lower()).strip()


def detect_take_groups(
    index: PhraseIndex, similarity: int = 75, window: int = 6
) -> TakeGroups:
    phrases = index.phrases
    normalised = [_normalise(phrase.text) for phrase in phrases]
    word_counts = [len(text.split()) for text in normalised]
    eligible = [count >= MIN_WORDS for count in word_counts]

    # Seed-anchored grouping (not union-find): every member of a group is
    # compared directly against the seed, never against another member. This
    # keeps groups from chaining together through a drifting sequence of
    # phrases where only *adjacent* pairs are similar — e.g. A~B~C~D~E~F where
    # A and F individually fall well below the threshold must never end up in
    # the same group just because each step along the way cleared it.
    assigned = [False] * len(phrases)
    groups: list[TakeGroup] = []

    for seed in range(len(phrases)):
        if not eligible[seed] or assigned[seed]:
            continue

        members = [seed]
        for candidate in range(seed + 1, min(seed + 1 + window, len(phrases))):
            if not eligible[candidate] or assigned[candidate]:
                continue
            threshold = (
                SHORT_PHRASE_SIMILARITY
                if word_counts[seed] < SHORT_PHRASE_WORDS
                or word_counts[candidate] < SHORT_PHRASE_WORDS
                else similarity
            )
            score = fuzz.token_sort_ratio(normalised[seed], normalised[candidate])
            if score >= threshold:
                members.append(candidate)

        if len(members) < 2:
            continue

        for member in members:
            assigned[member] = True
        groups.append(
            TakeGroup(
                group_id=f"take-{len(groups) + 1:02d}",
                phrase_ids=[phrases[member].phrase_id for member in members],
            )
        )

    return TakeGroups(groups=groups)
