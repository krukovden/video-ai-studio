"""Detect repeated attempts at the same line.

Recall is deterministic and cheap here; precision is the model's job later. The
detector only proposes candidate groups — which attempt is best is an editorial
judgement made during analysis.
"""
from __future__ import annotations

import re

from rapidfuzz import fuzz

from videoai.core.models import PhraseIndex, TakeGroup, TakeGroups

MIN_WORDS = 4


def _normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", text.lower()).strip()


def detect_take_groups(
    index: PhraseIndex, similarity: int = 80, window: int = 6
) -> TakeGroups:
    phrases = index.phrases
    normalised = [_normalise(phrase.text) for phrase in phrases]
    eligible = [len(text.split()) >= MIN_WORDS for text in normalised]

    parent = list(range(len(phrases)))

    def find(item: int) -> int:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    for i in range(len(phrases)):
        if not eligible[i]:
            continue
        for j in range(i + 1, min(i + 1 + window, len(phrases))):
            if not eligible[j]:
                continue
            if fuzz.token_sort_ratio(normalised[i], normalised[j]) >= similarity:
                union(i, j)

    members: dict[int, list[int]] = {}
    for position in range(len(phrases)):
        if eligible[position]:
            members.setdefault(find(position), []).append(position)

    groups: list[TakeGroup] = []
    for root in sorted(members):
        positions = members[root]
        if len(positions) < 2:
            continue
        groups.append(
            TakeGroup(
                group_id=f"take-{len(groups) + 1:02d}",
                phrase_ids=[phrases[position].phrase_id for position in positions],
            )
        )
    return TakeGroups(groups=groups)
