from videoai.core.models import Phrase, PhraseIndex
from videoai.logic.takes import detect_take_groups


def _index(*items: tuple[str, str, str]) -> PhraseIndex:
    phrases = []
    for ordinal, (phrase_id, clip_id, text) in enumerate(items):
        phrases.append(
            Phrase(
                phrase_id=phrase_id,
                clip_id=clip_id,
                start=float(ordinal),
                end=float(ordinal) + 0.8,
                text=text,
                word_start=ordinal,
                word_end=ordinal + 1,
            )
        )
    return PhraseIndex(phrases=phrases)


def test_exact_repeat_is_grouped():
    index = _index(
        ("clip-01#001", "clip-01", "This is the mega wrex truck"),
        ("clip-01#002", "clip-01", "This is the mega wrex truck"),
    )
    groups = detect_take_groups(index)
    assert len(groups.groups) == 1
    assert groups.groups[0].phrase_ids == ["clip-01#001", "clip-01#002"]


def test_paraphrased_repeat_is_grouped():
    index = _index(
        ("clip-01#001", "clip-01", "this is the mega wrex monster truck"),
        ("clip-01#002", "clip-01", "this is the mega wrex truck monster"),
    )
    groups = detect_take_groups(index, similarity=80)
    assert len(groups.groups) == 1


def test_unrelated_phrases_are_not_grouped():
    index = _index(
        ("clip-01#001", "clip-01", "look at the wheels"),
        ("clip-01#002", "clip-01", "now I will open the box"),
    )
    assert detect_take_groups(index).groups == []


def test_repeats_are_detected_across_clips():
    index = _index(
        ("clip-01#001", "clip-01", "welcome to my channel"),
        ("clip-02#001", "clip-02", "welcome to my channel"),
    )
    groups = detect_take_groups(index)
    assert len(groups.groups) == 1
    assert groups.groups[0].phrase_ids == ["clip-01#001", "clip-02#001"]


def test_repeat_beyond_window_is_ignored():
    # Filler text uses disjoint word sets (no two fillers, and no filler and
    # "welcome to my channel", share a word) so none of it accidentally
    # resembles anything else — a filler like "filler sentence number N" would
    # be >90% similar to its neighbour ("...number N+1") and confound this
    # test's actual point, which is the window cutoff, not filler similarity.
    filler_texts = [
        "banana kite yellow zoo",
        "plum ladder frost cave",
        "cotton otter violet drum",
        "desert falcon amber gate",
        "glacier pepper willow lamp",
        "comet ribbon walnut brook",
        "harbor cinder maple dune",
        "quartz lantern ember reef",
    ]
    items = [("clip-01#001", "clip-01", "welcome to my channel")]
    items += [
        (f"clip-01#{i + 2:03d}", "clip-01", text) for i, text in enumerate(filler_texts)
    ]
    items.append(("clip-01#010", "clip-01", "welcome to my channel"))
    assert detect_take_groups(_index(*items), window=3).groups == []


def test_three_attempts_form_one_group():
    index = _index(
        ("clip-01#001", "clip-01", "hello everyone welcome back"),
        ("clip-01#002", "clip-01", "hello everyone welcome back"),
        ("clip-01#003", "clip-01", "hello everyone welcome back"),
    )
    groups = detect_take_groups(index)
    assert len(groups.groups) == 1
    assert len(groups.groups[0].phrase_ids) == 3


def test_group_of_returns_group_id_or_none():
    index = _index(
        ("clip-01#001", "clip-01", "hello everyone welcome back"),
        ("clip-01#002", "clip-01", "hello everyone welcome back"),
        ("clip-01#003", "clip-01", "completely different content here"),
    )
    groups = detect_take_groups(index)
    assert groups.group_of("clip-01#001") == "take-01"
    assert groups.group_of("clip-01#003") is None


def test_single_word_phrases_are_never_grouped():
    # A single word carries no signal even when identical, so it stays
    # excluded regardless of MIN_WORDS or the similarity threshold.
    index = _index(
        ("clip-01#001", "clip-01", "yes"),
        ("clip-01#002", "clip-01", "yes"),
    )
    assert detect_take_groups(index).groups == []


def test_drift_chain_never_links_endpoints_transitively():
    # Each adjacent pair in this chain scores >= 75 (the default threshold),
    # but the endpoints (index 0 and index 5) score only 72.7 — well below
    # it. Seed-anchored grouping must never let them land in the same group
    # just because a chain of intermediate near-matches connects them.
    index = _index(
        ("clip-01#001", "clip-01", "the big red truck is over there by the fence"),
        ("clip-01#002", "clip-01", "the big red truck is over there near the fence"),
        ("clip-01#003", "clip-01", "the big red truck is parked near the fence"),
        ("clip-01#004", "clip-01", "the big blue van is parked near the fence"),
        ("clip-01#005", "clip-01", "the big blue van is parked near the garden"),
        ("clip-01#006", "clip-01", "the big blue van is over there by the garden"),
    )
    groups = detect_take_groups(index)
    for group in groups.groups:
        assert not ("clip-01#001" in group.phrase_ids and "clip-01#006" in group.phrase_ids)


def test_identical_short_phrases_are_grouped():
    index = _index(
        ("clip-01#001", "clip-01", "look at this"),
        ("clip-01#002", "clip-01", "look at this"),
    )
    groups = detect_take_groups(index)
    assert len(groups.groups) == 1
    assert groups.groups[0].phrase_ids == ["clip-01#001", "clip-01#002"]


def test_short_phrases_require_higher_similarity_than_long_ones():
    # "big red truck" vs "big red van" is 66.7 — above the default long-phrase
    # threshold (75) but below the short-phrase floor (90), so three-word
    # phrases must NOT group on that score alone.
    short = _index(
        ("clip-01#001", "clip-01", "big red truck"),
        ("clip-01#002", "clip-01", "big red van"),
    )
    assert detect_take_groups(short).groups == []

    # The same swap at four words scores 75.0 — at the default threshold —
    # and both phrases clear SHORT_PHRASE_WORDS, so they group.
    long = _index(
        ("clip-01#001", "clip-01", "the big red truck"),
        ("clip-01#002", "clip-01", "the big red van"),
    )
    groups = detect_take_groups(long)
    assert len(groups.groups) == 1


def test_word_dropped_retake_groups_at_default_threshold():
    # Scores 77.2 with token_sort_ratio — below the old default of 80 but
    # above the new default of 75.
    index = _index(
        ("clip-01#001", "clip-01", "this is the mega wrex monster truck"),
        ("clip-01#002", "clip-01", "this is the mega truck"),
    )
    groups = detect_take_groups(index)
    assert len(groups.groups) == 1
