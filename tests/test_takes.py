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
    items = [("clip-01#001", "clip-01", "welcome to my channel")]
    # Filler text is kept under MIN_WORDS so it is ineligible for comparison; at
    # 4+ words, "filler sentence number N" phrases are >90% similar to their
    # neighbours (only the digit differs), which would chain them into a
    # spurious group under token_sort_ratio and defeat the point of this test.
    items += [(f"clip-01#{i:03d}", "clip-01", f"filler number {i}") for i in range(2, 10)]
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


def test_short_phrases_are_skipped():
    index = _index(
        ("clip-01#001", "clip-01", "yes"),
        ("clip-01#002", "clip-01", "yes"),
    )
    assert detect_take_groups(index).groups == []
