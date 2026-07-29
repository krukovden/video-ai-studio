from videoai.core.models import ClipTranscript, Transcript, Word
from videoai.logic.phrases import build_phrases, pack_transcript


def _transcript(*clips: tuple[str, list[tuple[str, float, float]]]) -> Transcript:
    return Transcript(
        provider="mock",
        clips=[
            ClipTranscript(
                clip_id=clip_id,
                words=[Word(text=t, start=s, end=e) for t, s, e in words],
            )
            for clip_id, words in clips
        ],
    )


def test_phrases_split_on_gap():
    transcript = _transcript(
        ("clip-01", [("Look", 0.0, 0.3), ("here", 0.35, 0.7), ("Wow", 2.0, 2.4)])
    )
    index = build_phrases(transcript, gap=0.5, max_words=30)
    assert [p.text for p in index.phrases] == ["Look here", "Wow"]
    assert index.phrases[0].phrase_id == "clip-01#001"
    assert index.phrases[1].phrase_id == "clip-01#002"


def test_phrase_carries_time_and_word_range():
    transcript = _transcript(
        ("clip-01", [("Look", 0.0, 0.3), ("here", 0.35, 0.7), ("Wow", 2.0, 2.4)])
    )
    index = build_phrases(transcript, gap=0.5, max_words=30)
    first = index.phrases[0]
    assert first.start == 0.0 and first.end == 0.7
    assert (first.word_start, first.word_end) == (0, 2)
    second = index.phrases[1]
    assert (second.word_start, second.word_end) == (2, 3)


def test_phrases_split_on_max_words():
    words = [(f"w{i}", i * 0.1, i * 0.1 + 0.05) for i in range(10)]
    transcript = _transcript(("clip-01", words))
    index = build_phrases(transcript, gap=5.0, max_words=4)
    assert [len(p.text.split()) for p in index.phrases] == [4, 4, 2]


def test_phrases_are_numbered_per_clip():
    transcript = _transcript(
        ("clip-01", [("a", 0.0, 0.2)]),
        ("clip-02", [("b", 0.0, 0.2)]),
    )
    index = build_phrases(transcript, gap=0.5, max_words=30)
    assert [p.phrase_id for p in index.phrases] == ["clip-01#001", "clip-02#001"]


def test_clip_without_words_produces_no_phrases():
    transcript = _transcript(("clip-01", []))
    assert build_phrases(transcript, gap=0.5, max_words=30).phrases == []


def test_by_id_raises_for_unknown_phrase():
    index = build_phrases(_transcript(("clip-01", [("a", 0.0, 0.2)])), gap=0.5, max_words=30)
    assert index.by_id("clip-01#001").text == "a"
    try:
        index.by_id("clip-09#001")
    except KeyError as error:
        assert "clip-09#001" in str(error)
    else:
        raise AssertionError("expected KeyError")


def test_pack_transcript_is_compact_and_labelled():
    transcript = _transcript(
        ("clip-01", [("Look", 0.0, 0.3), ("here", 0.35, 0.7), ("Wow", 2.0, 2.4)])
    )
    packed = pack_transcript(build_phrases(transcript, gap=0.5, max_words=30))
    assert "## clip-01" in packed
    assert "[clip-01#001] 0.00-0.70 Look here" in packed
    assert "[clip-01#002] 2.00-2.40 Wow" in packed
