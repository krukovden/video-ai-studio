import pytest

from videoai.core.models import ClipInfo, ClipTranscript, Manifest, Transcript, Word
from videoai.logic.inserts import detect_inserts, is_insert_ref, resolve_insert_ref


def _manifest() -> Manifest:
    return Manifest(clips=[
        # 10 words over 10s: ordinary narration, well above the threshold.
        ClipInfo(clip_id="clip-01", path="/tmp/a.mp4", duration=10.0, width=1920,
                 height=1080, fps=30.0, has_audio=True, recorded_at=100.0),
        # 2 words over 10s: the camera is close and nobody is really talking.
        ClipInfo(clip_id="clip-09", path="/tmp/b.mp4", duration=10.0, width=1920,
                 height=1080, fps=30.0, has_audio=True, recorded_at=200.0),
        # No words at all: the close-up of the bubble popping.
        ClipInfo(clip_id="clip-10", path="/tmp/c.mp4", duration=7.0, width=1920,
                 height=1080, fps=30.0, has_audio=True),
    ])


def _transcript() -> Transcript:
    return Transcript(provider="mock", clips=[
        ClipTranscript(clip_id="clip-01", words=[
            Word(text=f"word{i}", start=float(i), end=float(i) + 0.5) for i in range(10)
        ]),
        ClipTranscript(clip_id="clip-09", words=[
            Word(text="look", start=1.0, end=1.4), Word(text="there", start=1.5, end=1.9),
        ]),
        ClipTranscript(clip_id="clip-10", words=[]),
    ])


def test_clip_below_the_density_threshold_is_an_insert_candidate():
    inserts = detect_inserts(_manifest(), _transcript(), max_words_per_second=0.5)
    assert "clip-09" in [insert.clip_id for insert in inserts]


def test_clip_above_the_density_threshold_is_not_a_candidate():
    inserts = detect_inserts(_manifest(), _transcript(), max_words_per_second=0.5)
    assert "clip-01" not in [insert.clip_id for insert in inserts]


def test_clip_with_no_words_is_a_candidate_at_any_threshold():
    """Zero density is silence by definition, not a value to compare: a threshold
    of zero must still leave a wordless clip usable as an insert."""
    inserts = detect_inserts(_manifest(), _transcript(), max_words_per_second=0.0)
    assert [insert.clip_id for insert in inserts] == ["clip-10"]


def test_clip_missing_from_the_transcript_entirely_counts_as_silent():
    manifest = Manifest(clips=[ClipInfo(clip_id="clip-20", path="/tmp/d.mp4", duration=5.0,
                                        width=1920, height=1080, fps=30.0, has_audio=False)])
    inserts = detect_inserts(manifest, Transcript(provider="mock", clips=[]), 0.5)
    assert [insert.clip_id for insert in inserts] == ["clip-20"]


def test_insert_carries_duration_recorded_at_and_density():
    inserts = detect_inserts(_manifest(), _transcript(), max_words_per_second=0.5)
    quiet = next(insert for insert in inserts if insert.clip_id == "clip-09")
    silent = next(insert for insert in inserts if insert.clip_id == "clip-10")
    assert quiet.duration == 10.0
    assert quiet.recorded_at == 200.0
    assert abs(quiet.speech_density - 0.2) < 1e-6
    assert silent.recorded_at is None
    assert silent.speech_density == 0.0


def test_zero_length_clip_does_not_divide_by_zero():
    manifest = Manifest(clips=[ClipInfo(clip_id="clip-30", path="/tmp/e.mp4", duration=0.0,
                                        width=1920, height=1080, fps=30.0, has_audio=True)])
    inserts = detect_inserts(manifest, Transcript(provider="mock", clips=[]), 0.5)
    assert inserts[0].speech_density == 0.0


def test_only_insert_prefixed_references_are_recognised():
    assert is_insert_ref("insert:clip-10")
    assert is_insert_ref("insert:clip-10@2-5")
    assert not is_insert_ref("clip-01#004")


def test_bare_reference_resolves_to_the_whole_clip():
    assert resolve_insert_ref("insert:clip-10", _manifest()) == ("clip-10", 0.0, 7.0)


def test_ranged_reference_resolves_to_exactly_that_span():
    assert resolve_insert_ref("insert:clip-10@2-5", _manifest()) == ("clip-10", 2.0, 5.0)


def test_unknown_insert_clip_id_raises_naming_the_reference():
    with pytest.raises(RuntimeError) as error:
        resolve_insert_ref("insert:clip-77", _manifest())
    assert "insert:clip-77" in str(error.value)
    assert not isinstance(error.value, KeyError)


def test_range_past_the_end_of_the_clip_raises_naming_the_reference():
    with pytest.raises(RuntimeError) as error:
        resolve_insert_ref("insert:clip-10@4-12", _manifest())
    assert "insert:clip-10@4-12" in str(error.value)


def test_inverted_range_raises_naming_the_reference():
    with pytest.raises(RuntimeError) as error:
        resolve_insert_ref("insert:clip-10@5-2", _manifest())
    assert "insert:clip-10@5-2" in str(error.value)
    assert "inverted" in str(error.value)


def test_malformed_range_raises_naming_the_reference():
    with pytest.raises(RuntimeError) as error:
        resolve_insert_ref("insert:clip-10@two-five", _manifest())
    assert "insert:clip-10@two-five" in str(error.value)
