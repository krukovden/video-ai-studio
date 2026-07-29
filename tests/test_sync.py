import math
import wave
from pathlib import Path

import numpy as np

from videoai.core.models import ClipInfo, Manifest
from videoai.logic.sync import audio_envelope, build_sync_map, estimate_offset


def _write_wav(path: Path, samples: np.ndarray, rate: int = 16000) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.clip(samples, -1.0, 1.0)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes((data * 32767).astype("<i2").tobytes())
    return path


def _bursts(duration: float, burst_times: list[float], rate: int = 16000) -> np.ndarray:
    samples = np.zeros(int(duration * rate), dtype=np.float64)
    for start in burst_times:
        begin = int(start * rate)
        end = min(len(samples), begin + int(0.2 * rate))
        if begin < len(samples):
            noise = np.random.default_rng(abs(int(start * 1000)) + 1).normal(0, 0.4, end - begin)
            samples[begin:end] = noise
    return samples


def test_audio_envelope_length_matches_duration(tmp_path: Path):
    path = _write_wav(tmp_path / "a.wav", _bursts(3.0, [0.5, 1.5]))
    envelope = audio_envelope(path, rate=100)
    assert 290 <= len(envelope) <= 310


def test_audio_envelope_is_loud_where_the_bursts_are(tmp_path: Path):
    path = _write_wav(tmp_path / "a.wav", _bursts(3.0, [1.0]))
    envelope = audio_envelope(path, rate=100)
    assert envelope[100:120].mean() > envelope[200:280].mean() * 5


def test_estimate_offset_recovers_a_known_shift(tmp_path: Path):
    signal = _bursts(8.0, [1.0, 2.7, 4.1, 6.3])
    reference = _write_wav(tmp_path / "ref.wav", signal)
    shifted = _write_wav(tmp_path / "other.wav", np.concatenate([np.zeros(16000 * 2), signal]))

    offset, confidence = estimate_offset(
        audio_envelope(reference), audio_envelope(shifted)
    )

    assert math.isclose(offset, -2.0, abs_tol=0.15)
    assert confidence > 2.0


def test_estimate_offset_reports_low_confidence_for_unrelated_audio(tmp_path: Path):
    first = _write_wav(tmp_path / "a.wav", _bursts(8.0, [1.0, 2.0, 3.0]))
    second = _write_wav(tmp_path / "b.wav", np.random.default_rng(7).normal(0, 0.3, 16000 * 8))

    _, confidence = estimate_offset(audio_envelope(first), audio_envelope(second))

    assert confidence < 2.0


def test_estimate_offset_confidence_stays_low_for_unrelated_audio_across_seeds(tmp_path: Path):
    # Guards the peak-to-runner-up confidence formula against silent regression:
    # a single fixed seed passing is not proof the formula generalises.
    reference = audio_envelope(_write_wav(tmp_path / "ref.wav", _bursts(8.0, [1.0, 2.0, 3.0])))
    for seed in range(5):
        other = _write_wav(
            tmp_path / f"other-{seed}.wav",
            np.random.default_rng(seed).normal(0, 0.3, 16000 * 8),
        )
        _, confidence = estimate_offset(reference, audio_envelope(other))
        assert confidence < 2.0, f"seed={seed} produced confidence={confidence}"


def _clip(clip_id: str, camera: str, recorded_at: float | None, duration: float = 10.0) -> ClipInfo:
    return ClipInfo(
        clip_id=clip_id, path=f"/tmp/{clip_id}.mov", duration=duration, width=1920,
        height=1080, fps=30.0, has_audio=True, camera=camera, recorded_at=recorded_at,
    )


def test_single_camera_places_clips_by_recorded_at():
    manifest = Manifest(clips=[
        _clip("clip-01", "main", 1000.0),
        _clip("clip-02", "main", 1030.0),
    ])

    sync = build_sync_map(manifest, envelope_of=lambda clip: None)

    assert sync.by_id("clip-01").global_start == 0.0
    assert sync.by_id("clip-02").global_start == 30.0
    assert sync.by_id("clip-02").method == "metadata"


def test_missing_recorded_at_falls_back_to_sequential_placement():
    manifest = Manifest(clips=[
        _clip("clip-01", "main", None, duration=10.0),
        _clip("clip-02", "main", None, duration=5.0),
    ])

    sync = build_sync_map(manifest, envelope_of=lambda clip: None)

    assert sync.by_id("clip-01").global_start == 0.0
    assert sync.by_id("clip-02").global_start == 10.0
    assert sync.by_id("clip-02").method == "sequential"


def test_metadata_placement_wins_over_manifest_order():
    # clip-01 comes first in the manifest but recorded later; its own
    # recorded_at must place it after clip-02, not before.
    manifest = Manifest(clips=[
        _clip("clip-01", "main", 2000.0, duration=10.0),
        _clip("clip-02", "main", 1000.0, duration=10.0),
    ])

    sync = build_sync_map(manifest, envelope_of=lambda clip: None)

    assert sync.by_id("clip-02").global_start == 0.0
    assert sync.by_id("clip-01").global_start == 1000.0
    assert sync.by_id("clip-01").method == "metadata"
    assert sync.by_id("clip-02").method == "metadata"


def test_untagged_clip_is_placed_sequentially_without_discarding_tagged_neighbours():
    # A clip with no recorded_at (re-encoded, AirDropped, edited) must not throw
    # away the real timestamps on its camera's other clips.
    manifest = Manifest(clips=[
        _clip("clip-01", "main", 1000.0, duration=10.0),
        _clip("clip-02", "main", None, duration=5.0),
        _clip("clip-03", "main", 1100.0, duration=10.0),
    ])

    sync = build_sync_map(manifest, envelope_of=lambda clip: None)

    assert sync.by_id("clip-01").global_start == 0.0
    assert sync.by_id("clip-01").method == "metadata"
    assert sync.by_id("clip-02").global_start == 10.0
    assert sync.by_id("clip-02").method == "sequential"
    assert sync.by_id("clip-03").global_start == 100.0
    assert sync.by_id("clip-03").method == "metadata"


def test_two_cameras_are_aligned_by_audio_when_clocks_disagree():
    # cam-b's clock is 5 s fast; its audio proves the true offset is 0.
    manifest = Manifest(clips=[
        _clip("clip-01", "cam-a", 1000.0, duration=20.0),
        _clip("clip-02", "cam-b", 1005.0, duration=20.0),
    ])
    rng = np.random.default_rng(3)
    shared = np.abs(rng.normal(0, 1.0, 2000))

    def envelope_of(clip: ClipInfo):
        return shared

    sync = build_sync_map(manifest, envelope_of=envelope_of)

    assert abs(sync.by_id("clip-01").global_start - sync.by_id("clip-02").global_start) < 0.2
    assert sync.by_id("clip-02").method == "audio"


def test_primary_camera_is_the_loud_one():
    from videoai.logic.sync import choose_primary_camera

    manifest = Manifest(clips=[
        _clip("clip-01", "cam-a", 1000.0),
        _clip("clip-02", "cam-b", 1000.0),
    ])
    quiet = np.full(500, 0.01)
    loud = np.full(500, 0.4)

    def envelope_of(clip: ClipInfo):
        return loud if clip.camera == "cam-b" else quiet

    assert choose_primary_camera(manifest, envelope_of) == "cam-b"


def test_primary_camera_override_is_honoured_and_validated():
    from videoai.logic.sync import choose_primary_camera

    manifest = Manifest(clips=[_clip("clip-01", "cam-a", 1000.0), _clip("clip-02", "cam-b", 1000.0)])
    envelope_of = lambda clip: np.full(10, 0.5)

    assert choose_primary_camera(manifest, envelope_of, override="cam-a") == "cam-a"
    try:
        choose_primary_camera(manifest, envelope_of, override="cam-z")
    except ValueError as error:
        assert "cam-z" in str(error)
    else:
        raise AssertionError("expected ValueError")


def test_primary_camera_without_audio_is_the_first_by_name():
    from videoai.logic.sync import choose_primary_camera

    manifest = Manifest(clips=[_clip("clip-01", "cam-b", None), _clip("clip-02", "cam-a", None)])

    assert choose_primary_camera(manifest, envelope_of=lambda clip: None) == "cam-a"


def test_overlaps_is_true_for_simultaneous_angles_and_false_for_sequential_takes():
    manifest = Manifest(clips=[
        _clip("clip-01", "cam-a", 1000.0, duration=20.0),
        _clip("clip-02", "cam-b", 1000.0, duration=20.0),
        _clip("clip-03", "cam-a", 1100.0, duration=20.0),
    ])

    sync = build_sync_map(manifest, envelope_of=lambda clip: None)

    assert sync.overlaps("clip-01", "clip-02", manifest) is True
    assert sync.overlaps("clip-01", "clip-03", manifest) is False
