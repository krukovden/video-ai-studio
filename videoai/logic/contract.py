"""Machine enforcement for the repository's production contract.

Every rule in `production-contract.yaml` is checked here against a measured
production report. A rule this module cannot check does not belong in that file:
an unenforced rule reads as a guarantee and is really a wish.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import yaml

from videoai.core.models import Timeline
from videoai.core.store import hash_parts

CONTRACT_FILENAME = "production-contract.yaml"

# Lossless x264 of real footage lands somewhere near a third of raw 4:2:0, and
# delivery keeps the segments, the picture master and the graphics track on disk
# at the same time. The factor below is deliberately an estimate: preflight is
# here to catch "there is 2 GB free" in seconds, not to predict the byte count.
_LOSSLESS_RAW_FRACTION = 0.35
_INTERMEDIATE_COPIES = 2.2


def contract_path(project_dir: Path) -> Path:
    local = project_dir / CONTRACT_FILENAME
    if local.is_file():
        return local
    root = Path(__file__).resolve().parents[2] / CONTRACT_FILENAME
    if not root.is_file():
        raise RuntimeError(f"production contract is missing: {root}")
    return root


def contract_text(project_dir: Path) -> str:
    return contract_path(project_dir).read_text(encoding="utf-8")


def contract_hash(project_dir: Path) -> str:
    """A digest of the resolved contract's content.

    Mixed into the polish stage fingerprint: editing a delivery rule has to
    re-run delivery, or the cached `final.mp4` was validated against rules that
    no longer exist.
    """
    return hash_parts(contract_text(project_dir))


def load_contract(project_dir: Path) -> dict:
    path = contract_path(project_dir)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict) or not data.get("version"):
        raise RuntimeError(f"invalid production contract: {path}")
    return data


def fallback_output_name(contract: dict) -> str:
    """What a deliberately degraded render is allowed to be called."""
    policy = contract.get("failure_policy") or {}
    name = policy.get("fallback_output_name")
    return name if isinstance(name, str) and name else "preview-fallback.mp4"


def has_closing_beat(timeline: Timeline) -> bool:
    language = " ".join(
        f"{clip.beat} {clip.quote} {clip.reason}"
        for clip in timeline.clips[-3:]
        if not clip.is_insert
    ).lower()
    signals = (
        "thank", "thanks", "bye", "goodbye", "see you", "recommend",
        "verdict", "final thought", "closing", "wrap up", "rating",
    )
    return any(signal in language for signal in signals)


def estimated_delivery_bytes(
    frame: tuple[int, int], fps: float, seconds: float
) -> int:
    """Roughly how much scratch space the delivery intermediates will need."""
    width, height = frame
    raw_per_second = width * height * 1.5 * max(1.0, fps)
    return int(raw_per_second * max(0.0, seconds) * _LOSSLESS_RAW_FRACTION * _INTERMEDIATE_COPIES)


def free_bytes(directory: Path) -> int:
    probe_at = directory
    while not probe_at.exists() and probe_at != probe_at.parent:
        probe_at = probe_at.parent
    return shutil.disk_usage(probe_at).free


def preflight(
    contract: dict,
    *,
    frame: tuple[int, int],
    output_height: int,
    music_dir: Path,
    track_count: int,
    closing_beat: bool,
    estimated_bytes: int,
    available_bytes: int,
) -> None:
    """Refuse a delivery that cannot satisfy the contract, before any cutting.

    Everything checked here is knowable from one source probe, the timeline and a
    directory listing. The alternative is discovering that the contract wants
    1080p from a 4:3 source, or that the disk is full, after a 4K render.
    """
    failures: list[str] = []
    required_output = contract.get("required_output") or {}
    width, height = frame
    wanted_width = required_output.get("width")
    wanted_height = required_output.get("height")
    if wanted_height is not None and height != wanted_height:
        failures.append(
            f"the contract requires a {wanted_height}px-tall delivery but "
            f"polish.output_height={output_height} produces {height}px"
        )
    if wanted_width is not None and width != wanted_width:
        failures.append(
            f"the contract requires a {wanted_width}x{wanted_height} delivery, but "
            f"the source's display aspect at {height}px tall gives {width}x{height}; "
            "either the footage is not that aspect or the contract's width is wrong"
        )
    required_features = contract.get("required_features") or {}
    if required_features.get("music") and track_count == 0:
        failures.append(
            f"the contract requires background music but {music_dir} holds no "
            "playable track"
        )
    if required_features.get("closing_beat") and not closing_beat:
        failures.append(
            "the contract requires a closing story beat, and the planned timeline "
            "ends without one; re-plan so the last section signs off"
        )
    if estimated_bytes and available_bytes < estimated_bytes:
        failures.append(
            f"delivery needs roughly {estimated_bytes / 1e9:.1f} GB of scratch space "
            f"for its lossless intermediates and only {available_bytes / 1e9:.1f} GB "
            "is free; clear space or lower polish.output_height"
        )
    if failures:
        raise RuntimeError(
            "production preflight failed:\n- " + "\n- ".join(failures)
        )


def validate_production_report(report: dict, project_dir: Path) -> None:
    contract = load_contract(project_dir)
    failures: list[str] = []
    required_output = contract.get("required_output") or {}
    if report.get("width") != required_output.get("width"):
        failures.append(
            f"width={report.get('width')}, required={required_output.get('width')}"
        )
    if report.get("height") != required_output.get("height"):
        failures.append(
            f"height={report.get('height')}, required={required_output.get('height')}"
        )
    features = report.get("features") or {}
    if required_output.get("full_decode") and not features.get("full_decode"):
        failures.append("the delivered file did not decode end to end")
    for feature, required in (contract.get("required_features") or {}).items():
        if not required:
            continue
        measured = features.get(feature)
        if not measured:
            failures.append(f"required feature missing: {feature}")
            continue
        # A rule may also name the floor its measurement has to clear, which is
        # how "ducking is enabled" becomes "the bed measurably drops under speech".
        if isinstance(required, dict) and required.get("minimum_db") is not None:
            minimum = float(required["minimum_db"])
            if float(measured) < minimum:
                failures.append(
                    f"{feature} measured {float(measured):.2f} dB, "
                    f"the contract requires at least {minimum:.2f} dB"
                )
    quality = report.get("quality") or {}
    expected_quality = contract.get("quality") or {}
    if expected_quality.get("source") and quality.get("source") != expected_quality["source"]:
        failures.append("delivery was not built from original sources")
    leaked = quality.get("proxy_inputs") or []
    if leaked:
        failures.append(
            "these delivery segments were cut from proxies rather than originals: "
            + ", ".join(str(path) for path in leaked)
        )
    maximum = expected_quality.get("maximum_lossy_video_generations")
    if maximum is not None and quality.get("lossy_video_generations", 999) > maximum:
        failures.append(
            f"lossy video generations={quality.get('lossy_video_generations')}, "
            f"maximum={maximum}"
        )
    if failures:
        raise RuntimeError("production contract failed:\n- " + "\n- ".join(failures))
