"""Machine enforcement for the repository's production contract."""
from __future__ import annotations

from pathlib import Path

import yaml

from videoai.core.models import Timeline


def contract_path(project_dir: Path) -> Path:
    local = project_dir / "production-contract.yaml"
    if local.is_file():
        return local
    root = Path(__file__).resolve().parents[2] / "production-contract.yaml"
    if not root.is_file():
        raise RuntimeError(f"production contract is missing: {root}")
    return root


def load_contract(project_dir: Path) -> dict:
    path = contract_path(project_dir)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict) or not data.get("version"):
        raise RuntimeError(f"invalid production contract: {path}")
    return data


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
    for feature, required in (contract.get("required_features") or {}).items():
        if required and not features.get(feature):
            failures.append(f"required feature missing: {feature}")
    quality = report.get("quality") or {}
    expected_quality = contract.get("quality") or {}
    if expected_quality.get("source") and quality.get("source") != expected_quality["source"]:
        failures.append("delivery was not built from original sources")
    maximum = expected_quality.get("maximum_lossy_video_generations")
    if maximum is not None and quality.get("lossy_video_generations", 999) > maximum:
        failures.append(
            f"lossy video generations={quality.get('lossy_video_generations')}, "
            f"maximum={maximum}"
        )
    if failures:
        raise RuntimeError("production contract failed:\n- " + "\n- ".join(failures))
