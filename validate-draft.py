"""Prove a rendered draft really contains the source footage its timeline claims.

For every timeline segment this pulls one frame out of the draft and the frame
the timeline says it came from out of the original source file, and compares
them. A draft can play perfectly and still be wrong — wrong clip, wrong offset,
silently dropped segment — and only this comparison catches that.

Run: uv run python validate-draft.py <project-dir>
Exit code 0 when every segment matches, 1 otherwise.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

MATCH_THRESHOLD = 0.90   # normalised correlation between the two frames
SAMPLE_INSET = 0.35      # sample this far into a segment, away from its edges


def probe(path: Path) -> dict:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json",
         "-show_format", "-show_streams", str(path)],
        capture_output=True, text=True, check=True,
    )
    return json.loads(result.stdout)


def grab(path: Path, at: float, dst: Path, height: int = 180) -> bool:
    result = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{at:.3f}", "-i", str(path),
         "-frames:v", "1", "-vf", f"scale=-2:{height}", str(dst)],
        capture_output=True,
    )
    return result.returncode == 0 and dst.exists() and dst.stat().st_size > 0


def similarity(first: Path, second: Path) -> float:
    """Normalised correlation of two frames, resized to a common small size.

    Compared on the grayscale image rather than a histogram: a histogram matches
    any frame with the same colours, which is most frames of the same scene.
    """
    a = cv2.imread(str(first), cv2.IMREAD_GRAYSCALE)
    b = cv2.imread(str(second), cv2.IMREAD_GRAYSCALE)
    if a is None or b is None:
        return 0.0
    size = (160, 90)
    a = cv2.resize(a, size).astype(np.float64)
    b = cv2.resize(b, size).astype(np.float64)
    a -= a.mean()
    b -= b.mean()
    denominator = float(np.sqrt((a * a).sum() * (b * b).sum()))
    if denominator == 0:
        return 1.0 if np.array_equal(a, b) else 0.0
    return float((a * b).sum() / denominator)


def main() -> int:
    project = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    work = project / "work"
    timeline_path = work / "05-timeline.json"
    manifest_path = work / "01-manifest.json"
    draft_path = project / "output" / "draft.mp4"

    for path in (timeline_path, manifest_path, draft_path):
        if not path.exists():
            print(f"MISSING: {path}")
            return 1

    timeline = json.loads(timeline_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sources = {clip["clip_id"]: clip for clip in manifest["clips"]}

    info = probe(draft_path)
    streams = {s["codec_type"]: s for s in info["streams"]}
    draft_duration = float(info["format"]["duration"])
    expected = sum(c["dur"] for c in timeline["clips"])

    print("=== FILE ===")
    print(f"  path       {draft_path}")
    print(f"  size       {draft_path.stat().st_size / 1_000_000:.1f} MB")
    print(f"  duration   {draft_duration:.2f}s (timeline says {expected:.2f}s, "
          f"drift {abs(draft_duration - expected):.2f}s)")
    if "video" in streams:
        v = streams["video"]
        print(f"  video      {v['codec_name']} {v['width']}x{v['height']}")
    else:
        print("  video      MISSING")
    if "audio" in streams:
        a = streams["audio"]
        print(f"  audio      {a['codec_name']} {a['sample_rate']}Hz {a['channels']}ch")
    else:
        print("  audio      MISSING")

    problems: list[str] = []
    if "video" not in streams:
        problems.append("draft has no video stream")
    if "audio" not in streams:
        problems.append("draft has no audio stream")

    # Where a segment really begins in the draft is the sum of the ENCODED
    # durations before it, not the sum of the requested ones: the encoder emits
    # a whole number of frames, so each segment runs a fraction long and the
    # difference accumulates. Reading the segment files keeps this check honest.
    segments_dir = work / "segments"
    segment_files = sorted(segments_dir.glob("seg-*.mp4"))
    positions: list[float] = []
    cursor = 0.0
    for path in segment_files:
        positions.append(cursor)
        cursor += float(probe(path)["format"]["duration"])
    encoded_total = cursor
    drift = encoded_total - expected
    print(f"  encoded    {encoded_total:.2f}s across {len(segment_files)} segments "
          f"(drift vs timeline {drift:+.2f}s)")
    if len(segment_files) != len(timeline["clips"]):
        problems.append(
            f"{len(segment_files)} segment files for {len(timeline['clips'])} timeline clips")

    print("\n=== SEGMENTS: draft frame vs original source frame ===")
    matched = 0
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        for index, clip in enumerate(timeline["clips"]):
            source = sources.get(clip["src"])
            if source is None:
                problems.append(f"segment {index}: unknown source {clip['src']}")
                continue
            original = Path(source["path"])
            if not original.exists():
                problems.append(f"segment {index}: source file missing {original}")
                continue

            inset = min(SAMPLE_INSET, clip["dur"] / 3)
            base = positions[index] if index < len(positions) else clip["start"]
            draft_at = base + inset
            source_at = clip["offset"] + inset

            draft_frame = tmp_dir / f"d{index}.png"
            source_frame = tmp_dir / f"s{index}.png"
            if not grab(draft_path, draft_at, draft_frame):
                problems.append(f"segment {index}: could not read draft at {draft_at:.2f}s")
                continue
            if not grab(original, source_at, source_frame):
                problems.append(f"segment {index}: could not read source at {source_at:.2f}s")
                continue

            score = similarity(draft_frame, source_frame)
            ok = score >= MATCH_THRESHOLD
            matched += ok
            mark = "OK  " if ok else "FAIL"
            print(f"  {mark} seg {index:2d}  draft@{draft_at:6.2f}s  "
                  f"{original.name}@{source_at:6.2f}s  similarity {score:.3f}")
            if not ok:
                problems.append(
                    f"segment {index}: draft frame does not match {original.name} "
                    f"at {source_at:.2f}s (similarity {score:.3f})")

    total = len(timeline["clips"])
    print(f"\n=== RESULT ===")
    print(f"  segments matched to their source: {matched}/{total}")
    if problems:
        print("  PROBLEMS:")
        for problem in problems:
            print(f"    - {problem}")
        return 1
    print("  every segment contains footage from the file and offset the timeline claims")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
