"""Print what the pipeline decided for a project, in reading order.

Run: uv run python inspect-result.py <project-dir>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def load(work: Path, name: str) -> dict | None:
    path = work / f"{name}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    project = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    work = project / "work"
    if not work.is_dir():
        print(f"no work directory in {project}")
        return 1

    manifest = load(work, "01-manifest")
    if manifest:
        total = sum(clip["duration"] for clip in manifest["clips"])
        print(f"SOURCE: {len(manifest['clips'])} clips, {total / 60:.1f} minutes")

    quality = load(work, "02-quality")
    if quality:
        unusable = [c["clip_id"] for c in quality["clips"] if not c["usable"]]
        unscored = [c["clip_id"] for c in quality["clips"] if not c.get("scored", True)]
        print(f"QUALITY: {len(unusable)} unusable, {len(unscored)} could not be scored")
        if unusable:
            print(f"  unusable: {', '.join(unusable)}")

    transcript = load(work, "03-transcript")
    if transcript:
        words = sum(len(c["words"]) for c in transcript["clips"])
        silent = [c["clip_id"] for c in transcript["clips"] if not c["words"]]
        print(f"SPEECH: {words} words recognised; {len(silent)} clips with no speech")

    phrases = load(work, "03b-phrases")
    takes = load(work, "03c-takes")
    if phrases:
        print(f"PHRASES: {len(phrases['phrases'])}")
    if takes and takes["groups"]:
        print(f"REPEATED TAKES: {len(takes['groups'])} groups")
        by_id = {p["phrase_id"]: p["text"] for p in (phrases or {}).get("phrases", [])}
        for group in takes["groups"][:10]:
            print(f"  {group['group_id']}:")
            for pid in group["phrase_ids"]:
                print(f"    {pid}: {by_id.get(pid, '')[:70]}")

    analysis = load(work, "04-analysis")
    if analysis:
        segments = analysis["segments"]
        failed = [s for s in segments if s["is_failed_take"]]
        shorts = [s for s in segments if s["shorts_candidate"]]
        helper = [s for s in segments if s.get("speaker") == "helper"]
        print(f"ANALYSIS: {len(segments)} phrases, {len(failed)} marked failed, "
              f"{len(shorts)} shorts candidates, {len(helper)} led by a helper")
        best = sorted(segments, key=lambda s: -s["delivery_score"])[:12]
        print("  strongest moments:")
        for s in best:
            print(f"    {s['phrase_id']} d={s['delivery_score']} v={s['visual_score']} "
                  f"{s['emotion']}: {s['text'][:60]}")

    plan = load(work, "05a-storyplan")
    if plan:
        print(f"\nTITLE: {plan['title']}")
        print(f"DESCRIPTION: {plan['description']}")
        print(f"TAGS: {', '.join(plan['tags'])}")
        print("SECTIONS:")
        for section in plan["sections"]:
            print(f"  [{section['name']}] {section['goal']}")
            print(f"      {len(section['phrase_ids'])} phrases")

    timeline = load(work, "05-timeline")
    if timeline:
        total = sum(c["dur"] for c in timeline["clips"])
        print(f"\nTIMELINE: {len(timeline['clips'])} cuts, {total:.1f}s "
              f"({total / 60:.1f} min)")
        for clip in timeline["clips"]:
            print(f"  {clip['src']} @{clip['start']:6.2f} +{clip['dur']:5.2f} "
                  f"[{clip['beat']}] {clip['quote'][:50]}")

    draft = load(work, "06-draft")
    if draft:
        print(f"\nDRAFT: {draft['path']} ({draft['duration']:.1f}s, "
              f"{draft['segment_count']} segments)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
