"""Keeping a written description of every source clip.

Watching footage is the one thing in this pipeline that costs money, and what a
clip contains does not change unless the clip does. So each file is described
once and the answer is kept; a later run reads it instead of paying for it
again, and only footage that is genuinely new or replaced is ever sent.

The cache is keyed by the file's identity, not by its `clip-NN` slot. Ids are
positional and shift whenever a clip is added, so keying by them would re-bill
the whole shoot for adding one video at the front.
"""
from __future__ import annotations

from videoai.core.models import ClipNote, ClipNotes


def clips_needing_description(
    cached: ClipNotes, clips: list[tuple[str, str]]
) -> list[str]:
    """Which clip ids have no usable description yet.

    `clips` is (clip_id, source_key) for everything in the manifest. A key that
    is already known is skipped no matter which slot it now occupies.
    """
    known = cached.keys()
    return [clip_id for clip_id, source_key in clips if source_key not in known]


def merge_notes(cached: ClipNotes, fresh: list[ClipNote]) -> ClipNotes:
    """The cache with `fresh` written into it, replacing any note for the same file."""
    replaced = {note.source_key for note in fresh}
    kept = [note for note in cached.notes if note.source_key not in replaced]
    return ClipNotes(provider=cached.provider, notes=kept + list(fresh))


def render_notes_markdown(notes: ClipNotes) -> str:
    """The creator-facing log: what is in every video, and when things happen.

    Written so it can be read on its own, away from the pipeline — this is the
    record of what the footage contains, and the thing to look at before asking
    for a different edit.
    """
    lines = [
        "# What is in each video",
        "",
        "Written once per file by a model that watched it. Kept and reused: a",
        "re-run does not send the same footage again, so re-editing costs",
        "nothing. Delete a clip's section (or the whole file) to have it",
        "described again.",
        "",
    ]
    if notes.provider:
        lines += [f"Described by: `{notes.provider}`", ""]

    total_in = sum(note.input_tokens for note in notes.notes)
    total_out = sum(note.output_tokens for note in notes.notes)
    if total_in or total_out:
        lines += [
            f"Cost so far: {total_in:,} input + {total_out:,} output tokens.",
            "",
        ]

    for note in notes.notes:
        heading = f"## {note.clip_id}"
        if note.source_name:
            heading += f" — {note.source_name}"
        lines += [heading, "", f"Length: {note.duration:.1f}s", ""]
        if note.summary:
            lines += [note.summary, ""]
        else:
            lines += ["*(not described)*", ""]
        if note.events:
            lines += ["| time | what happens | where |", "|---|---|---|"]
            for event in sorted(note.events, key=lambda item: item.at):
                where = event.where or "—"
                lines.append(f"| {event.at:.1f}s | {event.what} | {where} |")
            lines.append("")
    return "\n".join(lines)
