from pathlib import Path

import docx

from videoai.core.project import read_brief, resolve_clip_dir


def test_resolve_clip_dir_prefers_input(tmp_path: Path):
    (tmp_path / "input").mkdir()
    (tmp_path / "video").mkdir()
    assert resolve_clip_dir(tmp_path) == tmp_path / "input"


def test_resolve_clip_dir_falls_back_to_video(tmp_path: Path):
    (tmp_path / "video").mkdir()
    assert resolve_clip_dir(tmp_path) == tmp_path / "video"


def test_resolve_clip_dir_falls_back_to_project_dir(tmp_path: Path):
    assert resolve_clip_dir(tmp_path) == tmp_path


def test_read_brief_returns_empty_string_when_nothing_present(tmp_path: Path):
    assert read_brief(tmp_path) == ""


def test_read_brief_includes_project_yaml_and_notes(tmp_path: Path):
    (tmp_path / "project.yaml").write_text("title: Slime review\n", encoding="utf-8")
    (tmp_path / "notes.md").write_text("Keep the laugh at the end.", encoding="utf-8")

    brief = read_brief(tmp_path)

    assert "Slime review" in brief
    assert "Keep the laugh at the end." in brief


def test_read_brief_reads_docx_from_description(tmp_path: Path):
    description = tmp_path / "description"
    description.mkdir()
    document = docx.Document()
    document.add_paragraph("Pimple Popping Stress Toy")
    document.add_paragraph("Refillable, two in one.")
    document.save(description / "product.docx")

    brief = read_brief(tmp_path)

    assert "Pimple Popping Stress Toy" in brief
    assert "Refillable, two in one." in brief


def test_read_brief_reads_markdown_and_text_from_description(tmp_path: Path):
    description = tmp_path / "description"
    description.mkdir()
    (description / "a.md").write_text("markdown content", encoding="utf-8")
    (description / "b.txt").write_text("plain content", encoding="utf-8")

    brief = read_brief(tmp_path)

    assert "markdown content" in brief
    assert "plain content" in brief


def test_read_brief_ignores_macos_metadata_and_images(tmp_path: Path):
    description = tmp_path / "description"
    description.mkdir()
    (description / ".DS_Store").write_bytes(b"junk")
    (description / "._notes.md").write_bytes(b"junk")
    (description / "photo.jpeg").write_bytes(b"\xff\xd8\xff")
    (description / "real.md").write_text("real content", encoding="utf-8")

    brief = read_brief(tmp_path)

    assert brief.strip() == "real content"


def test_read_brief_survives_an_unreadable_docx(tmp_path: Path):
    description = tmp_path / "description"
    description.mkdir()
    (description / "broken.docx").write_bytes(b"not a real docx")
    (description / "real.md").write_text("real content", encoding="utf-8")

    brief = read_brief(tmp_path)

    assert "real content" in brief
    assert "broken.docx" in brief
