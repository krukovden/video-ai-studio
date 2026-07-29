import json
from pathlib import Path

from typer.testing import CliRunner

from videoai.cli import app

runner = CliRunner()


def test_run_produces_draft_from_a_folder_of_clips(tmp_path: Path, make_clip, monkeypatch):
    project = tmp_path / "project"
    (project / "input").mkdir(parents=True)
    make_clip("a.mp4", seconds=6.0).rename(project / "input" / "a.mp4")
    (project / "input" / "project.yaml").write_text("title: Test review\n", encoding="utf-8")

    config_path = tmp_path / "config.yaml"
    config_path.write_text("providers:\n  asr: mock\n  llm: mock\n", encoding="utf-8")

    words_payload = [
        {"text": "hello", "start": 0.5, "end": 0.9},
        {"text": "everyone", "start": 0.95, "end": 1.5},
        {"text": "look", "start": 3.0, "end": 3.3},
        {"text": "here", "start": 3.35, "end": 3.8},
    ]
    llm_payload = {
        "segments": [
            {"phrase_id": "clip-01#001", "content": "intro", "delivery_score": 9,
             "visual_score": 8, "emotion": "excited", "is_failed_take": False,
             "shorts_candidate": True},
            {"phrase_id": "clip-01#002", "content": "demo", "delivery_score": 7,
             "visual_score": 7, "emotion": "calm", "is_failed_take": False,
             "shorts_candidate": False},
        ],
        "title": "Test Review",
        "description": "A test review.",
        "tags": ["toys"],
        "sections": [
            {"name": "Hook", "goal": "open", "phrase_ids": ["clip-01#001"]},
            {"name": "Body", "goal": "show", "phrase_ids": ["clip-01#002"]},
        ],
    }
    llm_path = tmp_path / "llm.json"
    llm_path.write_text(json.dumps(llm_payload), encoding="utf-8")
    monkeypatch.setenv("VIDEOAI_MOCK_LLM", str(llm_path))

    # The mock ASR reads its sidecar next to the extracted audio, which ingest
    # creates on the first run; seed it by running ingest alone first.
    result = runner.invoke(app, ["run", str(project), "--config", str(config_path), "--stage", "ingest"])
    assert result.exit_code == 0, result.output
    sidecar = project / "work" / "media" / "clip-01.words.json"
    sidecar.write_text(json.dumps(words_payload), encoding="utf-8")

    result = runner.invoke(app, ["run", str(project), "--config", str(config_path)])

    assert result.exit_code == 0, result.output
    draft = project / "output" / "draft.mp4"
    assert draft.exists()
    assert (project / "work" / "05-timeline.json").exists()
    assert "render_draft" in result.output


def test_second_run_skips_cached_stages(tmp_path: Path, make_clip, monkeypatch):
    project = tmp_path / "project"
    (project / "input").mkdir(parents=True)
    make_clip("a.mp4", seconds=6.0).rename(project / "input" / "a.mp4")
    config_path = tmp_path / "config.yaml"
    config_path.write_text("providers:\n  asr: mock\n  llm: mock\n", encoding="utf-8")
    llm_path = tmp_path / "llm.json"
    llm_path.write_text(json.dumps({
        "segments": [{"phrase_id": "clip-01#001", "content": "intro", "delivery_score": 9,
                      "visual_score": 8, "emotion": "excited", "is_failed_take": False,
                      "shorts_candidate": False}],
        "title": "T", "description": "D", "tags": [],
        "sections": [{"name": "Hook", "goal": "open", "phrase_ids": ["clip-01#001"]}],
    }), encoding="utf-8")
    monkeypatch.setenv("VIDEOAI_MOCK_LLM", str(llm_path))

    runner.invoke(app, ["run", str(project), "--config", str(config_path), "--stage", "ingest"])
    (project / "work" / "media" / "clip-01.words.json").write_text(
        json.dumps([{"text": "hello", "start": 0.5, "end": 0.9},
                    {"text": "everyone", "start": 0.95, "end": 1.5}]),
        encoding="utf-8",
    )
    runner.invoke(app, ["run", str(project), "--config", str(config_path)])

    result = runner.invoke(app, ["run", str(project), "--config", str(config_path)])

    assert result.exit_code == 0, result.output
    assert "nothing to do" in result.output.lower()


def test_stages_command_lists_pipeline_order():
    result = runner.invoke(app, ["stages"])
    assert result.exit_code == 0
    for stage_id in ("ingest", "quality", "sync", "transcribe", "analyze", "plan", "render_draft"):
        assert stage_id in result.output


def test_run_with_video_folder_layout_produces_draft(tmp_path: Path, make_clip, monkeypatch):
    """The creator's real folders are `video/` + `project.yaml` + `description/`,
    not `input/`; this is the layout that used to abort with "no input directory"."""
    project = tmp_path / "project"
    (project / "video").mkdir(parents=True)
    make_clip("a.mp4", seconds=6.0).rename(project / "video" / "a.mp4")
    project.joinpath("project.yaml").write_text("title: Test review\n", encoding="utf-8")
    (project / "description").mkdir()
    (project / "description" / "notes.md").write_text("Keep it upbeat.\n", encoding="utf-8")

    config_path = tmp_path / "config.yaml"
    config_path.write_text("providers:\n  asr: mock\n  llm: mock\n", encoding="utf-8")

    words_payload = [
        {"text": "hello", "start": 0.5, "end": 0.9},
        {"text": "everyone", "start": 0.95, "end": 1.5},
        {"text": "look", "start": 3.0, "end": 3.3},
        {"text": "here", "start": 3.35, "end": 3.8},
    ]
    llm_payload = {
        "segments": [
            {"phrase_id": "clip-01#001", "content": "intro", "delivery_score": 9,
             "visual_score": 8, "emotion": "excited", "is_failed_take": False,
             "shorts_candidate": True},
            {"phrase_id": "clip-01#002", "content": "demo", "delivery_score": 7,
             "visual_score": 7, "emotion": "calm", "is_failed_take": False,
             "shorts_candidate": False},
        ],
        "title": "Test Review",
        "description": "A test review.",
        "tags": ["toys"],
        "sections": [
            {"name": "Hook", "goal": "open", "phrase_ids": ["clip-01#001"]},
            {"name": "Body", "goal": "show", "phrase_ids": ["clip-01#002"]},
        ],
    }
    llm_path = tmp_path / "llm.json"
    llm_path.write_text(json.dumps(llm_payload), encoding="utf-8")
    monkeypatch.setenv("VIDEOAI_MOCK_LLM", str(llm_path))

    result = runner.invoke(app, ["run", str(project), "--config", str(config_path), "--stage", "ingest"])
    assert result.exit_code == 0, result.output
    sidecar = project / "work" / "media" / "clip-01.words.json"
    sidecar.write_text(json.dumps(words_payload), encoding="utf-8")

    result = runner.invoke(app, ["run", str(project), "--config", str(config_path)])

    assert result.exit_code == 0, result.output
    draft = project / "output" / "draft.mp4"
    assert draft.exists()


def test_run_indexes_clips_from_per_camera_subfolders(tmp_path: Path, make_clip):
    project = tmp_path / "project"
    (project / "video" / "cam-a").mkdir(parents=True)
    (project / "video" / "cam-b").mkdir(parents=True)
    make_clip("a.mp4", seconds=3.0).rename(project / "video" / "cam-a" / "a.mp4")
    make_clip("b.mp4", seconds=3.0).rename(project / "video" / "cam-b" / "b.mp4")

    config_path = tmp_path / "config.yaml"
    config_path.write_text("providers:\n  asr: mock\n  llm: mock\n", encoding="utf-8")

    result = runner.invoke(app, ["run", str(project), "--config", str(config_path), "--stage", "ingest"])

    assert result.exit_code == 0, result.output
    manifest = json.loads((project / "work" / "01-manifest.json").read_text(encoding="utf-8"))
    cameras = {clip["camera"] for clip in manifest["clips"]}
    assert cameras == {"cam-a", "cam-b"}


def test_run_fails_with_no_video_files_anywhere(tmp_path: Path):
    project = tmp_path / "project"
    project.mkdir(parents=True)
    project.joinpath("project.yaml").write_text("title: Empty\n", encoding="utf-8")

    config_path = tmp_path / "config.yaml"
    config_path.write_text("providers:\n  asr: mock\n  llm: mock\n", encoding="utf-8")

    result = runner.invoke(app, ["run", str(project), "--config", str(config_path)])

    assert result.exit_code != 0
    # Rich wraps the error panel to terminal width and can break a long temp path
    # mid-word without inserting anything; strip padding and rejoin lines so the
    # assertion isn't sensitive to where a fixed-width panel happened to wrap.
    normalized = "".join(line.strip() for line in result.output.replace("│", "").splitlines())
    assert str(project) in normalized
    assert "input/" in normalized
    assert "video/" in normalized


def test_editing_brief_between_runs_reruns_analysis(tmp_path: Path, make_clip, monkeypatch):
    project = tmp_path / "project"
    (project / "video").mkdir(parents=True)
    make_clip("a.mp4", seconds=6.0).rename(project / "video" / "a.mp4")
    (project / "description").mkdir()
    notes = project / "description" / "notes.md"
    notes.write_text("Short brief.\n", encoding="utf-8")

    config_path = tmp_path / "config.yaml"
    config_path.write_text("providers:\n  asr: mock\n  llm: mock\n", encoding="utf-8")
    llm_path = tmp_path / "llm.json"
    llm_path.write_text(json.dumps({
        "segments": [{"phrase_id": "clip-01#001", "content": "intro", "delivery_score": 9,
                      "visual_score": 8, "emotion": "excited", "is_failed_take": False,
                      "shorts_candidate": False}],
        "title": "T", "description": "D", "tags": [],
        "sections": [{"name": "Hook", "goal": "open", "phrase_ids": ["clip-01#001"]}],
    }), encoding="utf-8")
    monkeypatch.setenv("VIDEOAI_MOCK_LLM", str(llm_path))

    runner.invoke(app, ["run", str(project), "--config", str(config_path), "--stage", "ingest"])
    (project / "work" / "media" / "clip-01.words.json").write_text(
        json.dumps([{"text": "hello", "start": 0.5, "end": 0.9},
                    {"text": "everyone", "start": 0.95, "end": 1.5}]),
        encoding="utf-8",
    )
    first = runner.invoke(app, ["run", str(project), "--config", str(config_path)])
    assert first.exit_code == 0, first.output

    cached = runner.invoke(app, ["run", str(project), "--config", str(config_path)])
    assert "nothing to do" in cached.output.lower()

    # Different length so the fingerprint changes even if the filesystem's mtime
    # resolution is too coarse to register the edit within the same test run.
    notes.write_text("A much longer brief that changes the creative direction entirely.\n", encoding="utf-8")

    result = runner.invoke(app, ["run", str(project), "--config", str(config_path)])

    assert result.exit_code == 0, result.output
    assert "analyze" in result.output
