import json
from pathlib import Path

from typer.testing import CliRunner

from videoai.cli import app

runner = CliRunner()


def _sidecar_for(project: Path, clip_id: str) -> Path:
    """Where the mock ASR expects this clip's words: next to the extracted audio.

    Derived media is named after the SOURCE's identity, not after the positional
    clip id, so the manifest is the only place that knows the file name.
    """
    manifest = json.loads((project / "work" / "01-manifest.json").read_text(encoding="utf-8"))
    clip = next(entry for entry in manifest["clips"] if entry["clip_id"] == clip_id)
    return Path(clip["audio_path"]).with_suffix(".words.json")


def _write_words(project: Path, clip_id: str, words: list[dict]) -> Path:
    sidecar = _sidecar_for(project, clip_id)
    sidecar.write_text(json.dumps(words), encoding="utf-8")
    return sidecar


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
    _write_words(project, "clip-01", words_payload)

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
    _write_words(project, "clip-01", [{"text": "hello", "start": 0.5, "end": 0.9},
                                      {"text": "everyone", "start": 0.95, "end": 1.5}])
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
    _write_words(project, "clip-01", words_payload)

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


# --- Finding I5: a stage failure must name the stage and how to re-run it,
# instead of surfacing as a raw traceback ---


def test_stage_failure_names_the_stage_and_exits_non_zero(tmp_path: Path, make_clip, monkeypatch):
    project = tmp_path / "project"
    (project / "video").mkdir(parents=True)
    make_clip("a.mp4", seconds=3.0).rename(project / "video" / "a.mp4")
    config_path = tmp_path / "config.yaml"
    config_path.write_text("providers:\n  asr: mock\n  llm: mock\n", encoding="utf-8")

    # No mock ASR sidecar exists, so `transcribe` raises FileNotFoundError.
    result = runner.invoke(app, ["run", str(project), "--config", str(config_path)])

    assert result.exit_code != 0
    combined = result.output + (result.stderr if result.stderr_bytes else "")
    assert "transcribe" in combined
    assert "--stage transcribe" in combined
    assert "Traceback" not in combined


def test_debug_flag_keeps_the_traceback(tmp_path: Path, make_clip):
    project = tmp_path / "project"
    (project / "video").mkdir(parents=True)
    make_clip("a.mp4", seconds=3.0).rename(project / "video" / "a.mp4")
    config_path = tmp_path / "config.yaml"
    config_path.write_text("providers:\n  asr: mock\n  llm: mock\n", encoding="utf-8")

    result = runner.invoke(app, ["run", str(project), "--config", str(config_path), "--debug"])

    assert result.exit_code != 0
    assert isinstance(result.exception, Exception)
    assert "transcribe" in str(result.exception)


def _executed_stages(output: str) -> set[str]:
    for line in output.splitlines():
        if line.startswith("Executed: "):
            return {stage.strip() for stage in line.removeprefix("Executed: ").split(",")}
    return set()


ALL_STAGE_IDS = {"ingest", "quality", "sync", "transcribe", "analyze", "plan", "render_draft"}


def test_editing_brief_reruns_only_analyze_plan_and_render(tmp_path: Path, make_clip, monkeypatch):
    """A brief edit is a normal working-loop action; it must not force re-probing,
    re-scoring and re-transcribing every clip — only the stages that read the brief
    (analyze, plan) and whatever depends on their output (render_draft, via the
    timeline) should re-run."""
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
    _write_words(project, "clip-01", [{"text": "hello", "start": 0.5, "end": 0.9},
                                      {"text": "everyone", "start": 0.95, "end": 1.5}])

    first = runner.invoke(app, ["run", str(project), "--config", str(config_path)])
    assert first.exit_code == 0, first.output
    # `--stage ingest` already ran (and cached) ingest above with an unchanged
    # media fingerprint, so this full run's cache correctly skips it.
    assert _executed_stages(first.output) == ALL_STAGE_IDS - {"ingest"}

    cached = runner.invoke(app, ["run", str(project), "--config", str(config_path)])
    assert cached.exit_code == 0, cached.output
    assert "nothing to do" in cached.output.lower()
    assert _executed_stages(cached.output) == set()

    # Different length so the fingerprint changes even if the filesystem's mtime
    # resolution is too coarse to register the edit within the same test run.
    notes.write_text("A much longer brief that changes the creative direction entirely.\n", encoding="utf-8")
    # A different brief produces a different edit; the mock LLM has to say so,
    # otherwise the identical timeline it returns correctly leaves the render cached.
    llm_path.write_text(json.dumps({
        "segments": [{"phrase_id": "clip-01#001", "content": "a different reading", "delivery_score": 7,
                      "visual_score": 8, "emotion": "calm", "is_failed_take": False,
                      "shorts_candidate": False}],
        "title": "T2", "description": "D2", "tags": [],
        "sections": [{"name": "Opening", "goal": "open differently", "phrase_ids": ["clip-01#001"]}],
    }), encoding="utf-8")

    after_brief_edit = runner.invoke(app, ["run", str(project), "--config", str(config_path)])
    assert after_brief_edit.exit_code == 0, after_brief_edit.output
    assert _executed_stages(after_brief_edit.output) == {"analyze", "plan", "render_draft"}

    unchanged_again = runner.invoke(app, ["run", str(project), "--config", str(config_path)])
    assert unchanged_again.exit_code == 0, unchanged_again.output
    assert "nothing to do" in unchanged_again.output.lower()
    assert _executed_stages(unchanged_again.output) == set()


def test_adding_a_video_file_reruns_every_stage(tmp_path: Path, make_clip, monkeypatch):
    """Unlike a brief edit, a new clip changes what every stage sees (a new clip to
    probe, score, sync and potentially transcribe), so it must invalidate everything."""
    project = tmp_path / "project"
    (project / "video" / "cam-a").mkdir(parents=True)
    (project / "video" / "cam-b").mkdir(parents=True)
    make_clip("a.mp4", seconds=6.0).rename(project / "video" / "cam-a" / "a.mp4")
    make_clip("b.mp4", seconds=3.0).rename(project / "video" / "cam-b" / "b.mp4")

    # Pin the primary camera so only cam-a's clip needs a mock ASR sidecar; cam-b
    # is a secondary angle and is never transcribed.
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "providers:\n  asr: mock\n  llm: mock\nsync:\n  primary_camera: cam-a\n", encoding="utf-8"
    )
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
    manifest = json.loads((project / "work" / "01-manifest.json").read_text(encoding="utf-8"))
    primary_clip_id = next(c["clip_id"] for c in manifest["clips"] if c["camera"] == "cam-a")
    _write_words(project, primary_clip_id, [{"text": "hello", "start": 0.5, "end": 0.9},
                                            {"text": "everyone", "start": 0.95, "end": 1.5}])

    first = runner.invoke(app, ["run", str(project), "--config", str(config_path)])
    assert first.exit_code == 0, first.output
    # `--stage ingest` already ran (and cached) ingest above with an unchanged
    # media fingerprint, so this full run's cache correctly skips it.
    assert _executed_stages(first.output) == ALL_STAGE_IDS - {"ingest"}

    cached = runner.invoke(app, ["run", str(project), "--config", str(config_path)])
    assert "nothing to do" in cached.output.lower()

    # A new clip on the secondary camera: no new sidecar needed (it's never
    # transcribed), but it must still invalidate every stage's media fingerprint.
    make_clip("c.mp4", seconds=2.0).rename(project / "video" / "cam-b" / "c.mp4")

    after_new_clip = runner.invoke(app, ["run", str(project), "--config", str(config_path)])

    assert after_new_clip.exit_code == 0, after_new_clip.output
    assert _executed_stages(after_new_clip.output) == ALL_STAGE_IDS


def test_changing_a_render_setting_reruns_only_the_render(tmp_path: Path, make_clip, monkeypatch):
    """Finding I1 end to end: a setting a stage reads is one of its inputs. Before
    the fix `draft_crf` was invisible to every fingerprint and the pipeline said
    "nothing to do"."""
    project = tmp_path / "project"
    (project / "video").mkdir(parents=True)
    make_clip("a.mp4", seconds=6.0).rename(project / "video" / "a.mp4")

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
    _write_words(project, "clip-01", [{"text": "hello", "start": 0.5, "end": 0.9},
                                      {"text": "everyone", "start": 0.95, "end": 1.5}])
    first = runner.invoke(app, ["run", str(project), "--config", str(config_path)])
    assert first.exit_code == 0, first.output
    cached = runner.invoke(app, ["run", str(project), "--config", str(config_path)])
    assert "nothing to do" in cached.output.lower()

    config_path.write_text(
        "providers:\n  asr: mock\n  llm: mock\nrender:\n  draft_crf: 30\n", encoding="utf-8"
    )

    result = runner.invoke(app, ["run", str(project), "--config", str(config_path)])

    assert result.exit_code == 0, result.output
    assert _executed_stages(result.output) == {"render_draft"}


def test_single_stage_run_warns_about_stale_downstream_stages(tmp_path: Path, make_clip, monkeypatch):
    """`--stage <id>` always re-runs that one stage regardless of cache state, which
    can leave `output/draft.mp4` built from an older timeline with nothing to say
    so. A single-stage run must name the now-stale downstream stages."""
    project = tmp_path / "project"
    (project / "video").mkdir(parents=True)
    make_clip("a.mp4", seconds=6.0).rename(project / "video" / "a.mp4")

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
    _write_words(project, "clip-01", [{"text": "hello", "start": 0.5, "end": 0.9},
                                      {"text": "everyone", "start": 0.95, "end": 1.5}])

    full = runner.invoke(app, ["run", str(project), "--config", str(config_path)])
    assert full.exit_code == 0, full.output
    # Nothing was skipped bypassing the cache, so the whole pipeline is in sync:
    # no stale-downstream notice should appear after a full run.
    assert "now depend on stale input" not in full.output

    # Re-run only ingest. Nothing about the source changed, so its freshly
    # computed manifest fingerprint is identical to before and downstream
    # artifacts remain valid — no false-positive staleness warning.
    noop_single_stage = runner.invoke(
        app, ["run", str(project), "--config", str(config_path), "--stage", "ingest"]
    )
    assert noop_single_stage.exit_code == 0, noop_single_stage.output
    assert "now depend on stale input" not in noop_single_stage.output

    # Add a clip (changes the media fingerprint), then re-run only ingest instead of
    # the full pipeline: everything downstream of the manifest is now stale.
    make_clip("b.mp4", seconds=2.0).rename(project / "video" / "b.mp4")

    single_stage = runner.invoke(
        app, ["run", str(project), "--config", str(config_path), "--stage", "ingest"]
    )

    assert single_stage.exit_code == 0, single_stage.output
    assert "now depend on stale input" in single_stage.output
    for stage_id in ("quality", "sync", "transcribe", "analyze", "plan", "render_draft"):
        assert stage_id in single_stage.output
    # The notice only names stale stages; it must not run them itself.
    assert not _sidecar_for(project, "clip-02").exists()
    assert _executed_stages(single_stage.output) == {"ingest"}
