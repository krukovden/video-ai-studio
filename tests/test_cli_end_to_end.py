import json
from datetime import datetime
from pathlib import Path

from typer.testing import CliRunner

import videoai.core.project as project_module
from videoai.cli import app

runner = CliRunner()

# Polish is off throughout: these tests are about the edit pipeline, and the
# polish render would encode a second video and reach for the repository's real
# music library. It has its own tests.
MOCK_CONFIG = "providers:\n  asr: mock\n  llm: mock\npolish:\n  enabled: false\n"


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


def _clean_visual(count: int = 2) -> dict:
    """Visual-check findings approving the first `count` selected segments.

    Any full run now passes through the visual gate, and the gate refuses a
    segment the model said nothing about — so every fixture that renders a draft
    has to answer for the segments its plan selects. Findings for an index the
    timeline does not have are ignored, so one generous fixture serves timelines
    of different lengths.
    """
    return {
        str(index): {
            "adult_prominent": False,
            "child_visible": True,
            "unusable": False,
            "note": "the child is holding the toy",
        }
        for index in range(count)
    }


def test_run_produces_draft_from_a_folder_of_clips(tmp_path: Path, make_clip, monkeypatch):
    project = tmp_path / "project"
    (project / "input").mkdir(parents=True)
    make_clip("a.mp4", seconds=6.0).rename(project / "input" / "a.mp4")
    (project / "input" / "project.yaml").write_text("title: Test review\n", encoding="utf-8")

    config_path = tmp_path / "config.yaml"
    config_path.write_text(MOCK_CONFIG, encoding="utf-8")

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
        "visual": _clean_visual(2),
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
    config_path.write_text(MOCK_CONFIG, encoding="utf-8")
    llm_path = tmp_path / "llm.json"
    llm_path.write_text(json.dumps({
        "segments": [{"phrase_id": "clip-01#001", "content": "intro", "delivery_score": 9,
                      "visual_score": 8, "emotion": "excited", "is_failed_take": False,
                      "shorts_candidate": False}],
        "visual": _clean_visual(2),
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
    for stage_id in (
        "ingest", "quality", "sync", "transcribe", "analyze", "describe", "plan", "visual_check",
        "effects", "render_draft", "export_edit", "polish",
    ):
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
    config_path.write_text(MOCK_CONFIG, encoding="utf-8")

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
        "visual": _clean_visual(2),
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
    config_path.write_text(MOCK_CONFIG, encoding="utf-8")

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
    config_path.write_text(MOCK_CONFIG, encoding="utf-8")

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
    config_path.write_text(MOCK_CONFIG, encoding="utf-8")

    # No mock ASR sidecar exists, so `transcribe` raises FileNotFoundError.
    result = runner.invoke(app, ["run", str(project), "--config", str(config_path)])

    assert result.exit_code != 0
    combined = result.output + (result.stderr if result.stderr_bytes else "")
    assert "transcribe" in combined
    assert "--stage transcribe" in combined
    assert "Traceback" not in combined


def test_stage_failure_re_run_command_includes_the_custom_config_path(
    tmp_path: Path, make_clip
):
    """Following the suggested re-run command verbatim after a run made with a
    custom --config must not silently fall back to config.yaml's defaults, which
    select different (and more expensive) providers."""
    project = tmp_path / "project"
    (project / "video").mkdir(parents=True)
    make_clip("a.mp4", seconds=3.0).rename(project / "video" / "a.mp4")
    config_path = tmp_path / "custom-config.yaml"
    config_path.write_text(MOCK_CONFIG, encoding="utf-8")

    # No mock ASR sidecar exists, so `transcribe` raises FileNotFoundError.
    result = runner.invoke(app, ["run", str(project), "--config", str(config_path)])

    assert result.exit_code != 0
    combined = result.output + (result.stderr if result.stderr_bytes else "")
    assert f"--config {config_path}" in combined
    assert "--stage transcribe" in combined


def test_produce_reports_a_stage_failure_exactly_as_run_does(tmp_path: Path, make_clip):
    """`produce` used to raise a BadParameter naming the stage and nothing else: no
    re-run command, no --debug hint. A creator should not have to know which of the
    two commands they typed to learn what to do next."""
    project = tmp_path / "project"
    (project / "video").mkdir(parents=True)
    make_clip("a.mp4", seconds=3.0).rename(project / "video" / "a.mp4")
    config_path = tmp_path / "produce-config.yaml"
    config_path.write_text(MOCK_CONFIG, encoding="utf-8")

    # No mock ASR sidecar exists, so `transcribe` raises FileNotFoundError.
    result = runner.invoke(app, ["produce", str(project), "--config", str(config_path)])

    assert result.exit_code != 0
    combined = result.output + (result.stderr if result.stderr_bytes else "")
    assert "Stage 'transcribe' failed" in combined
    assert f"videoai run {project} --config {config_path} --stage transcribe" in combined
    assert "--debug" in combined
    assert "Traceback" not in combined


def test_produce_debug_flag_keeps_the_traceback(tmp_path: Path, make_clip):
    project = tmp_path / "project"
    (project / "video").mkdir(parents=True)
    make_clip("a.mp4", seconds=3.0).rename(project / "video" / "a.mp4")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(MOCK_CONFIG, encoding="utf-8")

    result = runner.invoke(
        app, ["produce", str(project), "--config", str(config_path), "--debug"]
    )

    assert result.exit_code != 0
    assert isinstance(result.exception, Exception)
    assert "transcribe" in str(result.exception)


def test_debug_flag_keeps_the_traceback(tmp_path: Path, make_clip):
    project = tmp_path / "project"
    (project / "video").mkdir(parents=True)
    make_clip("a.mp4", seconds=3.0).rename(project / "video" / "a.mp4")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(MOCK_CONFIG, encoding="utf-8")

    result = runner.invoke(app, ["run", str(project), "--config", str(config_path), "--debug"])

    assert result.exit_code != 0
    assert isinstance(result.exception, Exception)
    assert "transcribe" in str(result.exception)


def _executed_stages(output: str) -> set[str]:
    for line in output.splitlines():
        if line.startswith("Executed: "):
            return {stage.strip() for stage in line.removeprefix("Executed: ").split(",")}
    return set()


ALL_STAGE_IDS = {
    "ingest", "quality", "sync", "transcribe", "analyze", "describe", "plan",
    "assemble", "visual_check", "effects", "render_draft", "export_edit", "polish",
}


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
    config_path.write_text(MOCK_CONFIG, encoding="utf-8")
    llm_path = tmp_path / "llm.json"
    llm_path.write_text(json.dumps({
        "segments": [{"phrase_id": "clip-01#001", "content": "intro", "delivery_score": 9,
                      "visual_score": 8, "emotion": "excited", "is_failed_take": False,
                      "shorts_candidate": False}],
        "visual": _clean_visual(2),
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
        "visual": _clean_visual(2),
        "title": "T2", "description": "D2", "tags": [],
        "sections": [{"name": "Opening", "goal": "open differently", "phrase_ids": ["clip-01#001"]}],
    }), encoding="utf-8")

    after_brief_edit = runner.invoke(app, ["run", str(project), "--config", str(config_path)])
    assert after_brief_edit.exit_code == 0, after_brief_edit.output
    assert _executed_stages(after_brief_edit.output) == {
        "analyze", "describe", "plan", "assemble", "visual_check", "effects",
        "render_draft", "export_edit", "polish",
    }

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
        MOCK_CONFIG + "sync:\n  primary_camera: cam-a\n", encoding="utf-8"
    )
    llm_path = tmp_path / "llm.json"
    llm_path.write_text(json.dumps({
        "segments": [{"phrase_id": "clip-01#001", "content": "intro", "delivery_score": 9,
                      "visual_score": 8, "emotion": "excited", "is_failed_take": False,
                      "shorts_candidate": False}],
        "visual": _clean_visual(2),
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


def test_changing_plan_exclude_phrases_reruns_only_plan_and_render(
    tmp_path: Path, make_clip, monkeypatch
):
    """The creator naming a bad moment as excluded is a plan-time decision, not a
    media or brief change: it must re-run planning and the render that depends on
    it, but must not force re-probing, re-scoring or re-transcribing every clip."""
    project = tmp_path / "project"
    (project / "video").mkdir(parents=True)
    make_clip("a.mp4", seconds=6.0).rename(project / "video" / "a.mp4")

    config_path = tmp_path / "config.yaml"
    config_path.write_text(MOCK_CONFIG, encoding="utf-8")
    llm_path = tmp_path / "llm.json"
    llm_path.write_text(json.dumps({
        "segments": [
            {"phrase_id": "clip-01#001", "content": "intro", "delivery_score": 9,
             "visual_score": 8, "emotion": "excited", "is_failed_take": False,
             "shorts_candidate": False},
            {"phrase_id": "clip-01#002", "content": "a bad moment", "delivery_score": 2,
             "visual_score": 3, "emotion": "neutral", "is_failed_take": False,
             "shorts_candidate": False},
        ],
        "visual": _clean_visual(2),
        "title": "T", "description": "D", "tags": [],
        "sections": [
            {"name": "Hook", "goal": "open", "phrase_ids": ["clip-01#001", "clip-01#002"]},
        ],
    }), encoding="utf-8")
    monkeypatch.setenv("VIDEOAI_MOCK_LLM", str(llm_path))

    runner.invoke(app, ["run", str(project), "--config", str(config_path), "--stage", "ingest"])
    _write_words(project, "clip-01", [
        {"text": "hello", "start": 0.5, "end": 0.9},
        {"text": "everyone", "start": 0.95, "end": 1.5},
        {"text": "bad", "start": 3.0, "end": 3.3},
        {"text": "moment", "start": 3.35, "end": 3.8},
    ])

    first = runner.invoke(app, ["run", str(project), "--config", str(config_path)])
    assert first.exit_code == 0, first.output
    assert _executed_stages(first.output) == ALL_STAGE_IDS - {"ingest"}

    cached = runner.invoke(app, ["run", str(project), "--config", str(config_path)])
    assert cached.exit_code == 0, cached.output
    assert "nothing to do" in cached.output.lower()
    assert _executed_stages(cached.output) == set()

    # Excluding a phrase changes what the planner can select; the mock LLM has to
    # say so (real footage's real model would simply stop choosing it once it is
    # hidden from the prompt), otherwise an identical timeline correctly leaves
    # the render cached.
    llm_path.write_text(json.dumps({
        "segments": [
            {"phrase_id": "clip-01#001", "content": "intro", "delivery_score": 9,
             "visual_score": 8, "emotion": "excited", "is_failed_take": False,
             "shorts_candidate": False},
            {"phrase_id": "clip-01#002", "content": "a bad moment", "delivery_score": 2,
             "visual_score": 3, "emotion": "neutral", "is_failed_take": False,
             "shorts_candidate": False},
        ],
        "visual": _clean_visual(2),
        "title": "T", "description": "D", "tags": [],
        "sections": [{"name": "Hook", "goal": "open", "phrase_ids": ["clip-01#001"]}],
    }), encoding="utf-8")
    config_path.write_text(
        MOCK_CONFIG + "plan:\n  exclude_phrases: [clip-01#002]\n",
        encoding="utf-8",
    )

    after_exclusion = runner.invoke(app, ["run", str(project), "--config", str(config_path)])

    assert after_exclusion.exit_code == 0, after_exclusion.output
    assert _executed_stages(after_exclusion.output) == {
        "plan", "assemble", "visual_check", "effects", "render_draft", "export_edit",
        "polish",
    }

    unchanged_again = runner.invoke(app, ["run", str(project), "--config", str(config_path)])
    assert unchanged_again.exit_code == 0, unchanged_again.output
    assert "nothing to do" in unchanged_again.output.lower()
    assert _executed_stages(unchanged_again.output) == set()


def test_changing_a_render_setting_reruns_only_the_render(tmp_path: Path, make_clip, monkeypatch):
    """Finding I1 end to end: a setting a stage reads is one of its inputs. Before
    the fix `draft_crf` was invisible to every fingerprint and the pipeline said
    "nothing to do"."""
    project = tmp_path / "project"
    (project / "video").mkdir(parents=True)
    make_clip("a.mp4", seconds=6.0).rename(project / "video" / "a.mp4")

    config_path = tmp_path / "config.yaml"
    config_path.write_text(MOCK_CONFIG, encoding="utf-8")
    llm_path = tmp_path / "llm.json"
    llm_path.write_text(json.dumps({
        "segments": [{"phrase_id": "clip-01#001", "content": "intro", "delivery_score": 9,
                      "visual_score": 8, "emotion": "excited", "is_failed_take": False,
                      "shorts_candidate": False}],
        "visual": _clean_visual(2),
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
        MOCK_CONFIG + "render:\n  draft_crf: 30\n", encoding="utf-8"
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
    config_path.write_text(MOCK_CONFIG, encoding="utf-8")
    llm_path = tmp_path / "llm.json"
    llm_path.write_text(json.dumps({
        "segments": [{"phrase_id": "clip-01#001", "content": "intro", "delivery_score": 9,
                      "visual_score": 8, "emotion": "excited", "is_failed_take": False,
                      "shorts_candidate": False}],
        "visual": _clean_visual(2),
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
    for stage_id in (
        "quality", "sync", "transcribe", "analyze", "plan", "visual_check",
        "render_draft", "export_edit", "polish",
    ):
        assert stage_id in single_stage.output
    # The notice only names stale stages; it must not run them itself.
    assert not _sidecar_for(project, "clip-02").exists()
    assert _executed_stages(single_stage.output) == {"ingest"}


# --- The self-correcting loop: plan, look at what was chosen, reject what is
# visually bad, re-plan without it ---

_CLEAN_SEGMENT = {"adult_prominent": False, "child_visible": True, "unusable": False,
                  "note": "the child is holding the toy"}
_ADULT_SEGMENT = {"adult_prominent": True, "child_visible": False, "unusable": False,
                  "note": "an adult walks through the frame"}

_TWO_PHRASE_WORDS = [
    {"text": "hello", "start": 0.5, "end": 0.9},
    {"text": "everyone", "start": 0.95, "end": 1.5},
    {"text": "look", "start": 3.0, "end": 3.3},
    {"text": "here", "start": 3.35, "end": 3.8},
]

_TWO_PHRASE_ANALYSIS = [
    {"phrase_id": "clip-01#001", "content": "intro", "delivery_score": 9,
     "visual_score": 8, "emotion": "excited", "is_failed_take": False,
     "shorts_candidate": True},
    {"phrase_id": "clip-01#002", "content": "demo", "delivery_score": 7,
     "visual_score": 7, "emotion": "calm", "is_failed_take": False,
     "shorts_candidate": False},
]


def _plan_doc(*phrase_ids: str) -> dict:
    return {
        "title": "T", "description": "D", "tags": [],
        "sections": [{"name": "Hook", "goal": "open", "phrase_ids": list(phrase_ids)}],
    }


def _seed_project(tmp_path: Path, make_clip, monkeypatch, payload: dict) -> tuple[Path, Path]:
    project = tmp_path / "project"
    (project / "video").mkdir(parents=True)
    make_clip("a.mp4", seconds=6.0).rename(project / "video" / "a.mp4")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(MOCK_CONFIG, encoding="utf-8")
    llm_path = tmp_path / "llm.json"
    llm_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setenv("VIDEOAI_MOCK_LLM", str(llm_path))

    seed = runner.invoke(app, ["run", str(project), "--config", str(config_path), "--stage", "ingest"])
    assert seed.exit_code == 0, seed.output
    _write_words(project, "clip-01", _TWO_PHRASE_WORDS)
    return project, config_path


def test_auto_fix_replans_without_the_visually_rejected_segment(
    tmp_path: Path, make_clip, monkeypatch
):
    """The whole point of the gate: the second plan, made without the shot that
    had an adult in it, renders a draft rather than the creator finding the adult
    themselves."""
    payload = {
        "segments": _TWO_PHRASE_ANALYSIS,
        # One document per planning round: the first pick is refused, the second
        # is made from a candidate list the rejected phrase has been removed from.
        "plan": [_plan_doc("clip-01#001", "clip-01#002"), _plan_doc("clip-01#001")],
        "visual": [{"0": _CLEAN_SEGMENT, "1": _ADULT_SEGMENT}, {"0": _CLEAN_SEGMENT}],
    }
    project, config_path = _seed_project(tmp_path, make_clip, monkeypatch, payload)

    result = runner.invoke(
        app, ["run", str(project), "--config", str(config_path), "--auto-fix", "2"]
    )

    assert result.exit_code == 0, result.output
    assert "Auto-fix round 1 of 2" in result.output
    assert "clip-01#002" in result.output
    assert "an adult walks through the frame" in result.output
    assert (project / "output" / "draft.mp4").exists()

    rejected = json.loads((project / "work" / "05c-rejected.json").read_text(encoding="utf-8"))
    assert rejected["phrase_ids"] == ["clip-01#002"]
    timeline = json.loads((project / "work" / "05-timeline.json").read_text(encoding="utf-8"))
    assert len(timeline["clips"]) == 1
    assert timeline["clips"][0]["quote"] == "hello everyone"


def test_auto_fix_zero_fails_immediately_and_names_every_rejected_segment(
    tmp_path: Path, make_clip, monkeypatch
):
    payload = {
        "segments": _TWO_PHRASE_ANALYSIS,
        "plan": _plan_doc("clip-01#001", "clip-01#002"),
        "visual": {"0": _ADULT_SEGMENT, "1": _ADULT_SEGMENT},
    }
    project, config_path = _seed_project(tmp_path, make_clip, monkeypatch, payload)

    result = runner.invoke(
        app, ["run", str(project), "--config", str(config_path), "--auto-fix", "0"]
    )

    assert result.exit_code != 0
    combined = result.output + (result.stderr if result.stderr_bytes else "")
    assert "Auto-fix round" not in combined
    assert "visual_check" in combined
    for phrase_id in ("clip-01#001", "clip-01#002"):
        assert phrase_id in combined
    assert "an adult walks through the frame" in combined
    assert not (project / "output" / "draft.mp4").exists()


def test_auto_fix_gives_up_after_its_cap_and_names_what_it_dropped(
    tmp_path: Path, make_clip, monkeypatch
):
    """A model that keeps choosing badly must not loop forever, and the run that
    gives up has to say which segments it refused."""
    payload = {
        "segments": _TWO_PHRASE_ANALYSIS,
        "plan": [_plan_doc("clip-01#001", "clip-01#002"), _plan_doc("clip-01#001")],
        "visual": [{"0": _CLEAN_SEGMENT, "1": _ADULT_SEGMENT}, {"0": _ADULT_SEGMENT}],
    }
    project, config_path = _seed_project(tmp_path, make_clip, monkeypatch, payload)

    result = runner.invoke(
        app, ["run", str(project), "--config", str(config_path), "--auto-fix", "1"]
    )

    assert result.exit_code != 0
    combined = result.output + (result.stderr if result.stderr_bytes else "")
    assert "Auto-fix round 1 of 1" in combined
    assert "gave up after 1" in combined
    assert "clip-01#001" in combined
    assert not (project / "output" / "draft.mp4").exists()

    # Both rounds' rejections survive, so a later run is not offered either shot.
    rejected = json.loads((project / "work" / "05c-rejected.json").read_text(encoding="utf-8"))
    assert rejected["phrase_ids"] == ["clip-01#002", "clip-01#001"]


# --- Per-run output snapshots: a re-run must not silently erase the previous
# final.mp4 with no history ---


class _FrozenClock:
    """Stands in for `datetime` inside `videoai.core.project`. The CLI itself has
    no `--now` knob (there is no legitimate reason a creator would want one), so
    freezing the clock the module reads is the only way to pin the snapshot
    folder's name from an end-to-end test."""

    def __init__(self, value: datetime) -> None:
        self._value = value

    def now(self) -> datetime:
        return self._value


def _seed_snapshot_project(tmp_path: Path, make_clip, monkeypatch) -> tuple[Path, Path]:
    project = tmp_path / "project"
    (project / "input").mkdir(parents=True)
    make_clip("a.mp4", seconds=6.0).rename(project / "input" / "a.mp4")

    config_path = tmp_path / "config.yaml"
    config_path.write_text(MOCK_CONFIG, encoding="utf-8")
    llm_path = tmp_path / "llm.json"
    llm_path.write_text(json.dumps({
        "segments": [{"phrase_id": "clip-01#001", "content": "intro", "delivery_score": 9,
                      "visual_score": 8, "emotion": "excited", "is_failed_take": False,
                      "shorts_candidate": False}],
        "visual": _clean_visual(2),
        "title": "T", "description": "D", "tags": [],
        "sections": [{"name": "Hook", "goal": "open", "phrase_ids": ["clip-01#001"]}],
    }), encoding="utf-8")
    monkeypatch.setenv("VIDEOAI_MOCK_LLM", str(llm_path))

    runner.invoke(app, ["run", str(project), "--config", str(config_path), "--stage", "ingest"])
    _write_words(project, "clip-01", [{"text": "hello", "start": 0.5, "end": 0.9},
                                      {"text": "everyone", "start": 0.95, "end": 1.5}])
    return project, config_path


def test_full_run_snapshots_output_with_the_owners_naming(tmp_path: Path, make_clip, monkeypatch):
    project, config_path = _seed_snapshot_project(tmp_path, make_clip, monkeypatch)
    monkeypatch.setattr(project_module, "datetime", _FrozenClock(datetime(2026, 8, 11, 15, 1)))

    result = runner.invoke(app, ["run", str(project), "--config", str(config_path)])

    assert result.exit_code == 0, result.output
    output_dir = project / "output"
    snapshot = output_dir / "11_Aug_15_01"
    assert snapshot.is_dir()
    root_files = {p.name for p in output_dir.iterdir() if p.is_file()}
    snapshot_files = {p.name for p in snapshot.iterdir() if p.is_file()}
    assert root_files == snapshot_files
    assert root_files  # the mock run actually produced deliverables to archive
    assert f"Snapshot: output/11_Aug_15_01/ ({len(root_files)} files)" in result.output


def test_second_run_in_the_same_minute_gets_a_suffixed_snapshot(tmp_path: Path, make_clip, monkeypatch):
    project, config_path = _seed_snapshot_project(tmp_path, make_clip, monkeypatch)
    monkeypatch.setattr(project_module, "datetime", _FrozenClock(datetime(2026, 8, 11, 15, 1)))

    first = runner.invoke(app, ["run", str(project), "--config", str(config_path)])
    assert first.exit_code == 0, first.output
    assert (project / "output" / "11_Aug_15_01").is_dir()

    # --force so the second invocation, inside the same frozen minute, actually
    # re-executes stages rather than reporting "nothing to do".
    second = runner.invoke(app, ["run", str(project), "--config", str(config_path), "--force"])

    assert second.exit_code == 0, second.output
    assert (project / "output" / "11_Aug_15_01_2").is_dir()


def test_nothing_to_do_run_creates_no_snapshot(tmp_path: Path, make_clip, monkeypatch):
    project, config_path = _seed_snapshot_project(tmp_path, make_clip, monkeypatch)

    first = runner.invoke(app, ["run", str(project), "--config", str(config_path)])
    assert first.exit_code == 0, first.output
    before = {p.name for p in (project / "output").iterdir() if p.is_dir()}
    assert before  # the first run did archive something

    second = runner.invoke(app, ["run", str(project), "--config", str(config_path)])

    assert second.exit_code == 0, second.output
    assert "nothing to do" in second.output.lower()
    assert "Snapshot:" not in second.output
    after = {p.name for p in (project / "output").iterdir() if p.is_dir()}
    assert after == before


def test_output_snapshots_false_disables_archiving(tmp_path: Path, make_clip, monkeypatch):
    project, config_path = _seed_snapshot_project(tmp_path, make_clip, monkeypatch)
    config_path.write_text(MOCK_CONFIG + "output_snapshots: false\n", encoding="utf-8")

    result = runner.invoke(app, ["run", str(project), "--config", str(config_path)])

    assert result.exit_code == 0, result.output
    assert "Snapshot:" not in result.output
    assert [p for p in (project / "output").iterdir() if p.is_dir()] == []


# --- The edit page is part of the workflow, not a command nobody is told about ---


def test_produce_names_the_edit_page_alongside_the_approval_it_asks_for(
    tmp_path: Path, make_clip, monkeypatch,
):
    """It stops here to be told what the creator wants, and until now the only
    thing it offered was yes. Reordering, dropping a shot and every accent
    decision live behind a command `produce` never mentioned."""
    project, config_path = _seed_snapshot_project(tmp_path, make_clip, monkeypatch)

    result = runner.invoke(app, ["produce", str(project), "--config", str(config_path)])

    assert result.exit_code == 0, result.output
    assert f"videoai edit {project} --config {config_path}" in result.output
    assert f"videoai approve {project} --config {config_path}" in result.output


def test_the_edit_page_draws_the_edit_that_was_actually_assembled(
    tmp_path: Path, make_clip, monkeypatch,
):
    project, config_path = _seed_snapshot_project(tmp_path, make_clip, monkeypatch)
    assert runner.invoke(
        app, ["run", str(project), "--config", str(config_path)]
    ).exit_code == 0

    result = runner.invoke(
        app, ["edit", str(project), "--config", str(config_path), "--write"]
    )

    assert result.exit_code == 0, result.output
    page = (project / "output" / "edit.html").read_text(encoding="utf-8")
    timeline = json.loads((project / "work" / "05-timeline.json").read_text(encoding="utf-8"))
    assert timeline["clips"]
    for clip in timeline["clips"]:
        assert clip["ref"] in page
    # A frame per shot, embedded rather than linked: the page is opened off a
    # disk, and these are frames of a child.
    assert "data:image/jpeg;base64," in page
