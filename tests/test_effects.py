"""The cartoon-effects system: the library, the placement stage, the compositor.

The version of this feature that was reverted put one video's syringe and its
"OH NO!" bubble straight into the renderer. These tests are mostly about the
seams that replaced it: that the library is data, that the placement stage
refuses anything it cannot render, that an empty plan is a real answer, and that
what the artifact promises is measurably on the delivered frames.
"""
from pathlib import Path
import json
import shutil
import subprocess

import pytest

from videoai.config import Config, EffectsSettings, PolishSettings
from videoai.core.models import (
    Analysis,
    ClipInfo,
    ClipTranscript,
    EffectEvent,
    EffectPlan,
    Manifest,
    SegmentAnalysis,
    StoryPlan,
    Timeline,
    TimelineClip,
    Transcript,
    Word,
)
from videoai.core.registry import StageContext
from videoai.core.store import ArtifactStore
from videoai.logic.effect_seeds import seed_library
from videoai.logic.effects import (
    GRID_CELLS,
    animation_transform,
    cell_anchor_point,
    default_library_dir,
    load_library,
    place_sprite,
    render_speech_bubble,
)
from videoai.stages.s05d_effects import build_effects_prompt, effects, parse_events
from videoai.stages.s06_render_draft import render_draft
from videoai.stages.s08_polish import polish

CONTRACT = """
version: 1
required_output: {width: 426, height: 240, full_decode: true}
required_features:
  intro: true
  outro: true
  section_titles: true
  captions: true
  music: true
  music_ducking: {minimum_db: 3.0}
  transitions: true
  closing_beat: true
quality:
  source: originals
  maximum_lossy_video_generations: 1
""".strip() + "\n"


# --------------------------------------------------------------------------- #
# The library
# --------------------------------------------------------------------------- #


def test_the_shipped_library_loads_and_every_sprite_file_is_really_there():
    library = load_library()

    assert library.directory == default_library_dir()
    assert set(library.names()) == {
        "comic_starburst", "speed_lines", "sparkle_stars", "sound_rings",
        "confetti_burst", "speech_bubble",
    }
    for sprite in library.sprites:
        assert library.path_of(sprite).is_file(), sprite.name
        assert sprite.tags, sprite.name
        assert sprite.expresses, sprite.name
        assert sprite.default_seconds > 0, sprite.name


def test_every_seeded_sprite_is_rgba_with_real_transparency_and_real_ink():
    """A sprite composited over footage is nothing but its alpha channel. A PNG
    that is opaque everywhere would paste a rectangle over the video, and one that
    is transparent everywhere would paste nothing at all."""
    from PIL import Image

    library = load_library()
    for sprite in library.sprites:
        image = Image.open(library.path_of(sprite))
        assert image.mode == "RGBA", sprite.name
        assert min(image.size) >= 256, sprite.name
        alpha = image.getchannel("A")
        histogram = alpha.histogram()
        transparent = histogram[0]
        opaque = histogram[255]
        total = image.width * image.height
        # Both ends of the channel are genuinely used, and neither dominates the
        # whole file: this is a drawing on a transparent ground.
        assert transparent > total * 0.20, sprite.name
        assert opaque > total * 0.02, sprite.name
        assert transparent + opaque < total, sprite.name  # antialiased edges exist


def test_the_library_catalogue_shows_the_model_words_and_never_a_file_name():
    """The model chooses by what a sprite expresses. Showing it file names would
    invite it to reason about pixels it cannot see."""
    catalogue = load_library().catalogue()

    assert "comic_starburst" in catalogue
    assert "impact" in catalogue
    assert "takes short text" in catalogue  # the nine-patch is flagged
    assert ".png" not in catalogue


def test_a_missing_library_is_empty_rather_than_an_error(tmp_path: Path):
    assert load_library(tmp_path / "nothing-here").sprites == ()


def test_seeding_a_library_elsewhere_reproduces_the_same_bytes(tmp_path: Path):
    """The drawings are deterministic, so the committed PNGs can be regenerated
    and the library's fingerprint means something."""
    seed_library(tmp_path / "effects")

    for sprite in load_library().sprites:
        assert (tmp_path / "effects" / sprite.file).read_bytes() == (
            default_library_dir() / sprite.file
        ).read_bytes(), sprite.name


# --------------------------------------------------------------------------- #
# Placement: parsing a reply into events
# --------------------------------------------------------------------------- #


def _reply(*events: dict) -> dict:
    return {"events": list(events)}


def _event(**overrides) -> dict:
    base = {
        "at_seconds": 5.0,
        "effect_name": "comic_starburst",
        "screen_position": "bottom-right",
        "scale": "medium",
        "text": "",
        "reason": "the bubble pops here",
    }
    return {**base, **overrides}


def test_a_canned_reply_becomes_validated_events_in_time_order():
    library = load_library()

    events = parse_events(
        _reply(
            _event(at_seconds=22.5, effect_name="sound_rings", reason="a loud squish"),
            _event(at_seconds=4.0, scale="large", screen_position="top-left"),
            _event(
                at_seconds=61.0,
                effect_name="speech_bubble",
                screen_position="middle-right",
                text="oh no!",
                reason="the child says oh no",
            ),
        ),
        library,
        duration=90.0,
        max_events=8,
    )

    assert [event.at_seconds for event in events] == [4.0, 22.5, 61.0]
    assert [event.effect_name for event in events] == [
        "comic_starburst", "sound_rings", "speech_bubble"
    ]
    assert events[1].scale == "medium"
    assert events[1].reason == "a loud squish"
    # The duration comes from the manifest, not from the model: how long an accent
    # runs is a property of the drawing.
    assert events[0].seconds == library.get("comic_starburst").default_seconds
    assert events[2].text == "oh no!"


def test_an_unknown_effect_name_is_refused_by_name():
    with pytest.raises(ValueError, match="unknown effect name: 'exploding_syringe'"):
        parse_events(
            _reply(_event(effect_name="exploding_syringe")),
            load_library(),
            duration=90.0,
            max_events=8,
        )


def test_a_time_past_the_end_of_the_edit_is_refused():
    with pytest.raises(ValueError, match=r"placed at 95.00s, outside the 90.00s edit"):
        parse_events(
            _reply(_event(at_seconds=95.0)), load_library(), duration=90.0, max_events=8
        )


def test_a_negative_time_is_refused():
    with pytest.raises(ValueError, match="outside the"):
        parse_events(
            _reply(_event(at_seconds=-3.0)), load_library(), duration=90.0, max_events=8
        )


def test_a_time_a_hundredth_past_the_end_is_clamped_rather_than_refused():
    """The model reads a clock printed to two decimals; the last frame's own
    rounding must not be treated as a hallucinated moment."""
    events = parse_events(
        _reply(_event(at_seconds=90.02)), load_library(), duration=90.0, max_events=8
    )

    assert events[0].at_seconds < 90.0


def test_more_events_than_the_configured_maximum_is_refused():
    reply = _reply(*[_event(at_seconds=float(index)) for index in range(9)])

    with pytest.raises(RuntimeError, match="returned 9 effect events and effects.max_events is 8"):
        parse_events(reply, load_library(), duration=90.0, max_events=8)


def test_an_unknown_screen_position_is_refused_by_name():
    with pytest.raises(ValueError, match="unknown screen_position 'over-the-face'"):
        parse_events(
            _reply(_event(screen_position="over-the-face")),
            load_library(),
            duration=90.0,
            max_events=8,
        )


def test_an_unknown_scale_is_refused_by_name():
    with pytest.raises(ValueError, match="unknown scale 'enormous'"):
        parse_events(
            _reply(_event(scale="enormous")), load_library(), duration=90.0, max_events=8
        )


def test_a_speech_bubble_with_no_text_is_refused():
    """It is stretched around its words; without them it has no size."""
    with pytest.raises(ValueError, match="text sprite and was given no text"):
        parse_events(
            _reply(_event(effect_name="speech_bubble", text="  ")),
            load_library(),
            duration=90.0,
            max_events=8,
        )


def test_text_on_a_sprite_that_cannot_render_it_is_dropped_not_refused():
    events = parse_events(
        _reply(_event(effect_name="sparkle_stars", text="WOW")),
        load_library(),
        duration=90.0,
        max_events=8,
    )

    assert events[0].text == ""


def test_an_empty_reply_is_a_valid_plan():
    """A calm review needs no accents, and a stage that could not say 'none' would
    guarantee every video got some."""
    assert parse_events({}, load_library(), duration=90.0, max_events=8) == []
    assert parse_events({"events": []}, load_library(), duration=90.0, max_events=8) == []


def test_an_events_value_that_is_not_a_list_is_refused():
    with pytest.raises(RuntimeError, match="'events' must be a list"):
        parse_events(
            {"events": {"at_seconds": 1}}, load_library(), duration=90.0, max_events=8
        )


# --------------------------------------------------------------------------- #
# Placement: the stage
# --------------------------------------------------------------------------- #


def _timeline() -> Timeline:
    return Timeline(
        fps=30,
        width=1920,
        height=1080,
        clips=[
            TimelineClip(
                src="clip-01", offset=0, dur=6, start=0, beat="Hook",
                quote="Look at this toy",
            ),
            TimelineClip(
                src="clip-02", offset=2, dur=4, start=6, beat="Popping Time",
                is_insert=True,
            ),
            TimelineClip(
                src="clip-01", offset=10, dur=5, start=10, beat="Closing",
                quote="Thanks for watching",
            ),
        ],
    )


def _effects_context(tmp_path: Path, config: Config) -> StageContext:
    work = tmp_path / "work"
    ctx = StageContext(
        project_dir=tmp_path,
        input_dir=tmp_path,
        work_dir=work,
        output_dir=tmp_path / "output",
        config=config,
        store=ArtifactStore(work),
    )
    ctx.store.write("05-timeline", _timeline(), fingerprint="timeline")
    ctx.store.write(
        "05a-storyplan",
        StoryPlan(title="Pop That Toy", description="A squishy review."),
        fingerprint="story",
    )
    ctx.store.write(
        "04-analysis",
        Analysis(
            provider="mock",
            segments=[
                SegmentAnalysis(
                    phrase_id="clip-01#001", clip_id="clip-01", start=0, end=6,
                    text="Look at this toy", emotion="excited",
                )
            ],
        ),
        fingerprint="analysis",
    )
    ctx.store.write(
        "03-transcript",
        Transcript(
            provider="mock",
            clips=[
                ClipTranscript(
                    clip_id="clip-01",
                    words=[
                        Word(text="Look", start=0.4, end=0.8),
                        Word(text="at", start=0.9, end=1.1),
                        Word(text="this", start=1.2, end=1.5),
                        Word(text="toy", start=1.6, end=2.0),
                    ],
                )
            ],
        ),
        fingerprint="transcript",
    )
    return ctx


def test_the_effects_stage_maps_a_canned_reply_to_an_artifact(
    tmp_path: Path, monkeypatch
):
    reply = {
        "effects": _reply(
            _event(at_seconds=7.5, effect_name="comic_starburst",
                   screen_position="middle-right", reason="the bubble pops"),
            _event(at_seconds=13.0, effect_name="confetti_burst",
                   screen_position="bottom-center", scale="large",
                   reason="he signs off happy"),
        )
    }
    llm = tmp_path / "llm.json"
    llm.write_text(json.dumps(reply), encoding="utf-8")
    monkeypatch.setenv("VIDEOAI_MOCK_LLM", str(llm))
    ctx = _effects_context(tmp_path, Config(providers={"asr": "mock", "llm": "mock"}))

    plan = effects(ctx)

    assert plan.provider == "mock"
    assert [(event.effect_name, event.at_seconds) for event in plan.events] == [
        ("comic_starburst", 7.5), ("confetti_burst", 13.0)
    ]
    assert plan.events[1].reason == "he signs off happy"


def test_the_effects_stage_makes_no_call_at_all_when_disabled(tmp_path: Path, monkeypatch):
    """`enabled: false` must not spend a model call to find out it is off."""
    monkeypatch.delenv("VIDEOAI_MOCK_LLM", raising=False)
    ctx = _effects_context(
        tmp_path,
        Config(
            providers={"asr": "mock", "llm": "mock"},
            effects=EffectsSettings(enabled=False),
        ),
    )

    # The mock provider raises without VIDEOAI_MOCK_LLM, so a call would fail here.
    plan = effects(ctx)

    assert plan.events == []
    assert plan.provider == ""


def test_the_prompt_carries_the_library_the_edit_the_words_and_the_inserts():
    library = load_library()
    analysis = Analysis(
        provider="mock",
        segments=[],
        inserts=[],
    )
    from videoai.core.models import InsertClip

    analysis = analysis.model_copy(
        update={
            "inserts": [
                InsertClip(
                    clip_id="clip-02", duration=9.0,
                    description="A hand popping one of the toy's bubbles",
                )
            ]
        }
    )
    transcript = Transcript(
        provider="mock",
        clips=[
            ClipTranscript(
                clip_id="clip-01", words=[Word(text="Look", start=0.4, end=0.8)]
            )
        ],
    )

    prompt = build_effects_prompt(
        library, _timeline(), transcript, analysis,
        StoryPlan(title="Pop That Toy"), "Keep it fun for six-year-olds.",
    )

    assert "comic_starburst" in prompt
    assert "Popping Time" in prompt                       # beats
    assert '"Look at this toy"' in prompt                 # quotes
    assert "Look@0.40" in prompt                          # word timings, in edit time
    assert "A hand popping one of the toy's bubbles" in prompt  # insert descriptions
    assert "Keep it fun for six-year-olds." in prompt      # the brief
    assert "NEVER cover the child's face" in prompt


# --------------------------------------------------------------------------- #
# Geometry and animation
# --------------------------------------------------------------------------- #


def test_every_grid_cell_lands_inside_the_safe_area():
    frame = (1920, 1080)
    for cell in GRID_CELLS:
        x, y = cell_anchor_point(cell, frame)
        assert 1920 * 0.07 <= x <= 1920 * 0.93, cell
        assert 1080 * 0.07 <= y <= 1080 * 0.93, cell


def test_a_sprite_in_a_corner_cell_is_pulled_back_inside_the_safe_area():
    frame = (1920, 1080)
    x, y = place_sprite((400, 400), "top-left", "center", frame)

    assert x >= round(1920 * 0.07)
    assert y >= round(1080 * 0.07)
    assert x + 400 <= 1920 - round(1080 * 0.07)


def test_a_bottom_anchored_sprite_sits_above_its_cells_point():
    frame = (1920, 1080)
    _, point_y = cell_anchor_point("middle-left", frame)
    _, y = place_sprite((300, 300), "middle-left", "bottom", frame)

    assert y + 300 == pytest.approx(point_y, abs=1)


def test_pop_in_overshoots_and_settles():
    early = animation_transform("pop-in", 0.30).scale
    settled = animation_transform("pop-in", 0.80).scale

    assert early > 1.10
    assert settled == pytest.approx(1.0)


def test_shake_only_rotates_and_decays():
    # 1/16 through is the first peak of the four-cycle wobble.
    early = animation_transform("shake", 0.0625)
    late = animation_transform("shake", 0.0625 + 0.75)

    assert abs(early.rotation) > 5.0
    assert early.scale == pytest.approx(1.0)
    assert abs(late.rotation) < abs(early.rotation)


def test_drift_up_rises_and_fades():
    late = animation_transform("drift-up", 0.9)

    assert late.dy < -0.2
    assert late.alpha < animation_transform("drift-up", 0.4).alpha


def test_every_animation_starts_and_ends_invisible():
    """The shared fade envelope: nothing ever appears or vanishes on one frame."""
    for animation in ("pop-in", "pulse", "shake", "drift-up", "none"):
        assert animation_transform(animation, 0.0).alpha == 0.0, animation
        assert animation_transform(animation, 1.0).alpha == 0.0, animation
        assert animation_transform(animation, 0.5).alpha > 0.5, animation


# --------------------------------------------------------------------------- #
# The nine-patch speech bubble
# --------------------------------------------------------------------------- #


def _bubble(tmp_path: Path, text: str, max_width: int = 640):
    from PIL import Image

    library = load_library()
    sprite = library.get("speech_bubble")
    assert sprite.nine_patch is not None
    dst = tmp_path / f"bubble-{abs(hash(text))}.png"
    render_speech_bubble(
        library.path_of(sprite), sprite.nine_patch, text, max_width, dst
    )
    return Image.open(dst).convert("RGBA"), sprite.nine_patch


def test_the_speech_bubble_renders_whatever_text_it_is_given(tmp_path: Path):
    """The words are dynamic: nothing about them is in the library, and two
    different phrases produce visibly different pictures at different sizes."""
    import numpy as np

    short, _ = _bubble(tmp_path, "OH NO!")
    long, spec = _bubble(tmp_path, "THAT IS SO MUCH PAINT COMING OUT OF IT")

    # Ink inside the bubble, in both cases: the text was really drawn.
    for image, label in ((short, "short"), (long, "long")):
        interior = np.array(image)[
            spec.text_top : image.height - spec.text_bottom,
            spec.text_left : image.width - spec.text_right,
        ]
        dark = (interior[:, :, 3] > 200) & (interior[:, :, :3].max(axis=2) < 90)
        assert dark.sum() > 200, label

    assert long.height > short.height  # more words needed more bubble
    assert long.width >= short.width


def test_nine_patch_stretching_keeps_the_corners_undistorted():
    """The whole point of a nine-patch: the bubble grows, the drawing does not
    smear. Compared on the stretched sprite before any text goes on it, because the
    text is deliberately allowed to sit over the corner bands' inner pixels."""
    import numpy as np
    from PIL import Image

    from videoai.logic.effects import nine_patch_resize

    library = load_library()
    sprite = library.get("speech_bubble")
    spec = sprite.nine_patch
    assert spec is not None
    source = Image.open(library.path_of(sprite)).convert("RGBA")

    narrow = np.array(nine_patch_resize(source, spec, (420, 340)))
    wide = np.array(nine_patch_resize(source, spec, (900, 620)))
    original = np.array(source)

    assert narrow.shape != wide.shape
    corners = {
        "top-left": (slice(0, spec.top), slice(0, spec.left)),
        "top-right": (slice(0, spec.top), slice(-spec.right, None)),
        "bottom-left": (slice(-spec.bottom, None), slice(0, spec.left)),
        "bottom-right": (slice(-spec.bottom, None), slice(-spec.right, None)),
    }
    for name, (rows, columns) in corners.items():
        assert np.array_equal(narrow[rows, columns], wide[rows, columns]), name
        # Not merely equal to each other: equal to the drawing they came from.
        assert np.array_equal(narrow[rows, columns], original[rows, columns]), name

    # And the tail really is inside the bottom-left band that was just proved
    # pixel-identical — it is the thing a stretched bottom edge would ruin first.
    tail_band = narrow[-spec.bottom :, : spec.left]
    below_the_body = tail_band[tail_band.shape[0] // 2 :]
    assert (below_the_body[:, :, 3] > 200).sum() > 100


def test_a_bubble_never_grows_past_the_width_it_was_given(tmp_path: Path):
    wide, _ = _bubble(tmp_path, "A" * 120, max_width=520)

    assert wide.width <= 520


# --------------------------------------------------------------------------- #
# Compositing onto the delivery
# --------------------------------------------------------------------------- #


def _delivery_context(
    tmp_path: Path, make_clip, config_effects: EffectsSettings
) -> tuple[StageContext, Path]:
    (tmp_path / "production-contract.yaml").write_text(CONTRACT, encoding="utf-8")
    source = make_clip("source.mp4", seconds=8.0, size="640x360")
    proxy = Path(shutil.copyfile(source, tmp_path / "source-proxy.mp4"))
    music_dir = tmp_path / "music"
    music_dir.mkdir()
    track = music_dir / "bed.mp3"
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", "sine=frequency=330:duration=4",
            "-c:a", "libmp3lame", str(track),
        ],
        check=True,
    )
    work = tmp_path / "work"
    output = tmp_path / "output"
    work.mkdir()
    output.mkdir()
    ctx = StageContext(
        project_dir=tmp_path,
        input_dir=tmp_path,
        work_dir=work,
        output_dir=output,
        config=Config(
            effects=config_effects,
            polish=PolishSettings(
                require_approval=False,
                intro_seconds=0.5,
                outro_seconds=0.5,
                title_seconds=0.5,
                captions_enabled=True,
                caption_words=2,
                transition_frames=3,
                music_dir=str(music_dir),
                output_height=240,
                hardware_encode=False,
            ),
        ),
        store=ArtifactStore(work),
    )
    ctx.store.write(
        "01-manifest",
        Manifest(
            clips=[
                ClipInfo(
                    clip_id="clip-01", path=str(source), proxy_path=str(proxy),
                    duration=8, width=640, height=360, fps=30, has_audio=True,
                )
            ]
        ),
        fingerprint="manifest",
    )
    ctx.store.write(
        "05-timeline",
        Timeline(
            fps=30,
            width=640,
            height=360,
            clips=[
                TimelineClip(src="clip-01", offset=0, dur=3, start=0, beat="Hook"),
                TimelineClip(
                    src="clip-01", offset=3, dur=3, start=3,
                    beat="Closing", quote="Thanks for watching",
                ),
            ],
        ),
        fingerprint="timeline",
    )
    ctx.store.write(
        "05a-storyplan", StoryPlan(title="Colour Pop Review"), fingerprint="story"
    )
    ctx.store.write(
        "03-transcript",
        Transcript(
            provider="test",
            clips=[
                ClipTranscript(
                    clip_id="clip-01",
                    words=[
                        Word(text="Hello", start=0.2, end=0.6),
                        Word(text="everyone", start=0.7, end=1.1),
                        Word(text="Look", start=3.2, end=3.5),
                        Word(text="inside", start=3.6, end=4.0),
                    ],
                )
            ],
        ),
        fingerprint="transcript",
    )
    draft = render_draft(ctx)
    ctx.store.write("06-draft", draft, fingerprint="draft")
    return ctx, tmp_path


def _frame_at(video: Path, at: float, dst: Path):
    from PIL import Image

    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error", "-ss", f"{at:.3f}",
            "-i", str(video), "-frames:v", "1", str(dst),
        ],
        check=True,
    )
    return Image.open(dst).convert("RGB")


# The event's own time on the timeline's clock, and where it lands on the
# delivery's once the half-second intro card is in front of it.
EVENT_AT = 1.4
QUIET_AT = 4.6


def test_a_composited_effect_changes_the_frame_it_is_on_and_nothing_else(
    tmp_path: Path, make_clip
):
    """The claim the artifact makes has to be visible on the delivered pixels.

    Two deliveries of the identical edit, one with an accent and one without: the
    frame at the accent's timestamp must differ, and a frame two seconds away must
    be bit-identical, or the effect is either invisible or all over the video.
    """
    import numpy as np

    plain_dir = tmp_path / "plain"
    with_effect_dir = tmp_path / "accented"
    plain_dir.mkdir()
    with_effect_dir.mkdir()

    plain_ctx, _ = _delivery_context(
        plain_dir, make_clip, EffectsSettings(enabled=True)
    )
    plain_ctx.store.write("05d-effects", EffectPlan(), fingerprint="none")
    plain = polish(plain_ctx)

    accented_ctx, _ = _delivery_context(
        with_effect_dir, make_clip, EffectsSettings(enabled=True)
    )
    accented_ctx.store.write(
        "05d-effects",
        EffectPlan(
            provider="mock",
            events=[
                EffectEvent(
                    at_seconds=EVENT_AT,
                    effect_name="comic_starburst",
                    screen_position="top-right",
                    scale="large",
                    seconds=0.8,
                    reason="the toy pops",
                )
            ],
        ),
        fingerprint="one",
    )
    accented = polish(accented_ctx)

    assert accented.effect_count == 1
    assert plain.effect_count == 0
    assert "comic_starburst" in accented.effects[0]

    intro = 0.5
    on_event = np.asarray(
        _frame_at(Path(accented.path), intro + EVENT_AT + 0.3, tmp_path / "a-on.png"),
        dtype=np.int16,
    )
    plain_on_event = np.asarray(
        _frame_at(Path(plain.path), intro + EVENT_AT + 0.3, tmp_path / "b-on.png"),
        dtype=np.int16,
    )
    changed = np.abs(on_event - plain_on_event)
    # A large accent over a 426x240 frame: a real, localised, unmistakable change.
    assert changed.max() > 60
    assert (changed.max(axis=2) > 30).mean() > 0.01

    off_event = np.asarray(
        _frame_at(Path(accented.path), intro + QUIET_AT, tmp_path / "a-off.png"),
        dtype=np.int16,
    )
    plain_off_event = np.asarray(
        _frame_at(Path(plain.path), intro + QUIET_AT, tmp_path / "b-off.png"),
        dtype=np.int16,
    )
    assert np.array_equal(off_event, plain_off_event)


def test_delivery_renders_with_an_empty_effects_plan(tmp_path: Path, make_clip):
    """An empty plan is the ordinary case for a calm video, and it must produce a
    contract-passing delivery rather than an absence of one."""
    ctx, project = _delivery_context(tmp_path, make_clip, EffectsSettings(enabled=True))
    ctx.store.write("05d-effects", EffectPlan(provider="mock"), fingerprint="empty")

    result = polish(ctx)

    assert Path(result.path).name == "final.mp4"
    assert result.effect_count == 0
    assert result.effects == []
    report = json.loads((project / "output" / "production-report.json").read_text())
    assert report["effects"]["applied"] == 0
    assert report["status"] == "passed"
    # Effects are seasoning: they are not among the features the contract checks.
    assert "effects" not in report["features"]


def test_delivery_renders_when_the_effects_artifact_was_never_written(
    tmp_path: Path, make_clip
):
    """An older project has no `05d-effects`. That is not a crash."""
    ctx, _ = _delivery_context(tmp_path, make_clip, EffectsSettings(enabled=True))

    assert not ctx.store.exists("05d-effects")
    result = polish(ctx)

    assert Path(result.path).is_file()
    assert result.effect_count == 0


def test_effects_disabled_composites_nothing_even_with_a_plan_on_disk(
    tmp_path: Path, make_clip
):
    ctx, _ = _delivery_context(tmp_path, make_clip, EffectsSettings(enabled=False))
    ctx.store.write(
        "05d-effects",
        EffectPlan(
            events=[
                EffectEvent(
                    at_seconds=1.4, effect_name="comic_starburst",
                    screen_position="top-right", scale="large", seconds=0.8,
                )
            ]
        ),
        fingerprint="one",
    )

    result = polish(ctx)

    assert result.effect_count == 0


def test_a_plan_naming_a_sprite_the_library_no_longer_has_is_refused(
    tmp_path: Path, make_clip
):
    """Silently skipping it would deliver a video missing the accents its own
    artifact promises."""
    ctx, _ = _delivery_context(tmp_path, make_clip, EffectsSettings(enabled=True))
    ctx.store.write(
        "05d-effects",
        EffectPlan(
            events=[
                EffectEvent(
                    at_seconds=1.0, effect_name="deleted_sprite",
                    screen_position="center",
                )
            ]
        ),
        fingerprint="stale",
    )

    with pytest.raises(RuntimeError, match="deleted_sprite"):
        polish(ctx)


def test_a_speech_bubble_event_is_composited_with_its_own_words(
    tmp_path: Path, make_clip
):
    """End to end for the nine-patch: the stretched bubble reaches the picture."""
    ctx, project = _delivery_context(
        tmp_path, make_clip, EffectsSettings(enabled=True)
    )
    ctx.store.write(
        "05d-effects",
        EffectPlan(
            events=[
                EffectEvent(
                    at_seconds=1.4, effect_name="speech_bubble",
                    screen_position="middle-right", scale="large", seconds=1.0,
                    text="OH NO!", reason="he says oh no",
                )
            ]
        ),
        fingerprint="bubble",
    )

    result = polish(ctx)

    assert result.effect_count == 1
    assert "speech_bubble" in result.effects[0]
    rendered = next(
        (project / "work" / "delivery").glob("effect-*-speech_bubble.png"), None
    )
    assert rendered is not None and rendered.is_file()
    report = json.loads((project / "output" / "production-report.json").read_text())
    assert report["effects"]["events"][0]["text"] == "OH NO!"
