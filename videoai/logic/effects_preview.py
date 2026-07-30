"""Showing the creator each accent on the frame it will actually land on.

An effect plan is a list of names and timestamps. Read as text it is impossible
to judge: "comic_starburst at 131.14s, bottom-right" tells you nothing about
whether it covers the child's face, points at the thing that is happening, or
sits in an empty corner. On this project's first delivery every one of them was
placed from a model's guess at a timeline it had never seen, and they landed
more or less at random.

So this renders the proposal: the real frame, at the real moment, with the real
sprite composited at the size and position the delivery would use — plus, beside
it, where the picture is actually moving at that instant. What the creator
approves is then what they get, and the two pictures side by side make a bad
placement obvious in a second.

Nothing here decides anything. It draws the plan so a human can accept it,
change it, or throw it out.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path

from videoai.core.models import EffectEvent
from videoai.logic.effects import (
    EffectLibrary,
    animation_transform,
    cell_anchor_point,
    place_sprite,
)
from videoai.logic.motion import GRID_CELLS, busiest_cell, cell_centre

# The instant of the animation to draw: past the pop-in overshoot, before the
# fade, which is what the accent looks like for most of its life.
PREVIEW_PROGRESS = 0.5


@dataclass(frozen=True)
class EffectProposal:
    """One accent, drawn on its own frame and ready to be judged."""

    index: int
    event: EffectEvent
    # Where the picture is actually moving at that moment, or None when nothing
    # is. Shown next to the chosen cell so a mismatch is visible rather than
    # buried in a log.
    measured_cell: str | None
    image_jpeg: bytes

    @property
    def agrees_with_motion(self) -> bool:
        return self.measured_cell is not None and self.measured_cell == self.event.screen_position


def compose_preview(
    frame_bgr,
    sprite_rgba,
    event: EffectEvent,
    library: EffectLibrary,
    measured_cell: str | None,
):
    """The frame with the accent on it, the grid drawn, and the motion marked."""
    import cv2
    import numpy as np

    picture = frame_bgr.copy()
    height, width = picture.shape[:2]

    # The nine cells, faint, so a position can be read off the picture.
    for index in (1, 2):
        cv2.line(picture, (width * index // 3, 0), (width * index // 3, height),
                 (255, 255, 255), 1, cv2.LINE_AA)
        cv2.line(picture, (0, height * index // 3), (width, height * index // 3),
                 (255, 255, 255), 1, cv2.LINE_AA)

    # Where the picture is really moving, as a hollow green box.
    if measured_cell:
        cx, cy = cell_centre(measured_cell, width, height)
        size = max(24, height // 9)
        cv2.rectangle(picture, (cx - size, cy - size), (cx + size, cy + size),
                      (0, 230, 0), 3, cv2.LINE_AA)

    # The accent itself, exactly as the delivery would place it.
    sprite = library.get(event.effect_name)
    transform = animation_transform(sprite.animation, PREVIEW_PROGRESS)
    scale = max(0.01, transform.scale)
    drawn = cv2.resize(
        sprite_rgba,
        (max(1, round(sprite_rgba.shape[1] * scale)),
         max(1, round(sprite_rgba.shape[0] * scale))),
        interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
    )
    x, y = place_sprite(
        (drawn.shape[1], drawn.shape[0]), event.screen_position, sprite.anchor,
        (width, height),
    )
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(width, x + drawn.shape[1]), min(height, y + drawn.shape[0])
    if x1 > x0 and y1 > y0:
        patch = drawn[y0 - y:y1 - y, x0 - x:x1 - x]
        alpha = (patch[..., 3:4].astype(np.float32) / 255.0) * transform.alpha
        picture[y0:y1, x0:x1] = (
            patch[..., :3].astype(np.float32) * alpha
            + picture[y0:y1, x0:x1].astype(np.float32) * (1 - alpha)
        ).astype(np.uint8)

    # The cell the plan chose, as a thin magenta cross, so "chosen" and
    # "measured" can be told apart even when they overlap.
    ax, ay = cell_anchor_point(event.screen_position, (width, height))
    cv2.drawMarker(picture, (ax, ay), (230, 0, 230), cv2.MARKER_CROSS,
                   max(20, height // 12), 2, cv2.LINE_AA)
    return picture


def render_preview_html(proposals: list[EffectProposal], title: str) -> str:
    """A single self-contained page: every accent, on its frame, with its reason."""
    cards: list[str] = []
    for proposal in proposals:
        event = proposal.event
        encoded = base64.b64encode(proposal.image_jpeg).decode("ascii")
        measured = proposal.measured_cell or "nothing moving"
        verdict = (
            '<span class="ok">on the action</span>'
            if proposal.agrees_with_motion
            else f'<span class="warn">motion is at <b>{measured}</b></span>'
        )
        text = f'<div class="says">says “{event.text}”</div>' if event.text else ""
        cards.append(f"""
    <section class="card" id="fx{proposal.index}">
      <img src="data:image/jpeg;base64,{encoded}" alt="frame at {event.at_seconds:.2f}s">
      <div class="meta">
        <h2>{proposal.index + 1}. {event.effect_name}</h2>
        <dl>
          <dt>at</dt><dd>{event.at_seconds:.2f}s</dd>
          <dt>placed</dt><dd>{event.screen_position} · {event.scale}</dd>
          <dt>check</dt><dd>{verdict}</dd>
          <dt>because</dt><dd>{event.reason}</dd>
        </dl>
        {text}
      </div>
    </section>""")

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Effects to approve — {title}</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font: 15px/1.55 -apple-system, system-ui, sans-serif; margin: 0;
         background: #12151a; color: #e8ecf1; }}
  header {{ padding: 28px 32px 8px; }}
  h1 {{ font-size: 22px; margin: 0 0 6px; }}
  .lede {{ color: #9fb0c3; max-width: 60ch; margin: 0 0 4px; }}
  .legend {{ color: #9fb0c3; font-size: 13px; margin: 10px 0 0; }}
  .legend b {{ color: #e8ecf1; }}
  .card {{ display: grid; grid-template-columns: minmax(320px, 2fr) 1fr; gap: 20px;
          align-items: start; padding: 20px 32px; border-top: 1px solid #232a34; }}
  @media (max-width: 820px) {{ .card {{ grid-template-columns: 1fr; }} }}
  img {{ width: 100%; border-radius: 10px; display: block; }}
  h2 {{ font-size: 17px; margin: 0 0 10px; }}
  dl {{ display: grid; grid-template-columns: auto 1fr; gap: 4px 12px; margin: 0; }}
  dt {{ color: #8b9bb0; }}
  dd {{ margin: 0; }}
  .ok {{ color: #7ee0a8; }}
  .warn {{ color: #ffc266; }}
  .says {{ margin-top: 10px; padding: 8px 10px; background: #1b2130;
           border-radius: 8px; }}
</style></head>
<body>
<header>
  <h1>Effects to approve — {title}</h1>
  <p class="lede">Each accent drawn on the frame it will land on, at the size and
  position the delivery would use. Nothing is rendered until you say so.</p>
  <p class="legend">
    <b style="color:#e06be0">magenta cross</b> = where the plan puts it ·
    <b style="color:#4fdc84">green box</b> = where the picture is actually moving ·
    faint grid = the nine positions available.
  </p>
</header>
{"".join(cards)}
</body></html>
"""
