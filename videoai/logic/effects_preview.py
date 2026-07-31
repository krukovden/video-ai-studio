"""The approval page: every accent on its own frame, and changeable there.

An effect plan read as text cannot be judged. "comic_starburst at 131.14s,
bottom-right" says nothing about whether it covers a face, points at what is
happening, or sits over a paper towel — and on this project's first delivery it
was the last of those, seven times over.

So the plan is drawn instead of described, and then made editable in place:
untick an accent to drop it, drag it to where it belongs, or swap it for a
different one. Nine grid cells are enough for a model to aim with and far too
coarse for someone who can see the shot; dragging gives a real coordinate.

The page writes nothing by itself. It produces a decisions document the pipeline
reads back, so what reaches the render is exactly what was approved — and the
frames, which are of a child, never leave the machine.
"""
from __future__ import annotations

import base64
import html
import json
from dataclasses import dataclass

from videoai.core.models import EffectEvent
from videoai.logic.effects import EffectLibrary, place_sprite

# The instant of the animation to draw: past a pop-in's overshoot and before the
# fade, which is what the accent looks like for most of its life.
PREVIEW_PROGRESS = 0.5


@dataclass(frozen=True)
class EffectProposal:
    """One accent, with everything the page needs to show and change it."""

    index: int
    event: EffectEvent
    # Where the picture is actually moving at that moment, as a fraction of the
    # frame, or None when nothing is. Drawn as a target so a bad placement is
    # visible rather than buried in a log.
    motion_xy: tuple[float, float] | None
    motion_cell: str | None
    # The frame, and the sprite as its own image so the page can move it.
    frame_jpeg: bytes
    sprite_png: bytes
    # The sprite's rendered size and placed position, as fractions of the frame,
    # so the page works at whatever size the browser shows it.
    sprite_width: float
    sprite_height: float
    start_x: float
    start_y: float
    # The delivery frame this was laid out against.
    frame_size: tuple[int, int] = (1920, 1080)


def proposal_geometry(
    event: EffectEvent,
    library: EffectLibrary,
    sprite_size: tuple[int, int],
    frame: tuple[int, int],
) -> tuple[float, float, float, float]:
    """The sprite's size and top-left, as fractions of the frame.

    Honours a dragged coordinate when there is one and falls back to the grid
    cell the plan proposed, so the page always opens showing exactly what the
    delivery would currently produce.
    """
    width, height = frame
    sprite_width, sprite_height = sprite_size
    if event.x is not None and event.y is not None:
        # A dragged point is the sprite's CENTRE: that is what a person aims at.
        left = event.x * width - sprite_width / 2
        top = event.y * height - sprite_height / 2
    else:
        sprite = library.get(event.effect_name)
        left, top = place_sprite(sprite_size, event.screen_position, sprite.anchor, frame)
    return (
        sprite_width / width,
        sprite_height / height,
        left / width,
        top / height,
    )


SERVED_HANDOFF = (
    "<b>Press Save when you are happy.</b> Your edits are kept in this browser "
    "as you go, and Save writes them straight into the plan — nothing else to do."
)

FILE_HANDOFF = (
    "<b>Your edits stay in this browser.</b> They are saved here automatically and "
    "survive a refresh, but nothing reaches the video until you press "
    "<b>Copy as text</b> and paste it back. To save directly instead, run "
    "<code>videoai approve-effects &lt;project&gt;</code>."
)


def render_preview_html(
    proposals: list[EffectProposal],
    title: str,
    sprite_choices: list[dict],
    save_url: str = "",
    token: str = "",
) -> str:
    """One self-contained page: the plan, drawn, and editable in place.

    `sprite_choices` carries every sprite the library offers — its picture and
    its aspect ratio — so choosing a different one actually changes the picture.
    An earlier version embedded only the sprite already planned, which meant the
    dropdown changed the answer while the frame went on showing the old drawing:
    a preview that stops previewing the moment you use it.
    """
    payload = [
        {
            "index": item.index,
            "at": round(item.event.at_seconds, 2),
            "name": item.event.effect_name,
            "scale": item.event.scale,
            "text": item.event.text,
            "reason": item.event.reason,
            "cell": item.event.screen_position,
            "keep": item.event.keep,
            "w": round(item.sprite_width, 5),
            "h": round(item.sprite_height, 5),
            "x": round(item.start_x, 5),
            "y": round(item.start_y, 5),
            "motion": (
                {
                    "x": round(item.motion_xy[0], 4),
                    "y": round(item.motion_xy[1], 4),
                    "cell": item.motion_cell,
                }
                if item.motion_xy
                else None
            ),
            # The frame's own shape, so the page can convert a sprite height
            # (a fraction of frame HEIGHT) into a CSS width (a fraction of frame
            # WIDTH) when a swap resizes it.
            "frame_w": item.frame_size[0],
            "frame_h": item.frame_size[1],
            "frame": base64.b64encode(item.frame_jpeg).decode("ascii"),
            "sprite": base64.b64encode(item.sprite_png).decode("ascii"),
        }
        for item in proposals
    ]
    return (
        _PAGE.replace("__SAVE_URL__", save_url)
        .replace("__TOKEN__", token)
        .replace("__HANDOFF__", SERVED_HANDOFF if save_url else FILE_HANDOFF)
        .replace("__TITLE__", html.escape(title))
        .replace("__SPRITES__", json.dumps(sprite_choices))
        .replace("__DATA__", json.dumps(payload))
    )


# Written out in full rather than assembled from fragments: this is one artefact
# a creator opens on their own machine, and a single readable template is easier
# to change than a string built in six places.
_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Approve effects — __TITLE__</title>
<style>
  :root { color-scheme: dark; --line:#232a34; --muted:#93a3b8; --bg:#12151a; }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:#e8ecf1;
         font:15px/1.55 -apple-system, system-ui, sans-serif; }
  header { padding:24px 28px 14px; border-bottom:1px solid var(--line);
           position:sticky; top:0; background:var(--bg); z-index:20; }
  h1 { font-size:21px; margin:0 0 6px; }
  .lede { color:var(--muted); margin:0; max-width:72ch; }
  .bar { display:flex; gap:10px; align-items:center; margin-top:14px; flex-wrap:wrap; }
  button { font:inherit; padding:8px 14px; border-radius:8px; border:1px solid var(--line);
           background:#1b2130; color:#e8ecf1; cursor:pointer; }
  button.primary { background:#2f6df6; border-color:#2f6df6; }
  button:disabled { opacity:.4; cursor:default; }
  button:not(:disabled):hover { filter:brightness(1.15); }
  .count { color:var(--muted); }
  .warn-box { margin:12px 0 0; padding:10px 13px; border-radius:8px; max-width:72ch;
              background:#2a2113; border:1px solid #6b5320; color:#ffd79a; }
  .warn-box code { background:#00000040; padding:1px 5px; border-radius:4px; }
  .restored { background:#132a1c; border-color:#2c6b46; color:#9ae6bb; }
  .scene { display:grid; grid-template-columns:minmax(340px,1.9fr) 1fr; gap:22px;
           padding:22px 28px; border-bottom:1px solid var(--line); }
  @media (max-width:900px){ .scene { grid-template-columns:1fr; } }
  .stage { position:relative; border-radius:10px; overflow:hidden; background:#000;
           user-select:none; touch-action:none; }
  .stage.dropped { opacity:.3; }
  .stage img.frame { width:100%; display:block; }
  .grid { position:absolute; inset:0; pointer-events:none;
          background-image:linear-gradient(to right,#ffffff30 1px,transparent 1px),
                           linear-gradient(to bottom,#ffffff30 1px,transparent 1px);
          background-size:33.333% 33.333%; }
  .motion { position:absolute; width:11%; aspect-ratio:1; transform:translate(-50%,-50%);
            border:3px solid #4fdc84; border-radius:6px; pointer-events:none; }
  .sprite { position:absolute; cursor:grab; }
  .sprite img { width:100%; display:block; pointer-events:none; }
  .sprite.dragging { cursor:grabbing; outline:2px dashed #2f6df6; outline-offset:3px; }
  .meta h2 { font-size:16px; margin:0 0 10px; }
  label.keep { display:flex; gap:8px; align-items:center; font-weight:600; margin-bottom:12px; }
  dl { display:grid; grid-template-columns:auto 1fr; gap:4px 12px; margin:0 0 12px; }
  dt { color:var(--muted); }
  dd { margin:0; }
  .row { display:flex; gap:8px; align-items:center; margin-bottom:8px; flex-wrap:wrap; }
  select { font:inherit; padding:6px 8px; border-radius:7px; background:#1b2130;
           color:#e8ecf1; border:1px solid var(--line); }
  .hint { color:var(--muted); font-size:13px; margin:6px 0 0; }
  .ok { color:#7ee0a8; } .warn { color:#ffc266; }
  #out { width:100%; height:170px; margin-top:10px; font:12px/1.4 ui-monospace,monospace;
         background:#0d1016; color:#9fe6b8; border:1px solid var(--line);
         border-radius:8px; padding:10px; display:none; }
</style></head>
<body>
<header>
  <h1>Approve effects — __TITLE__</h1>
  <p class="lede"><b>Untick</b> to drop an accent, <b>drag</b> it where it belongs,
  or <b>swap</b> it from the list. The green target is where the picture is actually
  moving at that moment.</p>
  <p class="warn-box" id="handoff">__HANDOFF__</p>
  <div class="bar">
    <button class="primary" id="savebtn" onclick="save()">Save</button>
    <button onclick="download()">Download instead</button>
    <button onclick="copyOut()">Copy as text</button>
    <button onclick="snapAll()">Snap all to motion</button>
    <button onclick="if(confirm('Throw away your changes?')){localStorage.removeItem(SAVE_KEY);location.reload();}">Reset</button>
    <span class="count" id="count"></span>
  </div>
  <textarea id="out" readonly></textarea>
</header>
<main id="scenes"></main>
<script>
const SPRITES = __SPRITES__;
const DATA = __DATA__;
// Sprite heights are a fraction of the frame's HEIGHT while CSS widths are a
// fraction of its WIDTH, so converting between them needs the frame's shape.
const FRAME_ASPECT = DATA.length ? (DATA[0].frame_h / DATA[0].frame_w) : (9 / 16);
const state = DATA.map(d => ({...d, x0:d.x, y0:d.y, name0:d.name}));

// The browser will not let this page write to disk, so every edit is kept in
// localStorage instead. Without it a refresh, a closed tab or a regenerated
// page threw away everything the creator had done — which is exactly what
// happened the first time somebody used this in anger.
const SAVE_KEY = 'videoai-effects-' + (DATA.length ? DATA[0].at : 'empty');

function persist() {
  try {
    localStorage.setItem(SAVE_KEY, JSON.stringify(
      state.map(s => ({index:s.index, keep:s.keep, name:s.name, x:s.x, y:s.y, w:s.w, h:s.h}))
    ));
  } catch (e) {}
}

function restore() {
  // True when there was anything saved to come back to. Deliberately not "does
  // it differ from the proposal": the creator wants to know their work survived,
  // and re-deciding to keep exactly what was proposed is still their decision.
  try {
    const saved = JSON.parse(localStorage.getItem(SAVE_KEY) || 'null');
    if (!Array.isArray(saved) || !saved.length) return false;
    saved.forEach(row => {
      const s = state.find(item => item.index === row.index);
      if (!s) return;
      s.keep = row.keep; s.name = row.name;
      s.x = row.x; s.y = row.y; s.w = row.w; s.h = row.h;
    });
    return true;
  } catch (e) { return false; }
}

const RESTORED = restore();

function el(tag, cls, inner) {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (inner !== undefined) n.innerHTML = inner;
  return n;
}

function build() {
  const root = document.getElementById('scenes');
  state.forEach((s, i) => {
    const scene = el('section', 'scene');
    const stage = el('div', 'stage');
    const frame = el('img', 'frame');
    frame.src = 'data:image/jpeg;base64,' + s.frame;
    stage.append(frame, el('div', 'grid'));

    if (s.motion) {
      const m = el('div', 'motion');
      m.style.left = (s.motion.x * 100) + '%';
      m.style.top = (s.motion.y * 100) + '%';
      stage.append(m);
    }

    const sprite = el('div', 'sprite');
    sprite.id = 'sprite' + i;
    // A restored swap has to show the restored drawing, not the proposed one.
    const restoredSprite = SPRITES.find(sp => sp.name === s.name);
    sprite.style.width = (s.w * 100) + '%';
    sprite.style.left = (s.x * 100) + '%';
    sprite.style.top = (s.y * 100) + '%';
    const img = el('img');
    img.src = 'data:image/png;base64,' +
              (restoredSprite && s.name !== s.name0 ? restoredSprite.data : s.sprite);
    sprite.append(img);
    stage.append(sprite);
    makeDraggable(sprite, stage, i);

    const verdict = s.motion
      ? (s.motion.cell === s.cell
          ? '<span class="ok">on the action</span>'
          : '<span class="warn">motion is at <b>' + s.motion.cell + '</b></span>')
      : '<span class="warn">nothing moving here</span>';

    const meta = el('div', 'meta');
    const keep = el('label', 'keep');
    const box = el('input');
    box.type = 'checkbox';
    box.checked = s.keep;
    box.onchange = () => {
      s.keep = box.checked;
      stage.classList.toggle('dropped', !box.checked);
      refresh();
    };
    keep.append(box, document.createTextNode('Use this accent'));

    meta.append(
      el('h2', null, (i + 1) + '. ' + s.name),
      keep,
      el('dl', null,
        '<dt>at</dt><dd>' + s.at.toFixed(2) + 's</dd>' +
        '<dt>proposed</dt><dd>' + s.cell + ' · ' + s.scale + '</dd>' +
        '<dt>check</dt><dd>' + verdict + '</dd>' +
        '<dt>because</dt><dd>' + s.reason + '</dd>' +
        (s.text ? '<dt>says</dt><dd>&ldquo;' + s.text + '&rdquo;</dd>' : ''))
    );

    const row = el('div', 'row');
    const pick = el('select');
    SPRITES.forEach(sp => {
      const o = el('option');
      o.value = sp.name; o.textContent = sp.name; o.selected = (sp.name === s.name);
      pick.append(o);
    });
    const note = el('p', 'hint');
    pick.onchange = () => {
      s.name = pick.value;
      const chosen = SPRITES.find(sp => sp.name === s.name);
      // Swap the drawing, and resize it the way the compositor would: height is
      // a fixed fraction of the frame for this scale, width follows the new
      // sprite's own aspect. Keep the centre so a swap does not also move it.
      const cx = s.x + s.w / 2, cy = s.y + s.h / 2;
      s.h = chosen.height_frac;
      s.w = chosen.height_frac * chosen.aspect * FRAME_ASPECT;
      s.x = Math.min(1 - s.w, Math.max(0, cx - s.w / 2));
      s.y = Math.min(1 - s.h, Math.max(0, cy - s.h / 2));
      img.src = 'data:image/png;base64,' + chosen.data;
      sprite.style.width = (s.w * 100) + '%';
      sprite.style.left = (s.x * 100) + '%';
      sprite.style.top = (s.y * 100) + '%';
      note.textContent = chosen.takes_text
        ? 'this one is a speech bubble — its real size comes from the words in it'
        : '';
      refresh();
    };
    const snap = el('button', null, 'Snap to motion');
    snap.disabled = !s.motion;
    snap.onclick = () => snapOne(i);
    row.append(pick, snap);

    meta.append(row, note,
      el('p', 'hint', 'Drag the accent to place it exactly — its centre is what gets used.'));

    scene.append(stage, meta);
    root.append(scene);
    stage.classList.toggle('dropped', !s.keep);
  });
  refresh();
}

function makeDraggable(node, stage, i) {
  let startX = 0, startY = 0, baseX = 0, baseY = 0;
  node.addEventListener('pointerdown', e => {
    e.preventDefault();
    node.setPointerCapture(e.pointerId);
    node.classList.add('dragging');
    const r = stage.getBoundingClientRect();
    startX = e.clientX; startY = e.clientY;
    baseX = state[i].x * r.width; baseY = state[i].y * r.height;
  });
  node.addEventListener('pointermove', e => {
    if (!node.classList.contains('dragging')) return;
    const r = stage.getBoundingClientRect();
    const s = state[i];
    s.x = Math.min(1 - s.w, Math.max(0, (baseX + e.clientX - startX) / r.width));
    s.y = Math.min(1 - s.h, Math.max(0, (baseY + e.clientY - startY) / r.height));
    node.style.left = (s.x * 100) + '%';
    node.style.top = (s.y * 100) + '%';
    refresh();
  });
  const stop = () => node.classList.remove('dragging');
  node.addEventListener('pointerup', stop);
  node.addEventListener('pointercancel', stop);
}

function snapOne(i) {
  const s = state[i];
  if (!s.motion) return;
  s.x = Math.min(1 - s.w, Math.max(0, s.motion.x - s.w / 2));
  s.y = Math.min(1 - s.h, Math.max(0, s.motion.y - s.h / 2));
  const n = document.getElementById('sprite' + i);
  n.style.left = (s.x * 100) + '%';
  n.style.top = (s.y * 100) + '%';
  refresh();
}

function snapAll() { state.forEach((s, i) => snapOne(i)); }

function moved(s) {
  return Math.abs(s.x - s.x0) > 0.002 || Math.abs(s.y - s.y0) > 0.002;
}

function decisions() {
  return {
    version: 1,
    events: state.map(s => ({
      index: s.index,
      // The moment this decision is about. Indices only mean anything against
      // the exact plan this page was drawn from; the time survives a re-plan.
      at: s.at,
      from_name: s.name0,
      keep: s.keep,
      effect_name: s.name,
      // The sprite's CENTRE, which is what a person aims at when dragging.
      x: +(s.x + s.w / 2).toFixed(4),
      y: +(s.y + s.h / 2).toFixed(4),
      moved: moved(s),
      swapped: s.name !== s.name0
    }))
  };
}

function refresh() {
  persist();
  const kept = state.filter(s => s.keep).length;
  document.getElementById('count').textContent =
    kept + ' of ' + state.length + ' kept · ' +
    state.filter(moved).length + ' moved · ' +
    state.filter(s => s.name !== s.name0).length + ' swapped';
  const out = document.getElementById('out');
  if (out.style.display === 'block') out.value = JSON.stringify(decisions(), null, 2);
}

// Set when the page is served by `videoai approve-effects`; empty when the file
// was opened straight off the disk, in which case there is nobody to post to.
const SAVE_URL = '__SAVE_URL__';
const TOKEN = '__TOKEN__';

async function save() {
  if (!SAVE_URL) {
    // Opened as a file. Fall back to the download so the button still does
    // something honest rather than failing silently.
    download();
    return;
  }
  const button = document.getElementById('savebtn');
  const wasLabel = button.textContent;
  button.textContent = 'Saving...';
  button.disabled = true;
  try {
    const response = await fetch(SAVE_URL, {
      method: 'POST',
      headers: {'Content-Type': 'application/json', 'X-Approval-Token': TOKEN},
      body: JSON.stringify(decisions())
    });
    const answer = await response.json();
    if (!answer.ok) throw new Error(answer.error || 'refused');
    const box = document.getElementById('handoff');
    box.className = 'warn-box restored';
    box.innerHTML = '<b>Saved.</b> ' + answer.summary +
      ' — this is now what the video will be rendered with.';
    button.textContent = 'Saved ✓';
  } catch (error) {
    const box = document.getElementById('handoff');
    box.className = 'warn-box';
    box.innerHTML = '<b>Could not save:</b> ' + error.message +
      '. The approval command may have stopped — use <b>Copy as text</b> instead.';
    button.textContent = wasLabel;
  } finally {
    button.disabled = false;
    setTimeout(() => { button.textContent = 'Save'; }, 2500);
  }
}

function download() {
  const blob = new Blob([JSON.stringify(decisions(), null, 2)], {type:'application/json'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'effects-decisions.json';
  a.click();
  URL.revokeObjectURL(a.href);
}

function copyOut() {
  const out = document.getElementById('out');
  out.style.display = 'block';
  out.value = JSON.stringify(decisions(), null, 2);
  out.select();
  try { document.execCommand('copy'); } catch (e) {}
}

build();
if (RESTORED) {
  const box = document.getElementById('handoff');
  box.className = 'warn-box restored';
  box.innerHTML = '<b>Your earlier edits were restored</b> from this browser. ' +
    'They still have to be sent back: press <b>Copy as text</b> and paste them, ' +
    'or <b>Download decisions</b> and run <code>videoai apply-effects</code>.';
}
</script>
</body></html>
"""
