"""The approval page: the whole edit, laid out and changeable in place.

An edit read as text cannot be judged. "comic_starburst at 131.14s,
bottom-right" says nothing about whether it covers a face, points at what is
happening, or sits over a paper towel — and on this project's first delivery it
was the last of those, seven times over. A running order read as a list of
phrase ids is no better: nobody can tell from `clip-01#007` that the sign-off is
in the middle of the video.

So the edit is drawn. Every shot in running order with the frame it plays, its
beat, its line and its length; drag one to move it, untick one to take it out.
Its accents are nested inside it, so a shot that moves visibly takes its badges
with it — which is the whole of why an accent is anchored to a shot rather than
to a second. Each badge can be swapped, dragged, resized and turned.

A shot switched off stays on the page, greyed and in position. A shot that
vanished would leave the creator no way back short of a re-plan, and no way to
see what they had already decided against.

The page writes nothing by itself. It produces a decisions document the pipeline
reads back into `05e-overrides`, so what reaches the render is exactly what was
approved — and the frames, which are of a child, never leave the machine.
"""
from __future__ import annotations

import base64
import html
import json
from dataclasses import dataclass

from videoai.core.models import EffectEvent
from videoai.logic.effects import (
    MAX_SCALE_FACTOR,
    MIN_SCALE_FACTOR,
    EffectLibrary,
    place_sprite,
)


@dataclass(frozen=True)
class ShotCard:
    """One timeline clip as the strip shows it."""

    ref: str
    beat: str
    quote: str
    duration: float
    # Whether the creator has this shot switched on. Read from the overrides
    # already on file, so a second visit opens showing the edit as it stands
    # rather than as the planner first proposed it.
    enabled: bool
    thumb_jpeg: bytes


@dataclass(frozen=True)
class EffectProposal:
    """One accent, with everything the page needs to show and change it."""

    index: int
    event: EffectEvent
    # The shot this accent is drawn under. Normally `event.clip_ref`; for an
    # accent from before refs existed it is whichever shot currently plays at its
    # second, so it still has somewhere to be nested. Display only — what a
    # decision is filed under comes from the event itself.
    clip_ref: str
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
    "as you go, and Save writes them straight into the edit — nothing else to do."
)

FILE_HANDOFF = (
    "<b>Your edits stay in this browser.</b> They are saved here automatically and "
    "survive a refresh, but nothing reaches the video until you press "
    "<b>Copy as text</b> and paste it back. To save directly instead, run "
    "<code>videoai edit &lt;project&gt;</code>."
)


def _embedded_json(payload) -> str:
    """`payload` as JSON safe to drop inside a `<script>` element.

    A quote lifted from the transcript is arbitrary text, and one containing
    `</script` would end the element early and leave the rest of the page as
    prose. Escaping the only sequence that can do that costs nothing and removes
    the whole class of failure.
    """
    return json.dumps(payload).replace("</", "<\\/")


def render_preview_html(
    shots: list[ShotCard],
    proposals: list[EffectProposal],
    title: str,
    sprite_choices: list[dict],
    project_key: str,
    save_url: str = "",
    token: str = "",
) -> str:
    """One self-contained page: the edit, drawn, and changeable in place.

    `sprite_choices` carries every sprite the library offers — its picture and
    its aspect ratio — so choosing a different one actually changes the picture.
    An earlier version embedded only the sprite already planned, which meant the
    dropdown changed the answer while the frame went on showing the old drawing:
    a preview that stops previewing the moment you use it.

    `project_key` scopes what the browser remembers. It used to be the first
    accent's timestamp, which meant two projects whose first accent happened to
    land on the same second shared one another's saved work, and a re-plan that
    moved that accent orphaned every edit in the tab without saying so.
    """
    shot_payload = [
        {
            "ref": shot.ref,
            "beat": shot.beat,
            "quote": shot.quote,
            "dur": round(shot.duration, 2),
            "enabled": shot.enabled,
            "thumb": base64.b64encode(shot.thumb_jpeg).decode("ascii"),
        }
        for shot in shots
    ]
    accent_payload = [
        {
            "index": item.index,
            "at": round(item.event.at_seconds, 2),
            # The shot this is nested under, and the identity a decision about it
            # is filed under. `anchor` is `EffectEvent.anchor_key` — the same
            # string the pipeline matches on — so what the browser remembers and
            # what the pipeline stores cannot drift apart.
            "shot": item.clip_ref,
            "anchor": item.event.anchor_key,
            "clip_ref": item.event.clip_ref,
            "at_in_clip": round(item.event.at_in_clip, 3),
            "name": item.event.effect_name,
            "scale": item.event.scale,
            "text": item.event.text,
            "reason": item.event.reason,
            "cell": item.event.screen_position,
            "keep": item.event.keep,
            "factor": round(item.event.scale_factor, 3),
            "rot": round(item.event.rotation, 1),
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
        .replace("__PROJECT__", html.escape(project_key))
        .replace("__MIN_FACTOR__", str(MIN_SCALE_FACTOR))
        .replace("__MAX_FACTOR__", str(MAX_SCALE_FACTOR))
        .replace("__SPRITES__", _embedded_json(sprite_choices))
        .replace("__SHOTS__", _embedded_json(shot_payload))
        .replace("__DATA__", _embedded_json(accent_payload))
    )


# Written out in full rather than assembled from fragments: this is one artefact
# a creator opens on their own machine, and a single readable template is easier
# to change than a string built in six places. It is laid out in the order the
# page is: styles, the strip of shots, the per-shot accent editors, then the
# state, the layout, and the handoff.
_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Edit — __TITLE__</title>
<style>
  :root { color-scheme: dark; --line:#232a34; --muted:#93a3b8; --bg:#12151a;
          --panel:#171c24; --accent:#2f6df6; }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:#e8ecf1;
         font:15px/1.55 -apple-system, system-ui, sans-serif; }
  header { padding:22px 28px 12px; border-bottom:1px solid var(--line); }
  h1 { font-size:21px; margin:0 0 6px; }
  .lede { color:var(--muted); margin:0; max-width:78ch; }
  .bar { display:flex; gap:10px; align-items:center; margin-top:14px; flex-wrap:wrap; }
  button { font:inherit; padding:8px 14px; border-radius:8px; border:1px solid var(--line);
           background:#1b2130; color:#e8ecf1; cursor:pointer; }
  button.primary { background:var(--accent); border-color:var(--accent); }
  button.tiny { padding:3px 9px; font-size:13px; }
  button:disabled { opacity:.4; cursor:default; }
  button:not(:disabled):hover { filter:brightness(1.15); }
  .count { color:var(--muted); }
  .warn-box { margin:12px 0 0; padding:10px 13px; border-radius:8px; max-width:78ch;
              background:#2a2113; border:1px solid #6b5320; color:#ffd79a; }
  .warn-box code { background:#00000040; padding:1px 5px; border-radius:4px; }
  .restored { background:#132a1c; border-color:#2c6b46; color:#9ae6bb; }
  #out { width:100%; height:170px; margin-top:10px; font:12px/1.4 ui-monospace,monospace;
         background:#0d1016; color:#9fe6b8; border:1px solid var(--line);
         border-radius:8px; padding:10px; display:none; }

  /* The running order: one card per shot, in a row you can drag things around. */
  .striprow { position:sticky; top:0; z-index:20; background:var(--bg);
              border-bottom:1px solid var(--line); padding:14px 28px 16px; }
  .strip { display:flex; gap:12px; overflow-x:auto; padding-bottom:6px; align-items:stretch; }
  .card { flex:0 0 210px; background:var(--panel); border:1px solid var(--line);
          border-radius:10px; padding:8px; cursor:grab; position:relative;
          display:flex; flex-direction:column; gap:6px; }
  .card.dragging { opacity:.45; cursor:grabbing; }
  .card.off { filter:grayscale(1); opacity:.45; }
  .card.off .use { filter:none; opacity:1; }
  .card img.thumb { width:100%; border-radius:6px; display:block; background:#000; }
  .card .pos { position:absolute; top:12px; left:12px; background:#000000b0;
               border-radius:6px; padding:1px 7px; font-size:13px; font-weight:600; }
  .card .beat { font-weight:600; font-size:14px; }
  .card .quote { color:var(--muted); font-size:13px; max-height:3.2em; overflow:hidden; }
  .card .dur { color:var(--muted); font-size:12px; }
  .card .chips { display:flex; gap:5px; flex-wrap:wrap; min-height:26px; align-items:center; }
  .card .chips img { height:24px; width:auto; }
  .card .chips img.gone { opacity:.28; }
  .card .chips .empty { color:#5d6b7d; font-size:12px; }
  .card .foot { display:flex; gap:6px; align-items:center; justify-content:space-between; }
  label.use { display:flex; gap:6px; align-items:center; font-size:13px; cursor:pointer; }

  /* One panel per shot, holding that shot's accents. */
  .shot { border-bottom:1px solid var(--line); padding:20px 28px; }
  .shot.off { opacity:.5; }
  .shot h2 { font-size:16px; margin:0 0 4px; }
  .shot h2 .pos { color:var(--muted); margin-right:8px; }
  .shot .line { color:var(--muted); margin:0 0 10px; }
  .shot .none { color:#5d6b7d; margin:0; font-size:14px; }
  .shot .offnote { color:#ffc266; margin:0 0 10px; }
  .scene { display:grid; grid-template-columns:minmax(340px,1.9fr) 1fr; gap:22px;
           padding:14px 0; }
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
  .sprite.dragging { cursor:grabbing; outline:2px dashed var(--accent); outline-offset:3px; }
  .meta h3 { font-size:15px; margin:0 0 10px; }
  label.keep { display:flex; gap:8px; align-items:center; font-weight:600; margin-bottom:12px; }
  dl { display:grid; grid-template-columns:auto 1fr; gap:4px 12px; margin:0 0 12px; }
  dt { color:var(--muted); }
  dd { margin:0; }
  .row { display:flex; gap:8px; align-items:center; margin-bottom:8px; flex-wrap:wrap; }
  .row > label { color:var(--muted); min-width:56px; }
  .row output { min-width:52px; font-variant-numeric:tabular-nums; }
  select { font:inherit; padding:6px 8px; border-radius:7px; background:#1b2130;
           color:#e8ecf1; border:1px solid var(--line); }
  input[type=range] { flex:1 1 130px; accent-color:var(--accent); }
  .hint { color:var(--muted); font-size:13px; margin:6px 0 0; }
  .ok { color:#7ee0a8; } .warn { color:#ffc266; }
</style></head>
<body>
<header>
  <h1>Edit — __TITLE__</h1>
  <p class="lede"><b>Drag a shot</b> in the row below to move it, or <b>untick</b> it to
  take it out — it stays where it is, greyed, so you can put it back. Each shot's
  accents sit under it: <b>swap</b>, <b>drag</b>, <b>resize</b> or <b>turn</b> any of them.
  The green target is where the picture is actually moving at that moment.</p>
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
<div class="striprow"><div class="strip" id="strip"></div></div>
<main id="shots"></main>
<script>
const SPRITES = __SPRITES__;
const SHOTS = __SHOTS__;
const DATA = __DATA__;
const MIN_FACTOR = __MIN_FACTOR__;
const MAX_FACTOR = __MAX_FACTOR__;
// Sprite heights are a fraction of the frame's HEIGHT while CSS widths are a
// fraction of its WIDTH, so converting between them needs the frame's shape.
const FRAME_ASPECT = DATA.length ? (DATA[0].frame_h / DATA[0].frame_w) : (9 / 16);

// A shot's on/off state and the running order are separate: switching one off
// leaves it exactly where it was, which is what lets a creator see and undo it.
const shots = SHOTS.map(s => ({...s, on:s.enabled}));
const shotOf = new Map(shots.map(s => [s.ref, s]));
let order = shots.map(s => s.ref);
const planned = order.slice();

// `w`/`h` are what the sprite measures ON SCREEN, which already includes the
// size the creator dragged it to last time. `bw`/`bh` are the same drawing at
// factor 1, so resizing is a multiplication rather than a running total that
// drifts every time the page is reopened.
const state = DATA.map(d => ({
  ...d,
  x0:d.x, y0:d.y, w0:d.w, h0:d.h, name0:d.name, factor0:d.factor, rot0:d.rot,
  bw: d.w / (d.factor || 1), bh: d.h / (d.factor || 1),
}));

// The browser will not let this page write to disk, so every edit is kept in
// localStorage instead. Without it a refresh, a closed tab or a regenerated
// page threw away everything the creator had done — which is exactly what
// happened the first time somebody used this in anger.
//
// Keyed on the PROJECT, and every row keyed on the thing it is about: a shot by
// its ref, an accent by its anchor. Keying the store on the first accent's
// timestamp made two unrelated projects share one another's work, and matching
// rows back by array index re-applied a decision to whatever had moved into that
// slot — the same bug that was fixed in the pipeline and left standing here.
const SAVE_KEY = 'videoai-edit-__PROJECT__';

function persist() {
  try {
    localStorage.setItem(SAVE_KEY, JSON.stringify({
      version: 2,
      order: order,
      shots: shots.map(s => ({ref:s.ref, on:s.on})),
      accents: state.map(s => ({
        anchor:s.anchor, keep:s.keep, name:s.name,
        x:s.x, y:s.y, w:s.w, h:s.h, bw:s.bw, bh:s.bh,
        factor:s.factor, rot:s.rot,
      })),
    }));
  } catch (e) {}
}

function mergeOrder(saved) {
  // A shot the plan grew since this browser last saw it keeps its PLANNED place,
  // immediately after the nearest earlier shot that is still there. Appending it
  // would put new material after the sign-off, which is not "leaving it alone".
  // Same rule as `apply_clip_overrides`, so the page and the pipeline agree.
  const known = new Set(order);
  const kept = saved.filter(ref => known.has(ref));
  const mentioned = new Set(saved);
  const trailing = new Map();
  let anchor = null;
  planned.forEach(ref => {
    if (mentioned.has(ref)) { anchor = ref; return; }
    if (!trailing.has(anchor)) trailing.set(anchor, []);
    trailing.get(anchor).push(ref);
  });
  const merged = (trailing.get(null) || []).slice();
  kept.forEach(ref => { merged.push(ref); merged.push(...(trailing.get(ref) || [])); });
  return merged;
}

function restore() {
  // True when there was anything saved to come back to. Deliberately not "does
  // it differ from the proposal": the creator wants to know their work survived,
  // and re-deciding to keep exactly what was proposed is still their decision.
  try {
    const saved = JSON.parse(localStorage.getItem(SAVE_KEY) || 'null');
    if (!saved || typeof saved !== 'object') return false;
    if (Array.isArray(saved.order)) order = mergeOrder(saved.order);
    (saved.shots || []).forEach(row => {
      const shot = shotOf.get(row.ref);
      if (shot) shot.on = row.on !== false;
    });
    (saved.accents || []).forEach(row => {
      const s = state.find(item => item.anchor === row.anchor);
      if (!s) return;
      s.keep = row.keep; s.name = row.name;
      s.x = row.x; s.y = row.y; s.w = row.w; s.h = row.h;
      if (row.bw) { s.bw = row.bw; s.bh = row.bh; }
      if (row.factor) s.factor = row.factor;
      if (row.rot !== undefined) s.rot = row.rot;
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

function text(tag, cls, words) {
  const n = el(tag, cls);
  n.textContent = words;
  return n;
}

function accentsOf(ref) { return state.filter(s => s.shot === ref); }

// --- the strip of shots ------------------------------------------------------

const cards = new Map();
const panels = new Map();
const chips = new Map();

function buildCard(shot) {
  const card = el('article', 'card');
  card.draggable = true;
  const thumb = el('img', 'thumb');
  thumb.src = 'data:image/jpeg;base64,' + shot.thumb;
  const chipRow = el('div', 'chips');
  card.append(
    thumb, el('div', 'pos'),
    text('div', 'beat', shot.beat || 'Shot'),
    text('div', 'quote', shot.quote ? '\\u201c' + shot.quote + '\\u201d' : ''),
    text('div', 'dur', shot.dur.toFixed(1) + 's'),
    chipRow
  );

  accentsOf(shot.ref).forEach(s => {
    const chip = el('img');
    chip.title = s.name;
    chips.set(s.anchor, chip);
    chipRow.append(chip);
  });
  if (!accentsOf(shot.ref).length) chipRow.append(text('span', 'empty', 'no accents'));

  const foot = el('div', 'foot');
  const use = el('label', 'use');
  const box = el('input');
  box.type = 'checkbox';
  box.checked = shot.on;
  box.onchange = () => { shot.on = box.checked; paintShot(shot.ref); refresh(); };
  use.append(box, document.createTextNode('Use'));
  // Buttons as well as dragging: a keyboard, a trackpad nobody likes and a
  // browser that will not drag are all cheaper to support than to argue with.
  const back = el('button', 'tiny', '\\u25c0');
  back.title = 'move earlier';
  back.onclick = () => moveShot(shot.ref, order.indexOf(shot.ref) - 1);
  const forward = el('button', 'tiny', '\\u25b6');
  forward.title = 'move later';
  forward.onclick = () => moveShot(shot.ref, order.indexOf(shot.ref) + 2);
  const nudge = el('div');
  nudge.append(back, forward);
  foot.append(use, nudge);
  card.append(foot);

  card.addEventListener('dragstart', event => {
    dragging = shot.ref;
    card.classList.add('dragging');
    event.dataTransfer.effectAllowed = 'move';
    // Firefox refuses to start a drag without payload, and never reads it back.
    event.dataTransfer.setData('text/plain', shot.ref);
  });
  card.addEventListener('dragend', () => {
    dragging = null;
    card.classList.remove('dragging');
    refresh();
  });
  card.addEventListener('dragover', event => {
    event.preventDefault();
    if (!dragging || dragging === shot.ref) return;
    const rect = card.getBoundingClientRect();
    const after = event.clientX > rect.left + rect.width / 2;
    moveShot(dragging, order.indexOf(shot.ref) + (after ? 1 : 0));
  });
  card.addEventListener('drop', event => event.preventDefault());
  return card;
}

let dragging = null;

function moveShot(ref, to) {
  const from = order.indexOf(ref);
  if (from < 0) return;
  const target = Math.max(0, Math.min(order.length - 1, to > from ? to - 1 : to));
  if (target === from) return;
  order.splice(from, 1);
  order.splice(target, 0, ref);
  relayout();
  refresh();
}

// --- one panel per shot, holding that shot's accents -------------------------

function buildPanel(shot) {
  const panel = el('section', 'shot');
  panel.append(el('h2', null, '<span class="pos"></span>' + escapeHtml(shot.beat || 'Shot')));
  panel.append(text('p', 'line', shot.quote ? '\\u201c' + shot.quote + '\\u201d' : ''));
  panel.append(text('p', 'offnote',
    'This shot is switched off: it will not be in the video, and neither will its accents.'));
  const mine = accentsOf(shot.ref);
  if (!mine.length) panel.append(text('p', 'none', 'No accents on this shot.'));
  mine.forEach(s => panel.append(buildScene(s)));
  return panel;
}

function buildScene(s) {
  const scene = el('section', 'scene');
  const stage = el('div', 'stage');
  const frame = el('img', 'frame');
  frame.src = 'data:image/jpeg;base64,' + s.frame;
  stage.append(frame, el('div', 'grid'));

  if (s.motion) {
    const target = el('div', 'motion');
    target.style.left = (s.motion.x * 100) + '%';
    target.style.top = (s.motion.y * 100) + '%';
    stage.append(target);
  }

  const sprite = el('div', 'sprite');
  const img = el('img');
  // A restored swap has to show the restored drawing, not the proposed one.
  const restored = SPRITES.find(sp => sp.name === s.name);
  img.src = 'data:image/png;base64,' +
            (restored && s.name !== s.name0 ? restored.data : s.sprite);
  sprite.append(img);
  stage.append(sprite);
  s.node = sprite;
  s.picture = img;
  paintSprite(s);
  makeDraggable(sprite, stage, s);

  const verdict = s.motion
    ? (s.motion.cell === s.cell
        ? '<span class="ok">on the action</span>'
        : '<span class="warn">motion is at <b>' + escapeHtml(s.motion.cell) + '</b></span>')
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
    text('h3', null, s.name),
    keep,
    el('dl', null,
      '<dt>at</dt><dd>' + s.at.toFixed(2) + 's</dd>' +
      '<dt>proposed</dt><dd>' + escapeHtml(s.cell) + ' \\u00b7 ' + escapeHtml(s.scale) + '</dd>' +
      '<dt>check</dt><dd>' + verdict + '</dd>' +
      '<dt>because</dt><dd>' + escapeHtml(s.reason) + '</dd>' +
      (s.text ? '<dt>says</dt><dd>&ldquo;' + escapeHtml(s.text) + '&rdquo;</dd>' : ''))
  );

  const note = el('p', 'hint');
  meta.append(buildSwapRow(s, note), note);
  meta.append(buildSizeRow(s), buildTurnRow(s));
  meta.append(text('p', 'hint',
    'Drag the accent to place it exactly \\u2014 its centre is what gets used.'));

  scene.append(stage, meta);
  stage.classList.toggle('dropped', !s.keep);
  return scene;
}

function buildSwapRow(s, note) {
  const row = el('div', 'row');
  const pick = el('select');
  SPRITES.forEach(sp => {
    const option = el('option');
    option.value = sp.name; option.textContent = sp.name;
    option.selected = (sp.name === s.name);
    pick.append(option);
  });
  pick.onchange = () => {
    s.name = pick.value;
    const chosen = SPRITES.find(sp => sp.name === s.name);
    // Swap the drawing and take its own shape with it: height is a fixed
    // fraction of the frame for this scale band, width follows the new sprite's
    // aspect, and the creator's own size multiplies both.
    s.bh = chosen.height_frac;
    s.bw = chosen.height_frac * chosen.aspect * FRAME_ASPECT;
    s.picture.src = 'data:image/png;base64,' + chosen.data;
    resize(s, s.factor);
    note.textContent = chosen.takes_text
      ? 'this one is a speech bubble \\u2014 its real size comes from the words in it'
      : '';
    refresh();
  };
  const snap = el('button', null, 'Snap to motion');
  snap.disabled = !s.motion;
  snap.onclick = () => snapOne(s);
  row.append(el('label', null, 'badge'), pick, snap);
  return row;
}

function buildSizeRow(s) {
  const row = el('div', 'row');
  const slider = el('input');
  slider.type = 'range';
  slider.min = Math.round(MIN_FACTOR * 100);
  slider.max = Math.round(MAX_FACTOR * 100);
  slider.step = 5;
  slider.value = Math.round(s.factor * 100);
  const readout = el('output');
  readout.textContent = slider.value + '%';
  slider.oninput = () => {
    resize(s, slider.value / 100);
    readout.textContent = Math.round(s.factor * 100) + '%';
    refresh();
  };
  const reset = el('button', 'tiny', 'reset');
  reset.onclick = () => {
    resize(s, 1);
    slider.value = 100;
    readout.textContent = '100%';
    refresh();
  };
  row.append(el('label', null, 'size'), slider, readout, reset);
  return row;
}

function buildTurnRow(s) {
  const row = el('div', 'row');
  const slider = el('input');
  slider.type = 'range';
  slider.min = -180; slider.max = 180; slider.step = 1;
  slider.value = s.rot;
  const readout = el('output');
  const show = () => { readout.textContent = Math.round(s.rot) + '\\u00b0'; };
  show();
  const turn = degrees => {
    s.rot = Math.max(-180, Math.min(180, s.rot + degrees));
    slider.value = s.rot;
    paintSprite(s);
    show();
    refresh();
  };
  slider.oninput = () => { s.rot = +slider.value; paintSprite(s); show(); refresh(); };
  const left = el('button', 'tiny', '\\u21ba 15\\u00b0');
  left.onclick = () => turn(-15);
  const right = el('button', 'tiny', '15\\u00b0 \\u21bb');
  right.onclick = () => turn(15);
  row.append(el('label', null, 'turn'), left, slider, right, readout);
  return row;
}

function resize(s, factor) {
  // Grow and shrink about the CENTRE: a resize that also slid the badge sideways
  // would undo the placement the creator had just dragged it to.
  const cx = s.x + s.w / 2, cy = s.y + s.h / 2;
  s.factor = Math.max(MIN_FACTOR, Math.min(MAX_FACTOR, factor));
  s.w = s.bw * s.factor;
  s.h = s.bh * s.factor;
  s.x = clamp(cx - s.w / 2, s.w);
  s.y = clamp(cy - s.h / 2, s.h);
  paintSprite(s);
}

function clamp(value, span) { return Math.min(Math.max(0, 1 - span), Math.max(0, value)); }

function paintSprite(s) {
  if (!s.node) return;
  s.node.style.width = (s.w * 100) + '%';
  s.node.style.left = (s.x * 100) + '%';
  s.node.style.top = (s.y * 100) + '%';
  s.node.style.transform = s.rot ? 'rotate(' + s.rot + 'deg)' : '';
}

function makeDraggable(node, stage, s) {
  let startX = 0, startY = 0, baseX = 0, baseY = 0;
  node.addEventListener('pointerdown', event => {
    event.preventDefault();
    node.setPointerCapture(event.pointerId);
    node.classList.add('dragging');
    const box = stage.getBoundingClientRect();
    startX = event.clientX; startY = event.clientY;
    baseX = s.x * box.width; baseY = s.y * box.height;
  });
  node.addEventListener('pointermove', event => {
    if (!node.classList.contains('dragging')) return;
    const box = stage.getBoundingClientRect();
    s.x = clamp((baseX + event.clientX - startX) / box.width, s.w);
    s.y = clamp((baseY + event.clientY - startY) / box.height, s.h);
    paintSprite(s);
    refresh();
  });
  const stop = () => node.classList.remove('dragging');
  node.addEventListener('pointerup', stop);
  node.addEventListener('pointercancel', stop);
}

function snapOne(s) {
  if (!s.motion) return;
  s.x = clamp(s.motion.x - s.w / 2, s.w);
  s.y = clamp(s.motion.y - s.h / 2, s.h);
  paintSprite(s);
  refresh();
}

function snapAll() { state.forEach(snapOne); }

// --- laying the page out in the running order --------------------------------

function relayout() {
  const strip = document.getElementById('strip');
  const main = document.getElementById('shots');
  order.forEach((ref, position) => {
    // Moving the EXISTING node rather than redrawing it: that is what carries a
    // shot's accent editors, their loaded frames and their sprites along with
    // it, and it is also what keeps a drag from being cancelled halfway.
    place(strip, cards.get(ref), position);
    place(main, panels.get(ref), position);
    const number = String(position + 1);
    cards.get(ref).querySelector('.pos').textContent = number;
    panels.get(ref).querySelector('.pos').textContent = number + '.';
  });
}

function place(parent, node, position) {
  if (parent.children[position] === node) return;
  parent.insertBefore(node, parent.children[position] || null);
}

function paintShot(ref) {
  const on = shotOf.get(ref).on;
  cards.get(ref).classList.toggle('off', !on);
  const panel = panels.get(ref);
  panel.classList.toggle('off', !on);
  panel.querySelector('.offnote').style.display = on ? 'none' : 'block';
}

function build() {
  shots.forEach(shot => {
    cards.set(shot.ref, buildCard(shot));
    panels.set(shot.ref, buildPanel(shot));
  });
  relayout();
  shots.forEach(shot => paintShot(shot.ref));
  refresh();
}

// --- what comes back ---------------------------------------------------------

function moved(s) {
  // Compared on the CENTRE, which is both what gets sent and what a person aims
  // at. The top-left shifts whenever the badge is resized, so comparing that
  // reported every resize as a move as well.
  return Math.abs((s.x + s.w / 2) - (s.x0 + s.w0 / 2)) > 0.002
      || Math.abs((s.y + s.h / 2) - (s.y0 + s.h0 / 2)) > 0.002;
}

function resized(s) { return Math.abs(s.factor - s.factor0) > 0.001; }
function turned(s) { return Math.abs(s.rot - s.rot0) > 0.01; }
function reordered() { return order.join('\\u0000') !== planned.join('\\u0000'); }

function decisions() {
  return {
    version: 2,
    // Position in this list is position in the edit, so a reorder and a shot
    // switched off are the same fact written the same way.
    clips: order.map(ref => ({ref: ref, enabled: shotOf.get(ref).on})),
    events: state.map(s => ({
      index: s.index,
      // The moment this decision is about. Indices only mean anything against
      // the exact plan this page was drawn from; the anchor survives a re-plan.
      at: s.at,
      clip_ref: s.clip_ref,
      at_in_clip: s.at_in_clip,
      from_name: s.name0,
      keep: s.keep,
      effect_name: s.name,
      // The sprite's CENTRE, which is what a person aims at when dragging.
      x: +(s.x + s.w / 2).toFixed(4),
      y: +(s.y + s.h / 2).toFixed(4),
      scale_factor: +s.factor.toFixed(3),
      rotation: +s.rot.toFixed(1),
      moved: moved(s),
      resized: resized(s),
      rotated: turned(s),
      swapped: s.name !== s.name0
    }))
  };
}

function refresh() {
  persist();
  state.forEach(s => {
    const chip = chips.get(s.anchor);
    if (!chip) return;
    const drawing = SPRITES.find(sp => sp.name === s.name);
    chip.src = 'data:image/png;base64,' + (drawing ? drawing.data : s.sprite);
    chip.classList.toggle('gone', !s.keep || !shotOf.get(s.shot).on);
  });
  const live = order.filter(ref => shotOf.get(ref).on);
  const kept = state.filter(s => s.keep && shotOf.get(s.shot).on).length;
  document.getElementById('count').textContent =
    live.length + ' of ' + shots.length + ' shots' +
    (reordered() ? ' \\u00b7 reordered' : '') + ' \\u00b7 ' +
    kept + ' of ' + state.length + ' accents kept \\u00b7 ' +
    state.filter(moved).length + ' moved \\u00b7 ' +
    state.filter(resized).length + ' resized \\u00b7 ' +
    state.filter(turned).length + ' turned \\u00b7 ' +
    state.filter(s => s.name !== s.name0).length + ' swapped';
  const out = document.getElementById('out');
  if (out.style.display === 'block') out.value = JSON.stringify(decisions(), null, 2);
}

function escapeHtml(value) {
  const node = document.createElement('span');
  node.textContent = value === undefined || value === null ? '' : value;
  return node.innerHTML;
}

// Set when the page is served by `videoai edit`; empty when the file was opened
// straight off the disk, in which case there is nobody to post to.
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
    box.innerHTML = '<b>Saved.</b> ' + escapeHtml(answer.summary) +
      ' \\u2014 this is now what the video will be rendered with.';
    button.textContent = 'Saved \\u2713';
  } catch (error) {
    const box = document.getElementById('handoff');
    box.className = 'warn-box';
    box.innerHTML = '<b>Could not save:</b> ' + escapeHtml(error.message) +
      '. The edit command may have stopped \\u2014 use <b>Copy as text</b> instead.';
    button.textContent = wasLabel;
  } finally {
    button.disabled = false;
    setTimeout(() => { button.textContent = 'Save'; }, 2500);
  }
}

function download() {
  const blob = new Blob([JSON.stringify(decisions(), null, 2)], {type:'application/json'});
  const link = document.createElement('a');
  link.href = URL.createObjectURL(blob);
  link.download = 'effects-decisions.json';
  link.click();
  URL.revokeObjectURL(link.href);
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
  // What to do next differs by how the page was opened, and telling a served
  // creator to copy text out when Save works is how a working button gets
  // ignored.
  box.innerHTML = '<b>Your earlier edits were restored</b> from this browser. ' +
    (SAVE_URL
      ? 'Press <b>Save</b> to write them into the edit.'
      : 'They still have to be sent back: press <b>Copy as text</b> and paste them, ' +
        'or <b>Download instead</b> and run <code>videoai apply-effects</code>.');
}
</script>
</body></html>
"""
