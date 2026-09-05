/* =====================================================================
   KRISH FACTORY — isometric scene renderer
   =====================================================================

   Everything here is drawn procedurally on a 2D canvas. No image assets, so
   the whole thing ships inside the repo and starts instantly on a VPS with no
   download step and no GPU requirement.

   Structure, and why:

     STATIC LAYER   platforms, gantries, signage, props, desks, rails.
                    Re-rendered only on resize, then blitted each frame. This is
                    ~90% of the drawing work and none of it changes per frame.

     DYNAMIC LAYER  monitors, characters, couriers, cargo pods, robot arms, the
                    strategy core, particles, light cones. Depth-sorted with the
                    agents so nothing floats in front of furniture it is behind.

     POST           vignette, scanlines, dust. Cheap, and does most of the work
                    of making flat shapes read as a lit room.

   The renderer is deliberately data-driven: SCENE describes the room, AGENT_DESKS
   places people. Swapping procedural drawing for sprite artwork later means
   replacing the draw* functions, not the layout.
   ===================================================================== */

"use strict";

/* ---------------------------------------------------------------- palette */
const PAL = {
  data: "#3ddbd9",
  research: "#a78bfa",
  build: "#5b9dff",
  validation: "#ffc75a",
  delivery: "#4ade80",
  system: "#8698bd",
  core: "#ff9f43",
  bad: "#ff6b81",
  metal: "#2b364f",
};

/* room description ------------------------------------------------------- */
/* The room is deliberately COMPACT. A wide grid fitted to the viewport makes
   every tile tiny and the detail unreadable, which defeats the point of drawing
   desks and people at all. Fewer, larger tiles beat a bigger empty floor. */
const SCENE = {
  grid: { w: 20, h: 15 },
  spineY: 14.2, // the cargo rail everyone walks along
  zones: [
    { key: "system",     label: "OPERATIONS", colour: "#8698bd", x: 0,  y: 0,  w: 5, h: 5, level: 0.92 },
    { key: "data",       label: "DATA INTAKE", colour: "#3ddbd9", x: 0,  y: 6,  w: 5, h: 5, level: 0.5  },
    { key: "research",   label: "RESEARCH LAB", colour: "#a78bfa", x: 6,  y: 0,  w: 5, h: 5, level: 0.78 },
    { key: "build",      label: "BUILD BAY",   colour: "#5b9dff", x: 6,  y: 6,  w: 5, h: 4, level: 0.4  },
    { key: "validation", label: "VALIDATION",  colour: "#ffc75a", x: 12, y: 0,  w: 8, h: 8, level: 0.66 },
    { key: "delivery",   label: "DELIVERY DOCK", colour: "#4ade80", x: 12, y: 9, w: 8, h: 3, level: 0.28 },
  ],
  core: { x: 6, y: 11, w: 4, h: 3, level: 1.2 },
  glass: { x: 13, y: 12.3, w: 6, h: 2 },
  tanks: [ [0.7, 11.4], [1.6, 11.4], [2.5, 11.4], [3.4, 11.4] ],
  racks: [ [4.1, 7.2], [4.1, 8.4], [4.1, 9.6] ],
  arms:  [ [12.7, 11.3], [19.1, 11.3] ],
  lamps: [ [2, 2], [2, 8], [8.5, 2], [8.5, 8], [16, 3], [16, 10.5], [8, 12.4] ],
};

const AGENT_DESKS = {
  orchestrator: [[1, 1]], memory: [[3.4, 1]], monitor: [[1, 3.4]], librarian: [[3.4, 3.4]],
  market_data: [[1.9, 7.4]],
  researcher: [[7, 1]], quant_analyst: [[9.4, 1]], architect: [[8.2, 3.4]],
  developer: [[8.2, 7.4]],
  tester:     [[12.9, 1],   [15.2, 1],   [17.5, 1],   [19.2, 1]],
  tuner:      [[12.9, 3.2], [15.2, 3.2], [17.5, 3.2], [19.2, 3.2]],
  robustness: [[12.9, 5.4], [15.2, 5.4], [17.5, 5.4], [19.2, 5.4]],
  risk: [[14, 7.2]], judge: [[18, 7.2]],
  doc_writer: [[13.4, 10]], packager: [[16, 10]], delivery: [[18.6, 10]],
};

/* ------------------------------------------------------------- projection */
/** Tallest thing above the deck: core platform + hologram + ceiling fixtures.
 *  Used to reserve vertical space so the room is centred on what is actually
 *  drawn, not on the flat grid. */
const HEAD_ROOM = 3.0;

/** A textbook isometric diamond is always 2:1. A browser viewport is more like
 *  3:1, so a "correct" projection leaves a third of the canvas empty and forces
 *  every object to be tiny. Stretching the horizontal axis trades geometric
 *  purity for a room that fills the screen and objects you can actually see. */
const STRETCH = 1.42;

class Iso {
  constructor() { this.s = 26; this.ox = 0; this.oy = 0; this.zoom = 1; this.pan = { x: 0, y: 0 }; }

  /** horizontal unit — stretched */
  get ux() { return this.s * this.zoom * STRETCH; }
  /** vertical unit — also the unit for heights, lifts and object sizes */
  get uy() { return this.s * this.zoom; }

  /** Fit the room to the viewport using the *rendered* bounding box.
   *  An iso grid of GWxGH is (GW+GH) tiles wide and half that tall, plus the
   *  head room above it. Centring on the flat grid instead leaves the room
   *  stranded in a corner with half the canvas empty. */
  fit(w, h) {
    const { w: GW, h: GH } = SCENE.grid;
    const span = GW + GH;
    const padX = 40, padY = 26;
    this.s = Math.max(13, Math.min(
      (w - padX * 2) / (span * STRETCH),
      (h - padY * 2) / (span * 0.5 + HEAD_ROOM),
    ));
    this.zoom = 1;
    this.ox = w / 2 - ((GW - GH) / 2) * this.ux;
    // vertical centre of the drawn content, head room included
    this.oy = h / 2 - (span * 0.5 * this.uy - HEAD_ROOM * this.uy) / 2;
  }
  /* Pan is deliberately NOT applied here. It is applied once as a canvas
     translate when drawing, so panning never invalidates the pre-rendered
     static layer - otherwise dragging would re-render the whole room per
     mouse-move, which is exactly as slow as it sounds. */
  p(gx, gy, lift = 0) {
    return {
      x: this.ox + (gx - gy) * this.ux,
      y: this.oy + (gx + gy) * this.uy * 0.5 - lift * this.uy,
    };
  }
  get unit() { return this.uy; }
}

/* ------------------------------------------------------------- utilities */
const rgba = (hex, a) => {
  const h = hex.replace("#", "");
  const v = h.length === 3 ? h.split("").map((c) => c + c).join("") : h;
  const n = parseInt(v, 16);
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
};
const mix = (a, b, t) => {
  const pa = parseInt(a.replace("#", ""), 16), pb = parseInt(b.replace("#", ""), 16);
  const r = Math.round((((pa >> 16) & 255) * (1 - t)) + (((pb >> 16) & 255) * t));
  const g = Math.round((((pa >> 8) & 255) * (1 - t)) + (((pb >> 8) & 255) * t));
  const bl = Math.round(((pa & 255) * (1 - t)) + ((pb & 255) * t));
  return `rgb(${r},${g},${bl})`;
};
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const ease = (t) => t * t * (3 - 2 * t);

/* =====================================================================
   primitive: an isometric box with three shaded faces and an optional
   emissive rim. Every solid object in the room is built from these.
   ===================================================================== */
function box(g, iso, gx, gy, w, d, hgt, base, opts = {}) {
  const p = iso.p(gx, gy, opts.lift || 0);
  const rx = w * iso.ux, ry = d * iso.uy * 0.5, hh = hgt * iso.uy;
  const top = opts.top || mix(base, "#ffffff", 0.22);
  const right = opts.right || mix(base, "#000000", 0.18);
  const left = opts.left || mix(base, "#000000", 0.42);

  g.beginPath();
  g.moveTo(p.x, p.y - hh);
  g.lineTo(p.x + rx, p.y + ry - hh);
  g.lineTo(p.x, p.y + ry * 2 - hh);
  g.lineTo(p.x - rx, p.y + ry - hh);
  g.closePath();
  g.fillStyle = top; g.fill();
  if (opts.topStroke) { g.strokeStyle = opts.topStroke; g.lineWidth = 1; g.stroke(); }

  if (hgt > 0) {
    g.beginPath();
    g.moveTo(p.x + rx, p.y + ry - hh);
    g.lineTo(p.x, p.y + ry * 2 - hh);
    g.lineTo(p.x, p.y + ry * 2);
    g.lineTo(p.x + rx, p.y + ry);
    g.closePath(); g.fillStyle = right; g.fill();

    g.beginPath();
    g.moveTo(p.x - rx, p.y + ry - hh);
    g.lineTo(p.x, p.y + ry * 2 - hh);
    g.lineTo(p.x, p.y + ry * 2);
    g.lineTo(p.x - rx, p.y + ry);
    g.closePath(); g.fillStyle = left; g.fill();

    if (opts.rim) {
      g.strokeStyle = opts.rim; g.lineWidth = 1.4;
      g.shadowColor = opts.rim; g.shadowBlur = 9;
      g.beginPath();
      g.moveTo(p.x - rx, p.y + ry - hh);
      g.lineTo(p.x, p.y - hh);
      g.lineTo(p.x + rx, p.y + ry - hh);
      g.lineTo(p.x, p.y + ry * 2 - hh);
      g.closePath(); g.stroke();
      g.shadowBlur = 0;
    }
  }
  return p;
}

/* a flat diamond, used for floors, light pools and glass */
function diamond(g, iso, gx, gy, w, d, fill, lift = 0, stroke = null) {
  const p = iso.p(gx, gy, lift);
  const rx = w * iso.ux, ry = d * iso.uy * 0.5;
  g.beginPath();
  g.moveTo(p.x, p.y);
  g.lineTo(p.x + rx, p.y + ry);
  g.lineTo(p.x, p.y + ry * 2);
  g.lineTo(p.x - rx, p.y + ry);
  g.closePath();
  if (fill) { g.fillStyle = fill; g.fill(); }
  if (stroke) { g.strokeStyle = stroke; g.lineWidth = 1; g.stroke(); }
  return p;
}

/* =====================================================================
   STATIC LAYER
   ===================================================================== */
function buildStatic(iso, W, H, dpr) {
  const c = document.createElement("canvas");
  c.width = W * dpr; c.height = H * dpr;
  const g = c.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0);

  drawVoid(g, iso);
  // far to near, so nearer platforms overlap the ones behind
  const zones = [...SCENE.zones].sort((a, b) => (a.x + a.y) - (b.x + b.y));
  for (const z of zones) drawPlatform(g, iso, z);
  drawGlassFloor(g, iso);
  drawRail(g, iso);
  drawCorePlatform(g, iso);
  for (const z of zones) drawSign(g, iso, z);
  drawGantries(g, iso);
  drawTanks(g, iso);
  drawRacks(g, iso);
  drawDesks(g, iso);
  return c;
}

/* the dark space the platforms float in, with a faint grid so scale reads */
function drawVoid(g, iso) {
  const { w: GW, h: GH } = SCENE.grid;
  for (let y = -1; y < GH + 2; y += 1) {
    for (let x = -1; x < GW + 2; x += 1) {
      diamond(g, iso, x, y, 1, 1, (x + y) % 2 ? "rgba(9,13,24,.5)" : "rgba(6,9,18,.5)", -0.35,
        "rgba(90,120,180,.03)");
    }
  }
}

function drawPlatform(g, iso, z) {
  const s = iso.unit;
  const lvl = z.level;

  // support pillars, drawn first so the slab sits on them
  const legs = [[z.x + 0.4, z.y + 0.4], [z.x + z.w - 0.4, z.y + 0.4],
                [z.x + 0.4, z.y + z.h - 0.4], [z.x + z.w - 0.4, z.y + z.h - 0.4]];
  for (const [lx, ly] of legs)
    box(g, iso, lx, ly, 0.22, 0.22, lvl, "#141c30", { top: "#1b2540" });

  // slab with thickness
  box(g, iso, z.x + z.w / 2, z.y + z.h / 2, z.w / 2, z.h / 2, 0.16, "#141d33", {
    lift: lvl,
    top: "#18223c",
    rim: rgba(z.colour, 0.55),
  });

  // deck: checker tiles tinted by department
  for (let y = 0; y < z.h; y++) {
    for (let x = 0; x < z.w; x++) {
      const checker = (x + y) % 2 === 0;
      diamond(g, iso, z.x + x, z.y + y, 1, 1,
        rgba(z.colour, checker ? 0.19 : 0.1), lvl + 0.16,
        "rgba(150,185,245,.085)");
    }
  }
  // hazard stripe along the front edge
  for (let x = 0; x < z.w; x += 0.5) {
    diamond(g, iso, z.x + x, z.y + z.h - 0.12, 0.25, 0.1,
      (x * 2) % 2 < 1 ? rgba(z.colour, 0.5) : "rgba(10,14,26,.6)", lvl + 0.17);
  }
}

function drawSign(g, iso, z) {
  const s = iso.unit;
  // Sits behind the zone's back edge and low, so it labels the department
  // without colliding with the name plates under each desk.
  const p = iso.p(z.x + z.w / 2, z.y - 0.9, z.level + 0.75);
  const label = z.label;
  g.font = `700 ${clamp(s * 0.34, 8, 12)}px ui-monospace,Menlo,Consolas,monospace`;
  const tw = g.measureText(label).width;
  const pad = 9, hgt = clamp(s * 0.7, 15, 24);

  // hanging bracket
  g.strokeStyle = "rgba(120,150,200,.35)"; g.lineWidth = 1.6;
  g.beginPath(); g.moveTo(p.x, p.y + hgt); g.lineTo(p.x, p.y + hgt + s * 0.55); g.stroke();

  // lit panel
  g.fillStyle = "rgba(7,11,21,.92)";
  g.strokeStyle = rgba(z.colour, 0.75); g.lineWidth = 1.3;
  g.shadowColor = rgba(z.colour, 0.9); g.shadowBlur = 16;
  roundRect(g, p.x - tw / 2 - pad, p.y, tw + pad * 2, hgt, 4);
  g.fill(); g.stroke();
  g.shadowBlur = 0;
  g.fillStyle = rgba(z.colour, 0.95);
  g.textAlign = "center"; g.textBaseline = "middle";
  g.fillText(label, p.x, p.y + hgt / 2 + 0.5);
  g.textBaseline = "alphabetic";
}

/* industrial trusses spanning the corridors — pure set dressing, big payoff */
function drawGantries(g, iso) {
  const spans = [
    { x1: 6.5, x2: 7.5, y1: 0, y2: 12, lift: 1.9 },
    { x1: 14.5, x2: 15.5, y1: 0, y2: 13, lift: 2.1 },
    { x1: 0, x2: 27, y1: 13.2, y2: 13.9, lift: 2.4 },
  ];
  for (const sp of spans) {
    const steps = Math.max(2, Math.round((sp.y2 - sp.y1) * 2));
    for (let i = 0; i <= steps; i++) {
      const t = i / steps;
      const gy = sp.y1 + (sp.y2 - sp.y1) * t;
      const gx = sp.x1 + (sp.x2 - sp.x1) * t;
      box(g, iso, gx, gy, 0.5, 0.14, 0.09, "#1a2338", { lift: sp.lift, top: "#222d4a" });
      if (i % 3 === 0)
        box(g, iso, gx, gy, 0.1, 0.1, sp.lift, "#121a2c", { top: "#1a2338" });
    }
  }
}

/* the glowing tubes in the data department — cached price history, visualised */
function drawTanks(g, iso) {
  for (const [tx, ty] of SCENE.tanks) {
    const lvl = 0.71;
    box(g, iso, tx, ty, 0.2, 0.2, 0.1, "#182034", { lift: lvl });
    const p = iso.p(tx, ty, lvl + 0.1);
    const s = iso.unit;
    const hgt = s * 1.15, wid = s * 0.3;
    const grad = g.createLinearGradient(p.x, p.y - hgt, p.x, p.y);
    grad.addColorStop(0, "rgba(167,139,250,.10)");
    grad.addColorStop(0.55, "rgba(167,139,250,.42)");
    grad.addColorStop(1, "rgba(61,219,217,.30)");
    g.fillStyle = grad;
    g.shadowColor = "rgba(167,139,250,.55)"; g.shadowBlur = 15;
    roundRect(g, p.x - wid / 2, p.y - hgt, wid, hgt, wid / 2); g.fill();
    g.shadowBlur = 0;
    g.strokeStyle = "rgba(190,205,255,.28)"; g.lineWidth = 1;
    roundRect(g, p.x - wid / 2, p.y - hgt, wid, hgt, wid / 2); g.stroke();
    box(g, iso, tx, ty, 0.22, 0.22, 0.08, "#232e4a", { lift: lvl + 1.25 });
  }
}

/* server racks — the blackboard, physically */
function drawRacks(g, iso) {
  for (const [rx, ry] of SCENE.racks) {
    const lvl = 0.71;
    const p = box(g, iso, rx, ry, 0.34, 0.34, 0.95, "#151d31", {
      lift: lvl, top: "#1d2740", rim: "rgba(61,219,217,.3)",
    });
    const s = iso.unit;
    for (let i = 0; i < 5; i++) {
      g.fillStyle = i % 2 ? "rgba(61,219,217,.5)" : "rgba(74,222,128,.45)";
      g.fillRect(p.x + s * 0.06, p.y + s * 0.1 - s * 0.95 + i * s * 0.16, s * 0.2, s * 0.03);
    }
  }
}

function drawGlassFloor(g, iso) {
  const gl = SCENE.glass;
  for (let y = 0; y < gl.h; y++) {
    for (let x = 0; x < gl.w; x++) {
      diamond(g, iso, gl.x + x, gl.y + y, 0.94, 0.94,
        (x + y) % 2 ? "rgba(70,110,220,.16)" : "rgba(50,90,200,.11)", 0,
        "rgba(130,175,255,.16)");
    }
  }
  // under-floor glow, so the glass reads as lit from below
  const p = iso.p(gl.x + gl.w / 2, gl.y + gl.h / 2, -0.1);
  const s = iso.unit;
  const rad = g.createRadialGradient(p.x, p.y, 0, p.x, p.y, s * gl.w * 0.7);
  rad.addColorStop(0, "rgba(80,130,255,.22)");
  rad.addColorStop(1, "transparent");
  g.fillStyle = rad;
  g.fillRect(p.x - s * gl.w, p.y - s * gl.h, s * gl.w * 2, s * gl.h * 2);
}

/* the cargo rail: the main corridor, and the track the pods run on */
function drawRail(g, iso) {
  const y = SCENE.spineY;
  for (let x = -0.5; x < SCENE.grid.w + 0.5; x += 1) {
    diamond(g, iso, x, y, 1, 0.62, "rgba(22,30,50,.9)", 0.02, "rgba(120,150,210,.07)");
  }
  // rails
  for (const off of [-0.22, 0.22]) {
    for (let x = -0.5; x < SCENE.grid.w + 0.5; x += 1) {
      diamond(g, iso, x, y + off, 1, 0.06, "rgba(150,175,225,.28)", 0.05);
    }
  }
  // sleepers
  for (let x = -0.5; x < SCENE.grid.w + 0.5; x += 0.5) {
    diamond(g, iso, x, y, 0.09, 0.32, "rgba(60,78,115,.5)", 0.03);
  }
}

function drawCorePlatform(g, iso) {
  const c = SCENE.core;
  const legs = [[c.x + 0.5, c.y + 0.5], [c.x + c.w - 0.5, c.y + 0.5],
                [c.x + 0.5, c.y + c.h - 0.5], [c.x + c.w - 0.5, c.y + c.h - 0.5]];
  for (const [lx, ly] of legs)
    box(g, iso, lx, ly, 0.26, 0.26, c.level, "#131b2e", { top: "#1c2540" });

  box(g, iso, c.x + c.w / 2, c.y + c.h / 2, c.w / 2, c.h / 2, 0.2, "#161f36", {
    lift: c.level, top: "#1b2540", rim: rgba(PAL.core, 0.5),
  });
  for (let y = 0; y < c.h; y++)
    for (let x = 0; x < c.w; x++)
      diamond(g, iso, c.x + x, c.y + y, 1, 1,
        (x + y) % 2 ? rgba(PAL.core, 0.07) : "rgba(14,19,33,.6)", c.level + 0.2,
        "rgba(255,159,67,.07)");

  // pedestal the hologram rises from
  box(g, iso, c.x + c.w / 2, c.y + c.h / 2, 0.9, 0.9, 0.28, "#20283f", {
    lift: c.level + 0.2, top: "#2a3456", rim: rgba(PAL.core, 0.8),
  });

  const p = iso.p(c.x + c.w / 2, c.y + c.h / 2, c.level + 1.15);
  g.font = `700 ${clamp(iso.unit * 0.34, 8, 12)}px ui-monospace,monospace`;
  g.textAlign = "center";
  g.fillStyle = rgba(PAL.core, 0.85);
  g.fillText("STRATEGY CORE", p.x, p.y + iso.unit * 1.6);
}

function drawDesks(g, iso) {
  const items = [];
  for (const [base, slots] of Object.entries(AGENT_DESKS)) {
    const zone = zoneOfDesk(slots[0]);
    slots.forEach(([x, y]) => items.push({ x, y, zone }));
  }
  items.sort((a, b) => (a.x + a.y) - (b.x + b.y));
  for (const it of items) drawWorkstation(g, iso, it.x, it.y, it.zone);
}

function zoneOfDesk(slot) {
  const [x, y] = slot;
  return SCENE.zones.find((z) => x >= z.x && x < z.x + z.w && y >= z.y && y < z.y + z.h)
      || SCENE.zones[0];
}

/* a cubicle: partition, desk, monitor shells, keyboard, chair, cable */
function drawWorkstation(g, iso, gx, gy, zone) {
  const lvl = zone.level + 0.16;
  const col = zone.colour;

  // partition panels behind the desk
  box(g, iso, gx - 0.62, gy - 0.5, 0.06, 0.55, 0.52, "#141d31",
    { lift: lvl, top: mix("#141d31", col, 0.16), rim: rgba(col, 0.18) });
  box(g, iso, gx - 0.1, gy - 0.95, 0.55, 0.06, 0.52, "#141d31",
    { lift: lvl, top: mix("#141d31", col, 0.16), rim: rgba(col, 0.18) });

  // desk
  box(g, iso, gx, gy - 0.1, 0.62, 0.42, 0.2, "#2c3b58",
    { lift: lvl, top: "#3a486e", rim: rgba(col, 0.22) });
  // monitor shells (screens are drawn live on the dynamic layer)
  box(g, iso, gx - 0.18, gy - 0.3, 0.17, 0.05, 0.3, "#0d1322", { lift: lvl + 0.2, top: "#161f34" });
  box(g, iso, gx + 0.17, gy - 0.3, 0.17, 0.05, 0.3, "#0d1322", { lift: lvl + 0.2, top: "#161f34" });
  // keyboard
  box(g, iso, gx, gy + 0.05, 0.2, 0.09, 0.03, "#1a2338", { lift: lvl + 0.2, top: "#26314f" });
  // chair
  box(g, iso, gx, gy + 0.62, 0.16, 0.16, 0.1, "#1b2438", { lift: lvl, top: "#242f4a" });
  box(g, iso, gx, gy + 0.78, 0.17, 0.05, 0.36, "#1b2438", { lift: lvl + 0.1, top: "#283452" });
  // cable to the floor
  const a = iso.p(gx, gy + 0.2, lvl), b = iso.p(gx + 0.3, gy + 0.5, 0);
  g.strokeStyle = "rgba(90,115,165,.35)"; g.lineWidth = 1.2;
  g.beginPath(); g.moveTo(a.x, a.y);
  g.quadraticCurveTo(a.x + 6, a.y + 22, b.x, b.y); g.stroke();
}

function roundRect(g, x, y, w, h, r) {
  g.beginPath();
  g.moveTo(x + r, y); g.lineTo(x + w - r, y); g.quadraticCurveTo(x + w, y, x + w, y + r);
  g.lineTo(x + w, y + h - r); g.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
  g.lineTo(x + r, y + h); g.quadraticCurveTo(x, y + h, x, y + h - r);
  g.lineTo(x, y + r); g.quadraticCurveTo(x, y, x + r, y); g.closePath();
}

/* =====================================================================
   DYNAMIC PIECES
   ===================================================================== */

/* three angled screens with a live-looking chart on each */
function drawScreens(g, iso, a, now) {
  const zone = a.zone;
  const lvl = zone.level + 0.36;
  const s = iso.unit;
  const working = a.status === "working" || a.status === "waiting";
  const dead = !a.alive || a.status === "stopped";
  const colour = a.status === "paused" ? PAL.validation
    : a.status === "error" ? PAL.bad : zone.colour;
  const glow = dead ? 0.14 : working ? 0.72 + Math.sin(now / 130 + a.gx * 3) * 0.16 : 0.3;

  for (const off of [-0.18, 0.17]) {
    const p = iso.p(a.gx + off, a.gy - 0.3, lvl);
    const w = s * 0.34, h = s * 0.26;
    g.save();
    g.translate(p.x, p.y);
    g.beginPath();
    g.moveTo(0, -h * 0.75); g.lineTo(w * 0.5, -h * 0.4);
    g.lineTo(0, h * 0.1); g.lineTo(-w * 0.5, -h * 0.4);
    g.closePath();
    g.fillStyle = rgba(colour, glow * 0.42);
    g.shadowColor = rgba(colour, 0.9); g.shadowBlur = working ? 20 : 6;
    g.fill(); g.shadowBlur = 0;

    if (!dead) {
      g.save(); g.clip();
      g.strokeStyle = rgba(colour, 0.95); g.lineWidth = 1.1;
      g.beginPath();
      const n = 12, phase = now / 420 + a.gx;
      for (let i = 0; i < n; i++) {
        const t = i / (n - 1);
        const xx = -w * 0.42 + t * w * 0.84;
        const yy = -h * 0.34 + Math.sin(phase + i * 0.7 + off * 9) * h * 0.16
                  + Math.sin(phase * 0.4 + i * 0.23) * h * 0.08;
        i ? g.lineTo(xx, yy) : g.moveTo(xx, yy);
      }
      g.stroke();
      if (working) {
        g.globalAlpha = 0.5; g.strokeStyle = "rgba(255,255,255,.7)"; g.lineWidth = 0.9;
        for (let i = 0; i < 2; i++) {
          const yy = -h * 0.6 + ((now / 300 + i * 0.5) % 1) * h * 0.6;
          g.beginPath(); g.moveTo(-w * 0.42, yy); g.lineTo(w * 0.42, yy); g.stroke();
        }
        g.globalAlpha = 1;
      }
      g.restore();
    }
    g.restore();
  }
}

/* a person: helmet with visor, torso, arms that type, legs that walk */
function drawPerson(g, iso, gx, gy, lift, colour, opt = {}) {
  const s = iso.unit;
  const p = iso.p(gx, gy, lift);
  const scale = s / 20;
  const typing = opt.typing ? Math.sin(opt.now / 90) : 0;
  const step = opt.walking ? Math.sin(opt.now / 110 + (opt.phase || 0)) : 0;
  const bob = opt.walking ? Math.abs(step) * 2.2 * scale : (opt.typing ? Math.sin(opt.now / 240) * 0.8 : 0);
  const y = p.y - bob;
  const dead = opt.dead;

  // contact shadow
  g.fillStyle = "rgba(0,0,0,.34)";
  g.beginPath(); g.ellipse(p.x, p.y + 2 * scale, 8 * scale, 3.6 * scale, 0, 0, 7); g.fill();

  const body = dead ? "#2b3550" : mix("#26314c", colour, 0.28);
  const limb = dead ? "#232c44" : mix("#1e2740", colour, 0.2);

  // legs
  g.strokeStyle = limb; g.lineWidth = 3.4 * scale; g.lineCap = "round";
  g.beginPath();
  g.moveTo(p.x - 2 * scale, y - 9 * scale);
  g.lineTo(p.x - 2 * scale + step * 4 * scale, y - 0.5 * scale); g.stroke();
  g.beginPath();
  g.moveTo(p.x + 2 * scale, y - 9 * scale);
  g.lineTo(p.x + 2 * scale - step * 4 * scale, y - 0.5 * scale); g.stroke();

  // torso
  g.fillStyle = body;
  roundRect(g, p.x - 5.2 * scale, y - 20 * scale, 10.4 * scale, 12 * scale, 3.2 * scale);
  g.fill();
  // hi-vis chest band
  g.fillStyle = rgba(colour, dead ? 0.2 : 0.8);
  g.fillRect(p.x - 5.2 * scale, y - 15.5 * scale, 10.4 * scale, 1.9 * scale);

  // arms
  g.strokeStyle = limb; g.lineWidth = 3 * scale;
  const reach = opt.typing ? 6 + typing * 1.6 : 4.5;
  g.beginPath();
  g.moveTo(p.x - 4.6 * scale, y - 18 * scale);
  g.lineTo(p.x - reach * scale, y - (11 - (opt.walking ? step * 2 : 0)) * scale); g.stroke();
  g.beginPath();
  g.moveTo(p.x + 4.6 * scale, y - 18 * scale);
  g.lineTo(p.x + reach * scale, y - (11 + (opt.walking ? step * 2 : 0)) * scale); g.stroke();

  // head + visor
  g.fillStyle = dead ? "#39456a" : "#d9e4ff";
  g.beginPath(); g.arc(p.x, y - 24 * scale, 4.4 * scale, 0, 7); g.fill();
  g.fillStyle = rgba(colour, dead ? 0.25 : 0.95);
  g.shadowColor = rgba(colour, 0.9); g.shadowBlur = dead ? 0 : 9 * scale;
  roundRect(g, p.x - 3.6 * scale, y - 25.6 * scale, 7.2 * scale, 3 * scale, 1.4 * scale);
  g.fill(); g.shadowBlur = 0;
  return { x: p.x, y };
}

/* the courier: a person carrying a glowing crate */
function drawCourier(g, iso, cr, pos, now) {
  const s = iso.unit, scale = s / 17;
  const anchor = drawPerson(g, iso, pos.gx, pos.gy, pos.lift ?? 0.04, cr.colour,
    { walking: true, now, phase: cr.phase });
  // crate held in front
  const cx = anchor.x + 8 * scale, cy = anchor.y - 15 * scale;
  g.fillStyle = rgba(cr.colour, 0.9);
  g.shadowColor = cr.colour; g.shadowBlur = 14;
  roundRect(g, cx - 4 * scale, cy - 4 * scale, 8 * scale, 8 * scale, 1.6 * scale);
  g.fill(); g.shadowBlur = 0;
  g.strokeStyle = "rgba(255,255,255,.6)"; g.lineWidth = 1;
  g.beginPath();
  g.moveTo(cx - 4 * scale, cy); g.lineTo(cx + 4 * scale, cy); g.stroke();
}

/* cargo pod that runs the rail when a package is delivered */
function drawPod(g, iso, pod, now) {
  const p = iso.p(pod.gx, SCENE.spineY, 0.16);
  const s = iso.unit;
  g.fillStyle = "rgba(0,0,0,.35)";
  g.beginPath(); g.ellipse(p.x, p.y + s * 0.2, s * 0.5, s * 0.16, 0, 0, 7); g.fill();
  box(g, iso, pod.gx, SCENE.spineY, 0.42, 0.24, 0.34, "#3a2a18",
    { lift: 0.1, top: "#6b4a24", rim: rgba(PAL.delivery, 0.75) });
  const q = iso.p(pod.gx, SCENE.spineY, 0.5);
  g.fillStyle = rgba(PAL.delivery, 0.5 + Math.sin(now / 140) * 0.25);
  g.shadowColor = PAL.delivery; g.shadowBlur = 18;
  roundRect(g, q.x - s * 0.12, q.y - s * 0.12, s * 0.24, s * 0.24, 3); g.fill();
  g.shadowBlur = 0;
}

/* robotic arm over the delivery dock, swings when packaging happens */
function drawArm(g, iso, ax, ay, activity, now) {
  const s = iso.unit;
  const baseLift = SCENE.zones.find((z) => z.key === "delivery").level + 0.16;
  box(g, iso, ax, ay, 0.24, 0.24, 0.2, "#1b2437", { lift: baseLift, top: "#26324f" });
  const swing = Math.sin(now / 260) * (0.35 + activity * 0.55);
  const jointA = iso.p(ax, ay, baseLift + 0.2);
  const elbow = { x: jointA.x + Math.cos(swing - 1.1) * s * 0.85,
                  y: jointA.y - Math.abs(Math.sin(swing - 1.1)) * s * 0.7 - s * 0.35 };
  const tip = { x: elbow.x + Math.cos(swing + 0.5) * s * 0.8,
                y: elbow.y + Math.sin(swing + 0.5) * s * 0.45 + s * 0.1 };
  g.lineCap = "round";
  g.strokeStyle = "#38456a"; g.lineWidth = s * 0.17;
  g.beginPath(); g.moveTo(jointA.x, jointA.y); g.lineTo(elbow.x, elbow.y); g.stroke();
  g.strokeStyle = "#2c3757"; g.lineWidth = s * 0.13;
  g.beginPath(); g.moveTo(elbow.x, elbow.y); g.lineTo(tip.x, tip.y); g.stroke();
  const c = activity > 0.05 ? PAL.delivery : "#4a5a80";
  g.fillStyle = rgba(c, 0.9); g.shadowColor = c; g.shadowBlur = activity > 0.05 ? 16 : 5;
  g.beginPath(); g.arc(tip.x, tip.y, s * 0.11, 0, 7); g.fill();
  g.shadowBlur = 0;
}

/* the strategy core: hologram cone, rotating rings, rising motes */
function drawCore(g, iso, state, now) {
  const c = SCENE.core;
  const s = iso.unit;
  const base = iso.p(c.x + c.w / 2, c.y + c.h / 2, c.level + 0.48);
  const pulse = state.pulse;
  const intensity = 0.5 + pulse * 0.5;

  // light cone
  const topY = base.y - s * 3.4;
  const grad = g.createLinearGradient(base.x, base.y, base.x, topY);
  grad.addColorStop(0, rgba(PAL.core, 0.42 * intensity));
  grad.addColorStop(1, "transparent");
  g.beginPath();
  g.moveTo(base.x - s * 0.55, base.y);
  g.lineTo(base.x + s * 0.55, base.y);
  g.lineTo(base.x + s * 1.5, topY);
  g.lineTo(base.x - s * 1.5, topY);
  g.closePath();
  g.fillStyle = grad; g.fill();

  // rotating rings
  for (let i = 0; i < 3; i++) {
    const t = now / (1400 + i * 420);
    const ry = s * (0.5 + i * 0.34);
    const lift = s * (0.9 + i * 0.85);
    g.strokeStyle = rgba(PAL.core, (0.5 - i * 0.11) * intensity);
    g.lineWidth = 1.5;
    g.shadowColor = rgba(PAL.core, 0.7); g.shadowBlur = 12;
    g.beginPath();
    g.ellipse(base.x, base.y - lift, ry, ry * 0.4, Math.sin(t) * 0.5, 0, 7);
    g.stroke(); g.shadowBlur = 0;
  }

  // motes rising: one per recent strategy event
  for (const m of state.motes) {
    const t = m.t;
    const y = base.y - t * s * 3.2;
    const spread = s * 0.2 + t * s * 1.2;
    const x = base.x + Math.cos(m.a + t * 3) * spread;
    g.fillStyle = rgba(m.colour, (1 - t) * 0.9);
    g.shadowColor = m.colour; g.shadowBlur = 10;
    g.beginPath(); g.arc(x, y, s * 0.06 * (1 - t * 0.4), 0, 7); g.fill();
    g.shadowBlur = 0;
  }

  // emitter
  g.fillStyle = rgba(PAL.core, 0.85 * intensity);
  g.shadowColor = PAL.core; g.shadowBlur = 26 * intensity;
  g.beginPath(); g.ellipse(base.x, base.y, s * 0.4, s * 0.18, 0, 0, 7); g.fill();
  g.shadowBlur = 0;
}

/* ceiling lamps casting pools of light on the decks */
function drawLights(g, iso, now) {
  for (const [lx, ly] of SCENE.lamps) {
    const zone = SCENE.zones.find((z) => lx >= z.x && lx < z.x + z.w && ly >= z.y && ly < z.y + z.h);
    const lvl = zone ? zone.level + 0.17 : 0.05;
    const colour = zone ? zone.colour : PAL.core;
    const p = iso.p(lx, ly, lvl);
    const s = iso.unit;
    const flick = 0.9 + Math.sin(now / 900 + lx) * 0.06;
    const rad = g.createRadialGradient(p.x, p.y + s * 0.5, 0, p.x, p.y + s * 0.5, s * 2.1);
    rad.addColorStop(0, rgba(colour, 0.15 * flick));
    rad.addColorStop(1, "transparent");
    g.fillStyle = rad;
    g.fillRect(p.x - s * 2.2, p.y - s * 1.4, s * 4.4, s * 3.6);
    // the fixture itself
    const f = iso.p(lx, ly, lvl + 2.5);
    g.strokeStyle = "rgba(110,140,195,.3)"; g.lineWidth = 1.2;
    g.beginPath(); g.moveTo(f.x, f.y - s * 0.5); g.lineTo(f.x, f.y); g.stroke();
    g.fillStyle = rgba(colour, 0.75);
    g.shadowColor = colour; g.shadowBlur = 14;
    roundRect(g, f.x - s * 0.3, f.y, s * 0.6, s * 0.1, 2); g.fill();
    g.shadowBlur = 0;
  }
}

/* pipes carrying light between departments — ambient life, not messages */
function drawPipes(g, iso, now) {
  const runs = [
    { a: [5.6, 3], b: [8.2, 3], lift: 1.5, colour: PAL.research },
    { a: [5.6, 9.5], b: [8.2, 9.5], lift: 1.1, colour: PAL.data },
    { a: [13.6, 4], b: [16.2, 4], lift: 1.4, colour: PAL.validation },
    { a: [13.6, 9.6], b: [16.2, 9.6], lift: 1.0, colour: PAL.build },
    { a: [21, 9.2], b: [21, 10.4], lift: 0.9, colour: PAL.delivery },
  ];
  const s = iso.unit;
  for (const r of runs) {
    const p1 = iso.p(r.a[0], r.a[1], r.lift);
    const p2 = iso.p(r.b[0], r.b[1], r.lift);
    g.strokeStyle = "rgba(60,78,118,.75)"; g.lineWidth = s * 0.16; g.lineCap = "round";
    g.beginPath(); g.moveTo(p1.x, p1.y); g.lineTo(p2.x, p2.y); g.stroke();
    g.strokeStyle = rgba(r.colour, 0.25); g.lineWidth = s * 0.07;
    g.beginPath(); g.moveTo(p1.x, p1.y); g.lineTo(p2.x, p2.y); g.stroke();
    for (let k = 0; k < 2; k++) {
      const t = ((now / 1500 + k * 0.5) % 1);
      const x = p1.x + (p2.x - p1.x) * t, y = p1.y + (p2.y - p1.y) * t;
      g.fillStyle = rgba(r.colour, 0.95);
      g.shadowColor = r.colour; g.shadowBlur = 12;
      g.beginPath(); g.arc(x, y, s * 0.07, 0, 7); g.fill();
      g.shadowBlur = 0;
    }
  }
}

/* post: vignette + scanlines + dust. Cheap, and does most of the "lit room" work */
function drawPost(g, W, H, now) {
  const v = g.createRadialGradient(W / 2, H * 0.45, Math.min(W, H) * 0.32,
                                   W / 2, H * 0.5, Math.max(W, H) * 0.78);
  v.addColorStop(0, "transparent");
  v.addColorStop(1, "rgba(2,4,10,.72)");
  g.fillStyle = v; g.fillRect(0, 0, W, H);

  g.globalAlpha = 0.035;
  g.fillStyle = "#8fb4ff";
  for (let y = (now / 55) % 4; y < H; y += 4) g.fillRect(0, y, W, 1);
  g.globalAlpha = 1;
}

export {
  SCENE, AGENT_DESKS, Iso, PAL,
  buildStatic, drawScreens, drawPerson, drawCourier, drawPod, drawArm,
  drawCore, drawLights, drawPipes, drawPost, zoneOfDesk, rgba, clamp, ease, roundRect,
};
