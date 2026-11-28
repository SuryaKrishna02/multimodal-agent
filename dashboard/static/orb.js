// The assistant orb ("Jarvis"): a blue hologram of light streaks on a turning sphere, drawn on a canvas.
// An original, procedural take on the "holographic assistant" look: random arcs on great circles of a sphere, a few
// long rings near the edge, a swirl at the centre and flickering data dots, in shades of blue. The canvas is
// transparent: the orb sits on whatever background it is used on, so its colours are deep enough to read on white.
//
// Every orb on the page shares one animation loop, and an orb that is hidden (display: none) is not drawn.
// data-state on the element sets the mood: idle, wake, listening, thinking or speaking. The mood eases from one
// state to the next, so the speed never jumps. With "reduce motion" on, each orb is drawn once per state.

// spin: turns of the sphere (rad/s). drift: how fast the streaks run along their circles. glow: brightness.
// pulse: radius swing (share of the radius). rate: pulse frequency (Hz).
const ORB_MOODS = {
  idle: { spin: 0.22, drift: 0.12, glow: 0.85, pulse: 0.012, rate: 0.25 },
  wake: { spin: 0.3, drift: 0.2, glow: 0.95, pulse: 0.02, rate: 0.4 },
  listening: { spin: 0.45, drift: 0.35, glow: 1.05, pulse: 0.045, rate: 1.4 },
  thinking: { spin: 1.6, drift: 1.5, glow: 1.1, pulse: 0.02, rate: 2 },
  speaking: { spin: 0.6, drift: 0.5, glow: 1.2, pulse: 0.06, rate: 4.5 },
};
// How much to draw at each size: small orbs get fewer pieces, so they read as a mark, not as noise.
// shell: arcs on the turning sphere. rings: tilted rings around the centre, some dashed like data. swirl: the bright
// spiral at the centre. rays: short streaks that break out past the edge. dots: flickering data points.
const ORB_DETAIL = {
  lg: { shell: 90, rings: 56, swirl: 3, rays: 46, dots: 120, glow: true },
  sm: { shell: 12, rings: 12, swirl: 2, rays: 0, dots: 0, glow: false },
  xs: { shell: 6, rings: 8, swirl: 2, rays: 0, dots: 0, glow: false },
};
const PITCH = 0.32;   // fixed tilt towards the viewer (rad), so the top of the sphere shows
const reduceMotion = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
const orbList = new Set();

const rnd = (a, b) => a + Math.random() * (b - a);
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm = (a) => { const l = Math.hypot(...a) || 1; return a.map((x) => x / l); };
function randomAxis(facing = 0) {   // facing 0..1 pulls the axis towards the viewer, so its circle looks round
  const z = rnd(facing, 1) * (Math.random() < 0.5 ? -1 : 1), t = rnd(0, 2 * Math.PI), r = Math.sqrt(1 - z * z);
  return [r * Math.cos(t), r * Math.sin(t), z];
}
function circle(axis, r, len, width) {   // one arc on a circle of radius r (sphere radii) around `axis`
  const u = norm(cross(axis, Math.abs(axis[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0]));
  return { u, v: cross(axis, u), r, a0: rnd(0, 2 * Math.PI), len, width, speed: rnd(0.4, 1.4) * (Math.random() < 0.5 ? -1 : 1),
    hue: rnd(200, 222), light: rnd(38, 54) };
}
const ring = (r, len) => ({ r, len, a0: rnd(0, 2 * Math.PI), speed: rnd(0.4, 1.3) * (Math.random() < 0.5 ? -1 : 1),
  width: rnd(0.6, 2.2), squash: rnd(0.78, 1), tilt: rnd(0, Math.PI), hue: rnd(198, 224), light: rnd(38, 54),
  dash: Math.random() < 0.45 ? [rnd(1, 4), rnd(2, 7)] : null });
function buildOrb(size) {
  const d = ORB_DETAIL[size];
  return {
    arcs: Array.from({ length: d.shell }, () => circle(randomAxis(), rnd(0.7, 1), rnd(0.15, 0.9), rnd(0.5, 1.4))),
    rings: Array.from({ length: d.rings }, () => ring(rnd(0.22, 1.06) ** 0.8, rnd(0.4, 2.8))),
    swirl: Array.from({ length: d.swirl }, () => ({ r: rnd(0.3, 0.5), a0: rnd(0, 2 * Math.PI), len: rnd(3, 5.5), width: rnd(0.8, 1.6) })),
    rays: Array.from({ length: d.rays }, () => ({ a: rnd(0, 2 * Math.PI), r: rnd(0.96, 1.08), len: rnd(0.04, 0.16), speed: rnd(-0.3, 0.3) })),
    dots: Array.from({ length: d.dots }, () => ({ p: randomAxis(), r: rnd(0.55, 1.05), seed: rnd(0, 100), k: rnd(1, 5) })),
    glow: d.glow,
  };
}

// Turn the sphere by `yaw`, tilt it by PITCH, and return [x, y, depth] with depth from -1 (back) to 1 (front).
function project(p, yaw) {
  const x = p[0] * Math.cos(yaw) + p[2] * Math.sin(yaw), z0 = -p[0] * Math.sin(yaw) + p[2] * Math.cos(yaw);
  const y = p[1] * Math.cos(PITCH) - z0 * Math.sin(PITCH), z = p[1] * Math.sin(PITCH) + z0 * Math.cos(PITCH);
  return [x, y, z];
}

function drawCore(ctx, c, size, glow) {   // a soft blue glow at the centre; the rest of the canvas stays clear
  const core = ctx.createRadialGradient(c, c, 0, c, c, size * 0.24);
  core.addColorStop(0, `rgba(40, 150, 255, ${0.3 * glow})`);
  core.addColorStop(1, "rgba(40, 150, 255, 0)");
  ctx.fillStyle = core;
  ctx.fillRect(0, 0, 2 * c, 2 * c);
}

function drawArc(ctx, a, o, R, c) {
  const pts = [];
  let depth = 0;
  for (let i = 0; i < 8; i++) {
    const t = a.a0 + o.phase * a.speed + (i / 7) * a.len;
    const p = project([0, 1, 2].map((k) => a.r * (Math.cos(t) * a.u[k] + Math.sin(t) * a.v[k])), o.yaw);
    pts.push(p);
    depth += p[2] / 8;
  }
  const alpha = o.mood.glow * (0.18 + 0.82 * ((depth + 1) / 2) ** 1.6);   // the back of the sphere is dim
  ctx.beginPath();
  pts.forEach(([x, y], i) => (i ? ctx.lineTo(c + x * R, c + y * R) : ctx.moveTo(c + x * R, c + y * R)));
  glowStroke(ctx, o, a.hue, a.light, alpha, a.width * o.scale);
}

// A stroke in two passes when the orb is large: a wide faint one (the glow, cheaper than shadowBlur), then the line.
function glowStroke(ctx, o, hue, light, alpha, width) {
  if (o.model.glow) {
    ctx.strokeStyle = `hsla(${hue}, 100%, 55%, ${alpha * 0.16})`;
    ctx.lineWidth = width * 3.2;
    ctx.stroke();
  }
  ctx.strokeStyle = `hsla(${hue}, 100%, ${light}%, ${alpha})`;
  ctx.lineWidth = width;
  ctx.stroke();
}

function drawRings(ctx, o, R, c) {   // tilted rings around the centre: the dish-like structure of the hologram
  for (const g of o.model.rings) {
    const start = g.a0 + o.phase * g.speed * 1.6;
    ctx.setLineDash(g.dash ? g.dash.map((x) => x * o.scale) : []);
    ctx.beginPath();
    ctx.ellipse(c, c, g.r * R, g.r * R * g.squash, g.tilt + o.yaw * 0.15, start, start + g.len);
    glowStroke(ctx, o, g.hue, g.light, o.mood.glow * (0.35 + 0.5 * (1 - g.r / 1.1)), g.width * o.scale);
  }
  ctx.setLineDash([]);
}

function drawSwirl(ctx, o, R, c) {   // a bright spiral that winds into the centre
  for (const s of o.model.swirl) {
    ctx.beginPath();
    for (let i = 0; i <= 24; i++) {
      const f = i / 24, ang = s.a0 + o.phase * 2.2 + f * s.len, rad = s.r * R * (1 - 0.75 * f);
      const x = c + Math.cos(ang) * rad, y = c + Math.sin(ang) * rad * 0.9;
      i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    }
    glowStroke(ctx, o, 210, 46, o.mood.glow * 0.6, s.width * o.scale);
  }
}

function drawRays(ctx, o, R, c) {   // short streaks that break out past the edge, like data leaving the sphere
  ctx.strokeStyle = `rgba(24, 120, 230, ${0.6 * o.mood.glow})`;
  ctx.lineWidth = 0.9 * o.scale;
  ctx.beginPath();
  for (const r of o.model.rays) {
    const a = r.a + o.phase * r.speed, x = Math.cos(a), y = Math.sin(a);
    ctx.moveTo(c + x * r.r * R, c + y * r.r * R);
    ctx.lineTo(c + x * (r.r + r.len) * R, c + y * (r.r + r.len) * R);
  }
  ctx.stroke();
}

function drawOrb(o, t) {
  const { ctx, px } = o;
  const c = px / 2;
  const m = o.mood;
  const wobble = o.state === "speaking" ? 0.5 * Math.sin(t * 11.3) + 0.5 * Math.sin(t * 7.1) : Math.sin(2 * Math.PI * m.rate * t);
  const R = px * 0.4 * (1 + m.pulse * wobble);
  ctx.clearRect(0, 0, px, px);
  drawCore(ctx, c, px, m.glow);
  ctx.lineCap = "round";
  for (const a of o.model.arcs) drawArc(ctx, a, o, R, c);
  drawRings(ctx, o, R, c);
  drawSwirl(ctx, o, R, c);
  if (o.model.rays.length) drawRays(ctx, o, R, c);
  for (const d of o.model.dots) {
    const [x, y, z] = project(d.p.map((v) => v * d.r), o.yaw);
    const a = m.glow * (0.35 + 0.65 * Math.abs(Math.sin(t * d.k + d.seed))) * (0.3 + 0.7 * (z + 1) / 2);
    ctx.fillStyle = `rgba(16, 110, 220, ${a})`;
    ctx.fillRect(c + x * R - o.scale, c + y * R - o.scale, 2 * o.scale, 2 * o.scale);
  }
}

function sizeOrb(o) {   // match the canvas to its box and the screen's pixel density
  const css = o.el.clientWidth;
  const px = Math.round(css * (window.devicePixelRatio || 1));
  if (!css || px === o.px) return;
  o.canvas.width = o.canvas.height = o.px = px;
  o.scale = Math.max(1.3, px / 132);   // stroke widths grow with the orb, and stay readable when it is small
}

let orbLast = 0;
function orbTick(now) {
  const t = now / 1000, dt = Math.min(0.1, t - orbLast || 0);
  orbLast = t;
  for (const o of orbList) {
    if (!o.el.isConnected) { orbList.delete(o); continue; }
    if (o.el.offsetParent === null) continue;   // hidden: draw nothing
    const state = o.el.dataset.state || "idle", target = ORB_MOODS[state] || ORB_MOODS.idle;
    if (reduceMotion && state === o.state && o.px) continue;
    o.state = state;
    for (const k in target) o.mood[k] += (target[k] - o.mood[k]) * (reduceMotion ? 1 : Math.min(1, dt * 2.5));   // ease, about 0.4 s
    o.yaw += o.mood.spin * dt;
    o.phase += o.mood.drift * dt;
    sizeOrb(o);
    if (o.px) drawOrb(o, reduceMotion ? 0 : t);
  }
  requestAnimationFrame(orbTick);
}

/** Put an animated orb into `el` (a .jarvis element; its class lg, sm or xs sets the detail). */
function mountOrb(el) {
  const size = el.classList.contains("lg") ? "lg" : el.classList.contains("xs") ? "xs" : "sm";
  const canvas = document.createElement("canvas");
  el.replaceChildren(canvas);
  orbList.add({ el, canvas, ctx: canvas.getContext("2d"), px: 0, scale: 1, yaw: rnd(0, 6), phase: 0, state: null,
    mood: { ...ORB_MOODS.idle }, model: buildOrb(size) });
}
requestAnimationFrame(orbTick);
