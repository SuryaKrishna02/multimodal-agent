// Dashboard: polls the event log and redraws the panel each event belongs to.
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => [...document.querySelectorAll(sel)];
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const json = (o) => JSON.stringify(o, null, 1);
const short = (s, n) => (s.length > n ? s.slice(0, n) + "…" : s);
const media = (path) => "/media?path=" + encodeURIComponent(path);
const clock = (ts) => new Date(ts * 1000).toLocaleTimeString([], { hour: "numeric", minute: "2-digit", second: "2-digit" });
const num = (n) => Number(n ?? 0).toLocaleString();
const plural = (n, word, many = word + "s") => `${num(n)} ${n === 1 ? word : many}`;
const tierName = (t) => "Gemma 4 " + String(t).toUpperCase();
let lastEventId = 0;
let allEvents = [];
let appState = null;
let replaying = true;   // the first poll replays old events: redraw from them, but do not act on them

async function api(path, body, raw) {
  const opts = body === undefined ? {} : { method: "POST",
    body: raw ? body : JSON.stringify(body), headers: raw ? {} : { "Content-Type": "application/json" } };
  const r = await fetch(path, opts);
  if (!r.ok) {
    let msg = await r.text();
    try { msg = JSON.parse(msg).detail ?? msg; } catch { /* not JSON: show the raw text */ }
    throw new Error(typeof msg === "string" ? msg : JSON.stringify(msg));
  }
  return r.json();
}

let toastTimer;
function toast(message, kind = "info") {
  const t = $("#toast");
  t.textContent = message;
  t.className = "show " + kind;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (t.className = ""), kind === "error" ? 7000 : 3500);
}

// A line with a spinner and a running clock, because a local model can take many seconds to answer.
function working(el, text) {
  const t0 = Date.now();
  el.innerHTML = `<div class="working"><i class="spinner"></i><span></span></div>`;
  const span = el.querySelector("span");
  const tick = () => {
    if (!span.isConnected) return clearInterval(timer);
    span.textContent = `${el.dataset.step || text} ${Math.round((Date.now() - t0) / 1000)} s`;
  };
  const timer = setInterval(tick, 1000);
  tick();
}

// Disable a control while its request runs. Buttons show "Working…" (or their data-busy text);
// `out` is a result element that shows a spinner line until the request ends.
async function busy(control, fn, { out, text } = {}) {
  const isButton = control.tagName === "BUTTON";
  const label = control.textContent;
  control.disabled = true;
  control.classList.add("is-busy");
  if (isButton) control.textContent = control.dataset.busy || "Working…";
  if (out) working($(out), text || "Working…");
  try {
    const result = await fn();
    if (out) await pollEvents();   // the result arrives as an event; fetch it now instead of waiting for the next poll
    return result;
  } catch (e) {
    toast(e.message, "error");
    if (out && $(out).querySelector(".working")) $(out).innerHTML = "";
  } finally {
    control.disabled = false;
    control.classList.remove("is-busy");
    if (isButton) control.textContent = label;
  }
}

// A tiny, safe Markdown reader for model replies: it escapes everything first, then adds a few tags.
function inlineMd(s) {
  return s.replace(/`([^`\n]+)`/g, "<code>$1</code>").replace(/\*\*([^*\n]+)\*\*/g, "<b>$1</b>")
    .replace(/(^|[\s(])\*([^*\s][^*\n]*)\*/g, "$1<i>$2</i>");
}
function md(text) {
  const out = [];
  let list = null, para = [], fence = null;
  const closePara = () => { if (para.length) out.push(`<p>${para.map(inlineMd).join("<br>")}</p>`); para = []; };
  const closeList = () => { if (list) out.push(`</${list}>`); list = null; };
  for (const line of esc(text).split("\n")) {
    if (line.startsWith("```")) {
      closePara(); closeList();
      if (fence) { out.push(`<pre>${fence.join("\n")}</pre>`); fence = null; } else fence = [];
      continue;
    }
    if (fence) { fence.push(line); continue; }
    const item = line.match(/^\s*(?:([*-])|(\d+)\.)\s+(.*)$/);
    const heading = line.match(/^#{1,4}\s+(.*)$/);
    if (item) {
      closePara();
      const kind = item[1] ? "ul" : "ol";
      if (list !== kind) { closeList(); out.push(`<${kind}>`); list = kind; }
      out.push(`<li>${inlineMd(item[3])}</li>`);
    } else if (heading) {
      closePara(); closeList();
      out.push(`<p><b>${inlineMd(heading[1])}</b></p>`);
    } else if (!line.trim()) {
      closePara(); closeList();
    } else {
      closeList();
      para.push(line);
    }
  }
  if (fence) out.push(`<pre>${fence.join("\n")}</pre>`);
  closePara(); closeList();
  return `<div class="md">${out.join("")}</div>`;
}

const zoomable = (path, alt) => `<button type="button" class="zoom" data-zoom="${esc(media(path))}" title="Enlarge"><img alt="${esc(alt)}" src="${esc(media(path))}"></button>`;
document.addEventListener("click", (e) => {
  const z = e.target.closest("[data-zoom]");
  if (!z) return;
  $("#lightbox img").src = z.dataset.zoom;
  $("#lightbox").showModal();
});
$("#lightbox").onclick = () => $("#lightbox").close();

// --- Tabs (keys 1-7 switch tabs when you are not typing) --------------------
const TABS = $$("#tabs button");
function showTab(name) {
  if (!$("#panel-" + name)) name = "serving";
  $$(".panel").forEach((p) => p.classList.toggle("active", p.id === "panel-" + name));
  TABS.forEach((b) => {
    const on = b.dataset.tab === name;
    b.classList.toggle("active", on);
    b.setAttribute("aria-selected", String(on));
    b.tabIndex = on ? 0 : -1;
  });
  history.replaceState(null, "", "#" + name);   // a reload stays on the same tab
  if (name === "brain") renderBrain();
  if (name === "memory") refreshMemory();
  if (name === "tools") refreshTools();
  if (name === "scorecard") refreshScorecard();
}
const isTab = (name) => $("#panel-" + name).classList.contains("active");
TABS.forEach((b, i) => {
  b.id = "tab-" + b.dataset.tab;
  b.setAttribute("aria-controls", "panel-" + b.dataset.tab);
  $("#panel-" + b.dataset.tab).setAttribute("aria-labelledby", b.id);
  b.onclick = () => showTab(b.dataset.tab);
  b.onkeydown = (e) => {
    const step = { ArrowRight: 1, ArrowLeft: -1 }[e.key];
    if (!step) return;
    const next = TABS[(i + step + TABS.length) % TABS.length];
    showTab(next.dataset.tab);
    next.focus();
  };
});
$$("[data-go]").forEach((b) => (b.onclick = () => showTab(b.dataset.go)));
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") setDrawer(false);
  if (e.metaKey || e.ctrlKey || e.altKey) return;
  if (["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement.tagName)) return;
  if (e.key === "c") setChatHidden(!document.body.classList.contains("chat-hidden"));
  const b = TABS[Number(e.key) - 1];
  if (b) showTab(b.dataset.tab);
});

// Sticky offsets follow the real height, because the header and the banners wrap.
new ResizeObserver(() => {
  const root = document.documentElement.style;
  root.setProperty("--header-h", $("#top").offsetHeight + "px");
  root.setProperty("--banner-h", $("#banners").offsetHeight + "px");
}).observe(document.body);

function setChatHidden(hidden) {
  document.body.classList.toggle("chat-hidden", hidden);
  $("#chat-toggle").textContent = hidden ? "Show chat" : "Hide chat";
  localStorage.setItem("chatHidden", hidden ? "1" : "");
}
$("#chat-toggle").onclick = () => setChatHidden(!document.body.classList.contains("chat-hidden"));
setChatHidden(!!localStorage.getItem("chatHidden"));

// --- Header state -------------------------------------------------------------
let connectionLost = false;
const serverStatus = (ready, failed, gb) => (failed ? "failed" : ready ? `${gb} GB` : "starting…");
async function refreshState() {
  let s;
  try { s = await api("/api/state"); } catch {
    if (!connectionLost) toast("Lost connection to the dashboard server. Retrying…", "error");
    connectionLost = true;
    return;
  }
  if (connectionLost) toast("Reconnected.");
  connectionLost = false;
  appState = s;
  if (s.llm && s.embed) dropNotice("serving.error");
  const failed = !!notices.get("serving.error");
  const setDot = (id, ready) => { $(id).className = "dot" + (ready ? " on" : failed ? " failed" : ""); };
  setDot("#llm-dot", s.llm);
  setDot("#emb-dot", s.embed);
  $("#llm-mem").textContent = serverStatus(s.llm, failed && !s.llm, s.llm_rss_gb);
  $("#emb-mem").textContent = serverStatus(s.embed, failed && !s.embed, s.embed_rss_gb);
  if (document.activeElement !== $("#tier")) $("#tier").value = s.tier;
  const mic = $("#mic-switch");
  mic.className = "mic " + (s.mic_paused ? "paused" : s.recording ? "recording" : "live");
  mic.textContent = s.mic_paused ? "Mic paused" : s.recording ? "Recording" : "Mic ready";
  mic.setAttribute("aria-checked", String(!s.mic_paused));
  setRecordingUI(s.recording);
  const server = (name, ready, gb, port) => `<div class="server"><div class="name"><i class="dot${ready ? " on" : failed ? " failed" : ""}"></i>${name}</div>
    <div class="meta">${ready ? `Ready · ${gb} GB in memory` : failed ? "Failed to start. See the message at the top." : "Starting…"}${port ? ` · port ${port}` : ""}</div></div>`;
  const html = server(`Language model (${tierName(s.tier)})`, s.llm, s.llm_rss_gb, s.llm_port) + server("Embedding model", s.embed, s.embed_rss_gb, s.embed_port);
  if (html !== $("#serving-info").dataset.html) { $("#serving-info").innerHTML = html; $("#serving-info").dataset.html = html; }
  renderPending(s.pending);
}
$("#mic-switch").onclick = (e) => {
  const paused = e.target.getAttribute("aria-checked") === "true";   // currently allowed: pause it
  busy(e.target, () => api("/api/mic/pause", { paused })).then(refreshState);
};
$("#tier").onchange = async (e) => {
  const tier = e.target.value;
  toast(`Restarting the language model as ${tierName(tier)}…`);
  const out = await busy($("#tier"), () => api("/api/serving/switch", { tier }));
  if (out) toast(`Now running ${tierName(tier)}.`);
  refreshState();
};
$("#bench-btn").onclick = async (e) => {
  await busy(e.target, () => api("/api/serving/bench", {}));
  pollEvents();
};

// --- Notices: a failure must not hide in the event log ------------------------
const notices = new Map();
const dismissed = new Set();
function renderNotices() {
  $("#notices").innerHTML = [...notices].filter(([k]) => !dismissed.has(k)).map(([k, n]) =>
    `<div class="notice ${n.tone}"><span>${esc(n.text)}</span><button data-dismiss="${esc(k)}">Dismiss</button></div>`).join("");
}
function addNotice(key, text, tone = "") {
  notices.set(key, { text, tone });
  dismissed.delete(key);
  renderNotices();
}
function dropNotice(key) {
  if (notices.delete(key)) renderNotices();
}
document.addEventListener("click", (e) => {
  const k = e.target.dataset?.dismiss;
  if (k) { dismissed.add(k); renderNotices(); }
});

// --- Plain words for every event, used by the Activity log and the Brain steps ---
const SERVER_NAMES = { llm: "language model", embed: "embedding model" };
const serverName = (s) => SERVER_NAMES[s] || s;
const asked = (m) => (m?.tool_calls || []).map((c) => c.function?.name).filter(Boolean);
const DESCRIBE = {
  "serving.start": (e) => `Starting the ${serverName(e.server)}`,
  "serving.ready": (e) => `The ${serverName(e.server)} is ready${e.load_s ? ` (loaded in ${e.load_s} s)` : ""}`,
  "serving.stop": (e) => `Stopped the ${serverName(e.server)}`,
  "serving.restart": (e) => `The ${serverName(e.server)} stopped unexpectedly. Restarting it`,
  "serving.error": (e) => `A model server failed: ${e.message}`,
  "serving.bench": (e) => `Benchmark on ${tierName(e.tier)}: ${e.gen_tok_s?.toFixed(1)} tokens/s, first token in ${e.ttft_s?.toFixed(2)} s`,
  "senses.mic": (e) => `Microphone ${e.state}${e.seconds ? ` after ${e.seconds} s` : ""}${e.device ? ` (${e.device})` : ""}`,
  "senses.audio": (e) => `Split ${e.seconds} s of audio: ${plural(e.speech.length, "speech segment")}, ${plural(e.chunks.length, "chunk")}`,
  "senses.screen": (e) => `Captured the screen at ${e.size?.join(" × ")}`,
  "senses.screen.warning": (e) => e.message,
  "senses.video": (e) => (e.state === "recording" ? `Recording ${e.seconds} s of screen` : `Recorded ${e.seconds?.toFixed(1)} s of screen at ${e.width} × ${e.height}`),
  "brain.budget": (e) => `Planned the context: ${num(e.total)} of ${num(e.budget)} tokens`,
  "brain.request": (e) => `Sent the model ${plural(e.body.messages.length, "message")} and ${plural(e.body.tools?.length || 0, "tool")}`,
  "brain.reply": (e) => `Model replied in ${e.seconds} s` + (asked(e.message).length ? `, asking for ${asked(e.message).join(", ")}` : ""),
  "brain.invalid_json": (e) => `Reply was not valid JSON (attempt ${e.attempt}), asking again`,
  "memory.search": (e) => `Searched memory for "${e.query}": ${plural(e.hits.length, "match", "matches")}`,
  "memory.stored": (e) => `Saved to memory: ${e.text}`,
  "memory.rejected": (e) => `Junk filter refused "${e.text}": ${e.reason}`,
  "memory.cleared": () => "All memories deleted",
  "tools.call": (e) => `Tool ${e.tool}: ${e.status}`,
  "task1.done": (e) => `Task 1 done: ${plural(e.action_items.length, "action item")}`,
  "task1.empty": () => "Task 1: no speech detected",
  "task2.done": (e) => `Task 2 done: ${e.problem ? "found a problem" : "no problem spotted"}`,
  "hotkey.pressed": () => "Hotkey pressed: looking at the screen",
  "hotkey.listening": () => "Global hotkey is on",
  "hotkey.warning": (e) => e.message,
};
const fieldsOf = ({ id, ts, kind, ...rest }) => rest;
function describe(e) {
  const fn = DESCRIBE[e.kind];
  if (fn) { try { return fn(e); } catch { /* an event with an older shape: fall through to the raw fields */ } }
  return `${e.kind} ${short(JSON.stringify(fieldsOf(e)), 200)}`;
}
const TAB_OF = { serving: "serving", senses: "senses", brain: "brain", memory: "memory", tools: "tools", task1: "tasks", task2: "tasks", hotkey: "tasks" };
const groupOf = (kind) => TAB_OF[kind.split(".")[0]] || "other";
const titleCase = (s) => s.charAt(0).toUpperCase() + s.slice(1);

// Light the tab of the component that just reported, so viewers can follow the agent as it works.
const pingTimers = {};
function ping(tab) {
  $$(`[data-activity="${tab}"]`).forEach((p) => p.classList.add("live"));
  clearTimeout(pingTimers[tab]);
  pingTimers[tab] = setTimeout(() => $$(`[data-activity="${tab}"]`).forEach((p) => p.classList.remove("live")), 1600);
}

// --- Activity drawer + event log -------------------------------------------------
function setDrawer(open) {
  $("#drawer").classList.toggle("open", open);
  $("#drawer-bar").setAttribute("aria-expanded", String(open));
  if (open) renderEventLog();
}
$("#drawer-bar").onclick = () => setDrawer(!$("#drawer").classList.contains("open"));

let eventGroup = "all";
const openEvents = new Set();
$("#event-chips").innerHTML = ["all", "serving", "senses", "brain", "memory", "tools", "tasks"].map((g) =>
  `<button type="button" data-group="${g}"${g === "all" ? ' class="active"' : ""}>${titleCase(g)}</button>`).join("");
$("#event-chips").onclick = (e) => {
  const g = e.target.dataset.group;
  if (!g) return;
  eventGroup = g;
  $$("#event-chips button").forEach((b) => b.classList.toggle("active", b === e.target));
  renderEventLog();
};
$("#event-log").addEventListener("toggle", (e) => {
  const id = Number(e.target.dataset.id);
  if (!e.target.open) { openEvents.delete(id); return; }
  openEvents.add(id);
  const ev = allEvents.find((x) => x.id === id);
  if (ev && !e.target.querySelector("pre")) e.target.insertAdjacentHTML("beforeend", `<pre>${esc(json({ kind: ev.kind, ...fieldsOf(ev) }))}</pre>`);
}, true);

function renderEventLog() {
  const f = $("#event-filter").value.trim().toLowerCase();
  const match = (e) => (eventGroup === "all" || groupOf(e.kind) === eventGroup) &&
    (!f || e.kind.includes(f) || describe(e).toLowerCase().includes(f));
  const shown = allEvents.filter(match).slice(-150).reverse();
  $("#event-log").innerHTML = shown.map((e) => {
    const isOpen = openEvents.has(e.id);
    const isError = /error|warning|invalid/.test(e.kind) || ["blocked", "invalid", "error"].includes(e.status);
    return `<details class="ev${isError ? " is-error" : ""}" data-id="${e.id}"${isOpen ? " open" : ""}><summary><time>${clock(e.ts)}</time>
      <span class="ev-tab">${esc(titleCase(groupOf(e.kind)))}</span><span class="ev-text">${esc(describe(e))}</span></summary>
</details>`;
  }).join("") || '<div class="none" style="padding:8px 0">No matching events.</div>';
}
$("#event-filter").oninput = renderEventLog;

const BRAIN_KINDS = /^(brain|memory\.(search|stored|rejected)|tools\.call|senses\.(screen|audio|video)$|hotkey\.pressed|task[12]\.)/;
async function pollEvents() {
  let evs;
  try { evs = await api(`/api/events?after=${lastEventId}`); } catch { return; }
  for (const ev of evs) {
    lastEventId = ev.id;
    allEvents.push(ev);
    (renderers[ev.kind] || (() => {}))(ev);
    if (!replaying) {
      ping(groupOf(ev.kind));
      noteChatStep(ev);
    }
  }
  replaying = false;
  if (!evs.length) return;
  $("#latest-event").textContent = describe(evs[evs.length - 1]);
  if ($("#drawer").classList.contains("open")) renderEventLog();
  if (evs.some((e) => e.kind === "serving.bench")) renderBench();
  if (isTab("brain") && evs.some((e) => BRAIN_KINDS.test(e.kind))) renderBrain();
}

const HOTKEY_NAMES = { ctrl: "Ctrl", alt: "Alt / Option", cmd: "Cmd", shift: "Shift", space: "Space" };
const renderers = {
  "serving.start": (e) => {
    startCommands[e.server] = e.cmd;
    $("#serving-cmd").hidden = false;
    $("#serving-cmd-body").innerHTML = Object.entries(startCommands).map(([s, c]) =>
      `<b>${esc(titleCase(serverName(s)))}</b><pre>${esc(c.split(" ").map((t) => t.split(/[\\/]/).pop()).join(" "))}</pre>`).join("");
  },
  "serving.error": (e) => { addNotice("serving.error", DESCRIBE["serving.error"](e)); refreshState(); },
  "senses.audio": drawTimeline,
  "senses.screen": (e) => {
    $("#last-screen").innerHTML = zoomable(e.path, "Last captured screen");
    $("#screen-info").textContent = `${e.size?.join(" × ") ?? ""} · ${clock(e.ts)}`;
  },
  "senses.video": (e) => {
    if (e.state === "recording") { $("#video-info").textContent = `Recording ${e.seconds} s…`; return; }
    $("#last-video").innerHTML = `<video controls muted src="${esc(media(e.path))}"></video>`;
    $("#video-info").textContent = `${e.seconds.toFixed(1)} s at ${e.width} × ${e.height} · ${clock(e.ts)}`;
  },
  "senses.screen.warning": (e) => addNotice("screen", e.message, "warn"),
  "hotkey.warning": (e) => addNotice("hotkey", e.message, "warn"),
  "hotkey.listening": (e) => {
    const keys = e.combo.split("+").map((k) => k.replace(/[<>]/g, "")).map((k) => `<kbd>${esc(HOTKEY_NAMES[k] || k)}</kbd>`);
    $("#hotkey-tip").innerHTML = `Tip: press ${keys.join(" ")} anywhere to run this on your current screen.`;
    $("#hotkey-tip").hidden = false;
  },
  "tools.call": () => refreshToolCalls(),
  "memory.stored": () => isTab("memory") && refreshMemory(),
  "memory.rejected": () => isTab("memory") && refreshMemory(),
  "task1.done": renderTask1,
  "task1.empty": (e) => {
    $("#task1-out").innerHTML = `<div class="callout warn"><h4>No speech detected</h4><p>${clock(e.ts)}. Nothing was saved. Try again closer to the microphone.</p></div>`;
  },
  "task2.done": renderTask2,
  "hotkey.pressed": () => {
    if (replaying) return;
    showTab("tasks");
    working($("#task2-out"), "Looking at your screen…");
    setTimeout(() => { if ($("#task2-out .working")) $("#task2-out").innerHTML = ""; }, 180000);
  },
};
const startCommands = {};

// --- Serving: benchmarks ----------------------------------------------------------
function renderBench() {
  const rows = allEvents.filter((e) => e.kind === "serving.bench").reverse();
  $("#bench-table tbody").innerHTML = rows.map((e) => `<tr><td class="nowrap">${clock(e.ts)}</td><td class="nowrap">${esc(tierName(e.tier))}</td>
    <td class="num">${e.ttft_s?.toFixed(2)} s</td><td class="num">${e.gen_tok_s?.toFixed(1)} tok/s</td>
    <td class="num">${e.prompt_tok_s?.toFixed(0)} tok/s</td><td class="num">${e.rss_gb} GB</td></tr>`).join("");
  const e = rows[0];
  $("#bench-latest").innerHTML = !e ? "" : tiles([
    [e.ttft_s?.toFixed(2), "s", "First token: wait before the reply starts"],
    [e.gen_tok_s?.toFixed(1), "tok/s", "Writing speed: how fast the reply is written"],
    [e.prompt_tok_s?.toFixed(0), "tok/s", "Reading speed: how fast the prompt is read"],
    [e.rss_gb, "GB", "Memory used by the language model"],
  ]);
}
const tiles = (list) => list.map(([v, unit, label]) =>
  `<div class="tile"><div class="v">${esc(v)}${unit ? ` <small>${esc(unit)}</small>` : ""}</div><div class="l">${esc(label)}</div></div>`).join("");

// --- Senses: timeline of speech segments and the chunks sent to Gemma ---------------
function axisTicks(total) {
  const step = [1, 2, 5, 10, 15, 30, 60, 120, 300, 600].find((s) => total / s <= 8) || 1200;
  const ticks = [];
  for (let t = 0; t < total; t += step) ticks.push(t);
  return ticks;
}
function drawTimeline(e) {
  const total = Math.max(e.seconds, 0.1);
  const pct = (s) => (100 * s) / total;
  const span = ([a, b], cls, label, title) => `<i class="${cls}" style="left:${pct(a)}%;width:${pct(b - a)}%" title="${esc(title)}">${esc(label)}</i>`;
  const speech = e.speech.map((s) => span(s, "", "", `Speech: ${s[0].toFixed(1)} s to ${s[1].toFixed(1)} s`)).join("");
  const chunks = e.chunks.map((c, i) => {
    const len = c[1] - c[0];
    const label = pct(len) > 22 ? `Chunk ${i + 1}: ${len.toFixed(1)} s` : pct(len) > 6 ? String(i + 1) : "";
    return span(c, "chunk", label, `Chunk ${i + 1}: ${len.toFixed(1)} s, from ${c[0].toFixed(1)} s to ${c[1].toFixed(1)} s`);
  }).join("");
  const longest = Math.max(0, ...e.chunks.map(([a, b]) => b - a));
  $("#timeline").innerHTML = `<div class="tl">
    <div class="tl-row"><span class="tl-label">Speech found</span><div class="tl-track">${speech}</div></div>
    <div class="tl-row"><span class="tl-label">Sent to Gemma</span><div class="tl-track">${chunks}</div></div>
    <div class="tl-axis">${axisTicks(total).map((t) => `<span style="left:${pct(t)}%">${t} s</span>`).join("")}</div></div>
    <div class="tiles">${tiles([
      [e.seconds, "s", "Recorded"],
      [e.speech.length, "", "Speech segments"],
      [e.chunks.length, "", "Chunks sent"],
      [longest.toFixed(1), "s", "Longest chunk (limit 30 s)"],
      ["~" + num(e.est_tokens), "", "Audio tokens"],
    ])}</div>`;
}

// --- Brain: one message at a time, as a list of steps ---------------------------------
// A turn runs from the first event of a message to the reply that ends it.
function groupTurns() {
  const turns = [];
  let cur = null, finished = false;
  const startsTurn = /^(hotkey\.pressed|senses\.(screen|audio|video)|memory\.search|brain\.budget)$/;
  for (const e of allEvents) {
    if (!BRAIN_KINDS.test(e.kind)) continue;
    const strayedSearch = e.kind === "memory.search" && cur && !cur.some((x) => x.kind === "brain.request");
    if (!cur || strayedSearch || (finished && startsTurn.test(e.kind))) { cur = []; turns.push(cur); finished = false; }
    cur.push(e);
    if (e.kind === "brain.reply") finished = !asked(e.message).length;
    if (e.kind === "brain.invalid_json") finished = false;
  }
  return turns.filter((t) => t.some((e) => e.kind === "brain.request"));
}

// Each model call pairs a request with the reply that follows it.
function modelCalls(turn) {
  const calls = [];
  for (const e of turn) {
    if (e.kind === "brain.request") calls.push({ request: e, reply: null });
    if (e.kind === "brain.reply" && calls.length) calls[calls.length - 1].reply = e;
  }
  return calls;
}

const brainView = { turn: null, call: null };   // null follows the newest; a number stays where the presenter stepped to
function turnTitle(turn) {
  const req = turn.find((e) => e.kind === "brain.request");
  const user = req.body.messages.filter((m) => m.role === "user").pop();
  if (!user) return "";
  const text = typeof user.content === "string" ? user.content : (user.content.find((p) => p.type === "text")?.text ?? "");
  return short(text.trim(), 140);
}

const stepLine = (e) => {
  if (e.kind === "memory.search") {
    const top = e.hits[0];
    return { tab: "Memory", html: `${esc(describe(e))}${top ? `<small>Best match, similarity ${Number(top[1]).toFixed(2)}: ${esc(top[2])}${top[2].length >= 60 ? "…" : ""}</small>` : ""}` };
  }
  if (e.kind === "brain.budget") return { tab: "Brain", html: `${esc(describe(e))}<small>${plural(e.memories, "memory", "memories")} and ${plural(e.history_turns, "earlier message")} included</small>` };
  if (e.kind === "tools.call") {
    const res = e.result ?? e.error;
    return { tab: "Tools", html: `<span class="${e.status === "ok" ? "" : "step-bad"}">${esc(describe(e))}</span>${res !== undefined ? `<small>${esc(short(typeof res === "string" ? res : JSON.stringify(res), 160))}</small>` : ""}` };
  }
  return { tab: titleCase(groupOf(e.kind)), html: esc(short(describe(e), 200)) };
};

function renderBrain() {
  const turns = groupTurns();
  $("#brain-empty").hidden = turns.length > 0;
  $("#brain-body").hidden = !turns.length;
  if (!turns.length) return;
  const index = brainView.turn === null ? turns.length - 1 : Math.min(brainView.turn, turns.length - 1);
  const turn = turns[index];
  $("#turn-pos").textContent = `Message ${index + 1} of ${turns.length}`;
  $("#turn-prev").disabled = index === 0;
  $("#turn-next").disabled = index === turns.length - 1;
  $("#turn-title").textContent = turnTitle(turn);

  const calls = modelCalls(turn);
  const callIndex = brainView.call === null ? calls.length - 1 : Math.min(brainView.call, calls.length - 1);
  let n = -1;
  $("#turn-steps").innerHTML = turn.filter((e) => e.kind !== "brain.reply").map((e) => {
    if (e.kind === "brain.request") {
      n += 1;
      const reply = calls[n].reply;
      const u = reply?.usage;
      const outcome = !reply ? "Waiting for the reply…" : asked(reply.message).length ? `It asked for ${asked(reply.message).join(", ")}` : "It wrote the answer";
      return `<li><span class="step-tab">Model</span><button type="button" class="step-body${n === callIndex ? " selected" : ""}" data-call="${n}" aria-pressed="${n === callIndex}">
        Call ${n + 1}: ${esc(outcome)}<small>${reply ? `${reply.seconds} s` : ""}${u ? ` · read ${num(u.prompt_tokens)} tokens, wrote ${num(u.completion_tokens)}` : ""}${calls.length > 1 ? " · Press to see this call below" : ""}</small></button></li>`;
    }
    const line = stepLine(e);
    return `<li><span class="step-tab">${esc(line.tab)}</span><div class="step-body">${line.html}</div></li>`;
  }).join("");

  renderBudget(turn.find((e) => e.kind === "brain.budget"), calls[0]?.reply);
  const call = calls[callIndex];
  $("#req-label").textContent = calls.length > 1 ? `(call ${callIndex + 1} of ${calls.length})` : "";
  renderRequest(call.request.body);
  renderReply(call.reply);
}
$("#turn-prev").onclick = () => { const t = groupTurns().length; brainView.turn = Math.max(0, (brainView.turn ?? t - 1) - 1); brainView.call = null; renderBrain(); };
$("#turn-next").onclick = () => {
  const t = groupTurns().length;
  const next = (brainView.turn ?? t - 1) + 1;
  brainView.turn = next >= t - 1 ? null : next;   // back on the newest: follow new messages again
  brainView.call = null;
  renderBrain();
};
$("#turn-steps").onclick = (e) => {
  const b = e.target.closest("[data-call]");
  if (b) { brainView.call = Number(b.dataset.call); renderBrain(); }
};

const BUDGET_PARTS = [
  ["system+memories", "System prompt and memories", "--s1", false],
  ["tools", "Tool descriptions", "--s2", false],
  ["history", "Earlier messages", "--s3", true],
  ["user", "This message", "--s4", true],
];
function renderBudget(b, firstReply) {
  if (!b) { $("#budget").innerHTML = '<div class="none">No budget was recorded for this message.</div>'; return; }
  const usedPct = (100 * b.total) / b.budget;
  const parts = BUDGET_PARTS.map(([key, name, color, darkInk]) => ({ key, name, color, darkInk, value: b[key] || 0, share: b.total ? (100 * (b[key] || 0)) / b.total : 0 }));
  const seg = parts.filter((p) => p.value > 0).map((p) => `<div class="${p.darkInk ? "dark-ink" : ""}" style="flex:${p.value} 1 0;background:var(${p.color})" title="${esc(p.name)}: ${num(p.value)} tokens">${p.share >= 9 ? `${Math.round(p.share)}%` : ""}</div>`).join("");
  const notes = { "system+memories": b.memories ? plural(b.memories, "memory", "memories") + " included" : "No memories included", history: b.history_turns ? plural(b.history_turns, "earlier message") : "" };
  const rows = parts.map((p) => `<tr><td><i class="swatch" style="background:var(${p.color})"></i>${esc(p.name)}${notes[p.key] ? ` <span class="muted small">· ${esc(notes[p.key])}</span>` : ""}</td>
    <td class="num">${num(p.value)}</td><td class="num muted">${Math.round(p.share)}%</td></tr>`).join("");
  const actual = firstReply?.usage?.prompt_tokens;
  $("#budget").innerHTML = `<div class="budget-head"><span class="v">${num(b.total)}</span><span class="muted">of ${num(b.budget)} tokens used (${usedPct < 1 ? "under 1" : Math.round(usedPct)}%)</span></div>
    <div class="meter" role="img" aria-label="${Math.round(usedPct)}% of the context used"><i style="width:${Math.min(100, usedPct)}%"></i></div>
    <div class="seg-title">What filled those ${num(b.total)} tokens</div>
    <div class="seg" role="img" aria-label="${esc(parts.map((p) => `${p.name} ${Math.round(p.share)}%`).join(", "))}">${seg}</div>
    <table class="seg-table"><tbody>${rows}</tbody></table>
    <p class="hint" style="margin:10px 0 0">These are estimates, made before sending.${actual ? ` llama-server counted <b>${num(actual)}</b> prompt tokens for the first call.` : ""}
      The context is ${num(b.budget + 1024)} tokens, and 1,024 of them stay free for the reply.</p>`;
}

const ROLE_NAMES = { system: "System prompt", user: "User", assistant: "Model", tool: "Tool result" };
function partHtml(p) {
  if (p.type === "text" || p.text !== undefined) return `<div class="m-text">${esc(p.text)}</div>`;
  const kinds = { image_url: "Image", input_audio: "Audio clip", input_video: "Video" };
  const key = Object.keys(kinds).find((k) => p[k]);
  if (!key) return `<div class="m-text">${esc(JSON.stringify(p))}</div>`;
  const blob = p[key].url || p[key].data || "";
  const size = blob.match(/<([\d,]+) chars>/)?.[1];
  return `<span class="m-part">${kinds[key]}${size ? ` · ${size} characters of data, shortened here` : ""}</span>`;
}
function messageHtml(m) {
  const body = typeof m.content === "string" ? (m.content ? `<div class="m-text">${esc(m.content)}</div>` : "")
    : (m.content || []).map(partHtml).join("");
  const calls = (m.tool_calls || []).map((c) => `<div class="m-text">Asks for <code>${esc(c.function?.name)}(${esc(c.function?.arguments)})</code></div>`).join("");
  return `<div class="m m-${esc(m.role)}"><div class="m-role">${esc(ROLE_NAMES[m.role] || m.role)}</div>${body}${calls}</div>`;
}
function renderRequest(body) {
  const { messages, tools, ...params } = body;
  const toolNames = (tools || []).map((t) => t.function?.name).filter(Boolean);
  const paramKeys = Object.keys(params);
  $("#last-request").innerHTML = `<div class="m-list">${messages.map(messageHtml).join("")}</div>
    ${toolNames.length ? `<div class="kv"><b>Tools offered:</b>${toolNames.map((n) => `<code>${esc(n)}</code>`).join("")}</div>` : ""}
    ${paramKeys.length ? `<div class="kv"><b>Settings:</b>${paramKeys.map((k) => `<code>${esc(k)}</code>`).join("")}</div>` : ""}
    <details class="more"><summary>Exact JSON sent (media shortened)</summary><pre style="max-height:420px">${esc(json(body))}</pre></details>`;
}
function renderReply(reply) {
  if (!reply) { $("#last-reply").innerHTML = '<div class="none">Waiting for the reply…</div>'; return; }
  const content = reply.message?.content || "";
  const calls = asked(reply.message);
  const looksLikeJson = content.trim().startsWith("{");
  const t = reply.timings || {}, u = reply.usage || {};
  const main = looksLikeJson ? `<pre>${esc(content)}</pre>` : content ? md(content) : "";
  $("#last-reply").innerHTML = `<div class="reply-text">${main}${calls.length ? `<div class="m-text">Asks for ${calls.map((n) => `<code>${esc(n)}</code>`).join(" ")}</div>` : ""}</div>
    <div class="tiles compact">${tiles([
      [reply.seconds, "s", "Total time"],
      [num(u.prompt_tokens), "", "Tokens read"],
      [num(u.completion_tokens), "", "Tokens written"],
      [t.predicted_per_second?.toFixed(1) ?? "–", "tok/s", "Writing speed"],
    ])}</div>
    <details class="more"><summary>Exact reply and timings</summary><pre style="max-height:320px">${esc(json({ seconds: reply.seconds, message: reply.message, timings: reply.timings, usage: reply.usage }))}</pre></details>`;
}

// --- Chat -------------------------------------------------------------------
// Clear stays visible, so people can find it. It is off when there is nothing to clear, and while a reply
// is on its way, because the server would add that reply to the conversation that was just cleared.
const syncChatClear = () => { $("#chat-clear").disabled = !!chatPending || !$("#chat-log").children.length; };
function addMsg(cls, html) {
  $("#chat-empty").hidden = true;
  const log = $("#chat-log");
  log.insertAdjacentHTML("beforeend", `<div class="msg ${cls}">${html}</div>`);
  log.scrollTop = log.scrollHeight;
  syncChatClear();
  return log.lastElementChild;
}
const callChip = (c) => `<span class="chip ${c.status === "ok" ? "ok" : c.status === "pending" ? "pending" : "bad"}"${c.call_id ? ` data-call-id="${c.call_id}"` : ""}>${esc(c.tool)} · ${esc(c.status === "pending" ? "needs your approval" : c.status)}</span>`;

// What the agent is doing now, shown in the reply bubble while the model works.
let chatPending = null;
const CHAT_STEPS = { "memory.search": () => "Searching memory…", "brain.request": () => "Waiting for the model…", "tools.call": (e) => `Ran ${e.tool}…` };
function noteChatStep(ev) {
  if (chatPending && CHAT_STEPS[ev.kind]) chatPending.dataset.step = CHAT_STEPS[ev.kind](ev);
}

$$("[data-suggest]").forEach((b) => (b.onclick = () => { $("#chat-input").value = b.dataset.suggest; $("#chat-input").focus(); }));
$("#chat-form").onsubmit = async (e) => {
  e.preventDefault();
  const text = $("#chat-input").value.trim();
  if (!text) return;
  if (appState && !appState.llm) { toast("The language model is not ready yet. Check the Serving tab.", "error"); return; }
  const withScreen = $("#chat-screen").checked;
  $("#chat-input").value = "";
  $("#chat-screen").checked = false;   // one screenshot per tick: never keep sending the screen by accident
  addMsg("user", `${esc(text)}${withScreen ? '<div class="msg-meta">with a screenshot</div>' : ""}`);
  const reply = addMsg("agent", "");
  reply.dataset.step = withScreen ? "Capturing your screen…" : "Thinking…";
  working(reply, "Thinking…");
  chatPending = reply;
  syncChatClear();
  const t0 = Date.now();
  const out = await busy(e.submitter ?? $("#chat-form button"), () => api("/api/chat", { text, with_screen: withScreen }));
  chatPending = null;
  syncChatClear();
  $("#chat-input").focus();
  if (!out) { reply.className = "msg agent error"; reply.textContent = "No reply: the request failed."; return; }
  const calls = out.tool_calls.map(callChip).join("");
  reply.innerHTML = `${md(out.text)}${calls ? `<div class="calls">${calls}</div>` : ""}<div class="msg-meta">${((Date.now() - t0) / 1000).toFixed(1)} s</div>`;
  $("#chat-log").scrollTop = $("#chat-log").scrollHeight;
};
$("#chat-clear").onclick = (e) => busy(e.target, async () => {
  await api("/api/chat/clear", {});
  $("#chat-log").innerHTML = "";
  $("#chat-empty").hidden = false;
  toast("Chat cleared. Long-term memories stay.");
}).finally(syncChatClear);
// The server keeps the conversation over a reload, so show the same one the model still has.
async function loadChat() {
  let history;
  try { history = await api("/api/chat"); } catch { return; }
  for (const m of history) {
    if (m.approval) addApprovalMsg(m.approval.tool, m.approval);
    else addMsg(m.role === "user" ? "user" : "agent", m.role === "user" ? esc(m.content) : md(m.content));
  }
}

// --- Tasks ------------------------------------------------------------------
let recordingSince = null;
const clockText = (s) => `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
function setRecordingUI(recording) {
  if (recording && !recordingSince) recordingSince = Date.now();
  if (!recording) recordingSince = null;
  $$("#rec-btn, #rec-btn2").forEach((b) => {
    if (b.disabled) return;   // busy() restores its own label
    b.textContent = recording ? `Stop and get action items (${clockText(Math.round((Date.now() - recordingSince) / 1000))})` : "Record voice note";
    b.classList.toggle("recording", recording);
    b.classList.toggle("primary", !recording);
  });
}
setInterval(() => { if (recordingSince) setRecordingUI(true); }, 1000);
async function toggleRecording(button) {
  let s;
  try { s = await api("/api/state"); } catch (e) { toast(e.message, "error"); return; }
  if (!s.recording) {
    await busy(button, () => api("/api/mic/start", {}));
  } else {
    const done = await busy(button, () => api("/api/mic/stop", {}), { out: "#task1-out", text: "Listening back and finding action items…" });
    if (done && !isTab("tasks")) toast("Done. The action items are in the Tasks tab (press 6).");
  }
  refreshState();
}
$("#rec-btn").onclick = (e) => toggleRecording(e.target);
$("#rec-btn2").onclick = (e) => toggleRecording(e.target);
const sendFile = (url, out, text) => async (e) => {
  const input = e.target;
  await busy(input.parentElement, () => api(url, input.files[0], true), { out, text });
  input.value = "";   // allow choosing the same file again
};
$("#wav-file").onchange = sendFile("/api/task1/file", "#task1-out", "Transcribing and finding action items…");
$("#wav-file2").onchange = sendFile("/api/task1/file", "#task1-out", "Transcribing and finding action items…");
$("#img-file").onchange = sendFile("/api/task2/file", "#task2-out", "Looking at the screenshot…");
$("#video-file").onchange = sendFile("/api/task2/video_file", "#task2-out", "Watching the video…");
$("#watch-btn").onclick = (e) => busy(e.target, () => api("/api/task2/watch", { seconds: 10 }), { out: "#task2-out", text: "Recording 10 s of your screen, then looking at it…" });
const lookNow = (e) => busy(e.target, () => api("/api/task2", {}), { out: "#task2-out", text: "Looking at your screen…" })
  .then((done) => { if (done && !isTab("tasks")) toast("Done. The answer is in the Tasks tab (press 6)."); });
$("#look-btn").onclick = lookNow;
$("#look-btn2").onclick = lookNow;

function renderTask1(e) {
  const items = e.action_items.map((a) => `<li><span>${esc(a.task)}${a.due ? ` <span class="muted">(due ${esc(a.due)})</span>` : ""}</span></li>`);
  const tries = e.attempts > 1 ? ` · the model needed ${e.attempts} tries to write valid JSON` : "";
  $("#task1-out").innerHTML = `<div class="result-meta">Last result · ${clock(e.ts)} · ${plural(e.chunks.length, "audio chunk")}${tries} · saved to memory</div>
    <h4>Action items</h4>${items.length ? `<ul class="todo">${items.join("")}</ul>` : '<p class="muted">None found.</p>'}
    <h4>Transcript</h4><p>${esc(e.transcript)}</p>`;
}
function renderTask2(e) {
  // An uploaded video sends no senses.video event, so the Seeing card gets it from here. A recording keeps its details.
  if (e.video && !$("#last-video video")?.src.endsWith(encodeURIComponent(e.video))) {
    $("#last-video").innerHTML = `<video controls muted src="${esc(media(e.video))}"></video>`;
    $("#video-info").textContent = `Uploaded · ${clock(e.ts)}`;
  }
  const shown =e.video ? `<video controls muted src="${esc(media(e.video))}"></video>` : zoomable(e.screenshot, "Screen that was analysed");
  const tries = e.attempts > 1 ? ` · the model needed ${e.attempts} tries to write valid JSON` : "";
  const problem = e.problem
    ? `<div class="callout warn"><h4>Problem</h4><p>${esc(e.problem)}</p></div>`
    : '<div class="callout ok"><h4>✓ No problem spotted</h4></div>';
  const fix = e.suggested_fix ? `<div class="callout info"><h4>Suggested fix</h4><p>${esc(e.suggested_fix)}</p></div>` : "";
  $("#task2-out").innerHTML = `<div class="result-meta">Last result · ${clock(e.ts)}${tries} · saved to memory</div>
    <div class="result-split"><figure>${shown}</figure><div><h4>What I see</h4><p>${esc(e.what_i_see)}</p>${problem}${fix}</div></div>`;
}

// --- Memory -------------------------------------------------------------------
const KIND_NAMES = { voice_note: "Voice note", action_item: "Action item", screen_help: "Screen", screen_video: "Screen video", note: "Note" };
const kindTag = (k) => `<span class="tag plain">${esc(KIND_NAMES[k] || k)}</span>`;
const isImage = (p) => /\.(png|jpe?g|webp)$/i.test(p || "");
const memRow = (m, extra = "") => `<div class="mem"><div><div class="mem-text">${esc(m.text)}</div>
  <div class="mem-meta">${kindTag(m.kind)}${m.when ? `<span>${esc(m.when)}</span>` : ""}${extra}</div></div>
  ${isImage(m.media) ? zoomable(m.media, "Screen saved with this memory") : ""}</div>`;
async function refreshMemory() {
  let rows;
  try { rows = await api("/api/memory"); } catch (e) { toast(e.message, "error"); return; }
  $("#mem-count").textContent = rows.length ? `(${num(rows.length)})` : "";
  $("#mem-clear").hidden = !rows.length;
  $("#mem-all").innerHTML = rows.map((m) => memRow(m)).join("")
    || '<div class="none">No memories yet. Record a voice note in Tasks, or ask the chat to remember something.</div>';
  const rejected = allEvents.filter((e) => e.kind === "memory.rejected").slice(-8).reverse();
  $("#mem-rejected").innerHTML = rejected.map((e) => `<div class="mem"><div><div class="mem-text">${esc(e.text)}</div>
    <div class="mem-meta">${kindTag(e.mem_kind)}<span>${esc(e.reason)}</span></div></div></div>`).join("")
    || '<div class="none">Nothing refused yet.</div>';
}
$("#mem-form").onsubmit = async (e) => {
  e.preventDefault();
  const q = $("#mem-q").value.trim();
  if (!q) return;
  const hits = await busy(e.submitter ?? $("#mem-form button"), () => api("/api/memory/search", { q }));
  if (!hits) return;
  const score = (h) => `<span class="score-meter" title="Similarity: higher means closer in meaning"><i style="--pct:${Math.max(0, h.score) * 100}%"></i>${h.score.toFixed(2)} similar</span>`;
  $("#mem-hits").innerHTML = `<div class="hint" style="margin-top:8px">Best ${plural(hits.length, "match", "matches")} for "${esc(q)}"</div>` +
    (hits.map((h) => memRow(h, score(h))).join("") || '<div class="none">No matching memories.</div>');
};
$("#mem-clear").onclick = (e) => {
  const count = $("#mem-count").textContent.replace(/[()]/g, "");
  if (!confirm(`Delete ${count === "1" ? "the 1 memory" : `all ${count} memories`}? This cannot be undone.`)) return;
  busy(e.target, async () => {
    await api("/api/memory/clear", {});
    $("#mem-hits").innerHTML = "";
    await refreshMemory();
    toast("All memories deleted.");
  });
};

// --- Tools --------------------------------------------------------------------
const TIER_NOTES = { safe: "run on their own", confirm: "wait for your approval", blocked: "never run" };
async function refreshTools() {
  let list;
  try { list = await api("/api/tools"); } catch (e) { toast(e.message, "error"); return; }
  $("#tool-list tbody").innerHTML = Object.keys(TIER_NOTES).map((tier) => {
    const group = list.filter((t) => t.tier === tier);
    if (!group.length) return "";
    return `<tr class="tier-row"><td colspan="3"><span class="tag ${tier}">${tier}</span> <span class="muted">${TIER_NOTES[tier]}</span></td></tr>` +
      group.map((t) => `<tr><td>${esc(t.name)}</td><td class="nowrap">${t.guarded ? '<span class="tag plain">guarded</span>' : ""}</td><td class="muted">${esc(t.description)}</td></tr>`).join("");
  }).join("");
  refreshToolCalls();
}
const argsText = (a) => (Object.keys(a || {}).length ? short(JSON.stringify(a), 80) : "");
const resultText = (v) => short(typeof v === "string" ? v : JSON.stringify(v) ?? "", 160);
function refreshToolCalls() {
  const calls = allEvents.filter((e) => e.kind === "tools.call").slice(-25).reverse();
  const tone = (s) => (s === "ok" ? "ok" : s === "pending" ? "warn" : "bad");
  $("#tool-calls tbody").innerHTML = calls.map((c) => `<tr><td class="nowrap muted">${clock(c.ts)}</td><td>${esc(c.tool)}</td>
    <td>${argsText(c.args) ? `<code>${esc(argsText(c.args))}</code>` : '<span class="muted">none</span>'}</td>
    <td><span class="tag ${tone(c.status)}">${esc(c.status)}</span></td><td class="muted">${esc(resultText(c.result ?? c.error))}</td></tr>`).join("");
}
let pendingKey = "";
// Approval shows every argument in full: nobody should approve a command they cannot read.
function renderPending(pending) {
  const key = JSON.stringify(pending);
  if (key === pendingKey) return;   // do not rebuild the buttons while someone is about to click one
  pendingKey = key;
  const args = (a) => Object.entries(a || {}).map(([k, v]) => `<div class="arg"><b>${esc(k)}</b><pre>${esc(typeof v === "string" ? v : json(v))}</pre></div>`).join("");
  const html = pending.map((p) => `<div class="ask">
    <div class="ask-title"><strong>Approval needed</strong>The agent wants to run <b>${esc(p.tool)}</b></div>
    <div class="ask-actions"><button class="primary small" data-decide="${p.call_id}" data-yes="1">Approve</button>
      <button class="danger small" data-decide="${p.call_id}">Deny</button></div>
    <div class="ask-args">${args(p.args)}</div>
    <p class="ask-note">Not done yet. Nothing runs until you press Approve.</p></div>`).join("");
  // CSS shows one of these: the chat panel, or the banner when the chat is out of view.
  for (const id of ["#chat-approvals", "#approval-banner"]) {
    $(id).innerHTML = html;
    $(id).hidden = !pending.length;
  }
  $("#tools-badge").hidden = !pending.length;
  $("#tools-badge").textContent = pending.length;
}
const approvalChip = (tool, out) => {
  const label = { ok: "approved", denied: "denied" }[out.status] || out.status;
  return `<span class="chip ${out.status === "ok" ? "ok" : "bad"}">${esc(tool)} · ${esc(label)}</span>`;
};
const approvalDetail = (out) => (out.status === "denied" ? "Nothing was run." : out.status === "ok" ? resultText(out.result) || "Finished with no output." : resultText(out.error));
const addApprovalMsg = (tool, out) => addMsg("agent", `<div class="calls">${approvalChip(tool, out)}</div><div class="msg-meta">${esc(approvalDetail(out))}</div>`);
// The chat has to show what happened after a decision, because the model does not reply again until the next message.
// The outcome replaces the "needs your approval" chip in the reply that asked. After a reload that chip is gone,
// so the outcome gets its own message, from the chat history that the server keeps for the model.
function showApprovalOutcome(callId, tool, out) {
  const chip = approvalChip(tool, out);
  const detail = approvalDetail(out);
  const waiting = $(`[data-call-id="${callId}"]`);
  if (!waiting) { addApprovalMsg(tool, out); return; }
  const calls = waiting.parentElement;
  const who = calls.children.length > 1 ? `${tool}: ` : "";   // several calls in one reply: say which one this is
  waiting.outerHTML = chip;
  calls.insertAdjacentHTML("afterend", `<div class="msg-meta">${esc(who + detail)}</div>`);
  $("#chat-log").scrollTop = $("#chat-log").scrollHeight;
}
document.addEventListener("click", async (e) => {
  const b = e.target.closest("[data-decide]");
  if (!b) return;
  const yes = !!b.dataset.yes;
  const callId = Number(b.dataset.decide);
  const tool = appState?.pending.find((p) => p.call_id === callId)?.tool ?? "Tool";
  b.closest(".ask").querySelectorAll("button").forEach((x) => (x.disabled = true));   // one decision per call
  const out = await busy(b, () => api("/api/tools/approve", { call_id: callId, yes }));
  if (out) {
    toast(yes ? "Approved." : "Denied.");
    showApprovalOutcome(callId, tool, out);
  }
  pendingKey = "";
  refreshState();
});

// --- Scorecard ------------------------------------------------------------------
// Plain names for the screen. The meanings come from the docstring of each evals/<component>/eval.py.
const COMPONENTS = {
  serving: ["Model serving", "Speed and memory of the local model server"],
  vad: ["Voice activity detection", "Does it fire on noise, or miss real speech?"],
  asr: ["Speech to text", "How many words does Gemma 4 audio get wrong?"],
  screen: ["Screen sampling", "Did the sampler keep a frame when something important happened?"],
  structured: ["Structured output", "How often is the JSON valid against the schema?"],
  tool_calls: ["Tool choice", "Does the model pick the right tool with the right arguments?"],
  memory: ["Memory", "Are the right memories found, and is junk kept out?"],
  tools: ["Tool safety", "Do safe calls succeed, and are unsafe calls stopped?"],
  task1: ["Task 1: voice note to action items", "Do the action items match the expected list?"],
  task2: ["Task 2: look at my screen", "Is the explanation and the fix correct?"],
};
// [name, what it measures, format]. "pct" shows a 0 to 1 share as a percentage.
const METRICS = {
  gen_tok_s: ["Writing speed", "Tokens per second while generating", "tok/s"],
  prompt_tok_s: ["Reading speed", "Tokens per second on a long prompt", "tok/s"],
  ttft_s: ["Time to first token", "Short prompt: the fastest it can start", "s"],
  ttft_2k_s: ["Time to first token, long prompt", "About 2,000 tokens: what a real request costs", "s"],
  peak_rss_gb: ["Peak memory", "Memory held by the model server", "GB"],
  system_used_gb: ["System memory in use", "The whole machine, including the browser and Meet", "GB"],
  false_trigger_rate: ["False triggers", "Non-speech frames marked as speech", "pct"],
  missed_speech_rate: ["Missed speech", "Speech frames it did not mark", "pct"],
  spurious_segments: ["Spurious segments", "Detected segments that overlap no labeled speech", ""],
  wer: ["Word error rate", "Wrong, missing or extra words, per reference word", "pct"],
  key_moment_recall: ["Key moments caught", "Key moments with a kept frame soon after", "pct"],
  kept_ratio: ["Frames kept", "Lower means fewer image tokens sent", "pct"],
  valid_rate: ["Valid JSON, schema enforced", "The server turns the schema into a grammar", "pct"],
  prompt_only_valid_rate: ["Valid JSON, schema in the prompt only", "For comparison: the schema is only described", "pct"],
  tool_accuracy: ["Right tool", "The right tool, or correctly no tool", "pct"],
  exact_rate: ["Right tool and arguments", "Every expected argument also matches", "pct"],
  selection_recall: ["Tool was offered", "The expected tool was among those offered to the model", "pct"],
  recall_at_3: ["Found in the top 3", "Relevant memories found in the top 3 results", "pct"],
  junk_stored_rate: ["Junk stored", "Junk items that got stored anyway", "pct"],
  good_rejected_rate: ["Good notes refused", "Good items the filter wrongly threw away", "pct"],
  success_rate: ["Safe calls succeed", "Calls expected to work that ran without error", "pct"],
  unsafe_blocked_rate: ["Unsafe calls stopped", "Calls expected to be blocked or confirmed that did not run", "pct"],
  policy_accuracy: ["Outcome as expected", "Calls whose outcome matched exactly", "pct"],
  precision: ["Precision", "Predicted items that match the expected list", "pct"],
  recall: ["Recall", "Expected items that were found", "pct"],
  correct_rate: ["Correct answers", "The problem call is right, and every required phrase is there", "pct"],
  correct_rate_image: ["Correct on screenshots", "Same check, screenshots only", "pct"],
  correct_rate_video: ["Correct on videos", "Same check, videos only", "pct"],
};
const metricInfo = (m) => METRICS[m] || [titleCase(m.replaceAll("_", " ")), "", ""];
function fmtValue(v, unit) {
  if (typeof v !== "number") return esc(v);
  if (unit === "pct") return `${+(v * 100).toFixed(1)}%`;
  const n = v >= 100 ? Math.round(v) : +v.toFixed(2);
  return unit ? `${n} ${unit}` : String(n);
}
function parseTarget(t) {
  const m = (t || "").match(/([<>]=?)\s*([\d.]+)/);
  return m ? { atMost: m[1].startsWith("<"), value: Number(m[2]) } : null;
}
const fmtTarget = (t, unit) => (t ? `${t.atMost ? "≤" : "≥"} ${fmtValue(t.value, unit)}` : "");

// A small bar for one result. The bar is the value and the tick is the target, so the margin is visible.
function bullet(cell, target, unit) {
  if (!target || typeof cell.value !== "number") return "";
  const max = unit === "pct" && !target.atMost ? 1 : Math.max(target.value, cell.value) * 1.25 || 1;
  const pos = (x) => Math.min(100, (100 * x) / max);
  return `<span class="bullet${cell.pass === false ? " bad" : ""}" title="Target ${fmtTarget(target, unit)}, measured ${fmtValue(cell.value, unit)}"><i style="width:${pos(cell.value)}%"></i><b style="left:calc(${pos(target.value)}% - 1px)"></b></span>`;
}
function scoreCell(cell, target, unit) {
  if (!cell) return '<td class="num"></td>';
  const mark = cell.pass === true ? '<em class="ok">✓ pass</em>' : cell.pass === false ? '<em class="bad">✗ fail</em>' : "";
  return `<td class="num"><span class="val${cell.pass === false ? " bad" : ""}${cell.pass === null ? " plain" : ""}">${fmtValue(cell.value, unit)}${mark}</span>${bullet(cell, target, unit)}</td>`;
}
const fmtWhen = (w) => new Date(w.replace(" ", "T")).toLocaleString([], { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" });
const TIER_KEYS = ["any", "e4b", "e2b"];
const cellsOf = (r) => TIER_KEYS.map((t) => r[t]).filter(Boolean);

function evalRows(rows) {
  const tiers = TIER_KEYS.filter((t) => rows.some((r) => r[t]));
  const head = tiers.map((t) => `<th class="num">${t === "any" ? "Result" : t.toUpperCase()}</th>`).join("");
  const body = [...rows].sort((a, b) => !!b.threshold - !!a.threshold).map((r) => {
    const [name, desc, unit] = metricInfo(r.metric);
    const target = parseTarget(r.threshold);
    return `<tr${target ? "" : ' class="extra"'}><td><div>${esc(name)}</div>${desc ? `<div class="metric-desc">${esc(desc)}</div>` : ""}</td>
      <td class="num">${target ? esc(fmtTarget(target, unit)) : '<span class="muted small">Measured only</span>'}</td>
      ${tiers.map((t) => scoreCell(r[t], target, unit)).join("")}</tr>`;
  }).join("");
  return `<div class="table-wrap"><table class="eval-table tiers-${tiers.length}"><thead><tr><th>Metric</th><th class="num">Target</th>${head}</tr></thead><tbody>${body}</tbody></table></div>`;
}
function sourceText(cell) {
  const entries = Object.entries(cell?.sources || {});
  if (!entries.length) return `${plural(cell?.n ?? 0, "run")}, measured live`;
  return entries.map(([src, n]) => `${num(n)} ${src}`).join(" + ") + " cases";
}
function evalBlock(component, rows, liveHtml, open) {
  const [name, desc] = COMPONENTS[component] || [component, ""];
  const notRun = rows[0].status === "not run";
  const cells = rows.flatMap(cellsOf).filter((c) => c.pass !== null && c.pass !== undefined);
  const pass = cells.filter((c) => c.pass).length, fail = cells.length - pass;
  const first = rows.flatMap(cellsOf)[0];
  const state = notRun ? ["none", "–"] : fail ? ["bad", "✗"] : ["", "✓"];
  const count = notRun ? "Not run yet" : fail ? `${fail} of ${cells.length} failing` : `${pass} of ${cells.length} met`;
  const body = notRun ? `<p class="hint">No results yet. Run <code>python cli.py evals --run ${esc(component)}</code>, or try one case now.</p>` : evalRows(rows);
  return `<details class="eval" data-component="${esc(component)}"${open ? " open" : ""}>
    <summary><span class="eval-state ${state[0]}" aria-hidden="true">${state[1]}</span>
      <span><span class="eval-name">${esc(name)}${name === component ? "" : `<code>${esc(component)}</code>`}</span><div class="eval-desc">${esc(desc)}</div></span>
      <span class="eval-count${fail ? " bad" : ""}">${count}</span>
      <span class="eval-src">${first ? `${esc(sourceText(first))}<br>${esc(fmtWhen(first.when))}` : ""}</span>
      <span class="eval-chev" aria-hidden="true">▶</span></summary>
    <div class="eval-body">${body}
      <div class="eval-try"><button class="small" data-live="${esc(component)}" title="Run one eval case now and show its input and output" data-busy="Running one case…">Try one case now</button></div>
      <div class="live-out" data-live-out="${esc(component)}">${liveHtml || ""}</div></div></details>`;
}
function renderScoreSummary(rows) {
  const all = rows.flatMap(cellsOf);
  const judged = all.filter((c) => c.pass === true || c.pass === false);
  const pass = judged.filter((c) => c.pass).length, fail = judged.length - pass;
  const notRun = rows.filter((r) => r.status === "not run").length;
  const components = [...new Set(rows.map((r) => r.component))];
  const sourceTotals = {};
  for (const c of components) {
    const cell = rows.filter((r) => r.component === c).flatMap(cellsOf)[0];
    for (const [src, n] of Object.entries(cell?.sources || {})) sourceTotals[src] = (sourceTotals[src] || 0) + n;
  }
  const whens = all.map((c) => c.when).filter(Boolean).sort();
  const state = fail ? `<div class="hero-state bad">✗ ${fail} failing</div>` : judged.length ? '<div class="hero-state ok">✓ All targets met</div>' : "";
  const src = Object.entries(sourceTotals).map(([s, n]) => `${num(n)} ${s}`).join(" + ") || "None yet";
  $("#score-summary").innerHTML = `<div class="score-top"><div><div class="hero">${pass}<small> of ${judged.length}</small></div>
      <div class="hero-label">target checks passed, counting each metric on each model</div>${state}</div>
    <div class="tiles">${tiles([
      [components.length - notRun, `of ${components.length}`, notRun ? `Components run, ${notRun} not run yet` : "Components run"],
      [all.length - judged.length, "", "Extra measurements with no target"],
      [src, "", "Test cases"],
      [whens.length ? fmtWhen(whens[whens.length - 1]) : "–", "", "Latest run"],
    ])}</div></div>
    ${sourceTotals.synthetic ? `<div class="source-note"><b>Read with care: ${sourceTotals.user ? "most" : "all"} of these cases are synthetic.</b> They are generated by code, and we know the right answer by construction. But text-to-speech is cleaner than a real voice, and a drawn terminal is cleaner than a real screen, so speech, voice detection and Task 1 scores are optimistic. A drop is a real regression; a pass does not prove it works for you.</div>` : ""}`;
}
const openEvals = new Set();
const autoOpened = new Set();
const liveOutputs = {};
const LEVEL_NAMES = { component: "Each part on its own", "end-to-end": "The whole agent, end to end" };
let scoreComponents = [];
async function refreshScorecard() {
  let rows;
  try { rows = await api("/api/scorecard"); } catch (e) { toast(e.message, "error"); return; }
  renderScoreSummary(rows);
  scoreComponents = [...new Set(rows.map((r) => r.component))];
  // A failing check opens itself the first time it appears, so it cannot be missed.
  for (const r of rows.filter((x) => cellsOf(x).some((c) => c.pass === false))) {
    if (!autoOpened.has(r.component)) openEvals.add(r.component);
    autoOpened.add(r.component);
  }
  let level = null;
  $("#score-list").innerHTML = scoreComponents.map((c) => {
    const own = rows.filter((r) => r.component === c);
    const head = own[0].level !== level ? `<div class="level">${esc(LEVEL_NAMES[own[0].level] || own[0].level)}</div>` : "";
    level = own[0].level;
    return head + evalBlock(c, own, liveOutputs[c], openEvals.has(c));
  }).join("");
  updateExpandLabel();
}
const updateExpandLabel = () => { $("#score-expand").textContent = openEvals.size >= scoreComponents.length && scoreComponents.length ? "Close all" : "Open all"; };
$("#score-list").addEventListener("toggle", (e) => {
  const c = e.target.dataset?.component;
  if (!c) return;
  if (e.target.open) openEvals.add(c); else openEvals.delete(c);
  updateExpandLabel();
}, true);
$("#score-expand").onclick = () => {
  const open = openEvals.size < scoreComponents.length;
  $$("#score-list .eval").forEach((d) => (d.open = open));
};
$("#score-refresh").onclick = (e) => busy(e.target, refreshScorecard);
$("#score-list").addEventListener("click", (e) => {
  const b = e.target.closest("[data-live]");
  if (b) runLive(b.dataset.live, b);
});

const MEDIA_KEYS = new Set(["image", "audio", "video"]);
function liveRecord(r) {
  const shown = (r.image ? zoomable(r.image, "Input of this eval case") : "") +
    (r.audio ? `<audio controls src="${esc(media(r.audio))}"></audio>` : "") +
    (r.video ? `<video controls muted src="${esc(media(r.video))}"></video>` : "");
  const rows = Object.entries(r).filter(([k]) => !MEDIA_KEYS.has(k)).map(([k, v]) => {
    const isText = typeof v === "string";
    return `<dt>${esc(titleCase(k.replaceAll("_", " ")))}</dt><dd class="${isText ? "" : "mono"}">${esc(isText ? v : JSON.stringify(v))}</dd>`;
  }).join("");
  return `${shown}<dl class="live-kv" style="margin:0">${rows}</dl>`;
}
const runLive = (component, button) => {
  const out = $(`[data-live-out="${component}"]`);
  working(out, "Running one case…");
  return busy(button, async () => {
    const recs = await api("/api/evals/live", { component, n: 1 });
    liveOutputs[component] = recs.map(liveRecord).join("<hr>");
    out.innerHTML = liveOutputs[component];
  }).finally(() => { if (out.querySelector(".working")) out.innerHTML = ""; });
};

showTab(location.hash.slice(1) || "serving");
setInterval(pollEvents, 700);
setInterval(refreshState, 2000);
pollEvents();
refreshState();
loadChat();
