// Dashboard: polls the event log and redraws the panel each event belongs to.
const $ = (sel) => document.querySelector(sel);
const esc = (s) => String(s ?? "").replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
const json = (o) => JSON.stringify(o, null, 1);
let lastEventId = 0;
let allEvents = [];

async function api(path, body, raw) {
  const opts = body === undefined ? {} : { method: "POST",
    body: raw ? body : JSON.stringify(body), headers: raw ? {} : { "Content-Type": "application/json" } };
  const r = await fetch(path, opts);
  if (!r.ok) throw new Error(`${path}: ${r.status} ${await r.text()}`);
  return r.json();
}

async function busy(button, fn) {   // disable a button while its request runs
  button.disabled = true;
  try { return await fn(); } catch (e) { alert(e.message); } finally { button.disabled = false; }
}

// --- Tabs (keys 1-7 switch tabs when you are not typing) --------------------
function showTab(name) {
  document.querySelectorAll(".panel").forEach((p) => p.classList.toggle("active", p.id === "panel-" + name));
  document.querySelectorAll("#tabs button").forEach((b) => b.classList.toggle("active", b.dataset.tab === name));
  if (name === "memory") refreshMemory();
  if (name === "tools") refreshTools();
  if (name === "scorecard") refreshScorecard();
}
document.querySelectorAll("#tabs button").forEach((b) => (b.onclick = () => showTab(b.dataset.tab)));
document.addEventListener("keydown", (e) => {
  if (["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement.tagName)) return;
  const b = document.querySelectorAll("#tabs button")[Number(e.key) - 1];
  if (b) showTab(b.dataset.tab);
});
showTab("serving");

// --- Header state -------------------------------------------------------------
async function refreshState() {
  const s = await api("/api/state");
  $("#llm-dot").classList.toggle("on", s.llm);
  $("#emb-dot").classList.toggle("on", s.embed);
  $("#llm-mem").textContent = s.llm ? `${s.llm_rss_gb} GB` : "starting...";
  $("#emb-mem").textContent = s.embed ? `${s.embed_rss_gb} GB` : "starting...";
  if (document.activeElement !== $("#tier")) $("#tier").value = s.tier;
  const mic = $("#mic-switch");
  mic.className = "mic " + (s.mic_paused ? "paused" : "live");
  mic.textContent = s.mic_paused ? "Mic: PAUSED" : s.recording ? "Mic: RECORDING" : "Mic: ON";
  $("#serving-info").innerHTML = `<table><tr><th>tier</th><td>${s.tier}</td></tr>
    <tr><th>LLM server :9931</th><td>${s.llm ? "ready" : "not ready"} · ${s.llm_rss_gb} GB resident</td></tr>
    <tr><th>Embedding server :9932</th><td>${s.embed ? "ready" : "not ready"} · ${s.embed_rss_gb} GB resident</td></tr></table>`;
  renderPending(s.pending);
}
$("#mic-switch").onclick = async () => {
  const paused = $("#mic-switch").classList.contains("live");
  await api("/api/mic/pause", { paused });
  refreshState();
};
$("#tier").onchange = (e) => busy($("#tier"), () => api("/api/serving/switch", { tier: e.target.value }));
$("#bench-btn").onclick = (e) => busy(e.target, () => api("/api/serving/bench", {}));

// --- Event log + per-panel renderers ------------------------------------------
async function pollEvents() {
  const evs = await api(`/api/events?after=${lastEventId}`);
  for (const ev of evs) {
    lastEventId = ev.id;
    allEvents.push(ev);
    (renderers[ev.kind] || (() => {}))(ev);
  }
  if (evs.length) renderEventLog();
}
function renderEventLog() {
  const f = $("#event-filter").value.trim();
  const shown = allEvents.filter((e) => !f || e.kind.includes(f)).slice(-150);
  $("#event-log").innerHTML = shown.reverse().map((e) => {
    const { id, ts, kind, ...rest } = e;
    const t = new Date(ts * 1000).toLocaleTimeString();
    return `<div class="ev">${t} <b>${esc(kind)}</b> ${esc(JSON.stringify(rest)).slice(0, 300)}</div>`;
  }).join("");
}
$("#event-filter").oninput = renderEventLog;

const renderers = {
  "serving.bench": (e) => {
    $("#bench-table tbody").insertAdjacentHTML("afterbegin", `<tr><td>${e.tier}</td><td>${e.ttft_s?.toFixed(2)} s</td>
      <td>${e.gen_tok_s?.toFixed(1)}</td><td>${e.prompt_tok_s?.toFixed(0)}</td><td>${e.rss_gb} GB</td></tr>`);
  },
  "senses.audio": drawTimeline,
  "senses.screen": (e) => { $("#last-screen").src = "/media?path=" + encodeURIComponent(e.path); },
  "senses.video": (e) => {
    if (e.state === "recording") { $("#video-info").textContent = `Recording ${e.seconds} s of screen...`; return; }
    $("#last-video").src = "/media?path=" + encodeURIComponent(e.path);
    $("#video-info").textContent = `${e.seconds.toFixed(1)} s at ${e.width}x${e.height}`;
  },
  "senses.screen.warning": (e) => { $("#screen-warning").textContent = e.message; },
  "brain.budget": (e) => renderBudget(e),
  "brain.request": (e) => { $("#last-request").textContent = json(e.body); },
  "brain.reply": (e) => {
    if ($("#budget-actual") && e.timings) $("#budget-actual").textContent = `${e.timings.prompt_n + (e.timings.cache_n || 0)} prompt tokens`;
    $("#last-reply").textContent = json({ seconds: e.seconds, message: e.message, timings: e.timings }); },
  "tools.call": () => refreshToolCalls(),
  "memory.stored": () => isTab("memory") && refreshMemory(),
  "task1.done": renderTask1,
  "task2.done": renderTask2,
  "hotkey.pressed": () => showTab("tasks"),
};
const isTab = (name) => $("#panel-" + name).classList.contains("active");

// --- Senses: timeline of speech segments (top) and chunks sent to Gemma (bottom) ---
function drawTimeline(e) {
  const total = Math.max(e.seconds, 1);
  const x = (s) => (s / total) * 1000;
  let svg = "";
  for (let t = 30; t < total; t += 30) svg += `<line x1="${x(t)}" x2="${x(t)}" y1="0" y2="90" stroke="var(--bad)" stroke-dasharray="4"/>`;
  for (const [a, b] of e.speech) svg += `<rect x="${x(a)}" y="12" width="${x(b) - x(a)}" height="26" fill="var(--speech)" rx="3"/>`;
  e.chunks.forEach(([a, b], i) => {
    svg += `<rect x="${x(a)}" y="50" width="${x(b) - x(a)}" height="26" fill="none" stroke="var(--chunk)" stroke-width="3" rx="3"/>`;
    svg += `<text x="${x(a) + 6}" y="68" font-size="13" fill="var(--chunk)">chunk ${i + 1}: ${(b - a).toFixed(1)} s</text>`;
  });
  $("#timeline").innerHTML = svg;
  $("#audio-info").textContent = `${e.seconds} s recorded · ${e.speech.length} speech segments (blue) · ` +
    `${e.chunks.length} chunks (green) · ~${e.est_tokens} audio tokens · red dashes = Gemma's 30 s limit`;
}

// --- Brain: token budget bar --------------------------------------------------
function renderBudget(tokens) {
  if (!tokens) return;
  const colors = { "system+memories": "#8a6", tools: "#a6a", history: "#a86", user: "#68c" };
  const parts = Object.entries(colors).map(([k, c]) =>
    `<div style="width:${Math.max(1, (100 * tokens[k]) / tokens.budget)}%;background:${c}" title="${k}"></div>`);
  const legend = Object.entries(colors).map(([k, c]) => `<span style="color:${c}">■</span> ${k}: ${tokens[k]}`).join(" &nbsp; ");
  $("#budget").innerHTML = `<div class="bar">${parts.join("")}</div>${legend}<br>
    <span class="muted">estimated ${tokens.total} of ${tokens.budget} tokens (context ${tokens.budget + 1024} minus 1024 reserved
    for the reply) · actual from llama-server: <b id="budget-actual">...</b></span>`;
}

// --- Chat -------------------------------------------------------------------
$("#chat-form").onsubmit = async (e) => {
  e.preventDefault();
  const text = $("#chat-input").value.trim();
  if (!text) return;
  $("#chat-input").value = "";
  const log = $("#chat-log");
  log.insertAdjacentHTML("beforeend", `<div class="msg user">${esc(text)}</div>`);
  const out = await busy(e.submitter, () => api("/api/chat", { text, with_screen: $("#chat-screen").checked }));
  if (!out) return;
  const calls = out.tool_calls.map((c) => `[${c.tool} → ${c.status}]`).join(" ");
  log.insertAdjacentHTML("beforeend", `<div class="msg">${esc(out.text)}${calls ? `<div class="calls">${esc(calls)}</div>` : ""}</div>`);
  log.scrollTop = log.scrollHeight;
};

// --- Tasks ------------------------------------------------------------------
async function toggleRecording(button) {
  const s = await api("/api/state");
  if (!s.recording) {
    await busy(button, () => api("/api/mic/start", {}));
    document.querySelectorAll("#rec-btn, #rec-btn2").forEach((b) => (b.textContent = "Stop and extract action items"));
  } else {
    document.querySelectorAll("#rec-btn, #rec-btn2").forEach((b) => (b.textContent = "Record voice note"));
    await busy(button, () => api("/api/mic/stop", {}));
  }
  refreshState();
}
$("#rec-btn").onclick = (e) => toggleRecording(e.target);
$("#rec-btn2").onclick = (e) => toggleRecording(e.target);
const sendFile = (url) => (e) => busy(e.target.parentElement, () => api(url, e.target.files[0], true));
$("#wav-file").onchange = sendFile("/api/task1/file");
$("#wav-file2").onchange = sendFile("/api/task1/file");
$("#img-file").onchange = sendFile("/api/task2/file");
$("#video-file").onchange = sendFile("/api/task2/video_file");
$("#watch-btn").onclick = (e) => busy(e.target, () => api("/api/task2/watch", { seconds: 10 }));
$("#look-btn").onclick = (e) => busy(e.target, () => api("/api/task2", {}));
$("#look-btn2").onclick = (e) => busy(e.target, () => api("/api/task2", {}));

function renderTask1(e) {
  const items = e.action_items.map((a) => `<li>${esc(a.task)}${a.due ? ` <span class="muted">(due ${esc(a.due)})</span>` : ""}</li>`);
  $("#task1-out").innerHTML = `<b>Transcript</b>\n${esc(e.transcript)}\n\n<b>Action items</b><ul>${items.join("")}</ul>` +
    `<span class="muted">${e.chunks.length} chunk(s) · JSON attempts: ${e.attempts} · saved to memory</span>`;
}
function renderTask2(e) {
  const media = e.video ? `<video controls muted src="/media?path=${encodeURIComponent(e.video)}"></video>`
    : `<img src="/media?path=${encodeURIComponent(e.screenshot)}">`;
  $("#task2-out").innerHTML = `${media}\n<b>What I see</b>\n${esc(e.what_i_see)}` +
    `\n\n<b>Problem</b>\n${esc(e.problem ?? "none")}\n\n<b>Suggested fix</b>\n${esc(e.suggested_fix ?? "-")}`;
}

// --- Memory -------------------------------------------------------------------
async function refreshMemory() {
  const rows = await api("/api/memory");
  $("#mem-all tbody").innerHTML = rows.map((m) =>
    `<tr><td>${m.id}</td><td>${m.when}</td><td>${esc(m.kind)}</td><td>${esc(m.text)}</td></tr>`).join("");
  const rejected = allEvents.filter((e) => e.kind === "memory.rejected").slice(-8).reverse();
  $("#mem-rejected tbody").innerHTML = rejected.map((e) => `<tr><td>${esc(e.text)}</td><td class="muted">${esc(e.reason)}</td></tr>`).join("")
    || "<tr><td class='muted'>Nothing rejected yet.</td></tr>";
}
$("#mem-form").onsubmit = async (e) => {
  e.preventDefault();
  const hits = await api("/api/memory/search", { q: $("#mem-q").value });
  $("#mem-hits tbody").innerHTML = hits.map((h) =>
    `<tr><td>${h.score.toFixed(3)}</td><td>${esc(h.kind)}</td><td>${esc(h.text)}</td></tr>`).join("");
};
$("#mem-clear").onclick = async () => { if (confirm("Delete all memories?")) { await api("/api/memory/clear", {}); refreshMemory(); } };

// --- Tools --------------------------------------------------------------------
async function refreshTools() {
  const list = await api("/api/tools");
  $("#tool-list tbody").innerHTML = list.map((t) => `<tr><td><b>${t.name}</b></td><td class="st-${t.tier === "safe" ? "ok" : t.tier === "confirm" ? "pending" : "blocked"}">${t.tier}${t.guarded ? " + guard" : ""}</td><td class="muted">${esc(t.description)}</td></tr>`).join("");
  refreshToolCalls();
}
function refreshToolCalls() {
  const calls = allEvents.filter((e) => e.kind === "tools.call").slice(-25).reverse();
  $("#tool-calls tbody").innerHTML = calls.map((c) => `<tr><td>${c.tool}</td><td>${esc(JSON.stringify(c.args))}</td>
    <td class="st-${c.status}">${c.status}</td><td>${esc(JSON.stringify(c.result ?? c.error)).slice(0, 160)}</td></tr>`).join("");
}
function renderPending(pending) {
  $("#pending").innerHTML = pending.length ? pending.map((p) => `<div class="row"><b>${p.tool}</b> <code>${esc(JSON.stringify(p.args))}</code>
    <button onclick="decide(${p.call_id}, true)">Approve</button><button class="danger" onclick="decide(${p.call_id}, false)">Deny</button></div>`).join("")
    : "Nothing pending.";
}
window.decide = async (call_id, yes) => { await api("/api/tools/approve", { call_id, yes }); refreshState(); };

// --- Scorecard ------------------------------------------------------------------
async function refreshScorecard() {
  const rows = await api("/api/scorecard");
  const cell = (c) => !c ? "" : `${typeof c.value === "number" ? +c.value.toFixed(3) : c.value} ` +
    (c.pass === true ? '<span class="pass">PASS</span>' : c.pass === false ? '<span class="fail">FAIL</span>' : "");
  const first = (r, i) => i === 0 || rows[i - 1].component !== r.component;   // first row of a component
  const runInfo = (r) => ["any", "e4b", "e2b"].filter((t) => r[t]).map((t) => `${t}: n=${r[t].n}, ${r[t].when}`).join("<br>");
  $("#scorecard tbody").innerHTML = rows.map((r, i) => `<tr${first(r, i) ? ' class="group"' : ""}>
    <td>${first(r, i) ? r.level : ""}</td>
    <td>${first(r, i) ? `<b>${r.component}</b><div class="muted small">${runInfo(r)}</div>` : ""}</td>
    <td>${r.status === "not run" ? '<span class="muted">not run yet</span>' : r.metric}</td><td>${r.threshold ?? ""}</td>
    <td>${cell(r.any)}</td><td>${cell(r.e4b)}</td><td>${cell(r.e2b)}</td>
    <td>${first(r, i) ? `<button class="small" onclick="runLive('${r.component}', this)">run 1 live</button>` : ""}</td></tr>`).join("");
}
$("#score-refresh").onclick = refreshScorecard;
window.runLive = (component, button) => busy(button, async () => {
  const recs = await api("/api/evals/live", { component, n: 1 });
  const media = recs.map((r) => (r.image ? `<img src="/media?path=${encodeURIComponent(r.image)}">` : "") +
    (r.audio ? `<audio controls src="/media?path=${encodeURIComponent(r.audio)}"></audio>` : "") +
    (r.video ? `<video controls muted src="/media?path=${encodeURIComponent(r.video)}"></video>` : "")).join("");
  $("#live-out").innerHTML = `<b>${component}</b>${media}<pre>${esc(json(recs))}</pre>`;
});

setInterval(pollEvents, 700);
setInterval(refreshState, 2000);
pollEvents();
refreshState();
