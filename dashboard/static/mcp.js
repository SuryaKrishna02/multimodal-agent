// Session 3: MCP servers in the Tools tab. Labels, connect, the tools each server offers, and their tiers.
// Uses the helpers of app.js ($, api, busy, esc, toast...).

Object.assign(DESCRIBE, {
  "mcp.config": (e) => `MCP config: ${Object.entries(e.servers).map(([n, l]) => `${n} (${l})`).join(", ")}`,
  "mcp.connected": (e) => `MCP server ${e.server} (${e.label}) offers ${plural(e.tools.length, "tool")}`,
  "mcp.enabled": (e) => `MCP server ${e.server}: the model may use ${e.tools.length ? e.tools.join(", ") : "no tools"}`,
  "mcp.tier": (e) => `MCP tool ${e.server}__${e.tool} is now ${e.tier}`,
  "mcp.error": (e) => `MCP server ${e.server}: ${e.error}`,
  "proactive.alert": (e) => `Spoke up: ${e.message} (${e.reason})`,
  "proactive.skipped": (e) => `Did not repeat ${e.key}: ${e.reason}`,
  "proactive.warning": (e) => e.message,
});
TAB_OF.mcp = "tools";

const LABEL_TAGS = { "on-device": "safe", "on-device (HTTP)": "safe", remote: "blocked" };
const hint = (a) => [a.read_only_hint && "read only", a.destructive_hint && "destructive"].filter(Boolean).join(", ");

function serverCard(s) {
  const tools = s.tools.map((t) => `<tr>
      <td><label><input type="checkbox" data-mcp-tool="${esc(s.name)}" value="${esc(t.name)}"${t.enabled ? " checked" : ""}> ${esc(t.name)}</label></td>
      <td><select data-mcp-tier="${esc(s.name)}" data-tool="${esc(t.name)}" aria-label="Tier of ${esc(t.name)}">
        ${["safe", "confirm", "blocked"].map((v) => `<option${v === t.tier ? " selected" : ""}>${v}</option>`).join("")}</select></td>
      <td class="muted">${esc(short(t.description, 120))}${hint(t.annotations) ? ` <small>(server says: ${esc(hint(t.annotations))})</small>` : ""}</td></tr>`).join("");
  return `<div class="mcp-server">
    <div class="row"><b>${esc(s.name)}</b><span class="tag ${LABEL_TAGS[s.label] || "plain"}">${esc(s.label)}</span>
      <span class="muted"><code>${esc(short(s.target, 90))}</code></span>
      ${s.secrets.length ? `<span class="muted">reads ${esc(s.secrets.join(", "))}</span>` : ""}
      <button class="small" data-mcp-connect="${esc(s.name)}" data-busy="Starting…">${s.connected ? "Reconnect" : "Connect"}</button>
      ${s.connected ? '<span class="tag ok">connected</span>' : ""}</div>
    ${s.label === "remote" ? '<p class="hint step-bad">Remote: every call to this server leaves this machine.</p>' : ""}
    ${tools ? `<div class="table-wrap"><table class="mcp-tools"><thead><tr><th>Model may use</th><th>Tier</th><th>What it does</th></tr></thead><tbody>${tools}</tbody></table></div>`
      : '<p class="hint">Connect to see its tools.</p>'}
  </div>`;
}

async function refreshMcp(view) {
  try { view ||= await api("/api/mcp"); } catch (e) { toast(e.message, "error"); return; }
  $("#mcp-servers").innerHTML = view.servers.map(serverCard).join("") || '<p class="hint">No servers. Paste a config.</p>';
  if (!$("#mcp-config").value) $("#mcp-config").value = view.config;
}
const oldToolsRefresh = TAB_REFRESH.tools;
TAB_REFRESH.tools = () => { oldToolsRefresh?.(); refreshMcp(); };

$("#mcp-servers").addEventListener("click", (ev) => {
  const b = ev.target.closest("[data-mcp-connect]");
  if (b) busy(b, () => api("/api/mcp/connect", { server: b.dataset.mcpConnect }).then(refreshMcp).then(refreshTools));
});
$("#mcp-servers").addEventListener("change", (ev) => {
  const box = ev.target.closest("[data-mcp-tool]");
  if (box) {
    const server = box.dataset.mcpTool;
    const chosen = $$(`[data-mcp-tool="${CSS.escape(server)}"]:checked`).map((c) => c.value);
    api("/api/mcp/enable", { server, tools: chosen }).then(refreshMcp).then(refreshTools).catch((e) => toast(e.message, "error"));
  }
  const sel = ev.target.closest("[data-mcp-tier]");
  if (sel) api("/api/mcp/tier", { server: sel.dataset.mcpTier, tool: sel.dataset.tool, tier: sel.value })
    .then(refreshMcp).then(refreshTools).catch((e) => toast(e.message, "error"));
});
$("#mcp-save").onclick = () => busy($("#mcp-save"), async () => {
  $("#mcp-problems").hidden = true;
  try {
    await refreshMcp(await api("/api/mcp/config", { text: $("#mcp-config").value }));
    toast("Config saved. Connect each server to see its tools.");
  } catch (e) {
    $("#mcp-problems").textContent = e.message;
    $("#mcp-problems").hidden = false;
  }
});

// --- Task 5 (Tasks tab): the screen watcher ---------------------------------------------------
Object.assign(DESCRIBE, {
  "task5.watch": (e) => `Task 5 screen watcher ${e.state}`,
  "task5.check": (e) => (e.alert ? `Task 5: ${e.what}` : "Task 5: nothing to report on the screen"),
});
TAB_OF.task5 = "tasks";
TAB_OF.proactive = "daemon";
function showT5(on) {
  const b = $("#t5-switch");
  b.classList.toggle("paused", !on);
  b.setAttribute("aria-checked", String(on));
  b.textContent = on ? "Watching on" : "Watching off";
}
$("#t5-switch").onclick = () => {
  const on = $("#t5-switch").getAttribute("aria-checked") !== "true";
  api("/api/task5/watch", { on }).then((s) => showT5(s.watching)).catch((e) => toast(e.message, "error"));
};
$("#t5-check").onclick = () => busy($("#t5-check"), async () => {
  const out = await api("/api/task5/check", {});
  $("#t5-out").innerHTML = out.alert
    ? `<p><b>Alert:</b> ${esc(out.what)}</p><p class="muted">Said: ${esc(out.say)}</p>`
    : `<p class="muted">Nothing to report.</p>`;
});
const oldTasksRefresh = TAB_REFRESH.tasks;
TAB_REFRESH.tasks = () => { oldTasksRefresh?.(); api("/api/task5").then((s) => showT5(s.watching)).catch(() => {}); };

// --- Session 3 rows of the scorecard ------------------------------------------------------
Object.assign(COMPONENTS, {
  tts: ["Voice out (Kokoro)", "How soon is the first sound ready, and does Whisper understand the speech?"],
  voice_stop: ["Stopping speech", "How long does the agent keep talking after you ask for silence?"],
  wake: ["Wake word", "Does \"Hey Sanjaya\" wake it, and does other talk leave it asleep?"],
  proactive: ["Proactive alerts", "Does it speak up about a failed test, and stay quiet otherwise?"],
  skills: ["Native skills", "Does a skill fire when it should, and stay quiet when it should not?"],
  mcp_select: ["MCP tool choice", "The right tool among 5, 20 and 40, offered all at once or the top 5"],
  task5: ["Task 5: proactive spoken alerts", "Watch a screen session and speak up once, when a test fails"],
  task6: ["Task 6: a spoken reminder through MCP", "A spoken request becomes the right reminder at the right time"],
});
Object.assign(METRICS, {
  first_audio_p50_s: ["First sound ready, median", "Kokoro writes the first spoken piece", "s"],
  first_audio_p95_s: ["First sound ready, 95th percentile", "The slow cases", "s"],
  rtf: ["Real-time factor", "Seconds of work per second of speech", "×"],
  stop_p95_ms: ["Stop latency, 95th percentile", "Stop request to silence", "ms"],
  local_p50_ms: ["Stop latency, this machine", "Median, the play_audio hook", "ms"],
  tab_p50_ms: ["Stop latency, dashboard tab", "Median, measured in the tab", "ms"],
  miss_rate: ["Missed wakes", "Times the phrase was said and it slept", "pct"],
  false_wakes: ["False wakes", "Wakes when nobody said the phrase", ""],
  false_wakes_per_hour: ["False wakes per hour", "On the recordings without the phrase (see negative_minutes)", ""],
  negative_minutes: ["Audio without the phrase", "Minutes that the false-wake rate is based on", "min"],
  false_alert_rate: ["False alerts", "Spoke up when nothing was wrong", "pct"],
  missed_alert_rate: ["Missed alerts", "Stayed quiet about a failed test", "pct"],
  mention_rate: ["Alert names it", "The alert named the failing test", "pct"],
  fire_recall: ["Fires when it should", "Prompts for a skill that called it", "pct"],
  false_fire_rate: ["Fires when it should not", "Prompts for no skill that called one", "pct"],
  acc_5: ["Right tool of 5", "All 5 offered", "pct"],
  acc_20: ["Right tool of 20", "All 20 offered", "pct"],
  acc_40: ["Right tool of 40", "All 40 offered", "pct"],
  sel_acc_20: ["Right tool of 20, top 5", "tools.select() offers 5", "pct"],
  sel_acc_40: ["Right tool of 40, top 5", "tools.select() offers 5", "pct"],
  sel_recall_40: ["Top 5 holds the right tool", "Out of 40", "pct"],
  detect_rate: ["Failures spoken", "At the failing screen or the next", "pct"],
  repeat_rate: ["Said twice", "The cool-down should prevent it", "pct"],
  requests_per_frame: ["Model calls per screen", "Unchanged screens are skipped", ""],
  success_rate: ["Right reminder", "Right text and due within 30 s of the asked time", "pct"],
  text_rate: ["Right reminder text", "Holds the word that was asked for", "pct"],
  time_rate: ["Right reminder time", "Due within 30 s of the asked time", "pct"],
  turn_s: ["Turn time", "Speech to text plus the model, median", "s"],
});
