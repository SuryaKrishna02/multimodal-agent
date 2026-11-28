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

// One server as a card that fits the task rail: a head line, then one row per tool (tick, name, tier, what it does).
const mcpTool = (s, t) => `<li class="mcp-tool">
    <label><input type="checkbox" data-mcp-tool="${esc(s.name)}" value="${esc(t.name)}"${t.enabled ? " checked" : ""}> <b>${esc(t.name)}</b></label>
    <select data-mcp-tier="${esc(s.name)}" data-tool="${esc(t.name)}" aria-label="Tier of ${esc(t.name)}">
      ${["safe", "confirm", "blocked"].map((v) => `<option${v === t.tier ? " selected" : ""}>${v}</option>`).join("")}</select>
    <p>${esc(short(t.description, 140))}${hint(t.annotations) ? ` <small>(server says: ${esc(hint(t.annotations))})</small>` : ""}</p></li>`;
function serverCard(s) {
  const on = s.tools.filter((t) => t.enabled).length;
  return `<div class="mcp-server${s.connected ? " connected" : ""}">
    <div class="mcp-head"><i class="dot${s.connected ? " on" : " off"}"></i><b>${esc(s.name)}</b>
      <span class="tag ${LABEL_TAGS[s.label] || "plain"}">${esc(s.label)}</span>
      <button class="small${s.connected ? "" : " primary"}" data-mcp-connect="${esc(s.name)}" data-busy="Starting…">${s.connected ? "Reconnect" : "Connect"}</button></div>
    <div class="mcp-meta"><code>${esc(short(s.target, 70))}</code>${s.secrets.length ? ` · reads ${esc(s.secrets.join(", "))}` : ""}
      ${s.tools.length ? ` · ${on} of ${plural(s.tools.length, "tool")} on` : ""}</div>
    ${s.label === "remote" ? '<p class="hint step-bad">Remote: every call to this server leaves this machine.</p>' : ""}
    ${s.tools.length ? `<ul class="mcp-tools">${s.tools.map((t) => mcpTool(s, t)).join("")}</ul>`
      : '<p class="hint">Connect to see its tools.</p>'}
    <div class="mcp-actions"><button type="button" class="link" data-mcp-edit="${esc(s.name)}">Edit</button>
      <button type="button" class="link danger-link" data-mcp-remove="${esc(s.name)}">Remove</button></div>
  </div>`;
}

let mcpView = null;   // the last GET /api/mcp: the servers, and the saved config the form adds to
async function refreshMcp(view) {
  try { view ||= await api("/api/mcp"); } catch (e) { toast(e.message, "error"); return; }
  mcpView = view;
  $("#mcp-servers").innerHTML = view.servers.map(serverCard).join("") || '<p class="hint">No servers yet. Add one below.</p>';
  $("#mcp-config").value = view.config;
  if (!view.servers.length) $("#mcp-add").open = true;
  const connected = view.servers.filter((s) => s.connected).length;
  setTaskState("mcp", view.servers.length ? `${connected} of ${view.servers.length} connected` : "None yet", connected ? "ok" : "warn");
  refreshTask6();
}
TASK_OPEN.mcp = refreshMcp;

// --- The Add a server form: fields and rows instead of JSON. It builds one mcpServers entry, adds it to the
// saved config and sends the whole config to POST /api/mcp/config, which checks it (names, ${NAME} secrets, URLs).
const mcpForm = { kind: "local", editing: null };
const VAR_NAME = /^[A-Za-z_][A-Za-z0-9_]*$/;
const savedServers = () => { try { return JSON.parse(mcpView?.config || "{}").mcpServers || {}; } catch { return {}; } };
const refOf = (v) => String(v ?? "").match(/^\$\{(.+)\}$/)?.[1] ?? String(v ?? "");
const ROW_KINDS = {   // each kind of row: its inputs, and the container it goes in
  arg: { box: "#mcp-args", fields: [["value", "-y"]] },
  env: { box: "#mcp-env", fields: [["key", "GITHUB_PERSONAL_ACCESS_TOKEN"], ["ref", "GITHUB_TOKEN"]] },
  header: { box: "#mcp-headers", fields: [["key", "Authorization"], ["ref", "SEARCH_AUTH"]] },
};
function addRow(kind, values = {}) {
  const k = ROW_KINDS[kind];
  const inputs = k.fields.map(([name, ph], i) => `${i ? '<span class="row-arrow" title="read from this environment variable">← $</span>' : ""}
    <input data-field="${name}" placeholder="${esc(ph)}" value="${esc(values[name] ?? "")}" spellcheck="false" autocomplete="off">`).join("");
  $(k.box).insertAdjacentHTML("beforeend", `<div class="form-row" data-row="${kind}">${inputs}
    <button type="button" class="row-x" aria-label="Remove this row" data-remove-row>×</button></div>`);
  showPreview();
}
const rowsOf = (kind) => $$(`${ROW_KINDS[kind].box} .form-row`).map((r) =>
  Object.fromEntries([...r.querySelectorAll("input")].map((i) => [i.dataset.field, i.value.trim()])));
function setMcpKind(kind) {
  mcpForm.kind = kind;
  $$("[data-mcp-kind]").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.mcpKind === kind)));
  $$("[data-mcp-panel]").forEach((p) => (p.hidden = !p.dataset.mcpPanel.split(" ").includes(kind)));
  showPreview();
}
// Build the entry from the form. Returns the entry and the problems found before sending (the server checks again).
function readMcpForm() {
  const name = $("#mcp-name").value.trim(), problems = [];
  if (!/^[A-Za-z0-9_-]{1,32}$/.test(name)) problems.push("Give the server a name: letters, digits, - and _ (at most 32).");
  const refs = (kind) => {
    const out = {};
    for (const r of rowsOf(kind).filter((x) => x.key || x.ref)) {
      const ref = r.ref || r.key;
      if (!r.key || !VAR_NAME.test(ref)) problems.push(`${kind === "env" ? "Variable" : "Header"} "${r.key || "?"}": give a name and an environment variable (letters, digits and _).`);
      else out[r.key] = `\${${ref}}`;
    }
    return out;
  };
  let entry;
  if (mcpForm.kind === "remote") {
    const url = $("#mcp-url").value.trim();
    if (!/^https?:\/\/\S+$/.test(url)) problems.push("The URL must start with http:// or https://.");
    entry = { url, ...(rowsOf("header").length ? { headers: refs("header") } : {}) };
  } else {
    const command = $("#mcp-command").value.trim(), cwd = $("#mcp-cwd").value.trim();
    if (!command) problems.push("Give the command that starts the server, for example npx.");
    const args = rowsOf("arg").map((r) => r.value).filter(Boolean), env = refs("env");
    entry = { command, ...(args.length ? { args } : {}), ...(Object.keys(env).length ? { env } : {}), ...(cwd ? { cwd } : {}) };
  }
  if (!mcpForm.editing && savedServers()[name]) problems.push(`A server called "${name}" already exists. Edit it, or pick another name.`);
  return { name, entry, problems };
}
function showPreview() {
  if (mcpForm.kind === "json") return;
  const { name, entry } = readMcpForm();
  $("#mcp-preview").textContent = JSON.stringify({ mcpServers: { [name || "name"]: entry } }, null, 2);
}
function showMcpProblems(text) {
  $("#mcp-problems").textContent = text;
  $("#mcp-problems").hidden = !text;
}
function resetMcpForm() {
  mcpForm.editing = null;
  ["#mcp-name", "#mcp-command", "#mcp-cwd", "#mcp-url"].forEach((id) => ($(id).value = ""));
  Object.values(ROW_KINDS).forEach((k) => ($(k.box).innerHTML = ""));
  $("#mcp-form-title").textContent = "+ Add a server";
  $("#mcp-add-btn").textContent = "Add server";
  $("#mcp-cancel").hidden = true;
  showMcpProblems("");
  setMcpKind("local");
}
function editMcpServer(name) {
  const entry = savedServers()[name];
  if (!entry) return;
  resetMcpForm();
  mcpForm.editing = name;
  $("#mcp-name").value = name;
  if (entry.url) {
    $("#mcp-url").value = entry.url;
    Object.entries(entry.headers || {}).forEach(([key, v]) => addRow("header", { key, ref: refOf(v) }));
  } else {
    $("#mcp-command").value = entry.command || "";
    $("#mcp-cwd").value = entry.cwd || "";
    (entry.args || []).forEach((value) => addRow("arg", { value }));
    Object.entries(entry.env || {}).forEach(([key, v]) => addRow("env", { key, ref: refOf(v) }));
  }
  setMcpKind(entry.url ? "remote" : "local");
  $("#mcp-form-title").textContent = `Edit ${name}`;
  $("#mcp-add-btn").textContent = "Save changes";
  $("#mcp-cancel").hidden = false;
  $("#mcp-add").open = true;
  $("#mcp-add").scrollIntoView({ block: "start", behavior: "smooth" });
}
// Send a whole server list. The server checks it and answers 400 with every problem; the form shows them.
async function saveMcpServers(servers, done) {
  showMcpProblems("");
  try {
    await refreshMcp(await api("/api/mcp/config", { text: JSON.stringify({ mcpServers: servers }, null, 1) }));
    toast(done);
    return true;
  } catch (e) {
    showMcpProblems(e.message);
    return false;
  }
}
$("#mcp-add").addEventListener("input", showPreview);
$("#mcp-add").addEventListener("click", (ev) => {
  const kind = ev.target.closest("[data-mcp-kind]")?.dataset.mcpKind;
  if (kind) setMcpKind(kind);
  const add = ev.target.closest("[data-add-row]")?.dataset.addRow;
  if (add) { addRow(add); $(`${ROW_KINDS[add].box} .form-row:last-child input`).focus(); }
  if (ev.target.closest("[data-remove-row]")) { ev.target.closest(".form-row").remove(); showPreview(); }
});
$("#mcp-cancel").onclick = () => { resetMcpForm(); $("#mcp-add").open = false; };
$("#mcp-add-btn").onclick = (ev) => {
  const { name, entry, problems } = readMcpForm();
  if (problems.length) { showMcpProblems(problems.join("\n")); return; }
  const servers = { ...savedServers() };
  if (mcpForm.editing && mcpForm.editing !== name) delete servers[mcpForm.editing];
  servers[name] = entry;
  const done = mcpForm.editing ? `Saved ${name}. Reconnect it to use the change.` : `Added ${name}. Press Connect to see its tools.`;
  busy(ev.target, async () => { if (await saveMcpServers(servers, done)) { resetMcpForm(); $("#mcp-add").open = false; } });
};

$("#mcp-servers").addEventListener("click", (ev) => {
  const b = ev.target.closest("[data-mcp-connect]");
  if (b) busy(b, () => api("/api/mcp/connect", { server: b.dataset.mcpConnect }).then(refreshMcp).then(refreshTools));
  const edit = ev.target.closest("[data-mcp-edit]")?.dataset.mcpEdit;
  if (edit) editMcpServer(edit);
  const remove = ev.target.closest("[data-mcp-remove]")?.dataset.mcpRemove;
  if (!remove) return;
  const servers = { ...savedServers() };
  delete servers[remove];
  if (!Object.keys(servers).length) { toast("Keep at least one server: edit this one instead.", "error"); return; }
  if (confirm(`Remove the server "${remove}"? Its tools leave the agent. You can add it again later.`)) {
    saveMcpServers(servers, `Removed ${remove}.`).then(refreshTools);
  }
});
setMcpKind("local");
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
  showMcpProblems("");
  try {
    await refreshMcp(await api("/api/mcp/config", { text: $("#mcp-config").value }));
    toast("Config saved. Connect each server to see its tools.");
  } catch (e) {
    showMcpProblems(e.message);
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
  setTaskState("task5", on ? "Watching" : "", "live");
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
const refreshT5 = () => api("/api/task5").then((s) => showT5(s.watching)).catch(() => {});
const oldTasksRefresh = TAB_REFRESH.tasks;
TAB_REFRESH.tasks = () => { oldTasksRefresh?.(); refreshT5(); };
TASK_OPEN.task5 = refreshT5;

// --- Task 6 (Agent page): what a spoken reminder needs, checked live ---------------------------
// The reminders server of this repo, connected, with add_reminder ticked; the daemon speaks the reminder when it is due.
async function refreshTask6() {
  const [view, daemon] = await Promise.all([api("/api/mcp").catch(() => null), api("/api/daemon").catch(() => null)]);
  const server = view?.servers.find((s) => s.name.includes("reminders"));
  const add = server?.tools.find((t) => t.name === "add_reminder");
  const checks = [
    [!!server?.connected, "The reminders MCP server is connected", "Connect it in MCP servers"],
    [!!add?.enabled, "The model may use add_reminder", "Tick it in MCP servers"],
    [!!add && add.tier !== "blocked", add?.tier === "confirm" ? "add_reminder is confirm: you approve each reminder" : "add_reminder is safe", "Set its tier in MCP servers"],
    [!!daemon?.status.alive, "The daemon runs, so a due reminder is spoken", "Start it in Always on"],
  ];
  $("#t6-checks").innerHTML = checks.map(([ok, text, fix], i) => `<li class="${ok ? "ok" : ""}"><span>${esc(text)}</span>${ok ? ""
    : ` <button type="button" class="link" ${i === 3 ? 'data-go="daemon"' : 'data-open-task="mcp"'}>${esc(fix)}</button>`}</li>`).join("");
  setTaskState("task6", checks.every(([ok]) => ok) ? "" : "Needs setup", "warn");
}
TASK_OPEN.task6 = refreshTask6;
refreshT5();
refreshMcp();   // the MCP button shows how many servers are connected; it also checks Task 6

// --- Session 3 rows of the scorecard ------------------------------------------------------
Object.assign(COMPONENTS, {
  tts: ["Voice out (Kokoro)", "How soon is the first sound ready, and does Whisper understand the speech?"],
  voice_stop: ["Stopping speech", "How long does the agent keep talking after you ask for silence?"],
  wake: ["Wake word", "Does \"Hey Jarvis\" wake it, and does other talk leave it asleep?"],
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
