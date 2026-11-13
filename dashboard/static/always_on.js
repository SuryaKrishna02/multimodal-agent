// Session 2 panels: Always on (daemon, feed, checkpoints), Evolve (versions, diffs, rollback), Tasks 3 and 4,
// and the consolidation card of the Memory tab. Uses the helpers of app.js ($, api, busy, esc, md, toast...).

// --- Plain words for the Session 2 events ---------------------------------------------
Object.assign(DESCRIBE, {
  "daemon.started": (e) => `Daemon started (pid ${e.pid})${e.recovered?.length ? `, recovered ${plural(e.recovered.length, "job")}` : ""}`,
  "daemon.stopped": (e) => `Daemon stopped (pid ${e.pid})`,
  "daemon.killed": (e) => `Daemon killed (pid ${e.pid}): this is the crash test`,
  "daemon.action": (e) => `Daemon ${e.status === "done" ? "finished" : "failed"} ${e.job} job #${e.job_id}: ${e.reason}`,
  "daemon.missed": (e) => `Schedule '${e.schedule}' ran ${Math.round(e.late_s / 60)} min late${e.skipped ? `, ${e.skipped} slots skipped` : ""}`,
  "daemon.warning": (e) => e.message,
  "daemon.listen": (e) => `Always-on listening switched ${e.want ? "on" : "off"}`,
  "checkpoint.queued": (e) => `Queued ${e.job} job #${e.job_id}: ${e.reason}`,
  "checkpoint.intent": (e) => `Job #${e.job_id}, step ${e.step}: intent recorded`,
  "checkpoint.execution": (e) => `Job #${e.job_id}, step ${e.step}: running`,
  "checkpoint.result": (e) => `Job #${e.job_id}, step ${e.step}: result recorded`,
  "checkpoint.skipped": (e) => `Job #${e.job_id}, step ${e.step}: already done before the crash, skipped`,
  "checkpoint.recovered": (e) => `Recovered job #${e.job_id} after a crash (${plural(e.steps_done.length, "step")} already done)`,
  "checkpoint.finished": (e) => `Job #${e.job_id} ${e.status}`,
  "senses.stt": (e) => `${e.engine === "whisper" ? "Whisper" : "Gemma"} transcribed ${e.audio_s} s in ${e.took_s} s: ${e.text}`,
  "senses.ambient": (e) => `Heard: ${e.text}`,
  "senses.ambient.state": (e) => `Always-on listening ${e.state}${e.device ? ` (${e.device})` : ""}`,
  "memory.consolidated": (e) => `Summarized ${e.rows} raw rows into ${plural(e.claims.length, "claim")} and ${plural(e.fact_ids.length, "fact")}`,
  "memory.superseded": (e) => `Fact #${e.new} replaces ${e.old.map((i) => "#" + i).join(", ")}`,
  "memory.forgotten": (e) => `Retention: deleted ${plural(e.deleted, "raw row")} that a summary covers`,
  "evolve.version": (e) => `New prompt version v${e.version} (from v${e.parent}): ${e.reason}`,
  "evolve.rollback": (e) => `Rolled back to prompt version v${e.version}`,
  "evolve.rejected": (e) => `Refused a prompt patch: ${e.reason}`,
  "evolve.feedback": (e) => `Feedback (${e.rating > 0 ? "good" : "bad"}, ${e.source}): ${e.text}`,
  "evolve.brief": (e) => `Wrote a brief with prompt v${e.version}`,
  "task3.answer": (e) => `Task 3 answered: ${e.answer}`,
});
Object.assign(COMPONENTS, {
  daemon: ["Daemon", "Does a killed daemon resume where it stopped, and does it stay small over hours?"],
  multimodal: ["Multimodal memory", "Find a screenshot or a clip by text: text vectors, media vectors or both"],
  consolidation: ["Memory consolidation", "Are summaries faithful to the raw log, and can you still find things after forgetting?"],
  task3: ["Task 3: ambient listener with recall", "Hear a conversation, then answer questions about it hours later"],
  task4: ["Task 4: self-improving daily brief", "Does the brief learn your preferences, on days it never got feedback on?"],
});
Object.assign(METRICS, {
  whisper_wer: ["Word error rate, Whisper", "Same clips, Whisper large-v3-turbo", "pct"],
  gemma_s: ["Seconds per clip, Gemma", "Median time to transcribe one clip, after a warm-up call", "s"],
  whisper_s: ["Seconds per clip, Whisper", "Median time to transcribe one clip, after a warm-up call", "s"],
  gemma_rtf: ["Real-time factor, Gemma", "Seconds of work per second of audio: under 1 keeps up with speech", "×"],
  whisper_rtf: ["Real-time factor, Whisper", "Seconds of work per second of audio: under 1 keeps up with speech", "×"],
  chance_recall: ["Chance: top 3", "What a random ranking scores (3 / items): compare every top-3 number with it", "pct"],
  chance_top1: ["Chance: first", "What a random ranking scores (1 / items)", "pct"],
  items_rejected: ["Items not stored", "Refused by the junk filter: their queries cannot succeed", ""],
  lost_hearing: ["Wrong: lost in hearing", "The transcripts lack the answer (speech to text or the segmenter)", ""],
  lost_retrieval: ["Wrong: lost in retrieval", "Heard, but not in the memories the model read (search or consolidation)", ""],
  lost_answer: ["Wrong: lost in the answer", "The model read it and still answered wrong, or invented an answer", ""],
  recovery_s: ["Recovery time", "Restart until the next unfinished step starts", "s"],
  redone_steps: ["Finished steps run again", "Must be 0: a finished step is skipped", ""],
  lost_jobs: ["Lost jobs", "Jobs that never finished after the restart", ""],
  soak_hours: ["Soak length", "Hours of the newest soak run", "h"],
  soak_awake_hours: ["Soak length, awake", "Hours the machine was awake: a sleeping machine runs nothing", "h"],
  rss_growth_mb_per_h: ["Memory growth", "Daemon memory over the soak, straight-line fit", "MB/h"],
  missed_events: ["Missed events", "Schedule slots or jobs that went missing in the soak", ""],
  faithfulness: ["Faithful claims", "Claims the judge finds supported by the rows they cite", "pct"],
  invalid_citation_rate: ["Invalid citations", "Cited ids that were not in the window (dropped)", "pct"],
  compression_ratio: ["Compression", "Raw characters per summary character", "×"],
  facts_per_raw_row: ["Facts per raw row", "Facts are an index next to the summary", ""],
  recall_before: ["Found before consolidation", "Relevant raw row in the top 3", "pct"],
  recall_after: ["Found after forgetting", "Relevant row, or a summary or fact that cites it, in the top 3", "pct"],
  accuracy: ["Correct answers", "Every required phrase, or 'I do not know' when nothing was said", "pct"],
  abstain_rate: ["Says 'I do not know'", "On questions about things nobody said", "pct"],
  cited_rate: ["Answers cite memory", "Answered questions that cite at least one memory", "pct"],
  dev_before: ["Dev days, before", "Preference checks passed with the seed prompt", "pct"],
  dev_after: ["Dev days, after", "After the last feedback round", "pct"],
  test_before: ["Held-out days, before", "Days that never got feedback, seed prompt", "pct"],
  test_after: ["Held-out days, after", "Same days, after the last round", "pct"],
  test_gain: ["Held-out gain", "After minus before, on days the reflection never saw", "pct"],
  regression_rate: ["Regressions", "Held-out checks that passed before and fail after", "pct"],
  locked_edits: ["Edits to locked blocks", "Versions whose locked blocks changed: must be 0", ""],
  patches_rejected: ["Patches refused", "Patches the validator would not apply", ""],
  overfit_dev_gain: ["Overfitting: rule followed on its day", "1 = the one-day rule was learned", ""],
  overfit_test_leak: ["Overfitting: leaked into other days", "Held-out briefs that mention the one-day details", "pct"],
  overfit_test_change: ["Overfitting: held-out change", "Held-out compliance after the overfit rounds, minus before", "pct"],
});
for (const m of ["image", "audio"]) for (const [mode, label] of [["text", "text vectors"], ["media", `${m} vectors`], ["both", "best of both"]]) {
  METRICS[`${m}_recall_${mode}`] = [`${titleCase(m)}: top 3, ${label}`, `Queries whose ${m} is in the top 3`, "pct"];
  METRICS[`${m}_top1_${mode}`] = [`${titleCase(m)}: first, ${label}`, `Queries whose ${m} comes first`, "pct"];
}

// --- Always on ----------------------------------------------------------------------------
let daemonAlive = false;
const ago = (ts) => (ts ? `${Math.max(0, Math.round(Date.now() / 1000 - ts))} s ago` : "never");
const clockOf = (ts) => (ts ? new Date(ts * 1000).toLocaleString([], { weekday: "short", hour: "numeric", minute: "2-digit" }) : "not yet");
async function refreshDaemon() {
  let d;
  try { d = await api("/api/daemon"); } catch { return; }
  const s = d.status;
  daemonAlive = s.alive;
  $("#daemon-dot").className = "dot" + (s.alive ? " on" : " failed");
  $("#daemon-state").textContent = s.alive ? `running, pid ${s.pid}` : "not running";
  $("#daemon-start").hidden = s.alive;
  $("#daemon-stop").hidden = $("#daemon-kill").hidden = !s.alive;
  $("#daemon-tiles").innerHTML = tiles([
    [s.alive ? "Running" : "Stopped", "", s.alive ? `Heartbeat ${s.age_s} s ago` : "Start it to run jobs on its own"],
    [s.job ? `#${s.job.id}` : "idle", "", s.job ? `Running a ${s.job.kind} job` : "No job running"],
    [s.queued ?? 0, "", "Jobs waiting"],
    [s.rss_mb ?? "–", "MB", "Daemon memory"],
  ]);
  const sw = $("#listen-switch");
  const listening = s.alive && s.listening;
  sw.className = "mic " + (listening ? "live" : "paused");
  sw.textContent = listening ? "Always-on listening on" : s.want_listen ? "Listening on (starts with the daemon)" : "Always-on listening off";
  sw.setAttribute("aria-checked", String(!!s.want_listen));
  $("#daemon-jobs").innerHTML = d.jobs.map(jobRow).join("") || '<div class="none">No jobs yet.</div>';
  $("#daemon-schedule").innerHTML = `<div class="table-wrap"><table><thead><tr><th>Name</th><th>Job</th><th>When</th><th>Last slot</th></tr></thead><tbody>${
    d.schedule.map((r) => `<tr><td>${esc(r.name)}</td><td><code>${esc(r.job)}</code></td><td>${esc(r.when.replace("every ", "every ").replace(/every (\d+)/, (_, n) => `every ${n / 60} min`))}</td><td class="muted">${esc(clockOf(r.last))}</td></tr>`).join("")}</tbody></table></div>`;
  renderFeed();
}
const STATUS_TONE = { done: "ok", failed: "bad", running: "warn", queued: "plain" };
const jobRow = (j) => `<details class="job" data-job="${j.id}"><summary><b>#${j.id}</b><code>${esc(j.kind)}</code>
  <span class="tag ${STATUS_TONE[j.status] || "plain"}">${esc(j.status)}</span>${j.attempts > 1 ? `<span class="tag warn">attempt ${j.attempts}</span>` : ""}
  <span class="muted">${esc(j.reason)}</span><span class="muted">${esc(clockOf(j.created))}</span></summary><div class="job-steps"></div></details>`;
$("#daemon-jobs").addEventListener("toggle", async (e) => {
  const box = e.target.querySelector?.(".job-steps");
  if (!e.target.open || !box) return;
  const j = await api(`/api/daemon/job/${e.target.dataset.job}`);
  const names = [...new Set(j.steps.map((s) => s.name))];
  const has = (n, p) => j.steps.filter((s) => s.name === n && s.phase === p).length;
  box.innerHTML = names.length ? `<div class="phases"><b>Step</b><b>Intent</b><b>Execution</b><b>Result</b><b>Output</b>${names.map((n) => {
    const res = j.steps.filter((s) => s.name === n && s.phase === "result").pop();
    const cell = (p) => { const k = has(n, p); return `<span class="ph ${k ? (res ? "done" : "open") : ""}">${k ? "✓" + (k > 1 ? ` ×${k}` : "") : "–"}</span>`; };
    return `<span class="ph">${esc(n)}</span>${cell("intent")}${cell("execution")}${cell("result")}<span class="muted">${esc(res ? short(JSON.stringify(res.data), 120) : "not finished")}</span>`;
  }).join("")}</div><p class="hint">×2 means the step started twice: it was running when the process died.</p>` : '<p class="hint">No steps recorded yet.</p>';
}, true);
const FEED_KINDS = new Set(["daemon.action", "daemon.missed", "checkpoint.recovered", "daemon.started", "daemon.killed", "memory.forgotten"]);
function renderFeed() {
  // Only what the daemon did, or a job you started here: eval runs write to the same log from the CLI.
  const rows = allEvents.filter((e) => FEED_KINDS.has(e.kind) && (e.source === "daemon" || e.source === "dashboard")).slice(-30).reverse();
  $("#daemon-feed").innerHTML = rows.map((e) => `<div class="feed-row"><time>${clock(e.ts)}</time><div><div>${esc(describe(e))}</div>
    ${e.summary ? `<div class="why">${esc(short(JSON.stringify(e.summary), 200))}</div>` : ""}</div>
    <span class="tag ${e.status === "failed" || e.kind === "daemon.killed" ? "bad" : e.kind === "daemon.missed" ? "warn" : "plain"}">${esc(e.source || "")}</span></div>`).join("")
    || '<div class="none">Nothing yet. Start the daemon, or press Summarize now in the Memory tab.</div>';
}
$("#daemon-start").onclick = (e) => busy(e.target, () => api("/api/daemon/start", {})).then(refreshDaemon);
$("#daemon-stop").onclick = (e) => busy(e.target, () => api("/api/daemon/stop", {})).then(() => setTimeout(refreshDaemon, 1500));
$("#daemon-kill").onclick = (e) => {
  if (!confirm("Kill the daemon with SIGKILL, as a crash would? A running job stops in the middle; start the daemon again to see it resume.")) return;
  busy(e.target, () => api("/api/daemon/stop", { kill: true })).then(() => setTimeout(refreshDaemon, 500));
};
$("#listen-switch").onclick = (e) => {
  const on = e.target.getAttribute("aria-checked") !== "true";
  busy(e.target, () => api("/api/daemon/listen", { on })).then(refreshDaemon);
};
TAB_REFRESH.daemon = refreshDaemon;

// --- Memory: consolidation card -----------------------------------------------------------------
async function refreshLevels() {
  let st;
  try { st = await api("/api/memory/stats"); } catch { return; }
  const L = st.levels;
  $("#mem-levels").innerHTML = tiles([
    [L.raw.rows, "raw", `${num(L.raw.chars)} characters · ${st.waiting} wait for a summary`],
    [L.summary.rows, "summaries", `${num(L.summary.chars)} characters`],
    [L.fact.rows, "facts", `${num(L.fact.chars)} characters`],
    [st.retention_days, "days", "Raw rows that a summary covers are deleted after this"],
  ]);
}
const runJob = (kind, label) => async (e) => {
  const out = await busy(e.target, () => api("/api/jobs", { kind }));
  if (!out) return;
  toast(out.queued ? `Queued for the daemon (job #${out.job_id}).` : `${label} done (job #${out.job_id}).`);
  refreshMemory();
  refreshLevels();
};
$("#consolidate-btn").onclick = runJob("consolidate", "Summarizing");
$("#forget-btn").onclick = runJob("forget", "Retention");
$("#mem-level").onchange = refreshMemory;
TAB_REFRESH.memory = refreshLevels;   // app.js reloads the list itself

// --- Evolve ---------------------------------------------------------------------------------
function promptBlocks(prompt, locked) {
  return prompt.split(/\n\n(?=### \[)/).map((block) => {
    const m = block.match(/^### \[(locked|editable):(\w+)\]\n([\s\S]*)$/);
    if (!m) return `<pre>${esc(block)}</pre>`;
    const isLocked = m[1] === "locked";
    return `<div class="prompt-block ${m[1]}"><span class="tag ${isLocked ? "plain" : "safe"}">${isLocked ? "🔒 locked" : "editable"}</span> <b>${esc(m[2])}</b><pre>${esc(m[3])}</pre></div>`;
  }).join("");
}
function diffHtml(text) {
  if (!text) return '<p class="hint">No change.</p>';
  return `<pre class="diff">${text.split("\n").map((l) => {
    const cls = l.startsWith("+") && !l.startsWith("+++") ? "add" : l.startsWith("-") && !l.startsWith("---") ? "del" : l.startsWith("@@") ? "hunk" : "";
    return `<span class="${cls}">${esc(l) || " "}</span>`;
  }).join("")}</pre>`;
}
async function showDiff(a, b) {
  const d = await api(`/api/evolve/diff?a=${a}&b=${b}`);
  $("#evolve-diff-card").hidden = false;
  $("#evolve-diff-label").textContent = `v${a} → v${b}`;
  $("#evolve-diff").innerHTML = diffHtml(d.diff);
}
async function refreshEvolve() {
  let s;
  try { s = await api("/api/evolve"); } catch { return; }
  $("#evolve-version").textContent = `v${s.active.id} · ${s.active.source}`;
  $("#evolve-locked").innerHTML = s.locked_broken.length
    ? `<span class="tag bad">locked blocks changed: ${esc(s.locked_broken.join(", "))}</span>`
    : `<span class="tag ok">🔒 ${s.locked.length} locked blocks untouched</span>`;
  $("#evolve-prompt").innerHTML = promptBlocks(s.prompt);
  $("#evolve-pending").textContent = s.pending ? `${plural(s.pending, "feedback row")} not used by a reflection yet.` : "All feedback has been used.";
  $("#evolve-versions").innerHTML = [...s.versions].reverse().map((v) => `<div class="version-row${v.active ? " active" : ""}">
    <b>v${v.id}${v.active ? " ●" : ""}</b><div><div>${esc(v.reason || "(no reason)")}</div><div class="muted small">${esc(v.source)} · from v${v.parent ?? "–"} · ${plural(v.preferences.length, "preference")} · ${esc(clockOf(v.ts))}</div></div>
    <span>${v.parent ? `<button class="small" data-diff="${v.parent},${v.id}">Diff</button>` : ""}${v.active ? "" : ` <button class="small" data-rollback="${v.id}">Make active</button>`}</span></div>`).join("");
  $("#evolve-feedback").innerHTML = s.feedback.map((f) => `<div class="feed-row"><time>${clock(f.ts)}</time><div>${f.rating > 0 ? "👍" : "👎"} ${esc(f.text)}
    <div class="why">${f.brief_id ? `brief #${f.brief_id} · ` : ""}${f.used_in ? `used in v${f.used_in}` : "not used yet"}</div></div><span class="tag ${f.source === "user" ? "safe" : "plain"}">${esc(f.source)}</span></div>`).join("")
    || '<div class="none">No feedback yet. Write a brief in the Tasks tab and tell it what to change.</div>';
}
$("#evolve-versions").onclick = async (e) => {
  const d = e.target.dataset;
  if (d.diff) showDiff(...d.diff.split(","));
  if (d.rollback) { await busy(e.target, () => api("/api/evolve/rollback", { version: Number(d.rollback) })); toast(`v${d.rollback} is active.`); refreshEvolve(); }
};
async function reflectNow(button, out) {
  const r = await busy(button, () => api("/api/evolve/reflect", {}), out ? { out, text: "Reading your feedback and patching the prompt…" } : {});
  if (!r) return;
  const html = r.version ? `<div class="callout ok"><h4>New version v${r.version}</h4><p>${esc(r.patch.reason)}</p></div>${diffHtml(r.diff)}`
    : `<div class="callout warn"><h4>No new version</h4><p>${esc(r.rejected)}</p></div>`;
  if (out) $(out).innerHTML = html;
  refreshEvolve();
  if (r.version) showDiff(evolveParent(r), r.version);
}
const evolveParent = (r) => Number((r.diff.match(/^--- v(\d+)/m) || [])[1] || r.version - 1);
$("#reflect-btn").onclick = (e) => reflectNow(e.target);
TAB_REFRESH.evolve = refreshEvolve;

// --- Task 3 -----------------------------------------------------------------------------------
$("#replay-btn").onclick = async (e) => {
  $("#task3-heard").innerHTML = "";
  const out = await busy(e.target, () => api("/api/task3/replay", { speed: Number($("#replay-speed").value) }));
  if (out) toast(out.queued ? `The daemon is replaying it (job #${out.job_id}). Transcripts appear below as they are heard.` : "Replay done. Now ask about it.");
};
renderers["senses.ambient"] = (e) => {
  if (!e.text || replaying) return;   // only what is heard from now on, not old runs (evals write here too)
  $("#task3-heard").insertAdjacentHTML("beforeend", `<div><span class="muted">${clock(e.ts)}</span> ${esc(e.text)}${e.memory_id ? "" : ' <span class="tag warn">refused by the junk filter</span>'}</div>`);
};
$("#ask-form").onsubmit = async (e) => {
  e.preventDefault();
  const question = $("#ask-q").value.trim() || $("#ask-q").placeholder;
  const r = await busy(e.submitter ?? $("#ask-form button"), () => api("/api/task3/ask", { question }), { out: "#task3-out", text: "Searching memory and answering…" });
  if (!r) return;
  const cites = r.memories.map((m) => `<div class="${r.memory_ids.includes(m.id) ? "hit" : ""}">${r.memory_ids.includes(m.id) ? "✓" : "·"} #${m.id} ${esc(short(m.text, 140))}</div>`).join("");
  $("#task3-out").innerHTML = `<div class="result-meta">${esc(question)}${r.window ? ` · searched "${esc(r.window)}" only` : ""}</div>
    <div class="callout info"><p>${esc(r.answer)}</p></div><div class="cites"><b>Memories it read</b> (✓ = cited)${cites || '<div class="none">None.</div>'}</div>`;
};

// --- Task 4 -----------------------------------------------------------------------------------
let lastBrief = null;
$("#brief-btn").onclick = async (e) => {
  const r = await busy(e.target, () => api("/api/task4/brief", {}), { out: "#task4-out", text: "Writing today's brief…" });
  if (!r) return;
  lastBrief = r;
  $("#task4-out").innerHTML = `<div class="result-meta">Brief #${r.brief_id} · prompt v${r.version} · from ${plural(r.memories.length, "memory", "memories")}</div>${md(r.markdown) || '<p class="muted">Empty brief.</p>'}`;
  $("#feedback-form").hidden = false;
};
$("#feedback-form").onclick = async (e) => {
  const rating = Number(e.target.dataset.rating || 0);
  if (!rating || !lastBrief) return;
  const text = $("#feedback-text").value.trim();
  if (rating < 0 && !text) { toast("Say what to change first.", "error"); return; }
  const r = await busy(e.target, () => api("/api/task4/feedback", { brief_id: lastBrief.brief_id, rating, text: text || "Good brief." }));
  if (!r) return;
  $("#feedback-text").value = "";
  $("#task4-reflect").innerHTML = `<p class="hint">Feedback saved (${plural(r.pending, "row")} waiting). <button class="small primary" id="reflect-inline" data-busy="Reflecting…">Reflect now</button></p>`;
  $("#reflect-inline").onclick = (ev) => reflectNow(ev.target, "#task4-reflect");
};

// --- Polling ------------------------------------------------------------------------------------
const SESSION2 = /^(daemon|checkpoint|evolve|memory\.(consolidated|forgotten|superseded))/;
setInterval(() => {
  if (isTab("daemon")) refreshDaemon();
  else api("/api/daemon").then((d) => { $("#daemon-dot").className = "dot" + (d.status.alive ? " on" : " failed"); }).catch(() => {});
}, 3000);
setInterval(() => {
  const recent = allEvents.slice(-20);
  if (isTab("evolve") && recent.some((e) => /^evolve/.test(e.kind) && Date.now() / 1000 - e.ts < 4)) refreshEvolve();
  if (isTab("memory") && recent.some((e) => SESSION2.test(e.kind) && Date.now() / 1000 - e.ts < 4)) refreshLevels();
}, 2000);
TAB_REFRESH[location.hash.slice(1)]?.();   // app.js showed the first tab before this file loaded
