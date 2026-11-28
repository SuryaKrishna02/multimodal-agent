// Session 3: the Skills pane below the tasks. Each skill folder, whether it loaded (or why not), its tier and effects,
// and its examples, which go into the message box. Uses the helpers of app.js ($, api, busy, esc, toast...).

Object.assign(DESCRIBE, {
  "skills.error": (e) => `Skill ${e.skill} was not loaded: ${(e.problems || []).join("; ")}`,
  "skills.reloaded": (e) => `Skills reloaded: ${e.loaded.join(", ") || "none"}${e.refused.length ? `; refused ${e.refused.join(", ")}` : ""}`,
});
TAB_OF.skills = "tools";

const SKILL_STATE = { loaded: ["ok", "loaded"], refused: ["bad", "not loaded"], new: ["warn", "new: press Reload"] };
function skillCard(s) {
  const [tone, label] = SKILL_STATE[s.state] || ["plain", s.state];
  const effects = s.effects.length ? s.effects.map((e) => `<span class="tag plain">${esc(e.replaceAll("_", " "))}</span>`).join("") : '<span class="muted small">no side effects</span>';
  const tries = s.loaded ? s.examples.map((x) => `<button type="button" class="chip-btn" data-suggest="${esc(x)}">${esc(x)}</button>`).join("") : "";
  const problems = s.problems.length ? `<ul class="skill-problems">${s.problems.map((p) => `<li>${esc(p)}</li>`).join("")}</ul>` : "";
  return `<div class="skill">
    <div class="mcp-head"><b>${esc(s.name)}</b>${s.version ? `<span class="muted small">v${esc(s.version)}</span>` : ""}
      ${s.tier ? `<span class="tag ${esc(s.tier)}">${esc(s.tier)}</span>` : ""}<span class="tag ${tone} push">${esc(label)}</span></div>
    <p class="skill-desc">${esc(s.description)}</p>
    <div class="skill-effects">${effects}</div>${problems}
    ${tries ? `<div class="skill-try"><span class="muted small">Try:</span>${tries}</div>` : ""}</div>`;
}

function showSkills(view) {
  $("#skills-list").innerHTML = view.skills.map(skillCard).join("") || '<p class="hint">No skills yet. Open "Add a skill" below.</p>';
  const loaded = view.skills.filter((s) => s.loaded).length;
  const refused = view.skills.filter((s) => s.state === "refused").length;
  setTaskState("skills", refused ? `${refused} not loaded` : `${loaded} loaded`, refused ? "bad" : "ok");
}
const refreshSkills = () => api("/api/skills").then(showSkills).catch((e) => toast(e.message, "error"));
$("#skills-reload").onclick = (e) => busy(e.target, async () => {
  const view = await api("/api/skills/reload", {});
  showSkills(view);
  refreshTools();   // a new skill is a new tool in the Tools list
  toast(`Skills reloaded: ${plural(view.skills.filter((s) => s.loaded).length, "skill")} loaded.`);
});
TASK_OPEN.skills = refreshSkills;
refreshSkills();

// --- Teach a new skill: /skill <description> in the chat, or the box in the Skills pane ------------------------
// The local model drafts it (POST /api/skills/draft). The draft shows in the conversation with its code: you can
// edit the name, the examples and the code, and you pick the tier. Nothing is written until you press Create
// (POST /api/skills/create), and the server checks everything again then.
Object.assign(DESCRIBE, {
  "skills.drafted": (e) => `Drafted a skill: ${e.name}${e.effects.length ? ` (${e.effects.join(", ")})` : ""}`,
  "skills.created": (e) => `Created the skill ${e.name} at tier ${e.tier}`,
});
const TIER_NOTES_SKILL = { safe: "runs on its own", confirm: "asks you before each call", blocked: "never runs" };
const RISKY = new Set(["writes", "runs_code", "network"]);
function draftCard(d) {
  const risky = d.effects.some((e) => RISKY.has(e));
  const tiers = ["safe", "confirm", "blocked"].map((t) => `<option value="${t}"${t === d.tier ? " selected" : ""}${t === "safe" && risky ? " disabled" : ""}>
    ${t}: ${TIER_NOTES_SKILL[t]}</option>`).join("");
  const effects = d.effects.length ? d.effects.map((e) => `<span class="tag ${RISKY.has(e) ? "warn" : "plain"}">${esc(e.replaceAll("_", " "))}</span>`).join("")
    : '<span class="tag ok">no side effects found</span>';
  return `<div class="skill-draft" data-params="${esc(JSON.stringify(d.params || []))}">
    <div class="brief-tag">Skill draft · nothing is written until you press Create</div>
    <label class="form-field">Name<input data-d="name" value="${esc(d.name)}" spellcheck="false"></label>
    <label class="form-field">What it does<input data-d="description" value="${esc(d.description)}"></label>
    <div class="form-field"><span>What the code does</span><div class="skill-effects">${effects}</div>
      <small>Found by reading the code's imports and calls. A skill that writes, runs programs or uses the network cannot be safe.</small></div>
    <label class="form-field">Tier<select data-d="tier">${tiers}</select></label>
    <label class="form-field">Examples <em>(one per line: tool choice reads them)</em><textarea data-d="examples" rows="3">${esc(d.examples.join("\n"))}</textarea></label>
    <label class="form-field">skill.py <em>(read it before you create it)</em><textarea data-d="code" class="code-edit" rows="12" spellcheck="false">${esc(d.code)}</textarea></label>
    <ul class="skill-problems"${d.problems.length ? "" : " hidden"}>${d.problems.map((p) => `<li>${esc(p)}</li>`).join("")}</ul>
    <div class="row"><button type="button" class="primary" data-create-skill data-busy="Creating…">Create skill</button>
      <button type="button" data-discard-skill>Discard</button></div></div>`;
}
async function draftSkill(description) {
  if (!description) { toast("Describe the skill after /skill, for example: /skill convert dollars to euros at a rate I give", "error"); return; }
  if (appState && !appState.llm) { toast("The language model is not ready yet.", "error"); return; }
  addMsg("user", userHtml(`/skill ${description}`));
  const reply = addMsg("agent", "");
  working(reply, "Drafting the skill…");
  reply.querySelector(".spinner").replaceWith(thinkingOrb());
  chatPending = reply;
  syncChatClear();
  renderOrb();
  try {
    reply.innerHTML = draftCard(await api("/api/skills/draft", { description }));
  } catch (e) {
    reply.className = "msg agent error";
    reply.textContent = `No draft: ${e.message}`;
  } finally {
    chatPending = null;
    syncChatClear();
    renderOrb();
    $("#chat-log").scrollTop = $("#chat-log").scrollHeight;
  }
}
SLASH.skill = { hint: "Teach Jarvis a new skill", usage: "/skill convert dollars to euros at a rate I give", run: draftSkill };

function readDraft(card) {
  const val = (k) => card.querySelector(`[data-d="${k}"]`).value;
  return { name: val("name").trim(), description: val("description").trim(), tier: val("tier"), code: val("code"),
    examples: val("examples").split("\n").map((x) => x.trim()).filter(Boolean), params: JSON.parse(card.dataset.params), effects: [] };
}
document.addEventListener("click", async (e) => {
  const card = e.target.closest(".skill-draft");
  if (!card) return;
  if (e.target.closest("[data-discard-skill]")) { card.innerHTML = '<p class="msg-meta">Draft discarded. Nothing was written.</p>'; return; }
  const b = e.target.closest("[data-create-skill]");
  if (!b) return;
  const problems = card.querySelector(".skill-problems");
  const view = await busy(b, async () => {
    try { return await api("/api/skills/create", readDraft(card)); } catch (err) {
      problems.innerHTML = err.message.split("\n").map((p) => `<li>${esc(p)}</li>`).join("");
      problems.hidden = false;
      return null;
    }
  });
  if (!view) return;
  const tries = view.examples.map((x) => `<button type="button" class="chip-btn" data-suggest="${esc(x)}">${esc(x)}</button>`).join("");
  card.innerHTML = `<div class="brief-tag">Skill created</div><p><b>${esc(view.name)}</b> is ready at tier <span class="tag ${esc(view.tier)}">${esc(view.tier)}</span>.
    It lives in <code>agent/skills/${esc(view.folder)}/</code>.</p><div class="skill-try"><span class="muted small">Try:</span>${tries}</div>`;
  toast(`Created the skill ${view.name}.`);
  refreshSkills();
  refreshTools();
});
$("#skill-describe").onsubmit = (e) => {
  e.preventDefault();
  const text = $("#skill-desc").value.trim();
  if (!text) { $("#skill-desc").focus(); return; }
  $("#skill-desc").value = "";
  draftSkill(text);
};
