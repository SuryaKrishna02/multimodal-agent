// Session 3: the agent's voice plays in this tab, because Google Meet shares only the audio of the presented tab.
// /api/voice/stream pushes every voice.* event. Each chunk is fetched, decoded and queued with Web Audio, so the
// sentences play back to back. Uses the helpers of app.js ($, api, busy, esc, short, clock, toast...).

Object.assign(DESCRIBE, {
  "voice.say": (e) => `Speaking${e.reason ? ` (${e.reason})` : ""}: ${short(e.text, 120)}`,
  "voice.chunk": (e) => `Wrote sentence ${e.n + 1} in ${e.synth_s} s (${e.audio_s} s of speech)`,
  "voice.played": (e) => `First sound ${e.since_say_s.toFixed(2)} s after the text, ${e.route === "tab" ? "in the dashboard tab" : "on this machine's speakers"}`,
  "voice.stop": (e) => `Stop speaking (${e.reason})`,
  "voice.stopped": (e) => `Silent ${Math.round(e.after_stop_s * 1000)} ms after the stop`,
  "voice.error": (e) => e.error,
  "voice.wake": (e) => `Wake word ${e.state}`,
  "voice.listening": (e) => `Listening for a request (${e.trigger})`,
  "voice.recorded": (e) => `Request recorded (${e.audio_s} s): thinking`,
  "voice.turn": (e) => (e.text ? `Heard "${short(e.text, 80)}" (speech to text ${e.stt_s} s, reply ${e.reply_s} s)${e.interrupted ? ", interrupted: not spoken" : ""}` : `No request heard after the ${e.trigger}`),
});
TAB_OF.voice = "voice";

const voice = { ctx: null, source: null, utts: {}, tail: 0, fadeS: 0.03, budget: {} };

// Only one tab plays the voice: two tabs would play every reply twice, and twice as much echo reaches the mic.
const voiceChannel = "BroadcastChannel" in window ? new BroadcastChannel("agent-voice") : null;
voiceChannel?.addEventListener("message", (m) => {
  if (m.data === "claim" && voice.source) {
    setVoice(false);
    toast("The voice now plays in another tab.");
  }
});

function setVoice(on) {
  const b = $("#voice-switch");
  if (on) {
    voiceChannel?.postMessage("claim");
    voice.ctx ||= new AudioContext();   // made inside the click: Chrome lets a page play sound only after a gesture
    voice.ctx.resume();
    voice.source = new EventSource("/api/voice/stream");
    voice.source.onmessage = (m) => onVoiceEvent(JSON.parse(m.data));
    voice.source.onopen = () => { b.textContent = "Voice on"; };
    voice.source.onerror = () => { b.textContent = "Voice reconnecting…"; };   // EventSource tries again by itself
  } else {
    voice.source?.close();
    voice.source = null;
    silence(null);
  }
  b.classList.toggle("paused", !on);
  b.setAttribute("aria-checked", String(on));
  b.textContent = on ? "Voice on" : "Voice off";
  $("#voice-route").textContent = on
    ? "Voice is on: the agent speaks in this tab. In Google Meet, present this tab and turn on \"Also share tab audio\"."
    : "Turn on \"Voice\" at the top to play the sound in this tab. Without it, the agent speaks through this machine's speakers.";
}
$("#voice-switch").onclick = () => setVoice(!voice.source);

function onVoiceEvent(e) {
  noteVoiceState(e);
  if (e.kind === "voice.say" && e.route === "tab") voice.utts[e.utt] = { sayTs: e.ts, chain: Promise.resolve(), nodes: [], gain: null, end: 0, stopped: false, played: false };
  if (e.kind === "voice.chunk" && e.route === "tab") queueChunk(e);
  if (e.kind === "voice.stop") silence(e);
}

// Fetch and decode chunks in order (one promise chain per utterance), then start each one where the last one ends.
function queueChunk(e) {
  const u = voice.utts[e.utt];
  if (!u || u.stopped) return;
  u.chain = u.chain.then(async () => {
    const r = await fetch(`/api/voice/audio/${e.utt}/${e.n}`);
    if (!r.ok) throw new Error(`chunk ${e.n}: ${r.status}`);
    const buffer = await voice.ctx.decodeAudioData(await r.arrayBuffer());
    if (u.stopped) return;
    if (!u.gain) { u.gain = voice.ctx.createGain(); u.gain.connect(voice.ctx.destination); }
    const node = voice.ctx.createBufferSource();
    node.buffer = buffer;
    node.connect(u.gain);
    const at = Math.max(voice.ctx.currentTime + 0.02, u.end, voice.tail);   // a new utterance waits for the one before
    node.start(at);
    u.end = voice.tail = at + buffer.duration;
    u.nodes.push(node);
    vstate.phase = null;   // thinking is over: this tab speaks now
    node.onended = () => { u.nodes = u.nodes.filter((n) => n !== node); showSpeaking(); };
    if (!u.played) {
      u.played = true;
      setTimeout(() => api("/api/voice/played", { utt: e.utt, say_ts: u.sayTs }).catch(() => {}), Math.max(0, (at - voice.ctx.currentTime) * 1000));
    }
    showSpeaking();
  }).catch((err) => toast("Could not play the voice: " + err.message, "error"));
}

// Fade out everything that plays, drop what waits, and report how long the silence took after the stop event.
function silence(stopEvent) {
  const now = voice.ctx?.currentTime ?? 0;
  for (const [utt, u] of Object.entries(voice.utts)) {
    if (u.stopped) continue;
    u.stopped = true;
    if (!u.nodes.length) continue;
    u.gain.gain.setValueAtTime(u.gain.gain.value, now);
    u.gain.gain.linearRampToValueAtTime(0, now + voice.fadeS);
    u.nodes.forEach((n) => n.stop(now + voice.fadeS));
    if (stopEvent) setTimeout(() => api("/api/voice/stopped", { utt, stop_ts: stopEvent.ts }).catch(() => {}), voice.fadeS * 1000);
  }
  voice.tail = 0;
  showSpeaking();
}

const showSpeaking = () => renderVoiceState();

// What the voice does now, shown in the header and on the Voice tab: waiting for the wake word, listening to a
// request, thinking, or speaking. Events come from the voice stream (fast, while Voice is on) and from the 1 s poll
// of the event log (always); `seen` makes sure each event counts once.
const vstate = { wake: false, phase: null, otherUntil: 0, seen: 0, timer: null };
const VSTATE_LABEL = { wake: 'Say "Hey Sanjaya"', listening: "Listening…", thinking: "Thinking…", speaking: "Speaking" };

function voiceState() {
  const here = Object.values(voice.utts).some((u) => u.nodes.length);   // this tab plays
  if (here || Date.now() / 1000 < vstate.otherUntil) return "speaking";
  return vstate.phase || (vstate.wake ? "wake" : "idle");
}

function renderVoiceState() {
  const s = voiceState();
  for (const el of document.querySelectorAll(".voice-state")) {
    el.dataset.state = s;
    el.hidden = s === "idle";
    el.querySelector(".vs-label").textContent = VSTATE_LABEL[s] || "";
  }
  $("#voice-stop").hidden = s !== "speaking";
}

// Speech that this tab does not play (this machine's speakers, or another tab): estimate when it ends from the
// length of each chunk, since the chunks play back to back.
function noteOtherSpeech(e) {
  if (e.route === "tab" && voice.source) return;   // this tab plays it, and knows exactly when it ends
  vstate.otherUntil = Math.max(vstate.otherUntil, Date.now() / 1000) + (e.audio_s || 0);
  clearTimeout(vstate.timer);
  vstate.timer = setTimeout(renderVoiceState, (vstate.otherUntil - Date.now() / 1000) * 1000 + 50);
}

function noteVoiceState(e) {
  if (replaying || e.id <= vstate.seen) return;
  vstate.seen = e.id;
  if (e.kind === "voice.wake") vstate.wake = e.state === "listening";
  if (e.kind === "voice.listening") vstate.phase = "listening";
  if (e.kind === "voice.recorded") vstate.phase = "thinking";
  if (e.kind === "voice.turn" && !e.text) vstate.phase = null;   // with text, it thinks until the speech starts
  if (e.kind === "voice.error" || (e.kind === "voice.wake" && e.state === "off")) vstate.phase = null;
  if (e.kind === "voice.stop") vstate.otherUntil = 0;
  if (e.kind === "voice.chunk" && !(e.route === "tab" && voice.source)) {
    vstate.phase = null;
    noteOtherSpeech(e);
  }
  renderVoiceState();
}
const stopSpeaking = () => api("/api/voice/stop", { reason: "button" }).catch((err) => toast(err.message, "error"));
$("#voice-stop").onclick = stopSpeaking;
document.addEventListener("keydown", (e) => {
  if (e.key !== "s" || e.metaKey || e.ctrlKey || e.altKey) return;
  if (["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement.tagName)) return;
  stopSpeaking();
});

$("#voice-form").onsubmit = (ev) => {
  ev.preventDefault();
  const button = ev.target.querySelector("button");
  busy(button, () => api("/api/voice/say", { text: $("#voice-text").value, voice: $("#voice-voice").value }));
};

function showWake(on, modelReady) {
  vstate.wake = on;
  renderVoiceState();
  const b = $("#wake-switch");
  b.classList.toggle("paused", !on);
  b.setAttribute("aria-checked", String(on));
  b.textContent = on ? "Wake word on" : "Wake word off";
  b.title = modelReady ? "Listen for \"Hey Sanjaya\" on this machine's microphone" : "The wake-word model is not trained yet: use Talk";
}
$("#wake-switch").onclick = () => {
  const on = $("#wake-switch").getAttribute("aria-checked") !== "true";
  api("/api/voice/wake", { on }).then((s) => showWake(s.wake, s.wake_model)).catch((err) => toast(err.message, "error"));
};
const talk = () => api("/api/voice/talk", {}).catch((err) => toast(err.message, "error"));
$("#talk-btn").onclick = talk;
document.addEventListener("keydown", (e) => {
  if (e.key !== "t" || e.metaKey || e.ctrlKey || e.altKey) return;
  if (["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement.tagName)) return;
  talk();
});

async function refreshVoice() {
  let s;
  try { s = await api("/api/voice/state"); } catch { return; }
  voice.budget = s.budget;
  voice.fadeS = (s.fade_ms ?? 30) / 1000;
  const select = $("#voice-voice");
  if (!select.options.length) {
    select.innerHTML = s.voices.map((v) => `<option${v === s.voice ? " selected" : ""}>${esc(v)}</option>`).join("");
  }
  if (s.error) addNotice("voice", s.error, "warn");
  showWake(s.wake, s.wake_model);
  renderVoice();
}
TAB_REFRESH.voice = refreshVoice;

// One row per utterance, built from the event log: when it was said, where, and how long each stage took.
function voiceRows() {
  const rows = {};
  for (const e of allEvents) {
    if (!e.kind.startsWith("voice.") || !e.utt) continue;
    const r = (rows[e.utt] ||= { utt: e.utt, chunks: [] });
    if (e.kind === "voice.say") Object.assign(r, { ts: e.ts, text: e.text, route: e.route, reason: e.reason });
    if (e.kind === "voice.chunk") r.chunks.push(e);
    if (e.kind === "voice.played") r.first = e.since_say_s;
    if (e.kind === "voice.stopped") r.stopped = e.after_stop_s;
  }
  return Object.values(rows).filter((r) => r.ts).reverse();
}

const secs = (s) => (s === undefined ? "–" : `${s.toFixed(2)} s`);
const overBudget = (s, key) => (s !== undefined && voice.budget[key] !== undefined && s > voice.budget[key] ? ' class="num step-bad"' : ' class="num"');

function renderVoice() {
  const rows = voiceRows();
  const firsts = rows.map((r) => r.first).filter((v) => v !== undefined);
  const stops = rows.map((r) => r.stopped).filter((v) => v !== undefined);
  const median = (xs) => (xs.length ? [...xs].sort((a, b) => a - b)[Math.floor(xs.length / 2)] : undefined);
  $("#voice-tiles").innerHTML = rows.length ? tiles([
    [secs(median(firsts)), "", `Median time to first sound (budget ${secs(voice.budget.tts_first + voice.budget.to_tab)})`],
    [stops.length ? `${Math.round(median(stops) * 1000)} ms` : "–", "", `Median stop latency (budget ${Math.round((voice.budget.stop ?? 0) * 1000)} ms)`],
    [String(rows.length), "", "Things said"],
  ]) : "";
  $("#voice-table tbody").innerHTML = rows.slice(0, 20).map((r) => `<tr><td>${clock(r.ts)}</td>
    <td>${esc(short(r.text, 90))}${r.reason ? `<small class="muted"> · ${esc(r.reason)}</small>` : ""}</td>
    <td>${r.route === "tab" ? "This tab" : "Speakers"}</td>
    <td${overBudget(r.chunks[0]?.synth_s, "tts_first")}>${secs(r.chunks[0]?.synth_s)}</td>
    <td class="num">${secs(r.first)}</td>
    <td${overBudget(r.stopped, "stop")}>${r.stopped === undefined ? "–" : Math.round(r.stopped * 1000) + " ms"}</td></tr>`).join("");
  $("#panel-voice .empty-note").hidden = rows.length > 0;
}

let voiceRedraw;
for (const kind of ["voice.say", "voice.chunk", "voice.played", "voice.stopped"]) {
  renderers[kind] = (e) => {
    noteVoiceState(e);
    clearTimeout(voiceRedraw);
    voiceRedraw = setTimeout(() => { if (isTab("voice")) renderVoice(); }, 200);
  };
}
renderers["voice.stop"] = noteVoiceState;
renderers["voice.wake"] = (e) => {
  noteVoiceState(e);
  if (!replaying) showWake(e.state === "listening", true);   // the switch follows a change made in another tab too
};
renderers["voice.recorded"] = (e) => { noteVoiceState(e); if (!replaying) $("#wake-status").textContent = "Thinking…"; };
renderers["voice.error"] = (e) => { noteVoiceState(e); if (!replaying) toast(e.error, "error"); };
renderers["voice.listening"] = (e) => { noteVoiceState(e); if (!replaying) $("#wake-status").textContent = "Listening… say your request."; };
renderers["voice.turn"] = (e) => {
  noteVoiceState(e);
  if (replaying) return;
  $("#wake-status").textContent = e.text ? `Heard: "${short(e.text, 60)}"` : "No request heard.";
  if (e.text) loadChat();   // the spoken request and its reply are in the chat now
};

// Show the wake-word state from the start, not only once the Voice tab opens.
api("/api/voice/state").then((s) => showWake(s.wake, s.wake_model)).catch(() => {});
