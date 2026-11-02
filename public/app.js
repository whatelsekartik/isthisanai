import { createConversation } from "./chat-core.mjs";
import { loadSettings, saveSettings, sanitise, defaultsFrom, formatValue } from "./settings.mjs";

const $ = (id) => document.getElementById(id);
const statusEl = $("status");
const statusText = $("status-text");
const threadEl = $("thread");
const emptyEl = $("empty");
const logEl = $("log");
const inputEl = $("input");
const sendEl = $("send");
const stopEl = $("stop");
const panel = $("settings");
const settingsBtn = $("settings-btn");
const slidersEl = $("sliders");

const SPECS = [
  { name: "max_tokens", label: "Max tokens", min: 1, max: 512, step: 1, default: 128 },
  { name: "temperature", label: "Temperature", min: 0, max: 2, step: 0.05, default: 0.8 },
  { name: "top_p", label: "Top-p", min: 0.05, max: 1, step: 0.01, default: 0.9 },
  { name: "top_k", label: "Top-k", min: 1, max: 100, step: 1, default: 40 },
  { name: "repetition_penalty", label: "Repetition penalty", min: 1, max: 2, step: 0.01, default: 1 },
];

let specs = SPECS;
let settings = loadSettings(localStorage, specs);

const convo = createConversation();
convo.setParams(settings);

/* ------------------------------------------------------------------ settings */

function buildSliders() {
  slidersEl.textContent = "";
  for (const spec of specs) {
    const wrap = document.createElement("div");
    wrap.className = "slider";

    const head = document.createElement("div");
    head.className = "slider-head";
    const label = document.createElement("label");
    label.htmlFor = `p-${spec.name}`;
    label.textContent = spec.label;
    const out = document.createElement("output");
    out.htmlFor = `p-${spec.name}`;
    out.id = `v-${spec.name}`;
    out.textContent = formatValue(spec.name, settings[spec.name]);
    head.append(label, out);

    const input = document.createElement("input");
    input.type = "range";
    input.id = `p-${spec.name}`;
    input.min = String(spec.min);
    input.max = String(spec.max);
    input.step = String(spec.step);
    input.value = String(settings[spec.name]);
    input.setAttribute("aria-valuetext", out.textContent);
    input.addEventListener("input", () => {
      const next = sanitise(specs, { ...settings, [spec.name]: Number(input.value) });
      settings = next;
      out.textContent = formatValue(spec.name, settings[spec.name]);
      input.setAttribute("aria-valuetext", out.textContent);
      saveSettings(localStorage, settings);
      convo.setParams(settings);
    });

    wrap.append(head, input);
    slidersEl.append(wrap);
  }
}

function syncSliders() {
  for (const spec of specs) {
    const input = $(`p-${spec.name}`);
    const out = $(`v-${spec.name}`);
    if (!input) continue;
    input.value = String(settings[spec.name]);
    out.textContent = formatValue(spec.name, settings[spec.name]);
    input.setAttribute("aria-valuetext", out.textContent);
  }
}

function togglePanel(open) {
  const next = open ?? panel.hidden;
  panel.hidden = !next;
  settingsBtn.setAttribute("aria-expanded", String(next));
  if (next) {
    const first = panel.querySelector("input");
    if (first) first.focus({ preventScroll: true });
  }
}

settingsBtn.addEventListener("click", (e) => {
  e.stopPropagation();
  togglePanel();
});

panel.addEventListener("click", (e) => e.stopPropagation());

document.addEventListener("click", () => {
  if (!panel.hidden) togglePanel(false);
});

document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && !panel.hidden) {
    togglePanel(false);
    settingsBtn.focus();
  }
});

$("reset").addEventListener("click", () => {
  settings = defaultsFrom(specs);
  saveSettings(localStorage, settings);
  syncSliders();
  convo.setParams(settings);
});

/* ------------------------------------------------------------------ rendering */

const nodes = new Map();
let pendingRender = 0;
let lastSnapshot = null;

function timeOf(ms) {
  return new Date(ms).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function makeMessage(msg) {
  const li = document.createElement("li");
  li.className = "msg";
  li.dataset.role = msg.role;
  li.dataset.id = msg.id;

  const text = document.createElement("div");
  text.className = "text";

  const meta = document.createElement("div");
  meta.className = "meta";

  li.append(text, meta);
  threadEl.append(li);
  const node = { li, text, meta, msg };
  nodes.set(msg.id, node);
  return node;
}

function syncMeta(node, msg) {
  const parts = [];
  if (msg.role === "assistant") {
    if (msg.content) {
      const copy = document.createElement("button");
      copy.type = "button";
      copy.textContent = "copy";
      copy.addEventListener("click", async () => {
        try {
          await navigator.clipboard.writeText(msg.content);
        } catch {
          const ta = document.createElement("textarea");
          ta.value = msg.content;
          document.body.append(ta);
          ta.select();
          document.execCommand("copy");
          ta.remove();
        }
        copy.textContent = "copied";
        setTimeout(() => { copy.textContent = "copy"; }, 1200);
      });
      parts.push(copy);
    }
    if (!node.msg.status || node.msg.status !== "streaming") {
      const retry = document.createElement("button");
      retry.type = "button";
      retry.textContent = "retry";
      retry.addEventListener("click", () => convo.retry(msg.id));
      parts.push(retry);
    }
    if (msg.status === "stopped") {
      const s = document.createElement("span");
      s.textContent = "stopped";
      parts.push(s);
    }
    if (msg.stats && msg.status === "done") {
      const s = document.createElement("span");
      s.className = "dot";
      s.textContent = `${msg.stats.tokens} tok · ${msg.stats.ms_per_token} ms/tok`;
      parts.push(s);
    }
    if (msg.startedAt) {
      const t = document.createElement("span");
      t.className = "dot";
      t.textContent = timeOf(msg.startedAt);
      parts.push(t);
    }
  }
  const stamp = parts.map((p) => p.textContent).join("|") + msg.status + msg.content.length;
  if (node.metaStamp === stamp) return;
  node.metaStamp = stamp;
  node.meta.textContent = "";
  node.meta.append(...parts);
}

function render() {
  pendingRender = 0;
  const snap = lastSnapshot;
  if (!snap) return;

  const nearBottom = logEl.scrollHeight - logEl.scrollTop - logEl.clientHeight < 120;
  const seen = new Set();

  snap.messages.forEach((msg, index) => {
    seen.add(msg.id);
    let node = nodes.get(msg.id);
    if (!node) {
      node = makeMessage(msg);
    }
    node.msg = msg;
    const streaming = msg.status === "streaming";
    const errorish = msg.status === "error" || msg.status === "empty";
    const body = errorish && !msg.content
      ? (msg.error ? `no reply — ${msg.error}` : "no reply")
      : msg.content;

    if (node.text.dataset.body !== body || node.text.dataset.streaming !== String(streaming)) {
      node.text.dataset.body = body;
      node.text.dataset.streaming = String(streaming);
      node.text.textContent = body;
      if (streaming) {
        const caret = document.createElement("span");
        caret.className = "caret";
        caret.setAttribute("aria-hidden", "true");
        node.text.append(caret);
      }
      if (errorish && msg.error && msg.content) {
        const note = document.createElement("div");
        note.className = "meta";
        note.style.opacity = "1";
        note.textContent = msg.error;
        node.text.append(note);
      }
    }
    node.li.dataset.status = msg.status || "done";
    node.li.style.order = String(index);
    syncMeta(node, msg);
  });

  for (const [id, node] of nodes) {
    if (!seen.has(id)) {
      node.li.remove();
      nodes.delete(id);
    }
  }

  // correct DOM order after retries/removals
  snap.messages.forEach((msg) => {
    const node = nodes.get(msg.id);
    if (node) threadEl.append(node.li);
  });

  const hasMessages = snap.messages.length > 0;
  emptyEl.hidden = hasMessages;
  threadEl.hidden = !hasMessages;
  threadEl.setAttribute("aria-live", snap.busy ? "off" : "polite");

  if (nearBottom) logEl.scrollTop = logEl.scrollHeight;
}

function schedule() {
  if (!pendingRender) pendingRender = requestAnimationFrame(render);
}

convo.subscribe((snap) => {
  lastSnapshot = snap;
  schedule();
  updateControls(snap);
});

/* ------------------------------------------------------------------ controls */

function updateControls(snap) {
  const busy = snap.busy;
  sendEl.disabled = busy || !inputEl.value.trim();
  stopEl.hidden = !busy;
  inputEl.setAttribute("aria-busy", String(busy));
  if (busy) {
    setStatus("generating", "thinking");
  } else if (statusEl.dataset.state === "error") {
    setStatus("error", "offline");
  } else {
    setStatus("ready", "ready");
  }
}

function setStatus(state, text) {
  statusEl.dataset.state = state;
  statusText.textContent = text;
}

/* ------------------------------------------------------------------ composer */

function autogrow() {
  inputEl.style.height = "auto";
  inputEl.style.height = `${Math.min(inputEl.scrollHeight, window.innerHeight * 0.4)}px`;
}

inputEl.addEventListener("input", () => {
  autogrow();
  sendEl.disabled = convo.busy || !inputEl.value.trim();
});

inputEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
    e.preventDefault();
    submit();
  }
});

$("composer").addEventListener("submit", (e) => {
  e.preventDefault();
  submit();
});

stopEl.addEventListener("click", () => convo.stop());

function submit() {
  const text = inputEl.value;
  if (!text.trim() || convo.busy) return;
  inputEl.value = "";
  autogrow();
  convo.send(text);
  sendEl.disabled = true;
  inputEl.focus();
}

/* ------------------------------------------------------------------ bootstrap */

/* The panel only offers what the engine actually implements, so say so once. */
function describeEngine() {
  const hint = $("engine-hint");
  if (hint) hint.textContent = "applies to the next reply";
}

async function loadSpecs() {
  try {
    const res = await fetch("/api/params");
    if (!res.ok) return;
    const data = await res.json();
    if (Array.isArray(data.params) && data.params.length) {
      specs = data.params;
      settings = loadSettings(localStorage, specs);
      convo.setParams(settings);
    }
  } catch {
    /* keep the built-in ranges; they match the server's */
  }
}

async function checkHealth() {
  try {
    const res = await fetch("/api/health", { cache: "no-store" });
    const data = await res.json();
    if (!data.ok) throw new Error(data.error || "unavailable");
    if (!convo.busy) setStatus("ready", "ready");
  } catch {
    if (!convo.busy) setStatus("error", "offline");
  }
}

(async () => {
  await loadSpecs();
  buildSliders();
  describeEngine();
  autogrow();
  checkHealth();
})();
