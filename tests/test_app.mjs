/**
 * End-to-end-ish UI test: loads the real `public/app.js` against a stub DOM and a stub
 * fetch, then drives the controls the way a person would.
 */
import test from "node:test";
import assert from "node:assert/strict";
import { createDom, settle } from "./dom.mjs";

const IDS = [
  "status", "status-text", "thread", "empty", "log", "input", "send", "stop",
  "settings", "settings-btn", "sliders", "reset", "engine-hint", "composer",
];

const PARAMS = [
  { name: "max_tokens", label: "Max tokens", min: 1, max: 512, step: 1, default: 128 },
  { name: "temperature", label: "Temperature", min: 0, max: 2, step: 0.05, default: 0.8 },
  { name: "top_p", label: "Top-p", min: 0.05, max: 1, step: 0.01, default: 0.9 },
  { name: "top_k", label: "Top-k", min: 1, max: 100, step: 1, default: 40 },
  { name: "repetition_penalty", label: "Repetition penalty", min: 1, max: 2, step: 0.01, default: 1 },
];

/** SSE body the stub fetch hands to the app. */
function sseStream(text, { delayMs = 0, signal } = {}) {
  const encoder = new TextEncoder();
  const frames = [
    `data: ${JSON.stringify({ type: "start", params: {} })}\n\n`,
    ...text.split(" ").map((w, i) => `data: ${JSON.stringify({ type: "token", text: (i ? " " : "") + w })}\n\n`),
    `data: ${JSON.stringify({ type: "done", tokens_generated: text.split(" ").length, ms_per_token: 4.2, tokens_per_second: 238 })}\n\n`,
  ];
  return new ReadableStream({
    async start(controller) {
      for (const frame of frames) {
        if (signal && signal.aborted) {
          const err = new Error("aborted");
          err.name = "AbortError";
          controller.error(err);
          return;
        }
        controller.enqueue(encoder.encode(frame));
        if (delayMs) await new Promise((r) => setTimeout(r, delayMs));
      }
      controller.close();
    },
  });
}

/** Initial hidden/disabled state mirrors public/index.html. */
const HIDDEN = ["thread", "stop", "settings"];

async function boot({ reply = "a dumb little chatbot", delayMs = 0, storage = new Map() } = {}) {
  const dom = createDom(IDS, { hidden: HIDDEN, disabled: ["send"], storage });
  const calls = [];
  globalThis.document = dom.document;
  globalThis.localStorage = dom.localStorage;
  globalThis.window = dom.window;
  Object.defineProperty(globalThis, "navigator", {
    value: { clipboard: { writeText: async () => {} } },
    configurable: true,
    writable: true,
  });
  // browsers run rAF on the next frame; defer so the app's "already scheduled" guard behaves
  globalThis.requestAnimationFrame = (fn) => setTimeout(() => fn(performance.now()), 0);
  globalThis.fetch = async (url, opts = {}) => {
    calls.push({ url, opts, body: opts.body ? JSON.parse(opts.body) : null });
    if (url === "/api/params") {
      return { ok: true, json: async () => ({ params: PARAMS, defaults: {} }) };
    }
    if (url === "/api/health") return { ok: true, json: async () => ({ ok: true, tensors: 34 }) };
    if (url === "/api/generate") {
      return { ok: true, body: sseStream(reply, { delayMs, signal: opts.signal }), json: async () => ({}) };
    }
    throw new Error(`unexpected fetch ${url}`);
  };

  // fresh module instance per test (app.js runs on import)
  const mod = await import(`../public/app.js?t=${Math.random()}`);
  await settle(5);
  const byId = { get: (id) => dom.document.getElementById(id) };
  return { dom, calls, byId, mod, storage: dom.localStorage };
}

/** The message body, without the copy/retry/timestamp row rendered next to it. */
const textOf = (li) => {
  const body = li.children.find((c) => c.classList.contains("text"));
  return body ? body.textContent.trim() : li.textContent.trim();
};

const typeInto = (byId, text) => {
  const input = byId.get("input");
  input.value = text;
  input.dispatchEvent({ type: "input" });
};

test("the panel is built from the server's parameter list", async () => {
  const { byId } = await boot();
  const sliders = byId.get("sliders");
  assert.equal(sliders.children.length, 5, "one row per parameter");
  for (const spec of PARAMS) {
    const input = byId.get(`p-${spec.name}`);
    assert.equal(input.min, String(spec.min));
    assert.equal(input.max, String(spec.max));
    assert.equal(input.step, String(spec.step));
    assert.equal(input.value, String(spec.default));
    assert.equal(byId.get(`v-${spec.name}`).textContent, spec.name === "max_tokens" || spec.name === "top_k"
      ? String(spec.default)
      : spec.default.toFixed(2));
  }
});

test("status becomes ready once the model answers /api/health", async () => {
  const { byId } = await boot();
  assert.equal(byId.get("status").dataset.state, "ready");
  assert.equal(byId.get("status-text").textContent, "ready");
});

test("typing enables send; whitespace does not", async () => {
  const { byId } = await boot();
  typeInto(byId, "   ");
  assert.equal(byId.get("send").disabled, true);
  typeInto(byId, "hello");
  assert.equal(byId.get("send").disabled, false);
});

test("sending renders the user message and streams the reply", async () => {
  const { byId, calls } = await boot({ reply: "the model said this" });
  typeInto(byId, "hi there");
  byId.get("composer").dispatchEvent({ type: "submit" });
  await settle(30);

  const thread = byId.get("thread");
  assert.equal(thread.hidden, false);
  assert.equal(byId.get("empty").hidden, true);
  assert.equal(thread.children.length, 2);
  assert.equal(thread.children[0].dataset.role, "user");
  assert.equal(textOf(thread.children[0]), "hi there");
  assert.equal(thread.children[1].dataset.role, "assistant");
  assert.equal(textOf(thread.children[1]), "the model said this");
  assert.equal(thread.children[1].dataset.status, "done");

  const gen = calls.find((c) => c.url === "/api/generate");
  assert.equal(gen.body.stream, true, "the UI asks for a stream");
  assert.deepEqual(gen.body.messages, [{ role: "user", content: "hi there" }]);
  assert.equal(gen.body.max_tokens, 128);
});

test("input clears and send disables while a reply is streaming", async () => {
  const { byId } = await boot({ reply: "slow words here", delayMs: 4 });
  typeInto(byId, "go");
  byId.get("composer").dispatchEvent({ type: "submit" });
  await settle(2);
  assert.equal(byId.get("input").value, "");
  assert.equal(byId.get("stop").hidden, false, "stop is offered while generating");
  assert.equal(byId.get("send").disabled, true);
  await settle(60);
  assert.equal(byId.get("stop").hidden, true);
});

test("stop aborts the request and keeps the partial reply", async () => {
  const { byId, calls } = await boot({ reply: "one two three four five", delayMs: 6 });
  typeInto(byId, "go");
  byId.get("composer").dispatchEvent({ type: "submit" });
  await settle(12);
  byId.get("stop").click();
  await settle(60);
  const gen = calls.find((c) => c.url === "/api/generate");
  assert.equal(gen.opts.signal.aborted, true, "the fetch was aborted");
  const assistant = byId.get("thread").children[1];
  assert.equal(assistant.dataset.status, "stopped");
  assert.ok(textOf(assistant).length > 0);
});

test("slider moves update the value, persist, and reach the next request", async () => {
  const { byId, calls, storage } = await boot();
  const temp = byId.get("p-temperature");
  temp.value = "1.5";
  temp.dispatchEvent({ type: "input" });

  assert.equal(byId.get("v-temperature").textContent, "1.50");
  const saved = JSON.parse(storage.getItem("cadence.settings.v1"));
  assert.equal(saved.temperature, 1.5);

  typeInto(byId, "hi");
  byId.get("composer").dispatchEvent({ type: "submit" });
  await settle(30);
  const gen = calls.find((c) => c.url === "/api/generate");
  assert.equal(gen.body.temperature, 1.5);
});

test("sliders snap to their step and clamp at the ends", async () => {
  const { byId, storage } = await boot();
  const k = byId.get("p-top_k");
  k.value = "1000";
  k.dispatchEvent({ type: "input" });
  assert.equal(byId.get("v-top_k").textContent, "100");
  const p = byId.get("p-top_p");
  p.value = "0";
  p.dispatchEvent({ type: "input" });
  assert.equal(byId.get("v-top_p").textContent, "0.05");
  const saved = JSON.parse(storage.getItem("cadence.settings.v1"));
  assert.equal(saved.top_k, 100);
  assert.equal(saved.top_p, 0.05);
});

test("reset restores defaults everywhere", async () => {
  const { byId, storage } = await boot();
  const temp = byId.get("p-temperature");
  temp.value = "1.75";
  temp.dispatchEvent({ type: "input" });
  byId.get("reset").click();
  assert.equal(byId.get("p-temperature").value, "0.8");
  assert.equal(byId.get("v-temperature").textContent, "0.80");
  assert.equal(JSON.parse(storage.getItem("cadence.settings.v1")).temperature, 0.8);
});

test("settings survive a reload (persisted values are used on boot)", async () => {
  // one storage map shared across two boots = a reload
  const storage = new Map();
  await boot({ storage });
  storage.set("cadence.settings.v1", JSON.stringify({ temperature: 1.9, top_k: 5 }));
  const second = await boot({ storage });
  assert.equal(second.byId.get("p-temperature").value, "1.9");
  assert.equal(second.byId.get("v-top_k").textContent, "5");
  typeInto(second.byId, "hi");
  second.byId.get("composer").dispatchEvent({ type: "submit" });
  await settle(30);
  const gen = second.calls.find((c) => c.url === "/api/generate");
  assert.equal(gen.body.temperature, 1.9);
  assert.equal(gen.body.top_k, 5);
});

test("Enter sends, Shift+Enter inserts a newline", async () => {
  const { byId } = await boot();
  typeInto(byId, "line");
  const shift = byId.get("input").dispatchEvent({ type: "keydown", key: "Enter", shiftKey: true });
  assert.equal(shift.defaultPrevented, false, "shift+enter is left to the textarea");
  const plain = byId.get("input").dispatchEvent({ type: "keydown", key: "Enter", shiftKey: false });
  assert.equal(plain.defaultPrevented, true, "enter is intercepted");
  await settle(20);
  assert.equal(byId.get("thread").children.length, 2);
});

test("the settings panel opens, closes on Escape, and closes on outside click", async () => {
  const { byId, dom } = await boot();
  assert.equal(byId.get("settings").hidden, true);
  byId.get("settings-btn").click();
  assert.equal(byId.get("settings").hidden, false);
  assert.equal(byId.get("settings-btn").getAttribute("aria-expanded"), "true");

  dom.document.dispatchEvent({ type: "keydown", key: "Escape" });
  assert.equal(byId.get("settings").hidden, true);

  byId.get("settings-btn").click();
  dom.document.dispatchEvent({ type: "click" });
  assert.equal(byId.get("settings").hidden, true);
});

test("a failed request shows an error instead of a made-up answer", async () => {
  const { byId } = await boot();
  globalThis.fetch = async (url, opts = {}) => {
    if (url === "/api/params") return { ok: true, json: async () => ({ params: PARAMS }) };
    if (url === "/api/health") return { ok: true, json: async () => ({ ok: true }) };
    return { ok: false, status: 503, json: async () => ({ error: "model unavailable" }) };
  };
  typeInto(byId, "hi");
  byId.get("composer").dispatchEvent({ type: "submit" });
  await settle(30);
  const assistant = byId.get("thread").children[1];
  assert.equal(assistant.dataset.status, "error");
  assert.match(textOf(assistant), /no reply/);
  assert.match(textOf(assistant), /model unavailable/);
});
