import test from "node:test";
import assert from "node:assert/strict";
import { createConversation } from "../public/chat-core.mjs";
import { createEventParser } from "../public/sse.mjs";
import { loadSettings, saveSettings, sanitise, STORAGE_KEY } from "../public/settings.mjs";

/** A transport that replays a fixed list of events, honouring abort like the real one. */
function fakeTransport(events, { delayMs = 0, onBody = () => {} } = {}) {
  return async ({ body, onEvent, signal }) => {
    onBody(body);
    for (const event of events) {
      if (signal.aborted) {
        const err = new Error("aborted");
        err.name = "AbortError";
        throw err;
      }
      onEvent(event);
      if (delayMs) await new Promise((r) => setTimeout(r, delayMs));
    }
  };
}

const wait = (ms = 0) => new Promise((r) => setTimeout(r, ms));

test("sending streams a reply into the assistant message", async () => {
  const seen = [];
  const convo = createConversation({
    transport: fakeTransport([
      { type: "start" },
      { type: "token", text: "hello" },
      { type: "token", text: " there" },
      { type: "done", tokens_generated: 2, ms_per_token: 5, tokens_per_second: 200 },
    ]),
  });
  convo.subscribe((s) => seen.push(s.messages.length ? { ...s.messages[1] } : null));
  convo.send("hi");
  await wait();

  const contents = seen.map((m) => m && m.content).filter((c) => c !== undefined);
  assert.ok(contents.includes("hello"), "partial text is visible while streaming");
  const final = seen[seen.length - 1];
  assert.equal(final.content, "hello there");
  assert.equal(final.status, "done");
});

test("conversation state after a successful turn", async () => {
  const convo = createConversation({
    transport: fakeTransport([
      { type: "token", text: "hello" },
      { type: "token", text: " there" },
      { type: "done", tokens_generated: 2, ms_per_token: 5, tokens_per_second: 200 },
    ]),
  });
  let last = null;
  convo.subscribe((s) => { last = s; });
  convo.send("hi");
  assert.equal(last.busy, true, "busy while streaming");
  await wait();
  assert.equal(last.busy, false);
  assert.equal(last.messages.length, 2);
  assert.equal(last.messages[0].role, "user");
  assert.equal(last.messages[0].content, "hi");
  assert.equal(last.messages[1].role, "assistant");
  assert.equal(last.messages[1].content, "hello there");
  assert.equal(last.messages[1].status, "done");
  assert.equal(last.messages[1].stats.tokens, 2);
});

test("the request carries the conversation and the current parameters", async () => {
  let sent = null;
  const convo = createConversation({
    transport: fakeTransport([{ type: "token", text: "ok" }, { type: "done" }], { onBody: (b) => { sent = b; } }),
  });
  convo.setParams({ max_tokens: 32, temperature: 0.2, top_p: 0.5, top_k: 7, repetition_penalty: 1.4 });
  convo.send("first");
  await wait();
  assert.deepEqual(sent.messages, [{ role: "user", content: "first" }]);
  assert.equal(sent.max_tokens, 32);
  assert.equal(sent.temperature, 0.2);
  assert.equal(sent.top_p, 0.5);
  assert.equal(sent.top_k, 7);
  assert.equal(sent.repetition_penalty, 1.4);

  convo.send("second");
  await wait();
  assert.deepEqual(sent.messages.map((m) => m.role), ["user", "assistant", "user"]);
  assert.equal(sent.messages[1].content, "ok");
});

test("stop aborts the stream and keeps what arrived", async () => {
  const convo = createConversation({
    transport: fakeTransport(
      [{ type: "token", text: "par" }, { type: "token", text: "tial" }, { type: "token", text: " more" }],
      { delayMs: 5 },
    ),
  });
  let last = null;
  convo.subscribe((s) => { last = s; });
  convo.send("go");
  await wait(2);
  assert.equal(convo.stop(), true);
  await wait(30);
  const msg = last.messages[1];
  assert.equal(msg.status, "stopped");
  assert.ok(msg.content.startsWith("par"), `content kept: ${JSON.stringify(msg.content)}`);
  assert.equal(last.busy, false);
});

test("a server error is surfaced, never replaced with invented text", async () => {
  const convo = createConversation({
    transport: fakeTransport([{ type: "error", error: "model unavailable" }]),
  });
  let last = null;
  convo.subscribe((s) => { last = s; });
  convo.send("hi");
  await wait();
  const msg = last.messages[1];
  assert.equal(msg.status, "error");
  assert.equal(msg.content, "");
  assert.equal(msg.error, "model unavailable");
  assert.equal(last.busy, false);
});

test("a thrown transport error is surfaced too", async () => {
  const convo = createConversation({
    transport: async () => { throw new Error("request failed (500)"); },
  });
  let last = null;
  convo.subscribe((s) => { last = s; });
  convo.send("hi");
  await wait();
  assert.equal(last.messages[1].status, "error");
  assert.equal(last.messages[1].error, "request failed (500)");
});

test("an empty reply is marked empty rather than shown as success", async () => {
  const convo = createConversation({ transport: fakeTransport([{ type: "done", tokens_generated: 0 }]) });
  let last = null;
  convo.subscribe((s) => { last = s; });
  convo.send("hi");
  await wait();
  assert.equal(last.messages[1].status, "empty");
  assert.ok(last.messages[1].error);
});

test("retry replaces the reply and keeps the earlier turns", async () => {
  let call = 0;
  let lastBody = null;
  const convo = createConversation({
    transport: async ({ body, onEvent }) => {
      lastBody = body;
      call++;
      onEvent({ type: "token", text: call === 1 ? "first answer" : "second answer" });
      onEvent({ type: "done", tokens_generated: 1 });
    },
  });
  let last = null;
  convo.subscribe((s) => { last = s; });
  convo.send("question");
  await wait();
  assert.equal(last.messages[1].content, "first answer");

  const replyId = last.messages[1].id;
  convo.retry(replyId);
  await wait();
  assert.equal(last.messages.length, 2, "retry must not append a third message");
  assert.equal(last.messages[1].content, "second answer");
  assert.deepEqual(lastBody.messages, [{ role: "user", content: "question" }]);
});

test("cannot send while a reply is streaming", async () => {
  const convo = createConversation({
    transport: fakeTransport([{ type: "token", text: "a" }, { type: "done" }], { delayMs: 5 }),
  });
  convo.send("one");
  const second = convo.send("two");
  assert.equal(second, null);
  await wait(30);
  let last = null;
  convo.subscribe((s) => { last = s; });
  assert.equal(last.messages.length, 2);
});

test("empty input is ignored", () => {
  const convo = createConversation({ transport: async () => {} });
  assert.equal(convo.send("   "), null);
  let last = null;
  convo.subscribe((s) => { last = s; });
  assert.equal(last.messages.length, 0);
});

test("clear empties the conversation", async () => {
  const convo = createConversation({
    transport: fakeTransport([{ type: "token", text: "x" }, { type: "done" }]),
  });
  convo.send("hi");
  await wait();
  assert.equal(convo.clear(), true);
  let last = null;
  convo.subscribe((s) => { last = s; });
  assert.equal(last.messages.length, 0);
});

/* ------------------------------------------------------------------ sse parsing */

test("sse parser reassembles frames split across chunks", () => {
  const events = [];
  const parser = createEventParser((e) => events.push(e));
  parser.push('data: {"type":"tok');
  parser.push('en","text":"hi"}\n\ndata: {"type":"do');
  parser.push('ne"}\n\n');
  assert.deepEqual(events, [{ type: "token", text: "hi" }, { type: "done" }]);
});

test("sse parser ignores comments and malformed frames", () => {
  const events = [];
  const parser = createEventParser((e) => events.push(e));
  parser.push(': keep-alive\n\n');
  parser.push("data: not json\n\n");
  parser.push('data: {"type":"token","text":"ok"}\n\n');
  assert.equal(events.length, 2);
  assert.equal(events[0].type, "error");
  assert.equal(events[1].text, "ok");
});

/* ------------------------------------------------------------------- settings */

function fakeStorage() {
  const map = new Map();
  return {
    getItem: (k) => (map.has(k) ? map.get(k) : null),
    setItem: (k, v) => map.set(k, String(v)),
    removeItem: (k) => map.delete(k),
    _map: map,
  };
}

test("settings persist and reload", () => {
  const storage = fakeStorage();
  const saved = { max_tokens: 64, temperature: 1.25, top_p: 0.75, top_k: 20, repetition_penalty: 1.2 };
  saveSettings(storage, saved);
  assert.deepEqual(loadSettings(storage, null), saved);
  assert.ok(storage._map.has(STORAGE_KEY));
});

test("settings fall back to defaults when storage is empty or corrupt", () => {
  assert.deepEqual(loadSettings(fakeStorage(), null), {
    max_tokens: 128, temperature: 0.8, top_p: 0.9, top_k: 40, repetition_penalty: 1,
  });
  const storage = fakeStorage();
  storage.setItem(STORAGE_KEY, "{not json");
  assert.equal(loadSettings(storage, null).top_k, 40);
  storage.setItem(STORAGE_KEY, JSON.stringify({ temperature: 99, top_k: "abc" }));
  const loaded = loadSettings(storage, null);
  assert.equal(loaded.temperature, 2);
  assert.equal(loaded.top_k, 40);
});

test("sanitise drops unknown keys and clamps stored values", () => {
  const out = sanitise(null, { temperature: -10, max_tokens: 9999, nonsense: 5 });
  assert.equal(out.temperature, 0);
  assert.equal(out.max_tokens, 512);
  assert.equal(out.nonsense, undefined);
});
