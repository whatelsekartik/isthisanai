/**
 * Handler-level tests: the real Vercel handlers, invoked with a fake req/res.
 * This closes the gap between "the UI sends the right body" and "the engine uses it".
 */
import test from "node:test";
import assert from "node:assert/strict";
import generate from "../api/generate.mjs";
import bench from "../api/bench.mjs";
import paramsRoute from "../api/params.mjs";
import health from "../api/health.mjs";
import { createEventParser } from "../public/sse.mjs";

function fakeRes() {
  const res = {
    statusCode: 200,
    headers: {},
    chunks: [],
    writableEnded: false,
    setHeader(k, v) { res.headers[k.toLowerCase()] = v; },
    flushHeaders() {},
    write(chunk) { res.chunks.push(chunk); return true; },
    end(chunk) { if (chunk) res.chunks.push(chunk); res.writableEnded = true; },
    on() {}, off() {},
    get text() { return res.chunks.join(""); },
  };
  return res;
}
const fakeReq = (body) => ({ method: "POST", body, on() {}, off() {} });

function parseSse(text) {
  const events = [];
  const parser = createEventParser((e) => events.push(e));
  parser.push(text);
  parser.flush();
  return events;
}

test("POST /api/generate streams start -> tokens -> done", async () => {
  const res = fakeRes();
  generate(fakeReq({ messages: [{ role: "user", content: "It is a truth" }], max_tokens: 6, stream: true }), res);
  assert.match(res.headers["content-type"], /text\/event-stream/);
  assert.match(res.headers["cache-control"], /no-store/);

  const events = parseSse(res.text);
  assert.equal(events[0].type, "start");
  assert.ok(events[0].params.max_tokens === 6);
  const tokens = events.filter((e) => e.type === "token");
  assert.ok(tokens.length > 0, "at least one token event");
  assert.ok(tokens.length <= 6, "never more token events than max_tokens");
  const done = events[events.length - 1];
  assert.equal(done.type, "done");
  assert.equal(done.tokens_generated, tokens.length);
  assert.ok(Number.isFinite(done.ms_per_token));
  assert.ok(res.writableEnded);
});

test("streamed text equals the non-streamed completion", async () => {
  const body = { messages: [{ role: "user", content: "The house was" }], max_tokens: 8, temperature: 0, top_k: 40 };

  const resA = fakeRes();
  generate(fakeReq({ ...body, stream: true }), resA);
  const streamed = parseSse(resA.text).filter((e) => e.type === "token").map((e) => e.text).join("");

  const resB = fakeRes();
  generate(fakeReq({ ...body }), resB);
  const json = JSON.parse(resB.text);
  assert.equal(streamed, json.completion);
});

test("non-streaming response keeps its documented shape", async () => {
  const res = fakeRes();
  generate(fakeReq({ prompt: "It is", max_tokens: 4 }), res);
  const json = JSON.parse(res.text);
  for (const key of ["prompt", "completion", "prompt_tokens", "tokens_generated", "params", "ms_per_token", "arch"]) {
    assert.ok(key in json, `missing ${key}`);
  }
  assert.equal(res.statusCode, 200);
});

test("parameters reach the engine and out-of-range values are clamped", async () => {
  const res = fakeRes();
  generate(fakeReq({ prompt: "It is", max_tokens: 9999, temperature: -5, top_p: 0, top_k: 0, repetition_penalty: 9 }), res);
  const json = JSON.parse(res.text);
  assert.deepEqual(json.params, {
    max_tokens: 512, temperature: 0, top_p: 0.05, top_k: 1, repetition_penalty: 2,
  });
});

test("max_tokens is respected exactly", async () => {
  for (const n of [1, 3, 12]) {
    const res = fakeRes();
    generate(fakeReq({ prompt: "It is a", max_tokens: n }), res);
    assert.equal(JSON.parse(res.text).tokens_generated, n, `max_tokens=${n}`);
  }
});

test("temperature change actually changes the output", async () => {
  const run = (temperature) => {
    const res = fakeRes();
    generate(fakeReq({ prompt: "It is a truth", max_tokens: 24, temperature }), res);
    return JSON.parse(res.text).completion;
  };
  const greedy1 = run(0);
  const greedy2 = run(0);
  assert.equal(greedy1, greedy2, "temperature 0 is deterministic");
  assert.notEqual(greedy1, run(1.6), "temperature 1.6 samples differently");
});

test("empty prompt is rejected without inventing an answer", async () => {
  const res = fakeRes();
  generate(fakeReq({ prompt: "   " }), res);
  assert.equal(res.statusCode, 400);
  assert.ok(JSON.parse(res.text).error);
});

test("GET /api/generate is rejected", async () => {
  const res = fakeRes();
  generate({ method: "GET", body: {}, on() {}, off() {} }, res);
  assert.equal(res.statusCode, 405);
});

test("/api/params advertises exactly the engine's parameters", async () => {
  const res = fakeRes();
  paramsRoute({ method: "GET" }, res);
  const json = JSON.parse(res.text);
  assert.equal(json.params.length, 5);
  for (const p of json.params) {
    assert.ok(p.min < p.max && p.step > 0 && p.default >= p.min && p.default <= p.max, p.name);
  }
});

test("/api/health reports a loaded model", async () => {
  const res = fakeRes();
  health({ method: "GET" }, res);
  const json = JSON.parse(res.text);
  assert.equal(json.ok, true);
  assert.ok(json.tensors > 0);
});

test("/api/bench distinguishes fresh measurements from stored ones", async () => {
  const first = fakeRes();
  bench({ method: "GET", url: "/api/bench?fresh=1" }, first);
  const live = JSON.parse(first.text);
  assert.equal(live.cached, false);
  assert.ok(live.measured_at);
  for (const key of ["weights_mb", "ms_per_token", "tokens_per_second", "state_bytes", "rss_mb"]) {
    assert.ok(key in live, `missing ${key}`);
  }

  const second = fakeRes();
  bench({ method: "GET", url: "/api/bench" }, second);
  const stored = JSON.parse(second.text);
  assert.equal(stored.cached, true);
  assert.equal(stored.measured_at, live.measured_at, "stored numbers are the ones measured earlier");
  assert.equal(stored.ms_per_token, live.ms_per_token);
});
