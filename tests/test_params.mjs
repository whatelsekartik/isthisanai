import test from "node:test";
import assert from "node:assert/strict";
import { PARAMS, DEFAULTS, normalise, parseParams } from "../lib/params.mjs";

test("every advertised parameter has a range, a step and a default", () => {
  assert.equal(PARAMS.length, 5);
  for (const p of PARAMS) {
    assert.ok(p.min <= p.default && p.default <= p.max, `${p.name} default inside range`);
    assert.ok(p.step > 0, `${p.name} has a step`);
    assert.equal(DEFAULTS[p.name], p.default);
  }
  assert.deepEqual(
    PARAMS.map((p) => p.name),
    ["max_tokens", "temperature", "top_p", "top_k", "repetition_penalty"],
  );
});

test("defaults match the documented table", () => {
  assert.deepEqual(DEFAULTS, {
    max_tokens: 128, temperature: 0.8, top_p: 0.9, top_k: 40, repetition_penalty: 1,
  });
});

test("values are clamped to range", () => {
  assert.equal(normalise("max_tokens", 9999), 512);
  assert.equal(normalise("max_tokens", 0), 1);
  assert.equal(normalise("temperature", -3), 0);
  assert.equal(normalise("temperature", 9), 2);
  assert.equal(normalise("top_p", 0), 0.05);
  assert.equal(normalise("top_k", 1000), 100);
  assert.equal(normalise("repetition_penalty", 0.1), 1);
  assert.equal(normalise("repetition_penalty", 5), 2);
});

test("values snap to the slider grid", () => {
  assert.equal(normalise("temperature", 0.833333), 0.85);
  assert.equal(normalise("top_p", 0.901), 0.9);
  assert.equal(normalise("repetition_penalty", 1.234), 1.23);
  assert.equal(normalise("max_tokens", 128.7), 129);
});

test("garbage falls back to the default instead of NaN", () => {
  assert.equal(normalise("temperature", undefined), 0.8);
  assert.equal(normalise("temperature", null), 0.8);
  assert.equal(normalise("temperature", "abc"), 0.8);
  assert.equal(normalise("top_k", ""), 40);
  assert.equal(normalise("max_tokens", Infinity), 128);   // non-finite -> default, not max
});

test("parseParams maps to the engine's option names", () => {
  const { params, options } = parseParams({
    max_tokens: 32, temperature: 0, top_p: 0.5, top_k: 7, repetition_penalty: 1.5,
  });
  assert.deepEqual(params, {
    max_tokens: 32, temperature: 0, top_p: 0.5, top_k: 7, repetition_penalty: 1.5,
  });
  assert.deepEqual(options, {
    maxTokens: 32, temperature: 0, topK: 7, topP: 0.5, repetitionPenalty: 1.5,
  });
});

test("parseParams ignores unknown keys and fills absent ones", () => {
  const { params } = parseParams({ nonsense: 1, top_k: "12" });
  assert.equal(params.top_k, 12);           // numeric strings are accepted
  assert.equal(params.nonsense, undefined);
  assert.equal(params.temperature, 0.8);
});

test("the engine really consumes every option we advertise", async () => {
  // guards against a slider that exists in the UI but is dropped before inference
  const { loadModel } = await import("../runtime/src/model.mjs");
  const model = loadModel(new URL("../runtime/model", import.meta.url).pathname);
  const ids = model.tok.encode("It is a truth");

  const seeded = (seed) => { let s = seed; return () => { s = (s * 1103515245 + 12345) & 0x7fffffff; return s / 0x7fffffff; }; };
  const run = (opts) => model.createSession().generate(ids, 24, { rnd: seeded(7), ...opts });

  const greedy = run({ temperature: 0, topK: 40, topP: 1, repetitionPenalty: 1 });
  const greedyTopK1 = run({ temperature: 0, topK: 1, topP: 1, repetitionPenalty: 1 });
  assert.deepEqual(greedy, greedyTopK1, "greedy decode should agree with top-k=1");

  const pen1 = run({ temperature: 1, topK: 100, topP: 1, repetitionPenalty: 1 });
  const pen2 = run({ temperature: 1, topK: 100, topP: 1, repetitionPenalty: 2 });
  assert.notDeepEqual(pen1, pen2, "repetition penalty has no effect on sampling");

  const nucleus = run({ temperature: 1, topK: 100, topP: 0.05 });
  const full = run({ temperature: 1, topK: 100, topP: 1 });
  assert.notDeepEqual(nucleus, full, "top-p has no effect on sampling");
});
