import test from "node:test";
import assert from "node:assert/strict";
import { sampleToken, candidateIndices, applyRepetitionPenalty } from "../runtime/src/sample.mjs";

const seeded = (seed) => { let s = seed; return () => { s = (s * 1103515245 + 12345) & 0x7fffffff; return s / 0x7fffffff; }; };
const logits = (...xs) => Float32Array.from(xs);

test("temperature 0 is greedy over the whole vocabulary", () => {
  const l = logits(-1, -1, 5, -1, 3);
  for (let i = 0; i < 50; i++) assert.equal(sampleToken(l, l.length, { temperature: 0 }), 2);
  // top-k/top-p must not change that: greedy means greedy
  assert.equal(sampleToken(l, l.length, { temperature: 0, topK: 1, topP: 0.05 }), 2);
});

test("top-k=1 is greedy and top-k=2 keeps exactly two candidates", () => {
  const l = logits(0, 4, 3, -2);
  assert.equal(sampleToken(l, l.length, { temperature: 1, topK: 1, topP: 1 }), 1);
  assert.deepEqual(candidateIndices(l, 2, 1), [1, 2]);
  assert.deepEqual(candidateIndices(l, 3, 1), [1, 2, 0]);
});

test("top-k larger than the vocabulary is harmless", () => {
  const l = logits(0, 1, 2);
  assert.equal(candidateIndices(l, 99, 1).length, 3);
  assert.deepEqual(candidateIndices(l, 99, 1), [2, 1, 0]);
});

test("top-p keeps the shortest prefix that reaches the mass", () => {
  // probabilities after softmax over these logits are close to [0.665, 0.245, 0.09]
  const l = logits(Math.log(0.665), Math.log(0.245), Math.log(0.09));
  assert.equal(candidateIndices(l, 3, 0.5).length, 1);   // 0.665 alone passes 0.5
  assert.equal(candidateIndices(l, 3, 0.7).length, 2);   // 0.665 + 0.245
  assert.equal(candidateIndices(l, 3, 0.999).length, 3);
  assert.equal(candidateIndices(l, 3, 1).length, 3);
});

test("sampling only ever returns candidates", () => {
  const l = logits(3, 2, 1, 0.5, 0);
  const rnd = seeded(11);
  const seen = new Set();
  for (let i = 0; i < 500; i++) seen.add(sampleToken(l, 5, { temperature: 1, topK: 2, topP: 1, rnd }));
  assert.deepEqual([...seen].sort(), [0, 1]);
});

test("top-p narrows the sampled set", () => {
  const l = logits(4, 3.4, 1, 0.5, 0);
  const rnd = seeded(3);
  const seen = new Set();
  for (let i = 0; i < 500; i++) seen.add(sampleToken(l, 5, { temperature: 1, topK: 5, topP: 0.6, rnd }));
  assert.ok(seen.size <= 2, `expected a narrow set, saw ${[...seen]}`);
});

test("repetition penalty pushes down the tokens it penalises", () => {
  const l = logits(2, 2.1, 0);
  const before = Array.from(l);
  applyRepetitionPenalty(l, [0], 2);
  assert.ok(l[0] < before[0], "positive logit is divided");
  assert.equal(l[1], before[1], "unseen tokens are untouched");
  assert.equal(l[2], before[2]);
  // negative logits are multiplied (pushed further down)
  const n = logits(-2, 0);
  applyRepetitionPenalty(n, [0], 2);
  assert.equal(n[0], -4);
});

test("repetition penalty of 1 is a genuine no-op", () => {
  const l = logits(1, 2, 3);
  const copy = Array.from(l);
  applyRepetitionPenalty(l, [0, 1, 2], 1);
  assert.deepEqual(Array.from(l), copy);
});

test("repetition penalty can flip the argmax", () => {
  const l = logits(3, 2.6);
  assert.equal(sampleToken(l, 2, { temperature: 0 }), 0);
  applyRepetitionPenalty(l, [0], 2);
  assert.equal(sampleToken(l, 2, { temperature: 0 }), 1);
});

test("sampling is deterministic given the same rng stream", () => {
  const l = logits(1.2, 0.9, 0.3, -0.4);
  const a = [];
  const b = [];
  const ra = seeded(5);
  const rb = seeded(5);
  for (let i = 0; i < 20; i++) a.push(sampleToken(l, 4, { temperature: 1.3, topK: 4, topP: 1, rnd: ra }));
  for (let i = 0; i < 20; i++) b.push(sampleToken(l, 4, { temperature: 1.3, topK: 4, topP: 1, rnd: rb }));
  assert.deepEqual(a, b);
});
