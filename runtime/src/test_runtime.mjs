#!/usr/bin/env node
/**
 * Runtime invariant tests (no framework, no dependencies).
 *
 *   node runtime/src/test_runtime.mjs
 *
 * 1. session isolation   — two interleaved sessions must produce exactly the outputs they
 *                          produce when run alone (no shared mutable state)
 * 2. determinism         — same seed, same text; different seed, different text
 * 3. bounded state       — the recurrent state does not grow with the number of tokens
 * 4. no NaN/Inf          — 512 tokens of greedy decoding stay finite (fixed-point overflow
 *                          or a bad shift shows up here immediately)
 */
import { loadModel } from "./model.mjs";

const dir = process.env.MODEL_DIR || new URL("../model", import.meta.url).pathname;
const model = loadModel(dir);
let failures = 0;
const check = (ok, name, detail = "") => {
  console.log(`[${ok ? "PASS" : "FAIL"}] ${name}${detail ? " — " + detail : ""}`);
  if (!ok) failures++;
};

function argmax(a) {
  let b = 0, bv = -Infinity;
  for (let i = 0; i < a.length; i++) if (a[i] > bv) { bv = a[i]; b = i; }
  return b;
}

// ---- 1. session isolation: interleave two prompts token by token -------------------
const p1 = model.tok.encode("The house was quiet and the world was calm");
const p2 = model.tok.encode("It is a truth universally acknowledged, that a single");

function runAlone(ids, n) {
  const s = model.createSession();
  const out = [];
  let lg = null;
  for (const id of ids) lg = s.step(id);
  for (let i = 0; i < n; i++) {
    const next = argmax(lg);
    out.push(next);
    lg = s.step(next);
  }
  return out;
}

const alone1 = runAlone(p1, 24);
const alone2 = runAlone(p2, 24);

const s1 = model.createSession(), s2 = model.createSession();
let l1 = null, l2 = null;
for (const id of p1) l1 = s1.step(id);
for (const id of p2) l2 = s2.step(id);
const inter1 = [], inter2 = [];
for (let i = 0; i < 24; i++) {
  const n1 = argmax(l1), n2 = argmax(l2);
  inter1.push(n1); inter2.push(n2);
  l1 = s1.step(n1); l2 = s2.step(n2);
}
check(alone1.join() === inter1.join(), "interleaved session A is unaffected by session B");
check(alone2.join() === inter2.join(), "interleaved session B is unaffected by session A");
check(alone1.join() !== alone2.join(), "different prompts give different continuations");

// ---- 2. determinism ---------------------------------------------------------------
const mk = (seed) => { let s = seed; return () => { s = (s * 1103515245 + 12345) & 0x7fffffff; return s / 0x7fffffff; }; };
// a hot temperature and no truncation, so the sampler actually has room to diverge
const gen = (seed) => model.createSession().generate(p1, 24, { temperature: 1.6, topK: 0, rnd: mk(seed) });
check(gen(1).join() === gen(1).join(), "same RNG seed -> identical tokens");
check(gen(1).join() !== gen(2).join(), "different RNG seed -> different tokens");

// ---- 3. bounded state --------------------------------------------------------------
const s3 = model.createSession();
const before = process.memoryUsage().heapUsed;
const dummy = p1[0];
for (let i = 0; i < 2048; i++) s3.step(dummy);
const after = process.memoryUsage().heapUsed;
const growthKb = (after - before) / 1024;
check(growthKb < 2048, "heap does not grow with token count",
      `${growthKb.toFixed(0)} KB after 2048 tokens`);
if (model.arch === "cadence") {
  const stateBytes = model.cfg.n_blocks * model.cfg.n_loops * model.H * model.dh * model.dh * 4;
  let live = 0;
  for (const arr of s3.S) live += arr.byteLength;
  check(live === stateBytes, "recurrent state is exactly the predicted size",
        `${live} B = ${model.cfg.n_blocks}*${model.cfg.n_loops}*${model.H}*${model.dh}^2*4`);
}

// ---- 4. numerical health over a long greedy decode ---------------------------------
const s4 = model.createSession();
let lg = null;
for (const id of p1) lg = s4.step(id);
let bad = 0;
for (let i = 0; i < 512; i++) {
  for (let k = 0; k < lg.length; k++) if (!Number.isFinite(lg[k])) { bad++; break; }
  lg = s4.step(argmax(lg));
}
check(bad === 0, "512 greedy tokens stay finite (no int32 overflow / bad shift)",
      `${bad} non-finite logit vectors`);

console.log(failures === 0 ? "\nall runtime invariants hold" : `\n${failures} FAILED`);
process.exit(failures === 0 ? 0 : 1);
