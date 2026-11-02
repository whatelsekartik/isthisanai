#!/usr/bin/env node
/**
 * Benchmark harness: cold start, prefill, steady-state decode, memory, state size.
 *   node runtime/src/bench.mjs [--dir runtime/model] [--n 64] [--json out.json]
 */
import fs from "node:fs";
import { loadModel } from "./model.mjs";

const args = process.argv.slice(2);
const flag = (n, d) => { const i = args.indexOf(`--${n}`); return i >= 0 ? args[i + 1] : d; };
const dir = flag("dir", process.env.MODEL_DIR || new URL("../model", import.meta.url).pathname);
const n = Number(flag("n", 64));
const prompt = "It is a truth universally acknowledged, that a single man in possession of a good fortune";

const tBoot = performance.now();
const model = loadModel(dir);
const loadMs = performance.now() - tBoot;
const session = model.createSession();
const ids = model.tok.encode(prompt);

// warm up (touches every weight page)
session.generate(ids.slice(0, 4), 4, { topK: 1 });

const s2 = model.createSession();
const tPrefill = performance.now();
let logits = null;
for (const id of ids) logits = s2.step(id);
const prefillMs = performance.now() - tPrefill;

const s3 = model.createSession();
for (const id of ids) s3.step(id);
const tDec = performance.now();
for (let i = 0; i < n; i++) {
  let best = 0, bv = -Infinity;
  for (let k = 0; k < logits.length; k++) if (logits[k] > bv) { bv = logits[k]; best = k; }
  logits = s3.step(best);
}
const decMs = performance.now() - tDec;

// context sweep: cost per token should be flat for Cadence (fixed state) and grow for
// the transformer (KV cache). This is the claim "O(1) per token", measured.
// the transformer's runtime has a fixed 256-token window (its position table is trained to
// seq_len), so its sweep must stay inside it; Cadence has no window at all.
const positions = model.arch === "cadence" ? [0, 512, 2048] : [0, 64, 200];
const sweep = [];
for (const startAt of positions) {
  const s4 = model.createSession();
  const dummy = model.tok.encode("the ")[0] ?? 0;
  for (let i = 0; i < startAt; i++) s4.step(dummy);
  const t = performance.now();
  const M = 32;
  for (let i = 0; i < M; i++) s4.step(dummy);
  sweep.push({ position: startAt, ms_per_token: +((performance.now() - t) / M).toFixed(3) });
}

const mem = process.memoryUsage();
const out = {
  arch: model.arch,
  params: model.info(),
  weights_mb: +(model.info().weight_bytes / 1e6).toFixed(2),
  file_mb: +(model.info().file_bytes / 1e6).toFixed(2),
  cold_start_ms: +loadMs.toFixed(1),
  prefill_tokens: ids.length,
  prefill_ms: +prefillMs.toFixed(1),
  decode_tokens: n,
  decode_ms: +decMs.toFixed(1),
  ms_per_token: +(decMs / n).toFixed(2),
  tokens_per_second: +(1000 * n / decMs).toFixed(1),
  rss_mb: +(mem.rss / 1e6).toFixed(1),
  heap_mb: +(mem.heapUsed / 1e6).toFixed(1),
  state_bytes: model.arch === "cadence"
    ? model.cfg.n_blocks * model.cfg.n_loops * model.H * model.dh * model.dh * 4
    : model.cfg.n_layers * model.cfg.seq_len * model.d * 4 * 2,
  node: process.version,
  platform: `${process.platform}/${process.arch}`,
  cpus: (await import("node:os")).cpus().length,
  context_sweep: sweep,
};
console.log(JSON.stringify(out, null, 1));
const j = flag("json");
if (j) fs.writeFileSync(j, JSON.stringify(out, null, 1));
