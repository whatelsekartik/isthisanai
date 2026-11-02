#!/usr/bin/env node
/**
 * CLI sampler.
 *   node runtime/src/generate.mjs "It is a truth" 120 [--temp 0.8] [--top-k 40] [--seed 7]
 */
import { loadModel } from "./model.mjs";

const args = process.argv.slice(2);
const prompt = args[0] && !args[0].startsWith("--") ? args[0] : "It is a truth";
const n = Number(args[1] && !args[1].startsWith("--") ? args[1] : 100);
const flag = (name, dflt) => {
  const i = args.indexOf(`--${name}`);
  return i >= 0 ? Number(args[i + 1]) : dflt;
};
const dir = process.env.MODEL_DIR || new URL("../model", import.meta.url).pathname;

let seed = flag("seed", 7);
const rnd = () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff; };

const t0 = performance.now();
const model = loadModel(dir);
const loadMs = performance.now() - t0;
const session = model.createSession();
const ids = model.tok.encode(prompt);
const t1 = performance.now();
const out = session.generate(ids, n, { temperature: flag("temp", 0.8), topK: flag("top-k", 40), rnd });
const ms = performance.now() - t1;

console.log(model.tok.decode(ids.concat(out)));
console.error(`\n[${model.arch}] ${model.info().weight_bytes / 1e6} MB · load ${loadMs.toFixed(1)} ms · ` +
  `${n} tokens in ${ms.toFixed(0)} ms = ${(1000 * n / ms).toFixed(1)} tok/s ` +
  `(${(ms / n).toFixed(2)} ms/token)`);
