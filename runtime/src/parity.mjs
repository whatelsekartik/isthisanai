#!/usr/bin/env node
/**
 * Compare the int8 JS runtime against the fp32 reference forward pass.
 *   python -m tools.dump_reference --out runs/reference.json
 *   node runtime/src/parity.mjs runs/reference.json [--dir runtime/model] [--json out]
 */
import fs from "node:fs";
import { loadModel } from "./model.mjs";

const args = process.argv.slice(2);
const refPath = args[0] || "runs/reference.json";
const flag = (n, d) => { const i = args.indexOf(`--${n}`); return i >= 0 ? args[i + 1] : d; };
const dir = flag("dir", process.env.MODEL_DIR || new URL("../model", import.meta.url).pathname);

const ref = JSON.parse(fs.readFileSync(refPath, "utf8"));
const model = loadModel(dir);
const s = model.createSession();
let logits = null;
for (const id of ref.ids) logits = s.step(id);
const js = Array.from(logits), py = ref.logits;

let maxAbs = 0, sumAbs = 0, dot = 0, nj = 0, np = 0;
for (let i = 0; i < py.length; i++) {
  const d = Math.abs(js[i] - py[i]);
  maxAbs = Math.max(maxAbs, d); sumAbs += d;
  dot += js[i] * py[i]; nj += js[i] * js[i]; np += py[i] * py[i];
}
const cos = dot / Math.sqrt(nj * np);
const jsTop = js.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0]).slice(0, 5).map((x) => x[1]);
const overlap = jsTop.filter((i) => ref.top5.includes(i)).length;
const out = {
  arch: model.arch, prompt: ref.prompt, ckpt_step: ref.ckpt_step,
  logit_mean_abs_diff: +(sumAbs / py.length).toFixed(4),
  logit_max_abs_diff: +maxAbs.toFixed(4),
  logit_cosine: +cos.toFixed(5),
  top1_match: jsTop[0] === ref.top5[0],
  top5_overlap: `${overlap}/5`,
  js_top5: jsTop.map((i) => model.tok.decode([i])),
  py_top5: ref.top5.map((i) => model.tok.decode([i])),
};
console.log(JSON.stringify(out, null, 1));
const j = flag("json");
if (j) fs.writeFileSync(j, JSON.stringify(out, null, 1));
