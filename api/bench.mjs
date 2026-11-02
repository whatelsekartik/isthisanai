/** GET /api/bench -> measured numbers from the process that serves this request. */
import os from "node:os";
import { getModel, json } from "../lib/api.mjs";

export const config = { maxDuration: 60 };

let cached = null;

export default function handler(req, res) {
  if (cached) return json(res, 200, { ...cached, cached: true });
  try {
    const t0 = performance.now();
    const { model, loadMs } = getModel();
    const bootMs = +(performance.now() - t0).toFixed(2);

    const prompt = "It is a truth universally acknowledged, that a single man in possession";
    const ids = model.tok.encode(prompt);

    const s = model.createSession();
    const tp = performance.now();
    let logits = null;
    for (const id of ids) logits = s.step(id);
    const prefillMs = performance.now() - tp;

    const s2 = model.createSession();
    for (const id of ids) s2.step(id);
    const N = 48;
    const td = performance.now();
    for (let i = 0; i < N; i++) {
      let best = 0, bv = -Infinity;
      for (let k = 0; k < logits.length; k++) if (logits[k] > bv) { bv = logits[k]; best = k; }
      logits = s2.step(best);
    }
    const decMs = performance.now() - td;
    const mem = process.memoryUsage();

    cached = {
      arch: model.arch,
      weights_mb: +(model.info().weight_bytes / 1e6).toFixed(2),
      cold_start_ms: bootMs,
      model_load_ms: loadMs,
      prefill_tokens: ids.length,
      prefill_ms: +prefillMs.toFixed(1),
      decode_tokens: N,
      decode_ms: +decMs.toFixed(1),
      ms_per_token: +(decMs / N).toFixed(2),
      tokens_per_second: +(1000 * N / decMs).toFixed(1),
      state_bytes: model.arch === "cadence"
        ? model.cfg.n_blocks * model.cfg.n_loops * model.H * model.dh * model.dh * 4
        : model.cfg.n_layers * model.cfg.seq_len * model.d * 8,
      rss_mb: +(mem.rss / 1e6).toFixed(1),
      heap_mb: +(mem.heapUsed / 1e6).toFixed(1),
      node: process.version,
      cpus: os.cpus().length,
      platform: `${process.platform}/${process.arch}`,
      quant: model.header.quant,
      ckpt_step: model.header.cfg && model.header.step,
    };
    json(res, 200, cached);
  } catch (err) {
    json(res, 500, { error: String(err && err.message || err) });
  }
}
