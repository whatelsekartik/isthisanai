import { getModel, json } from "../lib/api.mjs";

export default function handler(req, res) {
  try {
    const { model } = getModel();
    const mem = process.memoryUsage();
    json(res, 200, {
      ok: true,
      arch: model.arch,
      vocab: model.vocab,
      weights_bytes: model.info().weight_bytes,
      tensors: model.tensors.size,
      rss_mb: +(mem.rss / 1e6).toFixed(1),
      node: process.version,
    });
  } catch (err) {
    json(res, 500, { ok: false, error: String(err && err.message || err) });
  }
}
