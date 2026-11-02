/** POST /api/generate  {prompt, max_tokens, temperature, top_k} -> {text, timings} */
import { getModel, readBody, json } from "../lib/api.mjs";

export const config = { maxDuration: 60 };

export default function handler(req, res) {
  if (req.method === "OPTIONS") return json(res, 204, {});
  if (req.method !== "POST") return json(res, 405, { error: "POST only" });
  let body;
  try {
    body = readBody(req);
  } catch {
    return json(res, 400, { error: "invalid JSON body" });
  }
  const prompt = String(body.prompt ?? "It is a truth").slice(0, 2000);
  const maxTokens = Math.max(1, Math.min(200, Number(body.max_tokens ?? 64)));
  const temperature = Math.max(0.01, Math.min(2, Number(body.temperature ?? 0.8)));
  const topK = Math.max(0, Math.min(200, Number(body.top_k ?? 40)));

  try {
    const { model, loadMs, cold } = getModel();
    const t0 = performance.now();
    const session = model.createSession();
    const ids = model.tok.encode(prompt);
    const tTok = performance.now();
    const out = session.generate(ids, maxTokens, { temperature, topK });
    const tGen = performance.now();

    const mem = process.memoryUsage();
    json(res, 200, {
      prompt,
      completion: model.tok.decode(out),
      prompt_tokens: ids.length,
      tokens_generated: out.length,
      model_load_ms: loadMs,
      cold_start: cold,
      tokenize_ms: +(tTok - t0).toFixed(2),
      generate_ms: +(tGen - tTok).toFixed(0),
      ms_per_token: +((tGen - tTok) / Math.max(1, out.length)).toFixed(2),
      tokens_per_second: +(1000 * out.length / Math.max(1e-6, tGen - tTok)).toFixed(1),
      weights_mb: +(model.info().weight_bytes / 1e6).toFixed(2),
      state_bytes: model.arch === "cadence"
        ? model.cfg.n_blocks * model.cfg.n_loops * model.H * model.dh * model.dh * 4
        : model.cfg.n_layers * model.cfg.seq_len * model.d * 8,
      rss_mb: +(mem.rss / 1e6).toFixed(1),
      arch: model.arch,
    });
  } catch (err) {
    json(res, 500, { error: String(err && err.message || err) });
  }
}
