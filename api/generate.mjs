/**
 * POST /api/generate
 *
 *   body: { messages?, prompt?, max_tokens, temperature, top_p, top_k, repetition_penalty, stream? }
 *
 * Two response shapes, both supported by the same engine:
 *
 *   stream: true  -> text/event-stream, one JSON event per line
 *                    {type:"start", params, meta} · {type:"token", text} … {type:"done", stats}
 *                    (an `{type:"error"}` event ends the stream if generation fails)
 *   otherwise     -> one JSON object, exactly as before this change
 *
 * The non-streaming shape is kept byte-compatible because the CLI, the parity harness and
 * the existing tests all depend on it.
 */
import { getModel, readBody, json } from "../lib/api.mjs";
import { parseParams } from "../lib/params.mjs";

export const config = { maxDuration: 60 };

const MAX_PROMPT_CHARS = 4000;

/** UTF-8-safe incremental decoder: a token can end mid-codepoint. */
function createDecoder(tok) {
  const dec = new TextDecoder("utf-8");
  let pending = [];
  return {
    push(id) {
      const bytes = tok.bytesOf(id);
      if (!bytes || bytes.length === 0) return "";
      pending = pending.concat(bytes);
      const text = dec.decode(new Uint8Array(pending), { stream: true });
      pending = [];
      return text;
    },
    flush() {
      const text = dec.decode(new Uint8Array(pending), { stream: true }) + dec.decode();
      pending = [];
      return text;
    },
  };
}

function promptFrom(body) {
  if (Array.isArray(body.messages)) {
    return body.messages
      .filter((m) => m && typeof m.content === "string")
      .map((m) => (m.role === "assistant" ? `\nCadence: ${m.content}\n` : m.content))
      .join("\n")
      .slice(-MAX_PROMPT_CHARS);
  }
  return String(body.prompt ?? "");
}

export default function handler(req, res) {
  if (req.method === "OPTIONS") return json(res, 204, {});
  if (req.method !== "POST") return json(res, 405, { error: "POST only" });

  let body;
  try {
    body = readBody(req);
  } catch {
    return json(res, 400, { error: "invalid JSON body" });
  }

  const prompt = promptFrom(body);
  const { params, options } = parseParams(body);
  if (!prompt.trim()) return json(res, 400, { error: "prompt is empty" });

  const stream = body.stream === true;

  let model, loadMs, cold;
  try {
    ({ model, loadMs, cold } = getModel());
  } catch (err) {
    return json(res, 500, { error: String((err && err.message) || err) });
  }

  const ids = model.tok.encode(prompt);
  const stateBytes = model.arch === "cadence"
    ? model.cfg.n_blocks * model.cfg.n_loops * model.H * model.dh * model.dh * 4
    : model.cfg.n_layers * model.cfg.seq_len * model.d * 8;
  const meta = {
    arch: model.arch,
    model_load_ms: loadMs,
    cold_start: cold,
    prompt_tokens: ids.length,
    params,
    weights_mb: +(model.info().weight_bytes / 1e6).toFixed(2),
    state_bytes: stateBytes,
  };

  if (!stream) {
    try {
      const t0 = performance.now();
      const session = model.createSession();
      const out = session.generate(ids, options.maxTokens, options);
      const ms = performance.now() - t0;
      return json(res, 200, {
        prompt,
        completion: model.tok.decode(out),
        prompt_tokens: ids.length,
        tokens_generated: out.length,
        model_load_ms: loadMs,
        cold_start: cold,
        params,
        generate_ms: +ms.toFixed(0),
        ms_per_token: +(ms / Math.max(1, out.length)).toFixed(2),
        tokens_per_second: +(1000 * out.length / Math.max(1e-6, ms)).toFixed(1),
        weights_mb: meta.weights_mb,
        state_bytes: stateBytes,
        rss_mb: +(process.memoryUsage().rss / 1e6).toFixed(1),
        arch: model.arch,
      });
    } catch (err) {
      return json(res, 500, { error: String((err && err.message) || err) });
    }
  }

  // ---------------------------------------------------------------- streaming (SSE)
  res.statusCode = 200;
  res.setHeader("content-type", "text/event-stream; charset=utf-8");
  res.setHeader("cache-control", "no-store, no-transform");
  res.setHeader("connection", "keep-alive");
  res.setHeader("x-accel-buffering", "no");
  if (typeof res.flushHeaders === "function") res.flushHeaders();

  let closed = false;
  const onClose = () => { closed = true; };
  req.on?.("close", onClose);
  res.on?.("close", onClose);

  const send = (obj) => {
    if (closed) return false;
    try {
      res.write(`data: ${JSON.stringify(obj)}\n\n`);
      return true;
    } catch {
      closed = true;
      return false;
    }
  };

  const t0 = performance.now();
  let count = 0;
  let bytes = 0;
  const decoder = createDecoder(model.tok);

  try {
    send({ type: "start", ...meta });
    const session = model.createSession();
    session.generate(ids, options.maxTokens, {
      ...options,
      onToken(id) {
        const text = decoder.push(id);
        count++;
        if (text) { bytes += text.length; send({ type: "token", text }); }
        // stop as soon as the client goes away; the state is dropped with the session
        return !closed;
      },
    });
    const tail = decoder.flush();
    if (tail) send({ type: "token", text: tail });
    const ms = performance.now() - t0;
    send({
      type: "done",
      tokens_generated: count,
      stopped: closed,
      generate_ms: +ms.toFixed(0),
      ms_per_token: +(ms / Math.max(1, count)).toFixed(2),
      tokens_per_second: +(1000 * count / Math.max(1e-6, ms)).toFixed(1),
      chars: bytes,
    });
  } catch (err) {
    send({ type: "error", error: String((err && err.message) || err) });
  } finally {
    req.off?.("close", onClose);
    res.off?.("close", onClose);
    if (!res.writableEnded) res.end();
  }
}
