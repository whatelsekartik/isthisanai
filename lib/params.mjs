/**
 * Generation parameters: one place that defines the ranges, the defaults, and the clamping.
 *
 * The UI renders its sliders from the same table (`/api/params`), so a control can never
 * exist that the engine ignores, and a slider can never send a value the server silently
 * rewrites.
 */

export const PARAMS = [
  { name: "max_tokens", label: "Max tokens", min: 1, max: 512, step: 1, default: 128, api: "max_tokens" },
  { name: "temperature", label: "Temperature", min: 0, max: 2, step: 0.05, default: 0.8, api: "temperature" },
  { name: "top_p", label: "Top-p", min: 0.05, max: 1, step: 0.01, default: 0.9, api: "top_p" },
  { name: "top_k", label: "Top-k", min: 1, max: 100, step: 1, default: 40, api: "top_k" },
  { name: "repetition_penalty", label: "Repetition penalty", min: 1, max: 2, step: 0.01, default: 1.0, api: "repetition_penalty" },
];

export const DEFAULTS = Object.freeze(
  Object.fromEntries(PARAMS.map((p) => [p.name, p.default])),
);

/** Snap a value to the parameter's grid and clamp it to its range. */
export function normalise(name, value) {
  const spec = PARAMS.find((p) => p.name === name);
  if (!spec) throw new Error(`unknown parameter ${name}`);
  let v = typeof value === "number" ? value : parseFloat(value);
  if (!Number.isFinite(v)) v = spec.default;
  v = Math.min(spec.max, Math.max(spec.min, v));
  const steps = Math.round((v - spec.min) / spec.step);
  return Math.min(spec.max, Math.max(spec.min, +(spec.min + steps * spec.step).toFixed(4)));
}

/**
 * Turn a request body into engine options.
 * Returns `{ options, params }` where `params` is what was actually used (for the response
 * and the UI), so the client can show "you asked for X, we ran Y" without guessing.
 */
export function parseParams(body = {}) {
  const params = {};
  for (const spec of PARAMS) {
    const raw = body[spec.api] ?? body[spec.name];
    params[spec.name] = normalise(spec.name, raw === undefined ? spec.default : raw);
  }
  return {
    params,
    options: {
      maxTokens: params.max_tokens,
      temperature: params.temperature,
      topK: params.top_k,
      topP: params.top_p,
      repetitionPenalty: params.repetition_penalty,
    },
  };
}
