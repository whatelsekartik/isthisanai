/**
 * Settings: validation, persistence, reset.
 *
 * DOM-free on purpose — the tests drive it with a Map-backed fake localStorage. The ranges
 * come from the server (`/api/params`), so this module only sanitises and stores.
 */

export const STORAGE_KEY = "cadence.settings.v1";

const FALLBACK = {
  max_tokens: { min: 1, max: 512, step: 1, default: 128 },
  temperature: { min: 0, max: 2, step: 0.05, default: 0.8 },
  top_p: { min: 0.05, max: 1, step: 0.01, default: 0.9 },
  top_k: { min: 1, max: 100, step: 1, default: 40 },
  repetition_penalty: { min: 1, max: 2, step: 0.01, default: 1 },
};

export function defaultsFrom(specs) {
  const out = {};
  for (const s of specs || Object.keys(FALLBACK).map((name) => ({ name, ...FALLBACK[name] }))) {
    out[s.name] = s.default;
  }
  return out;
}

/** Clamp to range, snap to the step grid, drop anything unknown. */
export function sanitise(specs, values) {
  const out = defaultsFrom(specs);
  const list = specs || Object.keys(FALLBACK).map((name) => ({ name, ...FALLBACK[name] }));
  for (const spec of list) {
    const raw = values ? values[spec.name] : undefined;
    if (raw === undefined || raw === null || raw === "") continue;
    let v = Number(raw);
    if (!Number.isFinite(v)) continue;
    v = Math.min(spec.max, Math.max(spec.min, v));
    const steps = Math.round((v - spec.min) / spec.step);
    v = +(spec.min + steps * spec.step).toFixed(4);
    out[spec.name] = Math.min(spec.max, Math.max(spec.min, v));
  }
  return out;
}

export function loadSettings(storage, specs) {
  let raw = null;
  try {
    raw = storage && storage.getItem(STORAGE_KEY);
  } catch {
    raw = null;
  }
  if (!raw) return sanitise(specs, null);
  try {
    return sanitise(specs, JSON.parse(raw));
  } catch {
    return sanitise(specs, null);
  }
}

export function saveSettings(storage, settings) {
  try {
    storage && storage.setItem(STORAGE_KEY, JSON.stringify(settings));
    return true;
  } catch {
    return false; // private mode / quota: settings simply do not persist
  }
}

export function formatValue(name, value) {
  if (name === "max_tokens" || name === "top_k") return String(Math.round(value));
  if (name === "temperature" || name === "top_p") return value.toFixed(2);
  return value.toFixed(2);
}
