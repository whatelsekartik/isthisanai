/**
 * Token sampling: temperature, top-k, top-p (nucleus) and repetition penalty.
 *
 * Kept in its own module so the tests can exercise the exact code the runtime uses, and so
 * `model.mjs` stays about arithmetic rather than about decoding policy.
 *
 * All four controls are real filters applied in the documented order:
 *
 *   1. repetition penalty  — logits of tokens already present are pushed down
 *   2. temperature         — 0 means greedy (no randomness at all)
 *   3. top-k               — keep the k highest logits
 *   4. top-p               — keep the smallest prefix whose cumulative mass reaches p
 *
 * Defaults (`topP = 1`, `repetitionPenalty = 1`) are no-ops, so existing callers get exactly
 * the distribution they got before these options existed.
 */

/** Push down the logits of tokens that already appear in `recentIds`. */
export function applyRepetitionPenalty(logits, recentIds, penalty) {
  if (!(penalty > 1) || !recentIds || recentIds.length === 0) return logits;
  const seen = new Set();
  for (const id of recentIds) if (id >= 0 && id < logits.length) seen.add(id);
  for (const id of seen) {
    const v = logits[id];
    logits[id] = v > 0 ? v / penalty : v * penalty;
  }
  return logits;
}

/** Indices of the tokens kept by top-k / top-p, in descending logit order. */
export function candidateIndices(logits, topK, topP) {
  const n = logits.length;
  const k = Math.max(1, Math.min(topK || n, n));
  // partial selection keeps this O(n log k) for the common k=40, n=2304 case
  const idx = [];
  for (let i = 0; i < n; i++) {
    const v = logits[i];
    if (!Number.isFinite(v)) continue;
    if (idx.length < k) {
      idx.push(i);
      if (idx.length === k) idx.sort((a, b) => logits[b] - logits[a]);
    } else if (v > logits[idx[idx.length - 1]]) {
      idx[idx.length - 1] = i;
      // re-insert into the sorted tail
      for (let j = idx.length - 1; j > 0 && logits[idx[j]] > logits[idx[j - 1]]; j--) {
        const t = idx[j]; idx[j] = idx[j - 1]; idx[j - 1] = t;
      }
    }
  }
  if (idx.length < k) idx.sort((a, b) => logits[b] - logits[a]);
  if (!(topP > 0) || topP >= 1) return idx;

  // nucleus: smallest prefix whose normalised mass reaches topP
  let max = -Infinity;
  for (const i of idx) if (logits[i] > max) max = logits[i];
  let total = 0;
  const w = new Array(idx.length);
  for (let j = 0; j < idx.length; j++) {
    w[j] = Math.exp(logits[idx[j]] - max);
    total += w[j];
  }
  let acc = 0;
  let keep = idx.length;
  for (let j = 0; j < idx.length; j++) {
    acc += w[j] / total;
    if (acc >= topP) { keep = j + 1; break; }
  }
  return idx.slice(0, keep);
}

/**
 * Draw one token id.
 * @param {ArrayLike<number>} logits
 * @param {number} n           vocabulary size
 * @param {object} opts        {temperature, topK, topP, rnd}
 */
export function sampleToken(logits, n, { temperature = 0.8, topK = 40, topP = 1, rnd = Math.random } = {}) {
  // temperature 0 is "no sampling": take the argmax of everything, not of the kept set
  if (!(temperature > 0)) {
    let best = 0, bv = -Infinity;
    for (let i = 0; i < n; i++) {
      if (logits[i] > bv) { bv = logits[i]; best = i; }
    }
    return best;
  }
  const keep = candidateIndices(logits, topK, topP);
  const inv = 1 / temperature;
  let max = -Infinity;
  for (const i of keep) if (logits[i] > max) max = logits[i];
  let sum = 0;
  const p = new Float64Array(keep.length);
  for (let j = 0; j < keep.length; j++) {
    p[j] = Math.exp((logits[keep[j]] - max) * inv);
    sum += p[j];
  }
  let u = rnd() * sum;
  for (let j = 0; j < keep.length; j++) {
    u -= p[j];
    if (u <= 0) return keep[j];
  }
  return keep[keep.length - 1];
}
