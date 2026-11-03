/**
 * Zero-dependency inference runtime for the models trained in `src/`.
 *
 *   - int8 weights (per-output-row scale), int16 activations, float accumulation
 *   - Cadence: fixed-size recurrent state, *exact integer decay* (a bit shift),
 *     no KV cache, O(1) memory per token regardless of context
 *   - Transformer: standard causal attention with an fp32 KV cache (the baseline)
 *   - a Session object owns every mutable byte, so two requests never share state
 *
 * The arithmetic mirrors `CadenceMixer._scan` in src/cadence/model.py, which
 * tests/test_equivalence.py proves equals the chunk-parallel training path.
 */
import fs from "node:fs";
import { sampleToken, applyRepetitionPenalty } from "./sample.mjs";

const MAGIC = "CDN1";

/* ------------------------------------------------------------------------- loader */

export function loadModel(dir) {
  const buf = fs.readFileSync(`${dir}/model.bin`);
  if (buf.toString("utf8", 0, 4) !== MAGIC) throw new Error("bad magic");
  const hlen = buf.readUInt32LE(4);
  const header = JSON.parse(buf.toString("utf8", 8, 8 + hlen));
  const base = 8 + hlen;
  const tensors = new Map();
  for (const t of header.tensors) {
    const off = buf.byteOffset + base + t.offset;
    const data = t.dtype === "i8"
      ? new Int8Array(buf.buffer, off, t.nbytes)
      : new Float32Array(buf.buffer, off, t.nbytes / 4);
    let scales = null;
    if (t.scales >= 0) {
      scales = new Float32Array(buf.buffer, buf.byteOffset + base + t.scales, t.shape[0]);
    }
    tensors.set(t.name, { data, scales, shape: t.shape });
  }
  const tok = JSON.parse(fs.readFileSync(`${dir}/tokenizer.json`, "utf8"));
  return new Model(header, tensors, tok, buf.byteLength);
}

export class Model {
  constructor(header, tensors, tok, bytes) {
    this.header = header;
    this.tensors = tensors;
    this.cfg = header.cfg;
    this.arch = header.arch;
    this.vocab = header.vocab;
    this.bytes = bytes;
    this.tok = new Tokenizer(tok);
    this.d = this.cfg.d_model;
    this.H = this.cfg.n_heads;
    this.dh = this.d / this.H;
  }
  get(name) {
    const t = this.tensors.get(name);
    if (!t) throw new Error(`missing tensor ${name}`);
    return t;
  }
  createSession() {
    return this.arch === "cadence" ? new CadenceSession(this) : new TransformerSession(this);
  }
  info() {
    let w = 0, q = 0;
    for (const t of this.tensors.values()) {
      w += t.data.byteLength + (t.scales ? t.scales.byteLength : 0);
      if (t.data instanceof Int8Array) q++;
    }
    return { arch: this.arch, vocab: this.vocab, d_model: this.d, heads: this.H,
             weight_bytes: w, int8_tensors: q, file_bytes: this.bytes };
  }
}

/* ---------------------------------------------------------------------- tokenizer */

export class Tokenizer {
  constructor(cfg) {
    this.merges = cfg.merges;
    this.ranks = new Map();
    this.merges.forEach(([a, b], i) => this.ranks.set(a * 65536 + b, i));
    this.pat = new RegExp(cfg.pre_tokenizer, "gu");
    this.table = null;
    this.cache = new Map();
  }
  _expand() {
    const table = [];
    for (let i = 0; i < 256; i++) table.push([i]);
    for (const [a, b] of this.merges) table.push(table[a].concat(table[b]));
    this.table = table;
    return table;
  }
  encode(text) {
    const ids = [];
    const push = (word) => {
      let seq = this.cache.get(word);
      if (!seq) {
        let cur = Array.from(Buffer.from(word, "utf8"));
        for (;;) {
          let best = Infinity, pos = -1;
          for (let i = 0; i < cur.length - 1; i++) {
            const r = this.ranks.get(cur[i] * 65536 + cur[i + 1]);
            if (r !== undefined && r < best) { best = r; pos = i; }
          }
          if (pos < 0) break;
          cur = cur.slice(0, pos).concat([256 + best], cur.slice(pos + 2));
        }
        seq = cur;
        if (this.cache.size < 100000) this.cache.set(word, seq);
      }
      for (const id of seq) ids.push(id);
    };
    // NOTE: must iterate with exec (not String.match) so unmatched gaps are not dropped;
    // the Python tokenizer has the same property (see tests/test_tokenizer.py).
    this.pat.lastIndex = 0;
    let pos = 0, m;
    while ((m = this.pat.exec(text)) !== null) {
      if (m.index > pos) for (const b of Buffer.from(text.slice(pos, m.index), "utf8")) ids.push(b);
      push(m[0]);
      pos = m.index + m[0].length;
      if (m[0].length === 0) this.pat.lastIndex++;
    }
    if (pos < text.length) for (const b of Buffer.from(text.slice(pos), "utf8")) ids.push(b);
    return ids;
  }

  /** raw bytes of a single token id, for streaming decoders that must respect UTF-8 boundaries */
  bytesOf(id) {
    const table = this.table || this._expand();
    return table[id] || null;
  }

  decode(ids) {
    const table = this.table || this._expand();
    const out = [];
    for (const i of ids) {
      if (i < 0 || i >= table.length) continue;
      for (const b of table[i]) out.push(b);
    }
    return Buffer.from(out).toString("utf8");
  }
}

/* ------------------------------------------------------------------ small kernels */

const sigmoid = (v) => 1 / (1 + Math.exp(-v));
const silu = (v) => v / (1 + Math.exp(-v));

function rmsnormMul(x, w, scale, d, out, eps = 1e-6) {
  let ss = 0;
  for (let i = 0; i < d; i++) ss += x[i] * x[i];
  const r = 1 / Math.sqrt(ss / d + eps);
  for (let i = 0; i < d; i++) out[i] = x[i] * r * w[i] * scale[i];
}

/** quantise a float vector to int16 on a power-of-two grid; returns the exponent e (scale 2^e) */
function quantise(x, n, out) {
  let m = 0;
  for (let i = 0; i < n; i++) {
    const a = x[i] < 0 ? -x[i] : x[i];
    if (a > m) m = a;
  }
  if (m < 1e-12) { for (let i = 0; i < n; i++) out[i] = 0; return 0; }
  let e = Math.ceil(Math.log2(16384 / m));
  e = Math.max(-40, Math.min(40, e));
  const s = Math.pow(2, e);
  for (let i = 0; i < n; i++) {
    let v = Math.round(x[i] * s);
    if (v > 32767) v = 32767; else if (v < -32767) v = -32767;
    out[i] = v;
  }
  return e;
}

/** out = W x + b. W is int8 (row-major, one scale per row) or fp32 (no scales). */
function matvec(w, scales, bias, xq, eX, out, nOut, nIn, add) {
  const inv = Math.pow(2, -eX);
  if (scales) {
    for (let o = 0; o < nOut; o++) {
      const row = o * nIn;
      let acc = 0;
      for (let i = 0; i < nIn; i++) acc += w[row + i] * xq[i];
      let v = acc * inv * scales[o];
      if (bias) v += bias[o];
      out[o] = add ? out[o] + v : v;
    }
  } else {
    for (let o = 0; o < nOut; o++) {
      const row = o * nIn;
      let acc = 0;
      for (let i = 0; i < nIn; i++) acc += w[row + i] * xq[i];
      let v = acc * inv;
      if (bias) v += bias[o];
      out[o] = add ? out[o] + v : v;
    }
  }
  return out;
}

/* ----------------------------------------------------------------- Cadence model */

export class CadenceSession {
  constructor(m) {
    const c = m.cfg;
    this.m = m;
    this.c = c;
    this.d = m.d;
    this.H = m.H;
    this.dh = m.dh;
    this.K = c.conv_kernel;
    this.depth = c.n_blocks * c.n_loops;
    this.lo = c.bands.map((b) => b[0]);
    this.hi = c.bands.map((b) => b[1]);
    // Fixed-point constant: the state is stored as value * 2^16. That resolution matters:
    // the rank-1 write increments the state by (key * delta) >> (eK + eD - 16), so with a
    // 2^10 state the granularity would be ~0.06 in value units, and 6 % write error per
    // token compounds into a visibly different model (measured: rel 0.70 residual error).
    this.SS_LOG = 16;
    this.pos = 0;
    this.S = [];
    this.nov = [];
    this.convBuf = [];
    for (let l = 0; l < this.depth; l++) {
      this.S.push(new Int32Array(m.H * m.dh * m.dh));
      this.nov.push(new Float32Array(m.H));
      this.convBuf.push(new Float32Array(3 * m.d * (this.K - 1)));
    }
    const d = m.d, hh = c.mlp_hidden;
    this.x = new Float32Array(d);
    this.h = new Float32Array(d);
    this.p = new Float32Array(6 * d);
    this.qkv = new Float32Array(3 * d);
    this.out3 = new Float32Array(3 * d);
    this.y = new Float32Array(d);
    this.tmp = new Float32Array(Math.max(d, hh));
    this.gate = new Float32Array(m.H);
    this.xq = new Int16Array(Math.max(d, hh, 3 * d));
    this.g1 = new Float32Array(hh);
    this.g2 = new Float32Array(hh);
    this.logits = new Float32Array(m.vocab);
    this.kq = new Int16Array(m.dh);
    this.eq = new Int16Array(m.dh);
    this.qq = new Int16Array(m.dh);
    this.dq = new Int16Array(m.dh);
    this.acc = new Float64Array(m.dh);
    this.u = new Float32Array(m.dh);
    this.unit = new Float32Array(d);
    this.scaleBuf = new Float32Array(d);
  }

  step(tokId) {
    const m = this.m, c = this.c, d = this.d, H = this.H, dh = this.dh, K = this.K;
    const emb = m.get("embed.weight");
    const es = emb.scales[tokId];
    const row = tokId * d;
    for (let i = 0; i < d; i++) this.x[i] = emb.data[row + i] * es;

    if (this.trace) { this.trace.push(Array.from(this.x)); }
    for (let l = 0; l < this.depth; l++) {
      const b = l % c.n_blocks, loop = (l / c.n_blocks) | 0;
      const pre = `blocks.${b}.`;
      const mix = `${pre}mixer.`;
      const g = m.get(`${mix}gammas`).data;

      // ---- normalise, per-loop adapter, then the 6-way projection
      const n1 = m.get(`${pre}norm1.weight`).data;
      for (let i = 0; i < d; i++) this.scaleBuf[i] = 1 + g[loop * d + i];
      rmsnormMul(this.x, n1, this.scaleBuf, d, this.h);
      let sX = quantise(this.h, d, this.xq);
      const pw = m.get(`${mix}proj.weight`), pb = m.get(`${mix}proj.bias`);
      matvec(pw.data, pw.scales, pb.data, this.xq, sX, this.p, 6 * d, d, false);
      const gw = m.get(`${mix}gate.weight`), gb = m.get(`${mix}gate.bias`);
      matvec(gw.data, gw.scales, gb.data, this.xq, sX, this.gate, H, d, false);

      // ---- causal depthwise conv (kernel 4) over q,k,v, then SiLU
      const buf = this.convBuf[l];
      const K3 = 3 * d;
      const qkvNow = this.qkv;
      for (let i = 0; i < d; i++) {
        qkvNow[i] = this.p[i];
        qkvNow[d + i] = this.p[d + i];
        qkvNow[2 * d + i] = this.p[2 * d + i];
      }
      const cw = m.get(`${mix}conv.weight`), cb = m.get(`${mix}conv.bias`).data;
      // out[t] = sum_j w[j] x[t-K+1+j]: tap j is the (K-1-j)-th most recent input
      for (let ch = 0; ch < K3; ch++) {
        const sc = cw.scales ? cw.scales[ch] : 1, rowoff = ch * K;
        let acc = cb[ch];
        for (let j = 0; j < K - 1; j++) acc += cw.data[rowoff + j] * sc * buf[(K - 2 - j) * K3 + ch];
        acc += cw.data[rowoff + K - 1] * sc * qkvNow[ch];
        this.out3[ch] = silu(acc);
      }
      // shift the history window down by one token (oldest first), then insert the current
      // one at row 0. `copyWithin(0, 0, K3)` would be a no-op here — that bug left the
      // newest row frozen and made the convolution read stale activations.
      for (let j = K - 2; j >= 1; j--) buf.copyWithin(j * K3, (j - 1) * K3, j * K3);
      buf.set(qkvNow, 0);

      // ---- q,k are L2-normalised across the *full* width, matching
      // `l2norm(q, 1e-6).view(B,T,H,dh)` in src/cadence/model.py: one shared factor, so each
      // head's share of the norm is ~sqrt(dh/d) rather than 1. Normalising per head here
      // (the obvious choice, and what GDN-2 does) silently changed the model by ~4x in the
      // memory write and cost 0.99 -> 0.78 logit cosine against the fp32 reference.
      let sqAll = 0, skAll = 0;
      for (let i = 0; i < d; i++) {
        const qv = this.out3[i], kv = this.out3[d + i];
        sqAll += qv * qv; skAll += kv * kv;
      }
      const rqAll = 1 / Math.sqrt(sqAll + 1e-6);
      const rkAll = 1 / Math.sqrt(skAll + 1e-6);

      // ---- per-head memory
      const kappa = m.get(`${mix}kappa`).data[0];
      const hs = m.get(`${mix}head_scale`).data;
      for (let h = 0; h < H; h++) {
        const off = h * dh;
        const rq = rqAll, rk = rkAll;
        const S = this.S[l];
        const base = h * dh * dh;
        // retention index: dyadic, integer
        const uu = sigmoid(this.gate[h] + (c.novelty === "gate" ? kappa * this.nov[l][h] : 0));
        let a = Math.round(this.lo[h] + (this.hi[h] - this.lo[h]) * uu);
        if (a < 0) a = 0;
        const amax = c.n_dyadic - 1;
        if (a > amax) a = amax;
        // ---- exact integer decay: S <- round(S / 2^a)
        if (a > 0) {
          const add = 1 << (a - 1);
          for (let i = 0; i < dh * dh; i++) S[base + i] = (S[base + i] + add) >> a;
        }
        // ---- quantise e = sigmoid(erase) * k_hat, then k_hat, then q_hat
        for (let i = 0; i < dh; i++) {
          const kv = this.out3[d + off + i] * rk;
          this.unit[i] = sigmoid(this.p[3 * d + off + i]) * kv;
        }
        const eE = quantise(this.unit, dh, this.eq);
        for (let i = 0; i < dh; i++) this.unit[i] = this.out3[d + off + i] * rk;
        const eK = quantise(this.unit, dh, this.kq);
        for (let i = 0; i < dh; i++) this.unit[i] = this.out3[off + i] * rq;
        const eQ = quantise(this.unit, dh, this.qq);
        // ---- r = S^T e   (state scale SS * activation scale sE)
        for (let v = 0; v < dh; v++) this.acc[v] = 0;
        for (let i = 0; i < dh; i++) {
          const ei = this.eq[i], rowoff = base + i * dh;
          if (ei === 0) continue;
          for (let v = 0; v < dh; v++) this.acc[v] += S[rowoff + v] * ei;
        }
        const readScale = Math.pow(2, -(this.SS_LOG + eE));
        // ---- delta = write*v - r, quantised; then the rank-1 integer write
        for (let v = 0; v < dh; v++) {
          const uv = sigmoid(this.p[4 * d + off + v]) * this.out3[2 * d + off + v];
          const dv = uv - this.acc[v] * readScale;
          this.u[v] = dv;
          this.nov[l][h] = 0.875 * this.nov[l][h] + 0.125 * Math.abs(dv);
        }
        const eD = quantise(this.u, dh, this.dq);
        // The write increment lives in state units: S is stored as value * 2^10, the keys
        // are integers on 2^eK, the deltas on 2^eD.  So the product must be scaled by
        // 2^(10 - eK - eD): a left shift when the activations are small, a rounding right
        // shift when they are large (the usual case).  A negative left shift is a JS
        // footgun (`x << -16` shifts by 16), so both directions are handled explicitly.
        const kk = this.SS_LOG - eK - eD;
        const sh = kk < 0 ? -kk : 0;
        const mul = kk > 0 ? Math.pow(2, kk) : 0;
        const half = sh > 0 ? Math.pow(2, sh - 1) : 0;
        const div = sh > 0 ? Math.pow(2, sh) : 1;
        for (let i = 0; i < dh; i++) {
          const ki = this.kq[i], rowoff = base + i * dh;
          if (ki === 0) continue;
          for (let v = 0; v < dh; v++) {
            const p = ki * this.dq[v];
            const inc = sh > 0
              ? Math.trunc((p + (p >= 0 ? half : -half)) / div)
              : p * mul;
            let nv = S[rowoff + v] + inc;
            if (nv > 1073741823) nv = 1073741823;
            else if (nv < -1073741824) nv = -1073741824;
            S[rowoff + v] = nv;
          }
        }
        // ---- y = S^T q
        for (let v = 0; v < dh; v++) this.acc[v] = 0;
        for (let i = 0; i < dh; i++) {
          const qi = this.qq[i], rowoff = base + i * dh;
          if (qi === 0) continue;
          for (let v = 0; v < dh; v++) this.acc[v] += S[rowoff + v] * qi;
        }
        const yScale = Math.pow(2, -(this.SS_LOG + eQ));
        for (let v = 0; v < dh; v++) {
          this.y[off + v] = this.acc[v] * yScale * silu(this.p[5 * d + off + v]) * hs[h];
        }
      }

      // ---- output projection, residual
      const ow = m.get(`${mix}out.weight`);
      sX = quantise(this.y, d, this.xq);
      matvec(ow.data, ow.scales, null, this.xq, sX, this.tmp, d, d, false);
      for (let i = 0; i < d; i++) this.x[i] += this.tmp[i];

      // ---- gated MLP, residual
      const n2 = m.get(`${pre}norm2.weight`).data;
      const g2 = m.get(`${pre}mlp.gammas`).data;
      for (let i = 0; i < d; i++) this.scaleBuf[i] = 1 + g2[loop * d + i];
      rmsnormMul(this.x, n2, this.scaleBuf, d, this.h);
      sX = quantise(this.h, d, this.xq);
      const hh = c.mlp_hidden;
      const mw = m.get(`${pre}mlp.gate.weight`), uw = m.get(`${pre}mlp.up.weight`);
      matvec(mw.data, mw.scales, null, this.xq, sX, this.g1, hh, d, false);
      matvec(uw.data, uw.scales, null, this.xq, sX, this.g2, hh, d, false);
      for (let i = 0; i < hh; i++) this.g1[i] = silu(this.g1[i]) * this.g2[i];
      const sG = quantise(this.g1, hh, this.xq);
      const dw = m.get(`${pre}mlp.down.weight`);
      matvec(dw.data, dw.scales, null, this.xq, sG, this.tmp, d, hh, false);
      for (let i = 0; i < d; i++) this.x[i] += this.tmp[i];
      if (this.trace) this.trace.push(Array.from(this.x));
    }

    // ---- final norm and tied head
    const nf = m.get("norm_f.weight").data;
    for (let i = 0; i < d; i++) this.scaleBuf[i] = 1;
    rmsnormMul(this.x, nf, this.scaleBuf, d, this.h);
    const eL = quantise(this.h, d, this.xq);
    const emb2 = m.get("embed.weight");
    const V = this.m.vocab;
    const inv = Math.pow(2, -eL);
    for (let v = 0; v < V; v++) {
      const rowoff = v * d;
      let acc = 0;
      for (let i = 0; i < d; i++) acc += emb2.data[rowoff + i] * this.xq[i];
      this.logits[v] = acc * inv * emb2.scales[v];
    }
    this.pos++;
    return this.logits;
  }

  /**
   * Generate `n` tokens after `promptIds`.
   *
   * `onToken(id)` is called for every accepted token so a caller can stream; returning
   * `false` from it stops generation (used by the "stop" button).
   */
  generate(promptIds, n, {
    temperature = 0.8, topK = 40, topP = 1, repetitionPenalty = 1,
    rnd = Math.random, onToken = null,
  } = {}) {
    const out = [];
    const recent = promptIds.slice(-64);          // repetition window = last 64 context tokens
    let logits = null;
    for (const id of promptIds) logits = this.step(id);
    if (!logits) logits = this.step(0);
    for (let i = 0; i < n; i++) {
      applyRepetitionPenalty(logits, recent, repetitionPenalty);
      const id = sampleToken(logits, this.m.vocab, { temperature, topK, topP, rnd });
      out.push(id);
      recent.push(id);
      if (recent.length > 64) recent.shift();
      if (onToken && onToken(id) === false) break;
      logits = this.step(id);
    }
    return out;
  }
}

/* ------------------------------------------------- transformer baseline (fp32 KV cache) */

export class TransformerSession {
  constructor(m) {
    const c = m.cfg;
    this.m = m;
    this.c = c;
    this.d = m.d;
    this.H = m.H;
    this.dh = m.dh;
    this.depth = c.n_layers;
    this.maxLen = c.seq_len;
    this.pos = 0;
    this.kc = []; this.vc = [];
    for (let l = 0; l < this.depth; l++) {
      this.kc.push(new Float32Array(this.maxLen * m.d));
      this.vc.push(new Float32Array(this.maxLen * m.d));
    }
    const d = m.d, hh = c.mlp_hidden;
    this.x = new Float32Array(d);
    this.h = new Float32Array(d);
    this.qkv = new Float32Array(3 * d);
    this.y = new Float32Array(d);
    this.tmp = new Float32Array(Math.max(d, hh));
    this.xq = new Int16Array(Math.max(d, hh, 3 * d));
    this.g1 = new Float32Array(hh);
    this.g2 = new Float32Array(hh);
    this.logits = new Float32Array(m.vocab);
    this.scaleBuf = new Float32Array(d).fill(1);
  }

  generate(promptIds, n, {
    temperature = 0.8, topK = 40, topP = 1, repetitionPenalty = 1,
    rnd = Math.random, onToken = null,
  } = {}) {
    let logits = null;
    for (const id of promptIds) logits = this.step(id);
    const out = [];
    const recent = promptIds.slice(-64);
    for (let i = 0; i < n; i++) {
      applyRepetitionPenalty(logits, recent, repetitionPenalty);
      const id = sampleToken(logits, this.m.vocab, { temperature, topK, topP, rnd });
      out.push(id);
      recent.push(id);
      if (recent.length > 64) recent.shift();
      if (onToken && onToken(id) === false) break;
      logits = this.step(id);
    }
    return out;
  }
}
