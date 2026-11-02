# The inference runtime: int8, integer recurrence, one core

Design goal: **no Python, no PyTorch, no GPU, no BLAS** — one file of weights and a few
hundred lines of JavaScript that run anywhere Node 20+ runs (Vercel functions, a laptop, a
browser). Targets: < 10 MB of weights, O(1) memory per token, deterministic output.

## 1. Weight format (`runtime/model/model.bin`)

```
"CDN1"                       magic
u32                          header length
JSON header                  {"arch", "cfg", "vocab", "tensors":[{name,dtype,shape,offset,nbytes,scales}], "quant"}
tensor blobs                 8-byte aligned; int8 blobs are followed by their fp32 row scales
```

* Every 2-D weight matrix is **int8, symmetric, one fp32 scale per output row**.
* Everything else (biases, RMSNorm gains, per-loop adapters, per-head output scales, the
  surprise gain, band bounds) stays fp32 — that is ~2 % of the bytes.
* Tensors are memory-mapped straight out of one `Buffer` (`new Int8Array(buf.buffer, off,
  len)`), so loading is a `readFileSync` plus a header parse: no per-tensor copies, no
  dequantisation pass, no warm-up.

## 2. Arithmetic

Per matvec: activations are quantised to int16 on a **power-of-two** grid
(`e = ceil(log2(16384/max|x|))`), weights are int8 (or fp32 for small tensors — the loader
carries a `scales = null` flag and `matvec` takes the fp32 branch once per call),
accumulation is in doubles (exact for these magnitudes), and the output is
`acc · 2^{-e} · scale[row]`. Power-of-two activation
scales make the dequantisation a multiply by an exact power of two, which keeps the
error budget tight and lets the integer state path share the same trick.

The recurrence (the interesting part), per head, per token:

```
a      = round(lo_h + (hi_h - lo_h) · σ(gate_logit))       // 0..15, integer
S      = (S + 2^{a-1}) >> a                                 // exact, round-to-nearest
e      = σ(erase) · k̂        → int16, scale 2^eE
r      = (Σ_i S[i,v] · e_i) · 2^{-(16+eE)}                  // S is stored as value * 2^16
delta  = σ(write)·v - r      → int16, scale 2^eD
S     += (k̂_int · delta_int) >> (eK + eD - 16)              // one shift, amount from scales
y      = (Σ_i S[i,v] · q̂_i) · 2^{-(16+eQ)} · SiLU(o) · head_scale
```

Notes:

* **The decay is a bit shift.** `(S + 2^{a-1}) >> a` is exact integer division by `2^a`
  with round-to-nearest by construction; no multiply, no lookup table, no float, and the
  same result on every CPU (JS `>>` is a 32-bit arithmetic shift).
* The state is `Int32Array` with a **fixed** scale `2^-16`, so there is no per-step
  re-scaling and no accumulated scale drift. The resolution is not cosmetic: at `2^-10`
  each rank-1 write is quantised to 0.06 in value units (≈6 % per token) and the runtime
  drifts away from the fp32 model *even with perfectly correct code*.
* The rank-1 write is also an integer shift (`s` computed from the two activation
  exponents). All quantities stay inside int32 except the accumulation, which is done in
  doubles (exactly representable, since every term is an integer < 2^31).
* State per session: `depth · H · d_h² · 4` bytes = 6 · 4 · 48² · 4 = **221,184 B**, plus a
  3-token conv window and one float per head. Independent of context length.
* Values are clamped to `±2^30` on write; with the state at `2^-16` that is a value range of
  ±16 384, far outside anything the trained model produces, so the clamp never fires and
  the arithmetic cannot wrap.

## 3. Streaming and session isolation

`CadenceSession` owns every mutable array. A request creates a session, feeds the prompt,
samples, and drops it; two concurrent requests cannot share state. The transformer baseline
in the same file keeps an fp32 KV cache (for the comparison table only).

Streaming decode carries two things per layer: the state matrix and the last `K-1`
pre-activation q,k,v rows for the depthwise conv. Both are part of the session, which is why
"feed one token at a time" reproduces teacher forcing to 1e-16
(`tests/test_equivalence.py::test_streaming_equals_teacher_forcing`).

## 4. Why not Rust/WASM?

* The sandbox has no Rust toolchain and `static.rust-lang.org` is unreachable, so a Rust
  implementation could not be built or benchmarked here — shipping unmeasured code would
  violate the project's own rule.
* WASM SIMD would help the matvec (128-bit `i32x4.dot_i16x8_s` for int16, or widening
  int8 to int16 and using the same instruction): the scalar JS loop is
  ~1 int8 MAC per iteration with bounds checks, whereas a SIMD kernel does 8-16 per
  instruction, so a 3-6× speedup is plausible **and unmeasured**.
* The integer recurrence would not benefit at all: it is dominated by the `dh²` state
  updates in a straight-line loop with no vectorisable width.

The honest statement in the report is therefore: JS numbers are measured, the WASM numbers
are an estimate from first principles and are labelled as such.

## 5. Files

| file | role |
|---|---|
| `runtime/src/model.mjs` | loader, tokenizer, Cadence session, transformer session |
| `runtime/src/generate.mjs` | CLI sampler |
| `runtime/src/bench.mjs` | CLI benchmark (cold start, prefill, decode, memory) |
| `runtime/src/parity.mjs` | compares int8 JS logits against the fp32 PyTorch reference (cosine 1.00000, mean abs error 0.007) |
| `api/*.mjs` | Vercel function handlers (`/api/generate`, `/api/bench`, `/api/health`) |
| `public/index.html` | the page that calls those handlers |
| `dev.mjs` | local stand-in for `vercel dev` |
