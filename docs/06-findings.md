# Findings, limitations, and what to do next

This is the paper's "discussion" section: what the measurements actually support, what they
do not, and the order in which the open questions should be attacked.

## 1. What was learned

### 1.1 The decay is the last floating-point operation in the recurrence, and fixing that is cheap

The literature quantises weights, activations, and states (Jacob et al. 1712.05877,
LLM.int8, QS4D 2507.06079, iRNN 2022, ²GRU 2026), but the *state transition* of a gated
linear-attention LM — the per-token multiplicative decay — is still an fp multiply against a
sigmoid. Constraining the exponent to an integer grid

```
a_t = round(lo_h + (hi_h − lo_h)·σ(g_t)) ∈ {0..15},   α_t = 2^{−a_t}
```

turns the whole recurrence into integer arithmetic: `S = (S + 2^{a−1}) >> a`, a rank-1
integer write, and two integer dot products. The cost of the constraint is measured, not
hoped for: the model reaches val ppl **69.5** at 2.76 M tokens and its int8 deployment costs
**0.04 % perplexity** (fp32 73.89 → int8 73.93 on the export batch set).

The insight worth keeping: *exactness changes what "quantised" means.* Post-training
quantisation of weights is lossy by construction and needs correction (QAT, outlier
handling). A dyadic transition is lossless **by definition** — the only rounding is in the
2^{-e} scaling of activations, and that is shared with the matmuls. This is why the int8
runtime reproduces the fp32 model to a logit cosine of 1.00000 rather than "close enough".

### 1.2 The expensive part of a linear-attention LM on a CPU is not arithmetic, it is depth

At `d=192, H=4, d_h=48` a delta-rule block costs `d_h²` state traffic per head per token —
about 4 % of its MACs — but that work is *serial*: each token's state depends on the
previous token's, so it cannot be batched, vectorised across the token axis, or hidden by
the matmuls around it. Weight tying helps for the opposite reason: it makes depth cheap in
*bytes* while leaving it expensive in *serial steps*.

Measured consequence: Cadence applies six block computations per token (2 weight sets ×
3 loops) and costs 4.8–5.5 ms/token in JavaScript, versus 2.0–2.3 ms/token for a
3-layer transformer of the same width in the same runtime. Cadence is ~2.3× slower per
token, and the reason is legible in the parameter/MAC counts, not mysterious.

So the honest claim is not "faster". It is:

| property | Cadence | Transformer |
|---|---|---|
| cost per token as context grows | flat (5.06 → 5.32 ms, position 0 → 2048) | grows (2.00 → 2.26 ms inside a 256-token window) |
| session state | 216 KB, constant | 1.15 MB and context-dependent |
| context limit | none (fixed state) | 256 tokens (trained window) |
| val ppl at equal tokens | 69.5 | 107.4 (with 13 % more parameters) |
| val ppl at equal *training time* | 69.5 (1,201 s, 2.76 M tokens) | **56.3** (1,071 s, 13.8 M tokens) |

The last row is the uncomfortable one and it is the reason the equal-token comparison above
is not the headline: on CPU the transformer consumes tokens ~5× faster (batched matmul
versus an elementwise scan), so given the same wall-clock it sees 5× the data and ends up
with the better perplexity. Cadence's case therefore cannot be "better model per token of
training" — it has to be the inference contract in the rows above, which is what the runtime
delivers.

### 1.3 Chunk-parallel training is where per-channel gating dies

The WY/triangular-solve form of the delta rule is what makes training affordable on a CPU.
Adding per-channel decay to it produces terms `2^{c_{t,i} − c_{s,i}}` that are rank-`d_h`
tensors rather than scalars — the cheap diagonal algebra becomes a diagonal-plus-low-rank
problem, which is exactly why KDA (2510.26692) had to build a bespoke DPLR solver. Trying to
keep the cheap form with per-channel rates produced `exp2(+400)` and NaN in this codebase
before I moved to per-head scalars (measured, in `docs/02 §5.4`).

**Per-head retention is therefore a real capability loss**, taken deliberately to keep
training in fp32 and the runtime a single shift. It is the most obvious thing to improve
(see §3).

### 1.4 Numeric parity testing is not optional

The int8 runtime "worked" — it loaded, ran at 200 tokens/s, and produced plausible-looking
English — while being **materially the wrong model** (logit cosine 0.78 against fp32). Four
independent defects hid behind that:

| defect | symptom | why the smoke test missed it |
|---|---|---|
| small 2-D tensors int8-quantised but read as fp32 | 30× error in a residual branch | no exception, text still "wordy" |
| conv ring buffer `copyWithin(0,0,n)` no-op | stale activations, error growing with position | first token correct |
| position table read without its row scale | 30× error | same |
| q/k normalised per head instead of across the width | 4× error in the memory write | output still English-shaped |

The only test that caught them compared **logits** against the trainer, after which the fix
in each case was obvious. Any "we run our own model on CPU" claim without a parity number
should be treated as unverified. This project ships the number: cosine 1.00000, mean
absolute logit error 0.007, top-5 agreement 5/5 (`docs/05`).

### 1.5 Vercel is not the hard constraint; the bundle is

The function comfortably fits the limits: 1.63 MB of weights (budget ~10 MB), ~68 MB RSS
(limit 2 GB), 2.8 ms to read and parse the model against a 60 s `maxDuration`, ~200 tokens/s
on one core. Cold start is dominated by *file read + header parse*, both proportional to
artefact size, which is why the exporter refuses to store anything it does not have to
(all 11 large matrices int8, everything else fp32).

The real deployment risks are elsewhere:

* **Quality.** A 1.5 M-parameter model at ppl 70 is a demo, not a product.
* **Concurrency.** 216 KB of state per session means thousands of concurrent streams fit in
  memory, but this repo only *argues* that; it does not test it.
* **No streaming endpoint.** `/api/generate` returns the whole completion. Token-by-token
  streaming would need a streaming runtime and an edge-compatible handler.

## 2. What is *not* claimed

* Not a new primitive. The mixer is a gated delta rule (Schlag 2102.11174; Yang 2406.06484;
  GDN 2412.06464; GDN-2 2605.22791). Banded retention, the surprise gauge and depth tying are
  small design choices, not discoveries.
* Not a speed win. Cadence loses on raw tokens/s to a smaller-depth transformer in the same
  runtime (5.4 vs 2.1 ms/token at short context); it wins on *scaling* of cost with context
  (flat to 2048 tokens vs +33 % inside a 256-token window) and on state size (216 KB vs
  1.15 MB).
* Not better per unit of training time either: at equal wall-clock the transformer reaches
  ppl 56.3 against Cadence's 69.5. The equal-token win (69.5 vs 107.4) is real but narrow.
* Not a language-model quality result. 2.76 M tokens, 1.5 M parameters, one val document set,
  one seed per architecture. The perplexity comparison is a sanity check that training
  works, not a ranking of architecture families.
* Not proven for long context. The state is unbounded in principle; the corpus is not.

## 3. Next experiments, in priority order

1. **Dyadic ablation (the one that matters).** Train the same model with a free continuous
   decay `α = 2^{−a}` (no rounding, no band clamp) for the same tokens and compare val loss.
   That single number decides whether integer-exact retention is free or paid for. Designed,
   not run: it needs ~20 more minutes of the only CPU available, and the comparison is only
   meaningful at ≥ 3 seeds.
2. **Per-channel retention with a DPLR solver.** The capability left on the table by §1.3.
   Worth it only if the dyadic ablation shows the gate has headroom.
3. **Scale the corpus and the model.** 47 MB is already fetched; the binding constraint was
   time (1.2 s/step at `d=192`). A 6 M-parameter Cadence at 30 M tokens would give a
   perplexity number worth quoting.
4. **Recall benchmarks** (MQAR / needle) once ppl < 30, since associative recall is the
   actual differentiator of the delta rule over diagonal SSMs.
5. **WASM SIMD kernel** for the matvecs (int16 × int8 via `i32x4.dot_i16x8_s`). The measured
   JS throughput (~1 int8 MAC per loop iteration) suggests a 3–6× gain is available; that
   estimate is *not* measured, and no benchmark is claimed.
6. **Streaming + concurrency test** on the deployment: N parallel sessions, measure p95
   latency and memory, then decide whether a session-scoped state store is needed.
