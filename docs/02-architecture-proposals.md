# Phase 2 — Three candidate architectures, and the one that was built

Everything in this document is derived from the cost model in `01-literature-review.md`.
Numbers are computed for the *actual* configuration used in the experiments
(`d=192, H=4, d_h=48, depth 6, vocab 2304, int8 weights`), not for a 1B-parameter model.

## 0. Notation

| symbol | meaning |
|---|---|
| `d`, `H`, `d_h = d/H` | model width, heads, head width |
| `P`, `R` | physical blocks, loop count per block; depth `= P·R` |
| `S ∈ R^{d_h×d_h}` | per-head recurrent memory (key × value) |
| `k, q, v` | projected key/query/value, `k` and `q` L2-normalised (`k̂`, `q̂`) |
| `b, w ∈ [0,1]^{d_h}` | erase gate, write gate |
| `α = 2^{-a}` | **dyadic** retention factor, `a ∈ {0..15}` |
| `nov` | surprise gauge (leaked write error) |
| `C` | chunk length for training (32) |

---

## 1. Candidate A — **Cadence**: banded dyadic delta memory, depth-tied

**State.** One `d_h×d_h` matrix per head per block-application.

**Per-token recurrence** (this is the specification the JS/WASM runtime implements):

```
[x]  a  = clamp(round(lo_h + (hi_h − lo_h)·σ(g_t) + κ·nov_t), 0, 15)   # per head, integer
     α  = 2^{−a}
     S  ← α · S                                   # == S >> a in the integer runtime
     r  = Sᵀ (b_t ⊙ k̂_t)                          # read along the key about to be erased
     S  ← S + k̂_t (w_t ⊙ v_t − r)ᵀ                # delta (error-correcting) write
     y  = Sᵀ q̂_t ⊙ SiLU(o_t)                      # read along the query
     nov ← λ·nov + (1−λ)·mean|w_t ⊙ v_t − r|      # λ = 7/8, leak of the write error
```

`lo_h, hi_h` are the fixed **bands** of head `h` (e.g. heads 0,1 ∈ {0..6} → fast memory;
heads 2,3 ∈ {0..14} → slow memory).

**Information flow.** `σ(g)` chooses *how fast this head forgets*. Because the exponent is
an integer and the base is 2, forgetting is a shift: exact, branch-free, identical on any
CPU, and needing no multiply and no lookup table. `nov` pushes the retention towards
"slower" when the head is failing to predict what it is writing — surprise buys memory.

**Parameter count (measured).** 1,561,170 total, 1,118,802 non-embedding.

**Complexity.**

| quantity | value |
|---|---|
| MACs / token / layer | 592 900 |
| MACs / token (depth 6 + head) | ≈ 4.0 M |
| weights streamed / token (int8) | 1.12 MB |
| state / session | 6·4·48²·4 B = **216 KB** (int32) |
| state MACs / token | 36 864·6 = 221 184 (5.5 % of total) |
| serial depth | 6 block applications |
| unique weight sets | 2 (streamed once if L2-resident) |

**Training.** Chunk-parallel (derivation in §4); `O(L·C·d + L·d²)` per layer.

**Why it might work.** Delta-rule memory is the strongest per-FLOP associative memory in
the literature; bands give timescale diversity for free; two physical blocks give
depth 6 at the weight-traffic cost of depth 2.

**Risks.** (i) The dyadic constraint may cost loss versus a free log-space gate;
(ii) per-head (rather than per-channel) retention is coarser than KDA/GDN-2;
(iii) delta-rule state traffic is the largest of the three candidates.

---

## 2. Candidate B — **Lattice**: low-rank factored fast weights

**Idea.** Keep the delta rule but factor the memory: `S = U Vᵀ` with `U, V ∈ R^{d_h×r}`,
`r ≤ d_h`. Reads become `Sᵀx = V(Uᵀx)` — `2·d_h·r` MACs instead of `d_h²`. Writes project
the residual into the `r`-dimensional subspace: `U ← U + η_u k̂ (Vᵀres)ᵀ`, `V ← V + η_v …`.
To stop the basis collapsing, every `m` tokens re-orthonormalise `U` (cheap modified
Gram-Schmidt, `O(d_h·r²)` amortised), or maintain `U` directly as a Householder/Stiefel
parametrisation.

**Complexity at `r = 16`.** State 2·48·16 = 1 536 words (vs 2 304) = **-33 %**; read/write
MACs 3·2·48·16 = 4 608 per head (vs 9 216) = **-50 %**; total MACs/token ≈ 3.6 M.

**Verdict.** Attractive on paper, but: (i) the rank truncation removes exactly the
directions the delta rule uses to be *precise* (associative recall needs full rank to store
`d_h` independent pairs); (ii) the orthonormality maintenance adds a data-dependent
operation that is awkward in an integer runtime (needs a division / norm per column);
(iii) any accuracy loss is invisible at 4·10⁶ training tokens, so a fair evaluation would
need more compute than exists here. **Rejected: unmeasurable within budget, and it
reintroduces the non-integer operation the project is trying to remove.**

---

## 3. Candidate C — **Cascade**: pure multi-timescale diagonal memory

**Idea.** Drop the matrix entirely. Per channel keep a bank of `L` leaky integrators with
dyadic rates `2^{−1}, 2^{−2}, …, 2^{−L}`, update all in parallel (`x ← x + (u − x) >> ℓ`),
and read `y = Σ_ℓ γ_{i,ℓ} x_{i,ℓ}` with a learned `d×L` mixing matrix. A short causal conv
supplies local order information.

**Complexity at `L = 4`.** State `4d = 768` words per layer (**-92 %** vs A);
state MACs ≈ 1 500/token/layer; total ≈ 3.5 M MACs/token; all-integer, no divisions.

**Verdict.** The cheapest possible model and the most integer-friendly. But it is a
*diagonal* linear RNN: it can only decide "how much of value `i` was here recently", never
"which key was paired with which value". LRU/HGRN-lineage evidence shows a clear recall
gap versus the delta rule, and this project has no GPU budget to hide that gap at scale.
**Rejected as the primary architecture; retained as future work** — it is the obvious
ablation to run when compute allows, and it is included in the report's limitations.

---

## 4. Decision

Criteria and weights were fixed before scoring; scores are 0-10, higher is better.
"Cost" criteria are computed from the formulas above (objective), "fidelity" criteria are
judgements anchored in the literature cited in `01`.

| criterion | weight | A Cadence | B Lattice | C Cascade |
|---|---|---|---|---|
| associative-recall capacity (delta rule / full rank) | 0.25 | 9 | 6 | 3 |
| weights streamed per token | 0.15 | 8 (1.12 MB) | 8 | 8 |
| state traffic per token | 0.15 | 5 (216 KB, 5.5 % MACs) | 7 | 10 |
| integer exactness (no fp in the hot loop) | 0.15 | 10 | 6 (orthonormalisation) | 10 |
| trainability on CPU (chunk-parallel, stable) | 0.10 | 8 | 5 | 9 |
| implementation risk | 0.10 | 7 | 5 | 9 |
| multi-timescale capacity | 0.10 | 9 (bands + gate + gauge) | 8 | 7 |
| **weighted score** | | **7.85** | **6.35** | **6.55** |

Cadence wins, driven by recall capacity and integer exactness. The Transformer reference
of the same size (6 layers, `d=192`, `d_h=48`, same SwiGLU MLP, `V=2304`) is retained as
the **baseline** for Phase 4 rather than as a candidate: it is the thing to beat.

---

## 5. Cadence in full (the built system)

### 5.1 Block

```
h   = x ⊙ (1 + γ_r)                                  # per-loop adapter (P·R pairs)
q,k,v,e,w,o = W_in h + b_in                          # 6d outputs
[q,k,v] = SiLU(DepthwiseCausalConv4([q,k,v]))
k̂ = k/‖k‖,  q̂ = q/‖q‖
y   = Memory(q̂, k̂, v, σ(e), σ(w), g, nov)            # the recurrence above
x   = x + W_out y
x   = x + W_down( SiLU(W_gate (x⊙(1+γ'_r))) ⊙ W_up(x⊙(1+γ'_r)) )
```

2 physical blocks, 3 loops each, depth 6. `d=192, H=4, d_h=48, mlp=512, conv k=4`.

### 5.2 Retention bands

| head | band `[lo, hi]` | half-life range (tokens) | role |
|---|---|---|---|
| 0 | [0, 6] | 1.0 → 5.5 (init 2.5) | local order |
| 1 | [0, 6] | 1.0 → 5.5 | local order |
| 2 | [0, 14] | 1.0 → 10 000 (init 88) | document-level |
| 3 | [0, 14] | 1.0 → 10 000 | document-level |

Half-life = `a + 1` for `α = 2^{-a}` (`Σ_t α^t · t = α/(1-α)² ≈ 1.44·2^a`; the table quotes
`log₂`-style values, `2^a` tokens). The bands are a *prior*, not a constraint: within a
band the model chooses per token.

### 5.3 Surprise gauge

`nov_t` is a leaky average (`λ = 7/8`, exactly representable) of `mean|w⊙v − r|`, computed
**inside** the same chunk pass from the chunk-start state, so it is parallel over the chunk
and costs one `(B,C,H)` vector. It enters the retention index as `κ·nov` and is also
available to the runtime, where it is one `(H,)` accumulator per layer.

### 5.4 Chunk-parallel training algebra

Per chunk, let `c_t = −Σ_{s≤t} a_s` (log2, so `2^{c_t}` is the cumulative retention) and
`ẽ_t = b_t⊙k̂_t`. The recurrence expands to

```
S_t = 2^{c_t} S_0 + Σ_{s≤t} 2^{c_t−c_s} k̂_s w_sᵀ ,     w_s = u_s − r_s
r_t = 2^{c_t} ẽ_tᵀ S_0 + Σ_{s<t} 2^{c_t−c_s} (ẽ_tᵀ k̂_s) w_s
```

which is the linear system `(I + T) w = u − Ẽ S_0` with `T[t,s] = 2^{c_t−c_s}(ẽ_tᵀk̂_s)` for
`s<t`, and the outputs `y = Q̃ S_0 + (Q̃K̃ᵀ ∘ causal) w`, `S_C = 2^{c_C} S_0 + K̃ᵀ w`, where
`Ẽ = ẽ⊙2^{c}`, `Q̃ = q̂⊙2^{c}`, `K̃ = k̂⊙2^{c_C−c}`.

**Numerical note (a real constraint, not a preference).** Every rescaled term carries
`2^{c_t−c_s}` with `t ≥ s`, i.e. an exponent `≤ 0`, so all intermediates are bounded by
`O(C)`. A *per-channel* retention makes `c` a vector and the term
`2^{c_{t,i}−c_{s,i}}` a rank-`d_h` tensor — the cheap diagonal algebra is lost, which is
exactly why KDA (arXiv:2510.26692) needs a bespoke diagonal-plus-low-rank solver. I
therefore restrict retention to a **per-head scalar**, which also makes the runtime decay a
single shift. This costs expressivity and is listed as a limitation.

The chunk form is verified against the token recurrence by
`tests/test_equivalence.py` (max |Δ| ≈ 1e-6 in fp64/fp32).

### 5.5 Quantisation and the integer runtime

* Weights: symmetric int8 per output channel (`w/127·max|w|`), **trained with a
  straight-through estimator for the last 30 % of steps** (QAT), starting from an
  already-trained model so no separate quantisation-aware schedule is needed.
* Activations: int16 per-token dynamic scale (15-bit) inside each matvec, int32 accumulate.
* State: int32 with scale `2^{-16}`; the decay is `S = (S + 2^{a-1}) >> a` — exact,
  round-to-nearest, one instruction. (The exponent chosen matters: at `2^{-10}` the rank-1
  write quantises to 0.06 in value units, which is a 6 % error *per token* and compounds;
  `2^{-16}` brings it to 1.5e-5.)
* Delta write: `S += (k̂_int · resid_int) >> (e_k + e_d - 16)`, i.e. a single shift whose
  amount is read off the two power-of-two activation exponents.
* Activations: int16 on a power-of-two grid (`e = ceil(log2(16384/‖x‖_∞))`), so the
  dequantisation is an exact multiply by `2^{-e}`.
* Small tensors (per-loop adapters, the 4×d retention gate, conv kernels, biases, norms)
  stay fp32: quantising them saved 4 KB and cost more accuracy than it was worth — and, in
  one debugging session, was read back by the runtime as if it were int8.
* Note on `q, k`: they are L2-normalised across the **full width** before being split into
  heads (so a head holds ~`sqrt(d_h/d)` of unit norm), matching the trainer exactly. GDN-2
  normalises per head; doing that here silently changed the model by 4× in the memory write.

### 5.6 What measurement changed

Two design decisions were revised *because* the numbers said so:

1. **Per-channel retention → per-head retention.** The chunk-parallel algebra (§5.4) is
   only well conditioned when the retention factor is shared across a head's channels;
   with per-channel `a` the rescaled keys span `2^{±400}` and training produced NaN. Fixing
   it with a clamped, per-head scalar costs expressivity (documented as a limitation) and
   buys stable fp32 chunkwise training in the cheap diagonal form — which is precisely why
   KDA had to invent a DPLR solver to keep per-channel gating.
2. **Novelty from a chunk-start snapshot → a leaked error average computed inside the same
   chunk pass.** The snapshot version needed a second `O(C·d_h²)` read per chunk and was
   numerically fragile at chunk boundaries; the leaky average is one `(B,C,H)` vector and
   exactly `λ = 7/8`, which the integer runtime reproduces with a shift/add.

### 5.7 Ablations that were considered

| ablation | status |
|---|---|
| dyadic vs continuous retention (`α = σ(g)^2`, no rounding) | **planned, run if compute allows**; the honest test of whether the int-exactness costs loss |
| bands vs a single band for all heads | planned |
| surprise gauge off (`κ = 0`) | planned |
| 2 blocks × 3 loops vs 6 unique blocks | planned |
| transformer baseline, identical data/budget | **run** (Phase 4 report) |

Anything not run is reported as not run; no ablation number in this repository is
estimated.
