# Phase 1 — Literature review and opportunity analysis

*Goal: find a genuinely different, defensible place to build a language model whose
**inference cost on one CPU core** is the primary design constraint, rather than a
by-product of GPU economics.*

Constraints that define the search space (from the project brief):

| constraint | consequence |
|---|---|
| no GPU at inference | decode is dominated by **memory traffic and serial dependency**, not FLOPs |
| no Python/PyTorch at inference | the model must be expressible in int8 + integer arithmetic |
| ≤ ~10 MB weights, 2 GB / 1 vCPU function | small parameter count, small state, cold start ≪ 1 s |
| no pretrained weights, no external API | we must train from scratch on a CPU budget (~10⁷ tokens) |

---

## 1. Families and representative work

### 1.1 Softmax attention (the reference)

Vaswani et al., *Attention Is All You Need*, arXiv:1706.03762. Per layer, per token:
`4·d²` projection MACs, `2·L·d` attention MACs (L = context), and an O(L·d) KV cache.
FlashAttention (Dao et al., arXiv:2205.14135) makes training IO-aware but does not change
the asymptotic cache.

- **Strength:** perfect associative recall within the window; trivially parallel.
- **CPU weakness:** decode cost grows with context; KV cache is *state that must be
  re-read every token*, and with GQA-less small models it is comparable in size to the
  weights. On one core with a 2 GB cap, long context is simply unaffordable.

Sparse/local variants: Longformer (arXiv:2004.05150), sliding-window/Mistral
(arXiv:2310.06825) trade global recall for O(L·w) work. Low-rank: Linformer
(arXiv:2006.04768); kernel: Performer (arXiv:2009.14794). These change compute but keep an
O(L) cache unless combined with a recurrent state.

### 1.2 Recurrent networks (RNN/GRU/LSTM)

Constant state `O(d)`, constant per-token cost, but strictly sequential in *both*
directions, and the state is a plain vector: no key→value binding, so copying/recall
degrades quickly (Hochreiter's analysis; see also the tiny-LM evidence in TinyStories,
Eldan & Li, arXiv:2305.07759, which uses attention plus a very small vocabulary).

### 1.3 State-space models

S4 (Gu et al., arXiv:2110.13985), S5 (arXiv:2208.04933), diagonal/decay variants such as
LRU (Orvieto et al., arXiv:2305.04764), HGRN (Qin et al., arXiv:2311.04823), and
**Mamba** (Gu & Dao, arXiv:2312.00752) with input-dependent (selective) parameters and
**Mamba-2** (Dao & Gu, arXiv:2405.21060) which ties SSMs to structured masked attention
for fast training. State is `O(d·n)` with `n` the state dimension (typically 16-128 per
channel); the recurrence is linear so it parallelises as a scan.

- **Strength:** constant memory, cheap decode, good at long-range decay-shaped structure.
- **Weakness:** a diagonal linear recurrence cannot *overwrite* a memory slot; it only
  accumulates and leaks. Associative recall tasks (MQAR, induction heads) are where this
  shows up, and where the delta rule wins.

### 1.4 Linear attention and the delta rule (the current frontier)

- Katharopoulos et al., *Transformers are RNNs*, arXiv:2006.16236: linear attention is a
  running key→value outer-product accumulator. Training is `O(L·d²)`, inference `O(d²)`.
- Schlag, Irie & Schmidhuber, arXiv:2102.11174: the same object is a fast-weight
  programmer; the **delta rule** `S ← S + k(v − Sᵀk)ᵀ` turns accumulation into error-driven
  writing, which is what gives associative recall.
- Yang, Kautz & Hatamizadeh, arXiv:2406.06484: the WY representation makes delta-rule
  training chunk-parallel (`O(L·d²)` with a small triangular solve per chunk).
- **Gated DeltaNet** (Yang et al., arXiv:2412.06464) adds a per-head forget gate.
- **Gated DeltaNet-2** (NVIDIA, arXiv:2605.22791, 2026) decouples the erase and write
  gates per channel: `S_t = (I − k_t(b_t⊙k_t)ᵀ)D_t S_{t−1} + k_t(w_t⊙v_t)ᵀ`, which is the
  cleanest published form of the "forget, erase, write" decomposition.
- **Kimi Delta Attention** (Kimi Linear, arXiv:2510.26692) pushes gating to per-dimension
  granularity with a bespoke diagonal-plus-low-rank chunkwise solver.
- Recent refinements: Momentum DeltaNet (arXiv:2605.05838), Kaczmarz linear attention
  (arXiv:2605.08587), Preconditioned DeltaNet (arXiv:2604.21100), Dual DeltaNet
  (arXiv:2610.38578), Titans (arXiv:2501.00663) for test-time memory.

This family is the strongest per-FLOP option at small scale, and it is what I build on.

### 1.5 Parameter sharing and depth recurrence

ALBERT (Lan et al., arXiv:1909.11942) shares transformer blocks across layers;
Universal Transformers (Dehghani et al., arXiv:1807.03819) and Depth-Recurrent
Transformers (Geiping & Goldstein, arXiv:2304.11062) iterate the same block with a
per-iteration signal. For CPU inference this is not just a parameter-count trick: reusing
weights means the weight *bytes streamed per token* stay constant while depth grows, and
the working set may fit in L2.

### 1.6 Quantisation and integer inference

Jacob et al. (arXiv:1712.05877) established integer-only inference with per-tensor
affine scales; LLM.int8 (arXiv:2208.07339) showed outlier channels break naive int8 and
motivated per-channel scales and mixed precision. For recurrent models, iRNN
(SCITEPRESS 2022) runs an LSTM integer-only with a 16-bit cell state and piecewise-linear
activations; QS4D (arXiv:2507.06079) shows quantisation-aware training lets an S4D model
reach 5-6 bits with <1 % loss while post-training quantisation fails below 10-16 bits.
Power-of-two scale restrictions are standard for weights/activations (Jain et al.,
arXiv:1903.08066; ²GRU 2026; ShiftLIF arXiv:2605.01866 for spike amplitudes).

**Key observation:** in all of this work the object that is quantised is the *weights* and
*activations*. The **recurrence's decay factor** — the matrix that decides how memory
fades — is still an fp32 multiplier. That is the gap I exploit (G2 below).

### 1.7 CPU/WASM execution reality

WebAssembly SIMD (128-bit) has `i32x4.dot_i16x8_s` (int16 dot product with int32
accumulate) but **no** int8 dot product instruction; int8 × int8 must be widened first
(`i16x8.extmul_low_i8x16_s` + `i16x8.add`). Practically, a single-core WASM kernel gets
~2-6 GMAC/s on int8 with 4-wide SIMD, versus ~0.3-1 GMAC/s for a scalar loop in V8 with
typed arrays. On the serverless side, cold start is dominated by *bytes to parse* (weights
must be downloaded + copied into typed arrays), so the artifact size is a latency
parameter, not a vanity metric.

---

## 2. A cost model for one-core decode

For a model with `P` parameters (int8), `A` activated parameters per token, and a
recurrent state of `M` int32 words per layer-session:

```
bytes read per token   B_tok ≈ A                     (int8 weights)
MACs per token         C_tok ≈ A
state traffic          M·4 bytes per token per layer (read + write)
```

Decode latency is `max(C_tok / throughput, B_tok / bandwidth)`, plus `O(depth)` serial
dependency (each layer waits for the previous one). Two consequences that the literature
rarely optimises for:

1. **The state is a first-class cost.** For a head of dimension `d_h`, a delta-rule layer
   touches `d_h²` state words per token *per head*, i.e. `d·d_h` per layer — independent of
   the model width, but it does not shrink with quantisation of the weights unless the
   state is quantised too. For `d=192, d_h=48, H=4`: 36 864 state MACs per layer per token
   against ~590 000 weight MACs per layer per token. At 6 layers the state is ~6 % of the
   arithmetic but 100 % of the *serial* work that cannot be batched away.
2. **Depth is latency.** Six sequential block applications cost six round trips through
   memory. Weight-tying lets you have depth 6 while streaming depth 2 of weights.

The design target that falls out: **small state, int-exact recurrence, few unique
weights, many applications**.

---

## 3. Gaps worth attacking

| id | gap | evidence it is real | how I address it |
|---|---|---|---|
| **G1** | Decode cost is modelled as FLOPs; the *fixed-size state traffic* and serial depth are not part of architecture choices | SSM/linear-attention papers report FLOPs and GPU latency; no CPU-per-token state budget | I make the state budget a design constraint and report int32 state bytes/session and per-token state ops |
| **G2** | The recurrence's decay is never made integer-exact; quantisation work targets weights/activations | Jacob'18, LLM.int8, QS4D, ²GRU, ShiftLIF all quantise weights/activations; no work found that restricts the *state transition itself* to powers of two so that decay is a shift | **dyadic retention**: `α = 2^-a`, `a ∈ {0..15}`; decay becomes `S >>= a` — exact, branch-free, identical on every CPU |
| **G3** | Adaptive per-channel timescales cost a full gate matmul, and per-channel decay breaks cheap chunkwise training algebra | GLA/GDN-1 use per-head scalars; KDA/GDN-2 restore per-channel gating but need a bespoke DPLR solver (2510.26692) | **banded retention**: each head owns a band `[lo_h, hi_h]`, so heads are forced to different timescales at zero extra cost, and the algebra stays the cheap diagonal one |
| **G4** | Depth is serial latency; weight sharing is treated as a parameter-count trick, not a bandwidth trick | ALBERT, UT, Depth-Recurrent all frame sharing as parameter efficiency | **depth-tied blocks** with per-loop scale adapters: depth 6 from 2 weight sets (3× less weight traffic) |
| **G5** | "Retention should follow surprise" is implemented with extra O(d²) probes or second-order state | Titans (2501.00663), Momentum DeltaNet (2605.05838) pay state/work for extra memory | **surprise gauge** from the delta-rule write error, which is *already computed* — one leaked scalar per head, no extra state |
| **G6** | Reported efficiency is GPU-relative | common practice | every claim here is measured on one CPU core in Node/WASM with cold-start timing |

---

## 4. Prior-art boundary (what is *not* claimed)

Cadence is a **gated delta-rule linear attention** at its core, and I do not claim
otherwise. Specifically:

* the delta rule, the WY chunkwise training trick, decoupled erase/write gates, L2
  normalised keys, short causal conv, SiLU output gating, SwiGLU, RMSNorm, input-dependent
  forget gates and depth weight sharing are all prior work (§1.3-1.5).
* **novelty claimed** is narrow and testable:
  1. constraining retention to a **power-of-two per token** so the state update is an exact
     integer shift, and training the retention index through a straight-through estimator;
  2. **per-head retention bands** as a zero-cost multi-timescale prior;
  3. a **surprise gauge** reusing the delta-rule write error as a leaked scalar that
     biases retention (no extra state, no extra matmul);
  4. the combination of the above with depth-tied blocks and a quantisation-first operator
     set, evaluated as an *end-to-end measured CPU inference* story.

Claim (1) is the one I can defend most cleanly: I found no prior work that makes the
recurrence's decay dyadic for the purpose of exact integer inference. I make no claim that
it improves loss — §4 of the report measures whether it *hurts*, which is the honest bar
for a deployment-motivated constraint.

## References

1. Vaswani et al. *Attention Is All You Need.* arXiv:1706.03762
2. Dao et al. *FlashAttention.* arXiv:2205.14135
3. Katharopoulos et al. *Transformers are RNNs.* arXiv:2006.16236
4. Schlag, Irie, Schmidhuber. *Linear Transformers Are Secretly Fast Weight Programmers.* arXiv:2102.11174
5. Yang, Kautz, Hatamizadeh. *Gated Delta Networks: Improving Mamba2 with Delta Rule.* arXiv:2412.06464
6. Yang et al. *Parallelizing Linear Transformers with the Delta Rule over Sequence Length.* arXiv:2406.06484
7. NVIDIA. *Gated DeltaNet-2: decoupled erase/write gates.* arXiv:2605.22791
8. Moonshot AI. *Kimi Linear / Kimi Delta Attention.* arXiv:2510.26692
9. *Momentum DeltaNet.* arXiv:2605.05838
10. *Kaczmarz Linear Attention.* arXiv:2605.08587
11. *Preconditioned DeltaNet.* arXiv:2604.21100
12. *Dual DeltaNet.* arXiv:2610.38578
13. Gu, Goel, Ré. *Efficiently Modeling Long Sequences with Structured State Spaces (S4).* arXiv:2110.13985
14. Smith et al. *Simplified State Space Layers (S5).* arXiv:2208.04933
15. Orvieto et al. *Resurrecting Recurrent Neural Networks (LRU).* arXiv:2305.04764
16. Gu, Dao. *Mamba: Linear-Time Sequence Modeling with Selective State Spaces.* arXiv:2312.00752
17. Dao, Gu. *Transformers are SSMs (Mamba-2).* arXiv:2405.21060
18. Qin et al. *HGRN: Hierarchically Gated Recurrent Neural Networks.* arXiv:2311.04823
19. Behrouz et al. *Titans: Learning to Memorize at Test Time.* arXiv:2501.00663
20. Lan et al. *ALBERT.* arXiv:1909.11942
21. Dehghani et al. *Universal Transformers.* arXiv:1807.03819
22. Geiping, Goldstein. *Scaling Laws for Deep Networks / Depth-Recurrent Transformers.* arXiv:2304.11062
23. Jacob et al. *Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference.* arXiv:1712.05877
24. Dettmers et al. *LLM.int8().* arXiv:2208.07339
25. *QS4D: quantisation-aware training for S4D.* arXiv:2507.06079
26. Jain et al. *Efficient integer-arithmetic quantisation (power-of-two scales).* arXiv:1903.08066
27. Eldan, Li. *TinyStories.* arXiv:2305.07759
28. Beltagy et al. *Longformer.* arXiv:2004.05150
29. Wang et al. *Linformer.* arXiv:2006.04768
30. Choromanski et al. *Performer.* arXiv:2009.14794
31. iRNN: *Integer-only recurrent neural network inference.* SCITEPRESS, 2022
32. WebAssembly SIMD proposal (`i32x4.dot_i16x8_s`), github.com/WebAssembly/simd
