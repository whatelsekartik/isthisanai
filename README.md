# Cadence

A language model built for **one CPU core and 3 MB of RAM**, trained from scratch on public
domain books, and deployed as an int8 integer-arithmetic runtime with no Python, no PyTorch
and no GPU anywhere in the inference path.

* **Architecture:** a gated delta-rule linear attention ("fast-weight memory") with
  **power-of-two, per-head-banded retention**. The recurrence is
  `S ← (I − k(b⊙k)ᵀ)·2^{-a}·S + k(w⊙v)ᵀ` with `a ∈ {0..15}` chosen per token, per head, from a
  learned gate. Because `a` is an integer, the decay is `S >> a`: exact, branch-free, and
  bit-identical on every CPU.
* **Measured, not estimated** (this machine, one core, Node 22):

  | | Cadence | Transformer baseline (same tokens) |
  |---|---:|---:|
  | parameters | 1,561,170 | 1,770,816 |
  | val perplexity after 2.76 M tokens | **69.5** | 107.4 |
  | int8 artefact | **1.63 MB** | 1.86 MB |
  | cold start (read + parse) | **2.8 ms** | 2.6 ms |
  | decode | 4.8-5.5 ms/token | 2.0-2.3 ms/token (≤256 tokens) |
  | session state | **216 KB, constant** | 1,152 KB, grows with context |
  | int8 vs fp32 logit cosine | 1.00000 | 1.00000 |

  Cadence is *slower per token* at short context (it applies 6 block computations per token
  to the transformer's 3, and touches a `d_h²` state per head). What it buys is a cost that
  does not grow with context, a 5x smaller session, no fixed window, and better loss at an
  equal token budget — with a *larger* model on the other side of the comparison.

## Quick start

```bash
# 1. environment (CPU only)
python -m venv .venv && . .venv/bin/activate && pip install torch numpy

# 2. data: 47 MB of public-domain books -> BPE tokenizer -> uint16 token files
python -m tools.fetch_corpus            # GitHub mirrors of Project Gutenberg (no gutenberg.org)
python -m tools.prepare_data --vocab 2304

# 3. train Cadence and the matched transformer baseline (same data, same token budget)
python -m src.train --arch cadence --steps 900 --batch 12 --seq 256 --out runs/cadence_main
python -m src.train --arch transformer --n-layers 3 --steps 900 --batch 12 --seq 256 --out runs/tf_main

# 4. export int8 and check the numerical invariants
python -m tools.export --ckpt runs/cadence_main/best.pt --out runtime/model
python -m tools.dump_reference --out runs/reference.json
python -m pytest -q tests

# 5. run it
node runtime/src/generate.mjs "It is a truth" 80
node runtime/src/bench.mjs
node dev.mjs          # http://localhost:3000 (same handlers Vercel runs)
```

## What is measured, and what is claimed

| claim | evidence |
|---|---|
| trained from random init, no pretrained weights | `runs/*/metrics.json`, `src/cadence/model.py`, no download of any checkpoint |
| no external model API at inference | `runtime/src/model.mjs` has one dependency: `node:fs` |
| int8 weights, exact integer decay | `tools/export.py`, `docs/04-runtime.md`, `tests/test_equivalence.py` |
| chunk-parallel training == streaming inference | `tests/test_equivalence.py` (max |Δlogit| ≈ 1e-16) |
| throughput / latency / cold start / memory | `docs/03-benchmarks.md`, `runtime/src/bench.mjs`, `GET /api/bench` |
| the int8 JS engine computes what PyTorch trained | `docs/05`, `runtime/src/parity.mjs` (cosine 1.00000, mean logit error 0.007) |
| every claim above, re-checked automatically | `python -m tools.validate` (20/20) |
| honest comparison to a transformer at the same data and token budget | `docs/03-benchmarks.md` |

Not claimed: state-of-the-art quality, a new fundamental primitive, or any speed-up that was
not measured on this machine. The architecture is a delta-rule linear attention — prior art
is cited in `docs/01`, and the delta over it is spelled out in `docs/02` §4.

## Layout

```
docs/          01 literature review · 02 architecture proposals · 03 benchmarks · 04 runtime · 05 validation
src/cadence/   tokenizer.py (BPE) · model.py (Cadence + transformer baseline) · train.py
tools/         fetch_corpus · prepare_data · export · dump_reference
runtime/       src/model.mjs (int8 engine) · src/bench.mjs · src/generate.mjs · src/parity.mjs
api/           Vercel function handlers: /api/generate, /api/bench, /api/health
public/        the single page that calls them
tests/         equivalence, causality, streaming, retention-integrality
runs/          checkpoints, logs, metrics, quantisation reports
```
