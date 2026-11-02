# Phase 7 — validating the claims

The project makes seven claims. Each one is checked by an executable test, not by prose.
`python -m tools.validate` runs all of them and exits non-zero if any fails; the table below
is its own output (regenerate with `python -m tools.validate`, full log in
`runs/validation.md`).

```text
20/20 checks passed
```

| result | check | measured |
|---|---|---|
| PASS | checkpoint contains a state_dict and a step counter | step 900 |
| PASS | training metrics exist next to the checkpoint | 900 steps |
| PASS | all weights differ from a fresh random init | min relative distance 0.027 over 34 tensors |
| PASS | validation perplexity fell by >40% over the run | 200.4 -> 69.5 ppl (-65%) |
| PASS | training loss fell | 7.782 -> 4.049 |
| PASS | at least 1M tokens of gradient updates | 2,764,800 tokens |
| PASS | arch matches | cadence |
| PASS | model config matches the checkpoint exactly | — |
| PASS | vocabulary matches the embedding table | 2304 |
| PASS | every trained tensor is in the artefact, name for name | 34 tensors, 0 missing |
| PASS | artifact under the 10 MB target | 1.63 MB |
| PASS | spaces sit between words | 0.18 of characters |
| PASS | word lengths are plausible | mean 3.64 |
| PASS | frequent English words appear | 30% of tokens |
| PASS | output is not a repeated loop | type/token 0.62 |
| PASS | runtime imports only node builtins (fs/path/os/url) | `[]` |
| PASS | same seed -> identical text | — |
| PASS | different seed -> different text | — |
| PASS | generation returns text | `It is a truth to the same. I cannot` |
| PASS | per-token cost flat from position 0 to 2048 | 4.69 -> 4.89 ms/token (x1.04) |

## How each claim is actually established

### 1. "The weights came from random init, not from a published model"

* `src/cadence/model.py` is 400 lines of `nn.Linear`/`nn.Conv1d`/arithmetic — there is no
  `from_pretrained`, no `torch.hub`, no download of any checkpoint anywhere in the repo
  (`grep -rn "from_pretrained\|hub.load\|hf_hub" src tools` is empty).
* The tokenizer is trained from scratch on *our* corpus (`tools/prepare_data.py`), so even
  the vocabulary is ours — a foreign model's embedding table could not be used.
* The validation check compares every tensor against a freshly initialised model of the
  same config: the *minimum* relative distance over 34 tensors is 0.027, i.e. no tensor was
  left at (or near) its initial value, and none matches a random draw of the same shape.
* The training run is reproducible: `runs/cadence_main/config.json` holds the exact config,
  and `src/train.py` seeds Python, NumPy, PyTorch and the data loader from `--seed 1337`.

### 2. "Real training happened"

Val perplexity 200.4 → 69.5 over 900 steps / 2.76 M tokens, training loss 7.78 → 4.05, and
the curve is in `docs/03-benchmarks.md`. The int8 fake-quantisation phase is visible in the
log at step 630.

### 3. "The deployed function loads *our* weights"

`GET /api/health` reports the arch, vocabulary, tensor count and byte count read out of the
bundled `runtime/model/model.bin`; the validation check compares that artefact's tensor list
against the checkpoint's `state_dict` name for name (34/34). The int8 export is 1.63 MB.

### 4. "The model generates language"

Automated checks on the sampled text: 18 % of characters are spaces, mean word length 3.64,
30 % of words are in a 12-word common-English list, type/token ratio 0.62. That is enough to
reject "random bytes" and "repeated loop", and not enough to call it good writing. A sample:

> It is a truth universally acknowledged to his bulling-house and desteve her and the couple
> of his, dwitch-hearted times, and I did not talk of her; and she did not know the same
> inclusion.

Honest reading: locally fluent, globally incoherent, with invented words. At 1.5 M
parameters and 2.8 M tokens this is the expected quality; the point of the artefact is the
inference cost, not the prose.

### 5. "No external providers, no Python/GPU at inference"

The runtime (`runtime/src/*.mjs`, `lib/api.mjs`, `api/*.mjs`) imports only `node:fs`,
`node:path`, `node:os` and `node:url`, verified by scanning every import statement. There is
no `fetch`, no HTTP client, no database driver. The Vercel handlers read `model.bin` from
the deployment bundle. The only network traffic in the whole system is the one-off corpus
download in `tools/fetch_corpus.py`, which runs at *data preparation* time and is not part
of training or inference.

### 6. "The integer runtime computes the same function as the trainer"

`tests/test_equivalence.py` proves the PyTorch internal consistency (chunk-parallel
training vs token recurrence: 4.9e-16; streaming vs teacher forcing: 6.4e-16; causality
exact). `runtime/src/parity.mjs` then compares the *int8 JavaScript* implementation against
the fp32 PyTorch forward pass on the same prompt:

| quantity | Cadence | Transformer |
|---|---:|---:|
| logit cosine | 0.99999 | 1.00000 |
| mean abs logit error | 0.0132 | 0.0084 |
| max abs logit error | 0.0617 | 0.0577 |
| top-1 token agreement | yes | yes |
| top-5 overlap | 5/5 | 5/5 |

That is int8-quantisation noise and nothing else, which is the correct outcome: it means
the shipped model is the trained model.

**This check earned its keep.** It caught four separate defects that the "it runs and emits
words" smoke test happily passed:

1. small 2-D tensors (the per-loop adapters, 3×192) were being int8-quantised with 6 bits of
   range and then read back as raw floats — a 30× error in a residual branch;
2. the causal-conv ring buffer shifted with `copyWithin(0, 0, n)`, a no-op, so the newest
   row was never written and the convolution read stale activations;
3. the position table of the transformer was read without its per-row scale (another 30×
   error);
4. `q` and `k` were L2-normalised **per head** in the runtime but across the **full width**
   in the trainer, a factor-of-4 error in the memory write that cost 0.99 → 0.78 logit
   cosine.

None of those produced an exception or obviously broken text. Only the numeric comparison
against the reference did.

### 7. "Cost is independent of context length"

Measured by stepping tokens and timing 32-token windows at positions 0, 512 and 2048: 4.69 →
4.89 ms/token (±4 %, which is JIT warm-up and timer noise, not growth). The transformer baseline, measured the
same way inside its 256-token window, goes 2.00 → 2.26 ms/token and cannot go past 256 at
all because its position table is only trained there.

## Reproducing everything

```bash
python -m tools.fetch_corpus && python -m tools.prepare_data --vocab 2304
python -m src.train --arch cadence --steps 900 --batch 12 --seq 256 --out runs/cadence_main
python -m src.train --arch transformer --n-layers 3 --steps 900 --batch 12 --seq 256 --out runs/tf_main
python -m tools.export --ckpt runs/cadence_main/best.pt --out runtime/model
python -m tools.dump_reference --out runs/reference.json
python -m pytest -q tests
node runtime/src/parity.mjs runs/reference.json
node runtime/src/bench.mjs --json runs/bench_cadence.json
python -m tools.validate
python -m tools.make_report
```

## What is *not* validated

* **Long context.** The model has no window, but the corpus is books with ~250-token
  sentences; quality beyond a few thousand tokens is unmeasured, and the novelty gauge
  saturates (clamped at 4.0 by construction).
* **Recall benchmarks** (MQAR, needle-in-a-haystack). Not run: at ppl 70 there is no signal
  to measure, and building a synthetic suite is a separate project.
* **Batching and concurrency.** All numbers are single-session. The 216 KB session state is
  what makes concurrency cheap, but that is an argument, not a measurement. `runs/` contains
  no concurrency test.
* **The dyadic-retention ablation** (does constraining `α` to powers of two cost loss?) is
  designed in `docs/02 §5.6` and **not run**, because each training run here costs ~20
  minutes of the only CPU available. It is reported as not run, not estimated.
