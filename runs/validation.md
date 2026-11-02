# Validation output

`python -m tools.validate --ckpt runs/cadence_main/best.pt --model runtime/model`

| result | check | measured |
|---|---|---|
| PASS | checkpoint contains a state_dict and a step counter | step 900 |
| PASS | training metrics exist next to the checkpoint | 900 steps |
| PASS | all weights differ from a fresh random init | min relative distance 0.027 over 34 tensors |
| PASS | validation perplexity fell by >40% over the run | 200.4 -> 69.5 ppl (-65%) |
| PASS | training loss fell | 7.782 -> 4.049 |
| PASS | at least 1M tokens of gradient updates | 2,764,800 tokens |
| PASS | arch matches | cadence |
| PASS | model config matches the checkpoint exactly | - |
| PASS | vocabulary matches the embedding table | 2304 |
| PASS | every trained tensor is in the artefact, name for name | 34 tensors, 0 missing |
| PASS | artifact under the 10 MB target | 1.63 MB |
| PASS | spaces sit between words | 0.18 of characters |
| PASS | word lengths are plausible | mean 3.64 |
| PASS | frequent English words appear | 30% of tokens |
| PASS | output is not a repeated loop | type/token 0.62 |
| PASS | runtime imports only node builtins (fs/path/os/url) | [] |
| PASS | same seed -> identical text | - |
| PASS | different seed -> different text | - |
| PASS | generation returns text | 'It is a truth to the same. I cannot' |
| PASS | per-token cost flat from position 0 to 2048 | 5.61 -> 4.70 ms/token (x0.84) |

20/20 checks passed.
