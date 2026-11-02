"""Reproducible training loop for Cadence and for the matched Transformer baseline.

    python -m src.train --arch cadence --steps 800 --batch 32 --seq 256 --out runs/cadence
    python -m src.train --arch transformer --steps 800 --batch 32 --seq 256 --out runs/tf

Both archs see exactly the same tokens in the same order (same seed, same loader), so the
comparison is "same data, same token budget, same optimiser" and only the architecture and
parameter count differ (parameter count is reported, not matched away).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F

from src.cadence.model import CadenceConfig, TransformerConfig, build_model

# ------------------------------------------------------------------ quantised weights


class FakeQuantWeight(torch.autograd.Function):
    """Symmetric per-output-channel int8 quantisation with a straight-through gradient.

    Applied to the large weight matrices during the last part of training so that the
    exported int8 file is the distribution the model was trained in.
    """

    @staticmethod
    def forward(ctx, w):
        flat = w.reshape(w.shape[0], -1)
        scale = flat.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / 127.0
        q = (w / scale.view(-1, *([1] * (w.dim() - 1)))).round().clamp(-127, 127)
        return q * scale.view(-1, *([1] * (w.dim() - 1)))

    @staticmethod
    def backward(ctx, g):
        return g


def quantised_forward_hooks(model: torch.nn.Module):
    """Quantise the big weight matrices to int8 for the forward pass only (STE).

    The latent weight is stashed before the call and restored afterwards, so the
    parameter that the optimiser owns keeps full precision while the network sees the
    int8 grid - a straight-through estimator with no change to the module code.
    """
    handles = []
    for m in model.modules():
        if isinstance(m, torch.nn.Linear) and m.weight.numel() > 512:
            def pre(mod, inp, _m=m):
                if not getattr(mod, "_qat_on", False):
                    return
                mod._latent = mod.weight.detach().clone()
                mod.weight.data.copy_(FakeQuantWeight.apply(mod.weight).detach())

            def post(mod, inp, out):
                if getattr(mod, "_qat_on", False):
                    mod.weight.data.copy_(mod._latent)
                    del mod._latent

            m._qat_on = True
            handles += [m.register_forward_pre_hook(pre), m.register_forward_hook(post)]
    return handles


# ------------------------------------------------------------------------------ data


class TokenData:
    def __init__(self, root: str):
        self.meta = json.load(open(os.path.join(root, "meta.json")))
        self.train = np.memmap(os.path.join(root, "train.bin"), dtype=np.uint16, mode="r")
        self.val = np.memmap(os.path.join(root, "val.bin"), dtype=np.uint16, mode="r")
        self.vocab_size = self.meta["vocab_size"]

    def batch(self, split: str, batch: int, seq: int, rng: random.Random):
        arr = self.train if split == "train" else self.val
        ix = [rng.randrange(0, len(arr) - seq - 1) for _ in range(batch)]
        x = np.stack([np.asarray(arr[i:i + seq]) for i in ix])
        y = np.stack([np.asarray(arr[i + 1:i + 1 + seq]) for i in ix])
        return torch.from_numpy(x.astype(np.int64)), torch.from_numpy(y.astype(np.int64))


# ------------------------------------------------------------------------------ main


def lr_at(step: int, total: int, base: float, warmup: int, min_ratio: float) -> float:
    if step < warmup:
        return base * (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return base * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * p)))


def build_config(arch: str, args, vocab_size: int):
    if arch == "cadence":
        bands = tuple(tuple(int(x) for x in b.split(":")) for b in args.bands.split(","))
        return CadenceConfig(
            vocab_size=vocab_size, d_model=args.d_model, n_heads=args.n_heads,
            n_blocks=args.n_blocks, n_loops=args.n_loops, mlp_hidden=args.mlp_hidden,
            seq_len=args.seq, chunk=args.chunk, bands=bands, novelty=args.novelty,
            conv_kernel=args.conv_kernel)
    return TransformerConfig(vocab_size=vocab_size, d_model=args.d_model,
                             n_heads=args.n_heads, n_layers=args.n_layers,
                             mlp_hidden=args.mlp_hidden, seq_len=args.seq)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="cadence", choices=["cadence", "transformer"])
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--out", default="runs/cadence")
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1.2e-3)
    ap.add_argument("--min-lr-ratio", type=float, default=0.1)
    ap.add_argument("--warmup", type=int, default=40)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--save-every", type=int, default=200)
    ap.add_argument("--eval-batches", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--n-blocks", type=int, default=2)
    ap.add_argument("--n-loops", type=int, default=3)
    ap.add_argument("--n-layers", type=int, default=6)
    ap.add_argument("--mlp-hidden", type=int, default=683)
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--conv-kernel", type=int, default=4)
    ap.add_argument("--bands", default="0:6,0:6,0:14,0:14")
    ap.add_argument("--novelty", default="gate", choices=["gate", "none"])
    ap.add_argument("--qat-frac", type=float, default=0.35,
                    help="fraction of training with int8 fake-quantised weights")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--max-minutes", type=float, default=0.0)
    ap.add_argument("--sample-tokens", type=int, default=200)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.out, exist_ok=True)
    data = TokenData(args.data)
    cfg = build_config(args.arch, args, data.vocab_size)
    model = build_model(args.arch, cfg.to_dict())
    n_all, n_nonemb = model.n_params(), model.n_params(non_embedding=True)
    print(f"{args.arch}: {n_all:,} params ({n_nonemb:,} non-embedding)", flush=True)

    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.dim() < 2 else decay).append(p)
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95), eps=1e-8)

    step, best_val, t0 = 0, float("inf"), time.time()
    history = []
    cfg_path = os.path.join(args.out, "config.json")
    json.dump({"arch": args.arch, "cfg": cfg.to_dict(), "params": n_all,
               "params_non_embedding": n_nonemb, "args": vars(args)},
              open(cfg_path, "w"), indent=1)
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step = ck["step"]
        best_val = ck.get("best_val", best_val)
        history = ck.get("history", [])
        print(f"resumed from {args.resume} at step {step}", flush=True)

    rng = random.Random(args.seed * 7919 + step)
    qat_on = False
    handles = []

    def evaluate(n=args.eval_batches):
        model.eval()
        tot, cnt = 0.0, 0
        with torch.no_grad():
            erng = random.Random(12345)
            for _ in range(n):
                x, y = data.batch("val", args.batch, args.seq, erng)
                logits = model(x)
                tot += F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                       y.reshape(-1), reduction="sum").item()
                cnt += y.numel()
        model.train()
        return tot / cnt

    log = open(os.path.join(args.out, "log.jsonl"), "a")
    model.train()
    while step < args.steps:
        if args.max_minutes and (time.time() - t0) / 60 > args.max_minutes:
            print("time budget reached", flush=True)
            break
        if args.qat_frac > 0 and not qat_on and step >= args.steps * (1 - args.qat_frac):
            qat_on = True
            handles = quantised_forward_hooks(model)
            print(f"step {step}: int8 fake-quantisation of weights ON", flush=True)

        for g in opt.param_groups:
            g["lr"] = lr_at(step, args.steps, args.lr, args.warmup, args.min_lr_ratio)
        x, y = data.batch("train", args.batch, args.seq, rng)
        t = time.time()
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        dt = time.time() - t
        step += 1
        rec = {"step": step, "loss": round(loss.item(), 4), "lr": round(opt.param_groups[0]["lr"], 6),
               "grad_norm": round(float(gn), 4), "tok_s": round(args.batch * args.seq / dt),
               "elapsed": round(time.time() - t0, 1)}
        log.write(json.dumps(rec) + "\n")
        log.flush()
        if step % 20 == 0 or step == 1:
            print(json.dumps(rec), flush=True)

        if step % args.eval_every == 0 or step == args.steps:
            v = evaluate()
            rec = {"step": step, "val_loss": round(v, 5), "val_ppl": round(math.exp(v), 3),
                   "elapsed": round(time.time() - t0, 1)}
            history.append(rec)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print("EVAL " + json.dumps(rec), flush=True)
            if v < best_val:
                best_val = v
                torch.save({"model": model.state_dict(), "arch": args.arch,
                            "cfg": cfg.to_dict(), "step": step, "val_loss": v}, 
                           os.path.join(args.out, "best.pt"))
        if step % args.save_every == 0:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "arch": args.arch,
                        "cfg": cfg.to_dict(), "step": step, "best_val": best_val,
                        "history": history}, os.path.join(args.out, "last.pt"))

    for h in handles:
        h.remove()
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "arch": args.arch,
                "cfg": cfg.to_dict(), "step": step, "best_val": best_val, "history": history},
               os.path.join(args.out, "last.pt"))
    metrics = {"arch": args.arch, "params": n_all, "params_non_embedding": n_nonemb,
               "steps": step, "tokens": step * args.batch * args.seq,
               "best_val_loss": best_val, "best_val_ppl": math.exp(best_val),
               "history": history, "wall_seconds": round(time.time() - t0, 1),
               "qat": qat_on}
    json.dump(metrics, open(os.path.join(args.out, "metrics.json"), "w"), indent=1)
    print("DONE " + json.dumps({k: metrics[k] for k in
                                ("steps", "tokens", "best_val_loss", "best_val_ppl", "wall_seconds")}),
          flush=True)

    # a sample from the best checkpoint, for the record
    best = torch.load(os.path.join(args.out, "best.pt"), map_location="cpu")
    model.load_state_dict(best["model"])
    model.eval()
    from src.cadence.tokenizer import BPETokenizer
    tok = BPETokenizer.load(os.path.join(args.data, "tokenizer.json"))
    prompt = "It is a truth universally acknowledged"
    ids = torch.tensor([tok.encode(prompt)])
    out = model.generate(ids, args.sample_tokens, temperature=0.8, top_k=40)[0].tolist()
    text = tok.decode(out)
    open(os.path.join(args.out, "sample.txt"), "w").write(text)
    print("SAMPLE:", text[:400].replace("\n", " "), flush=True)


if __name__ == "__main__":
    main()
