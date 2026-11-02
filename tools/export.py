"""Export a trained checkpoint to the int8 runtime format (`model.bin`).

    python -m tools.export --ckpt runs/cadence_main/best.pt --out runtime/model

Format (little endian, "CDN1"):

    magic  4 bytes  "CDN1"
    hlen   u32      length of the JSON header
    header JSON     {"arch", "cfg", "vocab", "tensors":[...], "quant":{...}}
    data            tensor blobs, each 8-byte aligned

Every 2-D weight matrix with more than 512 parameters is stored as int8 with one fp32
scale per output row; everything else (biases, norms, per-loop gammas, band bounds,
per-head scale, the surprise gain) stays fp32. A JSON report records, per tensor, the
relative quantisation error and the end-to-end val loss before and after quantising.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F

from src.cadence.model import build_model


def quantize_rows(w: torch.Tensor):
    """Symmetric per-output-channel int8 quantisation."""
    flat = w.reshape(w.shape[0], -1).float()
    scale = flat.abs().amax(dim=1).clamp_min(1e-8) / 127.0
    q = (flat / scale.unsqueeze(1)).round().clamp(-127, 127).to(torch.int8)
    deq = q.float() * scale.unsqueeze(1)
    err = (deq - flat).norm() / flat.norm().clamp_min(1e-9)
    return q, scale, float(err)


def align(buf: bytearray, n: int = 8) -> None:
    while len(buf) % n:
        buf.append(0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/cadence_main/best.pt")
    ap.add_argument("--out", default="runtime/model")
    ap.add_argument("--tokenizer", default="data/processed/tokenizer.json")
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--val-batches", type=int, default=4)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--no-report", action="store_true")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu")
    arch, cfg = ck["arch"], ck["cfg"]
    model = build_model(arch, cfg)
    model.load_state_dict(ck["model"])
    model.eval()

    tensors = []
    for name, p in model.named_parameters():
        t = p.detach()
        # Only real matrices are quantised. Small 2-D tensors such as the per-loop
        # adapters ((loops, d) = (3,192) = 576) lose more accuracy than they save bytes,
        # and quantising them silently broke the runtime once (it read raw int8 as floats).
        if t.dim() >= 2 and t.numel() >= 4096:
            q, scale, err = quantize_rows(t)
            tensors.append(dict(name=name, dtype="i8", shape=list(q.shape),
                                data=q.numpy().tobytes(), scales=scale.numpy().astype("<f4").tobytes(),
                                rel_err=round(err, 6)))
        else:
            tensors.append(dict(name=name, dtype="f32", shape=list(t.shape),
                                data=t.float().numpy().astype("<f4").tobytes(),
                                scales=None, rel_err=0.0))

    body = bytearray()
    index = []
    for t in tensors:
        align(body)
        off = len(body)
        body += t["data"]
        soff = -1
        if t["scales"] is not None:
            align(body)
            soff = len(body)
            body += t["scales"]
        index.append(dict(name=t["name"], dtype=t["dtype"], shape=t["shape"],
                          offset=off, nbytes=len(t["data"]), scales=soff))

    header = dict(arch=arch, cfg=cfg, vocab=int(cfg["vocab_size"]), tensors=index,
                  quant=dict(scheme="int8 row-wise, fp32 per-row scale",
                             n_int8=sum(1 for t in tensors if t["dtype"] == "i8"),
                             n_f32=sum(1 for t in tensors if t["dtype"] == "f32"),
                             mean_rel_err=round(float(np.mean([t["rel_err"] for t in tensors
                                                               if t["dtype"] == "i8"])), 6)))
    hjson = json.dumps(header, separators=(",", ":")).encode()
    while (8 + len(hjson)) % 8:
        hjson += b" "

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "model.bin")
    with open(path, "wb") as f:
        f.write(b"CDN1")
        f.write(struct.pack("<I", len(hjson)))
        f.write(hjson)
        f.write(bytes(body))
    size = os.path.getsize(path)
    print(f"wrote {path}: {size/1e6:.2f} MB ({header['quant']['n_int8']} int8, "
          f"{header['quant']['n_f32']} fp32, mean rel err {header['quant']['mean_rel_err']})")

    if os.path.exists(args.tokenizer):
        shutil.copy(args.tokenizer, os.path.join(args.out, "tokenizer.json"))

    # ---------------------------------------------------------------- validation
    if not args.no_report:
        from src.train import TokenData
        data = TokenData(args.data)
        import random
        rng = random.Random(999)

        def val_loss(net):
            tot, cnt = 0.0, 0
            with torch.no_grad():
                for _ in range(args.val_batches):
                    x, y = data.batch("val", args.batch, args.seq, rng)
                    lg = net(x)
                    tot += F.cross_entropy(lg.reshape(-1, lg.size(-1)), y.reshape(-1),
                                           reduction="sum").item()
                    cnt += y.numel()
            return tot / cnt

        def quantise_inplace(net):
            with torch.no_grad():
                for name, p in net.named_parameters():
                    if p.dim() >= 2 and p.numel() >= 4096:
                        q, scale, _ = quantize_rows(p)
                        p.copy_((q.float() * scale.unsqueeze(1)).reshape(p.shape))

        rng = random.Random(999)
        fp = val_loss(model)
        quantise_inplace(model)
        rng = random.Random(999)
        q8 = val_loss(model)
        report = dict(arch=arch, params=int(sum(p.numel() for p in model.parameters())),
                      file_bytes=size, val_loss_fp32=round(fp, 5),
                      val_loss_int8=round(q8, 5),
                      val_ppl_fp32=round(float(np.exp(fp)), 3),
                      val_ppl_int8=round(float(np.exp(q8)), 3),
                      delta_ppl=round(float(np.exp(q8) - np.exp(fp)), 3),
                      ckpt_step=ck.get("step"), quant=header["quant"])
        json.dump(report, open(os.path.join(args.out, "quant_report.json"), "w"), indent=1)
        print(f"val loss fp32 {fp:.5f} -> int8 {q8:.5f}  "
              f"(ppl {np.exp(fp):.2f} -> {np.exp(q8):.2f})")


if __name__ == "__main__":
    main()
