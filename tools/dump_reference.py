"""Dump an fp32 reference forward pass for runtime parity.

    python -m tools.dump_reference --ckpt runs/cadence_main/best.pt \
        --prompt "It is a truth" --out runs/reference.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from src.cadence.model import build_model
from src.cadence.tokenizer import BPETokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/cadence_main/best.pt")
    ap.add_argument("--tokenizer", default="data/processed/tokenizer.json")
    ap.add_argument("--prompt", default="It is a truth universally acknowledged")
    ap.add_argument("--out", default="runs/reference.json")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu")
    cfg = dict(ck["cfg"])
    model = build_model(ck["arch"], cfg)
    model.load_state_dict(ck["model"])
    model.eval()
    if ck["arch"] == "cadence":
        model.cfg.mode = "token"          # the path the JS runtime implements
    tok = BPETokenizer.load(args.tokenizer)
    ids = tok.encode(args.prompt)
    with torch.no_grad():
        logits = model(torch.tensor([ids]))
    last = logits[0, -1].double().numpy()
    payload = {
        "arch": ck["arch"],
        "prompt": args.prompt,
        "ids": ids,
        "ckpt_step": ck.get("step"),
        "logits": [round(float(v), 5) for v in last],
        "top5": [int(i) for i in last.argsort()[-5:][::-1]],
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(payload, open(args.out, "w"))
    print(f"wrote {args.out}: {len(ids)} tokens, top5={payload['top5']}")


if __name__ == "__main__":
    main()
