"""Corpus preparation: clean text -> BPE tokenizer -> uint16 token binaries.

Steps
-----
1. concatenate `data/raw/books/*.txt` (+ `data/raw/tinyshakespeare.txt` if present)
2. strip leftover licence/metadata noise, normalise newlines, drop near-empty files
3. hold out a validation slice (by document, never mid-document for train/val leakage)
4. train a byte-level BPE tokenizer from scratch on the *train* slice
5. tokenize both slices into <split>.bin (uint16 little-endian, numpy memmap friendly)

Usage:
    python -m tools.prepare_data --vocab 2304 --out data/processed
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.cadence.tokenizer import BPETokenizer

NOISE = re.compile(
    r"(?im)^(produced by|transcriber|etext|end of the project|project gutenberg|"
    r"\*\*\* ?start|\*\*\* ?end|this ebook|title:|author:|release date|language:|"
    r"character set|encoding:|credits?:).*$"
)


def clean(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = NOISE.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--vocab", type=int, default=2304)
    ap.add_argument("--val-frac", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    docs = []
    for name in sorted(os.listdir(os.path.join(args.raw, "books"))):
        if not name.endswith(".txt"):
            continue
        p = os.path.join(args.raw, "books", name)
        t = clean(open(p, encoding="utf-8", errors="replace").read())
        if len(t) > 20_000:
            docs.append((name, t))
    shake = os.path.join(args.raw, "tinyshakespeare.txt")
    if os.path.exists(shake):
        docs.append(("tinyshakespeare", clean(open(shake, encoding="utf-8").read())))

    rng = random.Random(args.seed)
    rng.shuffle(docs)
    n_val = max(1, int(len(docs) * args.val_frac))
    val_docs, train_docs = docs[:n_val], docs[n_val:]
    train_text = "\n\n".join(t for _, t in train_docs)
    val_text = "\n\n".join(t for _, t in val_docs)
    print(f"docs train={len(train_docs)} val={len(val_docs)}  "
          f"chars train={len(train_text)/1e6:.1f}M val={len(val_text)/1e6:.1f}M", flush=True)

    os.makedirs(args.out, exist_ok=True)
    tok_path = os.path.join(args.out, "tokenizer.json")
    print(f"training BPE with vocab={args.vocab} ...", flush=True)
    tok = BPETokenizer().train(train_text, args.vocab)
    tok.save(tok_path)
    print(f"tokenizer vocab={tok.vocab_size} -> {tok_path}", flush=True)
    # report compression
    probe = train_text[:200_000]
    print(f"compression: {len(probe.encode('utf-8'))/len(tok.encode(probe)):.2f} bytes/token",
          flush=True)

    for split, text in (("train", train_text), ("val", val_text)):
        ids = np.array(tok.encode(text), dtype=np.uint16)
        p = os.path.join(args.out, f"{split}.bin")
        ids.tofile(p)
        print(f"{split}: {len(ids)/1e6:.2f}M tokens -> {p}", flush=True)

    meta = {
        "vocab_size": tok.vocab_size,
        "seed": args.seed,
        "val_docs": [n for n, _ in val_docs],
        "train_chars": len(train_text),
        "val_chars": len(val_text),
        "train_tokens": int(os.path.getsize(os.path.join(args.out, "train.bin")) // 2),
        "val_tokens": int(os.path.getsize(os.path.join(args.out, "val.bin")) // 2),
        "sources": [n for n, _ in docs],
    }
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps({k: v for k, v in meta.items() if k != "sources"}, indent=1))


if __name__ == "__main__":
    main()
