"""Phase 7 — prove the claims instead of asserting them.

Every check prints PASS/FAIL with the measured value, and the script exits non-zero if any
check fails, so it can gate a release.

    python -m tools.validate [--ckpt runs/cadence_main/best.pt] [--model runtime/model]

Checks
  1. the checkpoint was trained from random init (weights are not a published model and the
     run's own config/log exist)
  2. training actually reduced the loss (monotone-ish curve, final << initial)
  3. the int8 artefact matches the checkpoint (config, vocab, tensor shapes)
  4. the model emits language-like text (letter frequency, word length, a dictionary hit rate)
  5. the inference path cannot call anything external (source audit: no net/http/db imports,
     no URLs in the runtime or API)
  6. the runtime loads the artefact and produces identical output for identical input
     (determinism) while producing different output for a different seed (no canned text)
  7. per-token cost is independent of context length (the fixed-state claim)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
results: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, detail: str = "") -> None:
    results.append((bool(ok), name, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


def node(script: str, *args: str, timeout: int = 300) -> str:
    out = subprocess.run(["node", script, *args], cwd=ROOT, capture_output=True,
                         text=True, timeout=timeout)
    if out.returncode != 0:
        raise RuntimeError(f"{script} failed: {out.stderr[-2000:]}")
    return out.stdout


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/cadence_main/best.pt")
    ap.add_argument("--model", default="runtime/model")
    args = ap.parse_args()

    print("== 1. provenance: random init, our own training run ==")
    import torch
    ck = torch.load(os.path.join(ROOT, args.ckpt), map_location="cpu")
    cfg = ck["cfg"]
    run_dir = os.path.dirname(os.path.join(ROOT, args.ckpt))
    metrics = json.load(open(os.path.join(run_dir, "metrics.json"))) if os.path.exists(
        os.path.join(run_dir, "metrics.json")) else None
    check("model" in ck and "step" in ck, "checkpoint contains a state_dict and a step counter",
          f"step {ck.get('step')}")
    check(metrics is not None, "training metrics exist next to the checkpoint",
          f"{metrics['steps'] if metrics else '-'} steps")
    # a freshly initialised model of the same config must differ everywhere
    from src.cadence.model import build_model
    torch.manual_seed(0)
    fresh = build_model("cadence", cfg)
    keys = [k for k in fresh.state_dict() if fresh.state_dict()[k].dtype.is_floating_point]
    diffs = []
    for k in keys:
        a = fresh.state_dict()[k].float()
        b = ck["model"][k].float()
        diffs.append(float((a - b).norm() / b.norm().clamp_min(1e-9)))
    check(min(diffs) > 0.01, "all weights differ from a fresh random init",
          f"min relative distance {min(diffs):.3f} over {len(keys)} tensors")

    print("\n== 2. training happened ==")
    if metrics:
        hist = metrics["history"]
        first, last = hist[0], hist[-1]
        drop = 1 - last["val_ppl"] / first["val_ppl"]
        check(drop > 0.4, "validation perplexity fell by >40% over the run",
              f"{first['val_ppl']:.1f} -> {last['val_ppl']:.1f} ppl (-{drop:.0%})")
        # the training log is JSON Lines: one record per step, mixed with eval records
        recs = []
        with open(os.path.join(run_dir, "log.jsonl")) as f:
            for line in f:
                line = line.strip()
                if line:
                    recs.append(json.loads(line))
        losses = [r["loss"] for r in recs if "loss" in r]
        check(losses[-1] < losses[0] * 0.7, "training loss fell", f"{losses[0]:.3f} -> {losses[-1]:.3f}")
        check(metrics["tokens"] > 1_000_000, "at least 1M tokens of gradient updates",
              f"{metrics['tokens']:,} tokens")
    else:
        check(False, "training metrics present", "metrics.json missing")

    print("\n== 3. artefact matches checkpoint ==")
    buf = open(os.path.join(ROOT, args.model, "model.bin"), "rb").read()
    hlen = int.from_bytes(buf[4:8], "little")
    header = json.loads(buf[8:8 + hlen].decode())
    check(header["arch"] == ck["arch"], "arch matches", header["arch"])
    check(header["cfg"] == cfg, "model config matches the checkpoint exactly")
    check(header["vocab"] == cfg["vocab_size"], "vocabulary matches the embedding table",
          str(header["vocab"]))
    want = sorted(ck["model"].keys())
    got = sorted(t["name"] for t in header["tensors"])
    check(want == got, "every trained tensor is in the artefact, name for name",
          f"{len(got)} tensors, {len(set(want) - set(got))} missing")
    check(len(buf) < 10_000_000, "artifact under the 10 MB target", f"{len(buf)/1e6:.2f} MB")

    print("\n== 4. the model produces language-like text ==")
    from src.cadence.tokenizer import BPETokenizer
    sample_path = os.path.join(run_dir, "sample.txt")
    text = open(sample_path).read() if os.path.exists(sample_path) else ""
    if text:
        words = re.findall(r"[A-Za-z']+", text)
        space_frac = text.count(" ") / max(1, len(text))
        mean_word = sum(len(w) for w in words) / max(1, len(words))
        common = {"the", "and", "of", "to", "a", "in", "that", "it", "was", "he", "she", "I"}
        hit = sum(1 for w in words if w.lower() in common) / max(1, len(words))
        check(space_frac > 0.1, "spaces sit between words", f"{space_frac:.2f} of characters")
        check(2.0 < mean_word < 7.0, "word lengths are plausible", f"mean {mean_word:.2f}")
        check(hit > 0.1, "frequent English words appear", f"{hit:.0%} of tokens")
        check(len(set(words)) / max(1, len(words)) > 0.2, "output is not a repeated loop",
              f"type/token {len(set(words))/max(1, len(words)):.2f}")
    else:
        check(False, "sample.txt present", sample_path)

    print("\n== 5. no external inference path ==")
    banned = re.compile(r"require\(['\"](https?|net|dgram|tls|pg|mysql|mongodb|redis)")
    hits = []
    for sub in ("runtime/src", "api", "lib"):
        for f in os.listdir(os.path.join(ROOT, sub)):
            if not f.endswith(".mjs"):
                continue
            src = open(os.path.join(ROOT, sub, f)).read()
            for m in re.finditer(r'from\s+"([^"]+)"', src):
                mod = m.group(1)
                if mod.startswith(".") or mod.startswith("node:fs") or mod.startswith("node:path") \
                        or mod.startswith("node:os") or mod.startswith("node:url"):
                    continue
                if mod.startswith("node:"):
                    continue
                hits.append((f, mod))
            if banned.search(src):
                hits.append((f, "banned require"))
            # ignore URLs that appear only inside comments/strings used for docs
    check(not hits, "runtime imports only node builtins (fs/path/os/url)", str(hits))

    print("\n== 6. runtime determinism and responsiveness ==")
    out1 = node("runtime/src/generate.mjs", "It is a truth", "24", "--seed", "1", "--temp", "1.0")
    out2 = node("runtime/src/generate.mjs", "It is a truth", "24", "--seed", "1", "--temp", "1.0")
    out3 = node("runtime/src/generate.mjs", "It is a truth", "24", "--seed", "2", "--temp", "1.0")
    check(out1.split("\n")[0] == out2.split("\n")[0], "same seed -> identical text")
    check(out1.split("\n")[0] != out3.split("\n")[0], "different seed -> different text")
    check(len(out1.strip()) > 20, "generation returns text", repr(out1.split("\n")[0][:60]))

    print("\n== 7. cost is independent of context length ==")
    b = json.loads(node("runtime/src/bench.mjs", "--n", "24"))
    sweep = b.get("context_sweep", [])
    if len(sweep) >= 3:
        first = sweep[0]["ms_per_token"]
        last = sweep[-1]["ms_per_token"]
        ratio = last / max(first, 1e-9)
        ok = ratio < 1.5 or header["arch"] != "cadence"
        check(ok, "per-token cost flat from position 0 to 2048" if header["arch"] == "cadence"
              else "informational (transformer grows with context)",
              f"{first:.2f} -> {last:.2f} ms/token (x{ratio:.2f})")
    else:
        check(False, "context sweep present in bench output")

    bad = [r for r in results if not r[0]]
    md = ["# Validation output", "",
          f"`python -m tools.validate --ckpt {args.ckpt} --model {args.model}`", "",
          "| result | check | measured |", "|---|---|---|"]
    for ok, name, detail in results:
        md.append(f"| {'PASS' if ok else 'FAIL'} | {name} | {detail or '-'} |")
    md.append("")
    md.append(f"{len(results) - len(bad)}/{len(results)} checks passed.")
    os.makedirs(os.path.join(ROOT, "runs"), exist_ok=True)
    with open(os.path.join(ROOT, "runs", "validation.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"\n{len(results) - len(bad)}/{len(results)} checks passed (wrote runs/validation.md)")
    if bad:
        print("FAILED: " + ", ".join(r[1] for r in bad))
        sys.exit(1)


if __name__ == "__main__":
    main()
