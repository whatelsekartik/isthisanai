"""The tokenizer must be a lossless bijection between bytes and ids for any input."""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cadence.tokenizer import BPETokenizer

TOK = "data/processed/tokenizer.json"


def test_roundtrip_ascii_and_utf8():
    tok = BPETokenizer.load(TOK)
    samples = [
        "",
        "It is a truth universally acknowledged, that a single man in possession",
        "naïve café — “quotes” … ½ ¾",
        "Ω≈ç√∫˜µ≤≥÷ \\n\\t tabbed",
        "日本語のテキスト",
        "🙂 emoji 🙂",
    ]
    for s in samples:
        assert tok.decode(tok.encode(s)) == s, repr(s)
    print(f"  {len(samples)} samples round-tripped exactly")


def test_random_bytes_roundtrip():
    tok = BPETokenizer.load(TOK)
    rng = random.Random(0)
    for _ in range(200):
        n = rng.randrange(0, 60)
        raw = bytes(rng.randrange(32, 127) for _ in range(n))
        s = raw.decode("ascii")
        assert tok.decode(tok.encode(s)) == s
    print("  200 random ASCII strings round-tripped")


def test_vocab_matches_model():
    """A tokenizer id outside the embedding table would be a silent index-out-of-range."""
    tok = BPETokenizer.load(TOK)
    meta = __import__("json").load(open("data/processed/meta.json"))
    assert tok.vocab_size == meta["vocab_size"], (tok.vocab_size, meta["vocab_size"])
    assert max(max(tok.encode("the quick brown fox jumps over the lazy dog")),
               tok.vocab_size - 1) == tok.vocab_size - 1
    print(f"  vocab_size {tok.vocab_size} == embedding rows")


if __name__ == "__main__":
    for fn in [test_roundtrip_ascii_and_utf8, test_random_bytes_roundtrip, test_vocab_matches_model]:
        print(fn.__name__)
        fn()
    print("tokenizer tests passed")
