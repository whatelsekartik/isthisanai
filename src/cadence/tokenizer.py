"""Cadence byte-level BPE tokenizer.

Written from scratch for this project (no `tokenizers`, no `sentencepiece`, no
`tiktoken`). Two-stage:

1. A GPT-2-style regex pre-tokenizer splits text into word-ish chunks.
2. Byte-level BPE merges are learned on the *unique* words weighted by frequency
   (the standard trick that makes training O(types) rather than O(corpus)).

The vocabulary is exactly `256 + merges` so that encoding is lossless for arbitrary
UTF-8 input (every byte is representable). The vocabulary is stored as JSON so the
JS/WASM runtime can rebuild the merge table without Python.

Layout of ids:  0..255 = raw bytes, 256..255+M = learned merges in learn order.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from typing import Dict, Iterable, List, Sequence, Tuple

# GPT-2 split: contractions | letters with optional leading space | numbers |
# punctuation/symbols with optional leading space | underscore | whitespace runs.
#
# `_+` is listed explicitly: `_` is a word character but neither a letter (`[^\W\d_]`)
# nor a digit, so without this alternative it matches nothing and *silently disappears*
# from the output. Found by tests/test_tokenizer.py::test_random_bytes_roundtrip.
PAT = re.compile(
    r"'(?:[sdmt]|ll|ve|re)"
    r"| ?[^\W\d_]+"
    r"| ?\d+"
    r"| ?[^\s\w_]+"
    r"|_+"
    r"|\s+",
    re.UNICODE,
)


class BPETokenizer:
    def __init__(self, merges: Sequence[Tuple[int, int]] | None = None):
        self.merges: List[Tuple[int, int]] = list(merges or [])
        self._ranks: Dict[Tuple[int, int], int] = {
            p: i for i, p in enumerate(self.merges)
        }
        # byte -> id is identity; merge id = 256 + rank
        self._cache: Dict[str, Tuple[int, ...]] = {}

    # ---------------------------------------------------------------- training
    @staticmethod
    def _words(text: str) -> Counter:
        return Counter(PAT.findall(text))

    def train(self, text: str, vocab_size: int, verbose: bool = True) -> "BPETokenizer":
        """Learn BPE merges with incremental pair counting.

        Words are stored once with their frequency; for every merge we only revisit the
        word types that actually contain the merged pair, which keeps training linear in
        corpus size instead of quadratic in vocab size.
        """
        merges_needed = vocab_size - 256
        word_freqs = self._words(text)
        # id -> [tuple(ids), freq]
        words: Dict[int, list] = {}
        pair_counts: Counter = Counter()
        pair_words: Dict[Tuple[int, int], set] = {}
        for wid, (w, c) in enumerate(word_freqs.items()):
            if not w:
                continue
            ids = tuple(w.encode("utf-8"))
            words[wid] = [ids, c]
            for pr, k in Counter(zip(ids, ids[1:])).items():
                pair_counts[pr] += c * k
                pair_words.setdefault(pr, set()).add(wid)

        for step in range(merges_needed):
            if not pair_counts:
                break
            best, best_count = max(pair_counts.items(), key=lambda kv: (kv[1], -kv[0][0]))
            if best_count < 2:
                break
            new_id = 256 + len(self.merges)
            self.merges.append(best)
            self._ranks[best] = len(self.merges) - 1

            # rebuild only the affected word types
            for wid in list(pair_words.get(best, ())):
                ids, c = words[wid]
                for pr, k in Counter(zip(ids, ids[1:])).items():
                    pair_counts[pr] -= c * k
                    if pair_counts[pr] <= 0:
                        pair_counts.pop(pr, None)
                    s = pair_words.get(pr)
                    if s is not None:
                        s.discard(wid)
                        if not s:
                            pair_words.pop(pr, None)
                new_ids = self._apply_merge(ids, best, new_id)
                words[wid][0] = new_ids
                for pr, k in Counter(zip(new_ids, new_ids[1:])).items():
                    pair_counts[pr] += c * k
                    pair_words.setdefault(pr, set()).add(wid)
            pair_words.pop(best, None)
            if verbose and (step + 1) % 256 == 0:
                print(f"  merge {step + 1}/{merges_needed}: {best} count={best_count}", flush=True)
        self._cache.clear()
        return self

    @staticmethod
    def _apply_merge(ids: Tuple[int, ...], pair: Tuple[int, int], new_id: int) -> Tuple[int, ...]:
        out: List[int] = []
        i = 0
        n = len(ids)
        a, b = pair
        while i < n:
            if i < n - 1 and ids[i] == a and ids[i + 1] == b:
                out.append(new_id)
                i += 2
            else:
                out.append(ids[i])
                i += 1
        return tuple(out)

    # ---------------------------------------------------------------- encoding
    def _encode_word(self, word: str) -> Tuple[int, ...]:
        cached = self._cache.get(word)
        if cached is not None:
            return cached
        ids = list(word.encode("utf-8"))
        if len(ids) > 1:
            while True:
                best_rank = None
                best_pos = -1
                for i in range(len(ids) - 1):
                    r = self._ranks.get((ids[i], ids[i + 1]))
                    if r is not None and (best_rank is None or r < best_rank):
                        best_rank, best_pos = r, i
                if best_rank is None:
                    break
                a, b = ids[best_pos], ids[best_pos + 1]
                ids[best_pos:best_pos + 2] = [256 + best_rank]
                del a, b
        out = tuple(ids)
        if len(self._cache) < 1_000_000:
            self._cache[word] = out
        return out

    def encode(self, text: str) -> List[int]:
        """Lossless: every input character is covered by a match or by the gap fallback."""
        ids: List[int] = []
        pos = 0
        for m in PAT.finditer(text):
            if m.start() > pos:                      # defensive: never drop input
                ids.extend(text[pos:m.start()].encode("utf-8"))
            ids.extend(self._encode_word(m.group()))
            pos = m.end()
        if pos < len(text):
            ids.extend(text[pos:].encode("utf-8"))
        return ids

    def encode_batch(self, texts: Iterable[str]) -> List[List[int]]:
        return [self.encode(t) for t in texts]

    def decode(self, ids: Sequence[int]) -> str:
        """Inverse of encode. Ids >= 256 expand through the merge table."""
        if not self.merges:
            return bytes(int(i) & 0xFF for i in ids).decode("utf-8", "replace")
        table: List[List[int]] = [[i] for i in range(256)]
        for (a, b) in self.merges:
            table.append(table[a] + table[b])
        out = bytearray()
        for i in ids:
            if 0 <= i < len(table):
                out.extend(table[i])
            else:
                raise ValueError(f"id out of range: {i}")
        return out.decode("utf-8", "replace")

    # ------------------------------------------------------------ persistence
    @property
    def vocab_size(self) -> int:
        return 256 + len(self.merges)

    def save(self, path: str) -> None:
        payload = {
            "kind": "cadence-bpe-bytelevel-v1",
            "merges": [[int(a), int(b)] for a, b in self.merges],
            "pre_tokenizer": PAT.pattern,
        }
        with open(path, "w") as f:
            json.dump(payload, f)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        with open(path) as f:
            payload = json.load(f)
        return cls([tuple(m) for m in payload["merges"]])


if __name__ == "__main__":  # tiny smoke test
    tok = BPETokenizer().train("hello world hello there world" * 20, 300, verbose=False)
    s = "hello world, there!"
    enc = tok.encode(s)
    print(enc, tok.decode(enc))
    assert tok.decode(enc) == s
    print("roundtrip ok", tok.vocab_size)
