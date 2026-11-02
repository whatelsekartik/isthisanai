"""Chunk-parallel training path must agree with the token-by-token recurrence.

The token path (`_scan`) is the specification: it is what the JavaScript runtime
implements. The chunk path is an algebraic rearrangement used to train in parallel. If
they disagree, the deployed model is not the trained model.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.cadence.model import CadenceConfig, CadenceLM


def make(seed=0, **kw):
    torch.manual_seed(seed)
    cfg = CadenceConfig(vocab_size=97, d_model=32, n_heads=2, n_blocks=2, n_loops=2,
                        mlp_hidden=48, chunk=8, seq_len=16,
                        bands=((0, 4), (0, 8)), **kw)
    m = CadenceLM(cfg).double().eval()
    return m


def test_chunk_matches_token():
    m = make()
    idx = torch.randint(0, 97, (3, 16))
    m.cfg.mode = "chunk"
    logits_c, state_c = m(idx, return_state=True)
    tok = make()                      # same seed -> same weights
    tok.load_state_dict(m.state_dict())
    tok.cfg.mode = "token"
    logits_t, state_t = tok(idx, return_state=True)
    d_logits = (logits_c - logits_t).abs().max().item()
    d_state = max((a[0] - b[0]).abs().max().item() for a, b in zip(state_c, state_t))
    assert d_logits < 1e-5, f"logit mismatch {d_logits}"
    assert d_state < 1e-5, f"state mismatch {d_state}"
    print(f"  chunk vs token: max|Δlogit| = {d_logits:.2e}, max|Δstate| = {d_state:.2e}")


def test_streaming_equals_teacher_forcing():
    """Feeding one token at a time with carried state == one batched forward."""
    m = make()
    idx = torch.randint(0, 97, (1, 16))
    m.cfg.mode = "chunk"
    full = m(idx)
    for mm in m.modules():
        if hasattr(mm, "force_token"):
            mm.force_token = True
    state = [None] * m.cfg.depth
    steps = []
    for t in range(idx.shape[1]):
        lg, state = m(idx[:, t:t + 1], state, return_state=True)
        steps.append(lg[:, 0])
    stepwise = torch.stack(steps, dim=1)
    d = (full - stepwise).abs().max().item()
    assert d < 1e-6, f"streaming mismatch {d}"
    print(f"  streaming vs teacher forcing: max|Δlogit| = {d:.2e}")


def test_causality():
    """Changing a later token must not change earlier outputs."""
    m = make()
    idx = torch.randint(0, 97, (1, 16))
    a = m(idx)
    idx2 = idx.clone()
    idx2[0, 12:] = (idx2[0, 12:] + 7) % 97
    b = m(idx2)
    d_prefix = (a[:, :12] - b[:, :12]).abs().max().item()
    d_suffix = (a[:, 12:] - b[:, 12:]).abs().max().item()
    assert d_prefix < 1e-6, f"future token leaked: {d_prefix}"
    assert d_suffix > 1e-3, "test is vacuous: suffix did not change"
    print(f"  causality: prefix Δ = {d_prefix:.2e}, suffix Δ = {d_suffix:.3f}")


def test_retention_is_dyadic():
    """The decay index must be an integer in [0, n_dyadic-1] (so the runtime can shift)."""
    m = make()
    mx = m.blocks[0].mixer
    g = torch.randn(4, 7, 2) * 5
    a = mx.decay_index(g, torch.rand(4, 7, 2) * 3)
    assert a.shape == g.shape
    assert torch.all(a == a.round()), "retention exponent is not integral"
    assert a.min() >= 0 and a.max() <= m.cfg.n_dyadic - 1
    print(f"  dyadic retention: a ∈ [{a.min().item():.0f}, {a.max().item():.0f}]")


if __name__ == "__main__":
    for fn in [test_chunk_matches_token, test_streaming_equals_teacher_forcing,
               test_causality, test_retention_is_dyadic]:
        print(fn.__name__)
        fn()
    print("all equivalence tests passed")
