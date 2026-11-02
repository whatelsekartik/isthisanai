"""Cadence: a dyadic multi-timescale delta memory LM, plus matched baselines.

Everything here is written from scratch: no pretrained weights, no library layers
beyond `nn.Module`/`nn.Linear`/`nn.Conv1d` primitives.

Core idea (see docs/02-architecture-proposals.md):

  S_l,h <- A(g_l,h) * S_l,h                        per-channel decay, A = 2^-a, a in {0..15}
  r      = S^T (erase * k_hat)                     read along the erased key
  S     <- S + k_hat (write*v - r)^T               delta / fast-weight write
  y      = S^T q_hat * silu(o)                     read along the query

Two things make this different from a plain gated delta rule:

1. **Dyadic retention.** The decay is constrained to a power of two per channel,
   `a_t = clamp(round(lo + (hi-lo)*sigmoid(g_t) + kappa*novelty), lo, hi)`, so that at
   inference `S <- S >> a` is an exact integer operation - no multiplies, no floating
   point, bit-identical on every CPU. `lo/hi` come from a per-head *band* (fast heads and
   slow heads), so the model gets multiple timescales without extra state.
2. **Retention modulated by surprise.** A leaked running estimate of the write error
   `|write*v - r|` is fed back into the decay index, so a channel that is failing to
   predict is retained longer. The gauge is a running mean, which costs O(1) per token
   instead of the O(d^2) probe a snapshot-based gauge would need.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- config


@dataclass
class CadenceConfig:
    vocab_size: int = 4096
    d_model: int = 192
    n_heads: int = 4
    n_blocks: int = 2          # physical blocks (weights)
    n_loops: int = 3           # applications per block -> depth = n_blocks*n_loops
    mlp_hidden: int = 512
    conv_kernel: int = 4
    seq_len: int = 256
    # dyadic retention
    n_dyadic: int = 16         # decay indices 0..15  (2^0 .. 2^-15)
    bands: tuple = ((3, 15), (3, 15), (1, 6), (1, 6))  # per-head (lo, hi) retentiveness
    novelty: str = "ema"       # "ema" | "none"
    novelty_decay: float = 0.875  # 7/8: dyadic leak, exact in integers
    chunk: int = 32             # chunkwise-parallel training length
    mode: str = "chunk"         # "chunk" (training) | "token" (reference scan)
    gauge_init: float = 0.0    # kappa
    tie_embeddings: bool = True
    dropout: float = 0.0

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0
        assert self.n_dyadic <= 31
        assert len(self.bands) == self.n_heads, "one retention band per head"

    @property
    def d_head(self) -> int:
        return self.d_model // self.n_heads

    @property
    def depth(self) -> int:
        return self.n_blocks * self.n_loops

    def to_dict(self) -> dict:
        d = asdict(self)
        d["bands"] = [list(b) for b in self.bands]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "CadenceConfig":
        d = dict(d)
        d["bands"] = tuple(tuple(b) for b in d["bands"])
        return cls(**d)


# ------------------------------------------------------------------------- building blocks


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


def l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


def ste_round(x: torch.Tensor) -> torch.Tensor:
    """Round with a straight-through gradient (the quantisation-aware training trick)."""
    return x + (x.round() - x).detach()


class CadenceMixer(nn.Module):
    """Banded, dyadically-quantised delta memory with a surprise gauge.

    Retention is a *scalar per head per token* in {2^0 .. 2^-15}: this keeps the
    chunk-parallel training algebra well conditioned (a per-channel rate makes the
    rescaled keys overflow the fp32 exponent range) and it is also the cheapest form
    for the integer runtime, where the decay is a single arithmetic shift.
    Timescales come from the per-head *bands* (see `CadenceConfig.bands`).
    """

    def __init__(self, cfg: CadenceConfig):
        super().__init__()
        self.cfg = cfg
        d, dh, H = cfg.d_model, cfg.d_head, cfg.n_heads
        self.d, self.dh, self.H = d, dh, H
        self.proj = nn.Linear(d, 6 * d, bias=True)       # q k v | erase write outgate
        self.gate = nn.Linear(d, H, bias=True)           # retention index (per head)
        self.conv = nn.Conv1d(3 * d, 3 * d, kernel_size=cfg.conv_kernel,
                              groups=3 * d, bias=True)
        nn.init.constant_(self.conv.weight, 1.0 / cfg.conv_kernel)
        nn.init.zeros_(self.conv.bias)
        self.out = nn.Linear(d, d, bias=False)
        self.head_scale = nn.Parameter(torch.ones(H))
        self.gammas = nn.Parameter(torch.zeros(cfg.n_loops, d))
        nn.init.zeros_(self.gate.weight)                 # start at the band midpoint
        nn.init.zeros_(self.gate.bias)
        with torch.no_grad():
            self.proj.bias[3 * d:5 * d].fill_(2.0)       # sigma(2)=0.88 erase/write
        self.kappa = nn.Parameter(torch.tensor([cfg.gauge_init]))
        self.force_token = False        # incremental decoding uses the token recurrence
        lo = torch.tensor([b[0] for b in cfg.bands], dtype=torch.float32)
        hi = torch.tensor([b[1] for b in cfg.bands], dtype=torch.float32)
        self.register_buffer("band_lo", lo, persistent=False)
        self.register_buffer("band_hi", hi, persistent=False)
        lag = torch.arange(cfg.chunk).float()
        leak = torch.full((cfg.chunk,), cfg.novelty_decay)
        causal = (lag.unsqueeze(1) - lag.unsqueeze(0)) >= 0
        self.register_buffer(
            "nov_kernel", (1 - leak).view(1, -1) * torch.where(causal, leak.pow(
                torch.abs(lag.unsqueeze(1) - lag.unsqueeze(0))), torch.zeros(())),
            persistent=False)

    # ------------------------------------------------------------------ retention
    def decay_index(self, g: torch.Tensor, novelty: torch.Tensor | None) -> torch.Tensor:
        """g: (B,T,H) retention logit -> a: (B,T,H) integer exponent in [0, n_dyadic-1]."""
        cfg = self.cfg
        u = torch.sigmoid(g)
        if novelty is not None and cfg.novelty == "gate":
            u = u + novelty * self.kappa
        a = self.band_lo + (self.band_hi - self.band_lo) * u
        a = a.clamp(0.0, cfg.n_dyadic - 1.0)
        return ste_round(a) if self.training else a.round()

    # --------------------------------------------------- token-by-token (reference)
    def _scan(self, S, q, k, v, erase, write, og, a, nov):
        B, Ts, H, dh = q.shape
        y = q.new_empty(B, Ts, H, dh)
        leak = self.cfg.novelty_decay
        for t in range(Ts):
            kt, qt, vt = k[:, t], q[:, t], v[:, t]
            et = erase[:, t] * kt
            ut = write[:, t] * vt
            S = S * torch.pow(2.0, -a[:, t]).unsqueeze(-1).unsqueeze(-1)
            r = torch.matmul(S.transpose(-1, -2), et.unsqueeze(-1)).squeeze(-1)
            resid = ut - r
            S = S + kt.unsqueeze(-1) * resid.unsqueeze(-2)
            yt = torch.matmul(S.transpose(-1, -2), qt.unsqueeze(-1)).squeeze(-1)
            y[:, t] = yt * og[:, t] * self.head_scale.view(1, -1, 1)
            nov = leak * nov + (1 - leak) * resid.detach().abs().mean(-1)
        return y, S, nov

    # ------------------------------------------------------ chunk-parallel (training)
    def _chunkwise(self, S, q, k, v, erase, write, og, a):
        """(B,C,H,dh) tensors; a: (B,C,H) integer retention exponents.

        Uses the bounded form of the WY/triangular solve: every rescaled quantity is
        multiplied by 2^(c_s - c_r) with s >= r, so nothing exceeds O(C).
        """
        B, C, H, dh = q.shape
        dev = q.device
        # NB: torch.eye(n, dtype=bool).tril(-1) returns all-False (the bool kernel treats
        # the matrix as already triangular); build the masks from ones instead.
        ones = torch.ones(C, C, device=dev, dtype=torch.bool)
        strict, causal = ones.tril(-1), ones.tril(0)

        c = torch.cumsum(-a.to(q.dtype), dim=1)                   # (B,C,H) <= 0 (fp32 by default)
        rho = torch.exp2(c)                                       # 2^c
        Et = erase * k * rho.unsqueeze(-1)                        # (B,C,H,dh)
        Qd = q * rho.unsqueeze(-1)
        U = write * v
        r0 = torch.einsum("bchd,bhde->bche", Et, S)               # (B,C,H,dh)
        uw = U - r0
        inst = uw.detach().abs().mean(-1)                         # (B,C,H) surprise

        y = U.new_empty(B, C, H, dh)
        S_new = torch.empty_like(S)
        Ek = erase * k                                            # raw ẽ (B,C,H,dh)
        for h in range(H):
            kh, eh, qh, uh = k[:, :, h], Ek[:, :, h], q[:, :, h], uw[:, :, h]
            # 2^(c_t - c_s) for t >= s; clamped before masking (exp2(+400) = inf, inf*0 = nan)
            f = torch.exp2((c[:, :, h].unsqueeze(2) - c[:, :, h].unsqueeze(1)).clamp(max=0.0))
            Kt = kh * torch.exp2(c[:, -1:, h] - c[:, :, h]).unsqueeze(-1)   # 2^(c_C-c_s) k̂
            Tmat = torch.matmul(eh, kh.transpose(1, 2)) * f * strict
            w = torch.linalg.solve_triangular(
                Tmat + torch.eye(C, device=dev), uh, upper=False, unitriangular=True)
            QK = torch.matmul(qh, kh.transpose(1, 2)) * f * causal
            y[:, :, h] = torch.matmul(qh * rho[:, :, h].unsqueeze(-1), S[:, h]) \
                + torch.matmul(QK, w)
            S_new[:, h] = (torch.exp2(c[:, -1:, h]).unsqueeze(-1) * S[:, h]
                           + torch.matmul(Kt.transpose(1, 2), w))
        y = y * og * self.head_scale.view(1, 1, -1, 1)
        return y, S_new

    # ------------------------------------------------------------------------ fwd
    def forward(self, x: torch.Tensor, loop: int, state=None):
        """state is (S, conv_window) or a bare S tensor.

        `conv_window` holds the last `conv_kernel-1` *pre-activation* q,k,v rows of this
        loop, so that single-token streaming decode feeds the same FIR filter that training
        does. Without it, feeding one token at a time would silently change the model.
        """
        cfg, d, dh, H = self.cfg, self.d, self.dh, self.H
        K = cfg.conv_kernel
        B, T, _ = x.shape
        S, buf = (state, None) if not isinstance(state, tuple) else state
        h = x * (1.0 + self.gammas[loop])
        q, k, v, eg, wg, og = self.proj(h).split(d, dim=-1)
        qkv_t = torch.cat([q, k, v], dim=-1)                      # (B,T,3d) pre-activation

        if T == 1:
            # streaming: prepend the carried window instead of left-padding
            if buf is None:
                buf = x.new_zeros(B, K - 1, 3 * d)
            win = torch.cat([buf, qkv_t], dim=1)                   # (B,K,3d), oldest first
            wc = self.conv.weight.view(3 * d, K)                   # depthwise, 1 tap per row-k
            out = (win * wc.unsqueeze(0).transpose(1, 2)).sum(1) + self.conv.bias
            buf = win[:, 1:].contiguous()
        else:
            qkv = F.pad(qkv_t.transpose(1, 2), (K - 1, 0))
            out = self.conv(qkv)[..., :T].transpose(1, 2)
            buf = qkv_t[:, -(K - 1):].contiguous()

        q, k, v = out.split(d, dim=-1)
        q, k, v = F.silu(q), F.silu(k), F.silu(v)
        q = l2norm(q, 1e-6).view(B, T, H, dh)
        k = l2norm(k, 1e-6).view(B, T, H, dh)
        v = v.view(B, T, H, dh)
        erase = torch.sigmoid(eg).view(B, T, H, dh)
        write = torch.sigmoid(wg).view(B, T, H, dh)
        og = F.silu(og).view(B, T, H, dh)
        gate = self.gate(h)                                        # (B,T,H)

        S = x.new_zeros(B, H, dh, dh) if S is None else S
        if cfg.mode == "token" or self.force_token:
            nov = x.new_zeros(B, H)
            a = self.decay_index(gate, None)
            y, S, _ = self._scan(S, q, k, v, erase, write, og, a, nov)
            return self.out(y.reshape(B, T, d)), (S, buf)

        C = cfg.chunk
        assert T % C == 0, f"sequence length {T} must be a multiple of chunk {C}"
        lag = torch.arange(C, device=x.device).view(1, C, 1) + 1
        carry = torch.pow(torch.tensor(cfg.novelty_decay, device=x.device), lag)
        ys = []
        nov = x.new_zeros(B, H)
        for start in range(0, T, C):
            sl = slice(start, start + C)
            # surprise gauge, computed from the chunk-start state before any write
            r0 = torch.einsum("bchd,bhde->bche", erase[:, sl] * k[:, sl], S)
            inst = (write[:, sl] * v[:, sl] - r0).detach().abs().mean(-1)     # (B,C,H)
            nov = (torch.einsum("ts,bsh->bth", self.nov_kernel.to(x.device), inst)
                   + carry * nov.unsqueeze(1))
            a = self.decay_index(gate[:, sl], nov if cfg.novelty == "gate" else None)
            y_c, S = self._chunkwise(S, q[:, sl], k[:, sl], v[:, sl], erase[:, sl],
                                     write[:, sl], og[:, sl], a)
            ys.append(y_c)
            nov = nov[:, -1]
        y = torch.cat(ys, dim=1)
        return self.out(y.reshape(B, T, d)), (S, buf)


class SwiGLU(nn.Module):
    def __init__(self, cfg: CadenceConfig):
        super().__init__()
        d, hh = cfg.d_model, cfg.mlp_hidden
        self.gate = nn.Linear(d, hh, bias=False)
        self.up = nn.Linear(d, hh, bias=False)
        self.down = nn.Linear(hh, d, bias=False)
        self.gammas = nn.Parameter(torch.zeros(cfg.n_loops, d))

    def forward(self, x: torch.Tensor, loop: int) -> torch.Tensor:
        h = x * (1.0 + self.gammas[loop])
        return self.down(F.silu(self.gate(h)) * self.up(h))


class CadenceBlock(nn.Module):
    def __init__(self, cfg: CadenceConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.mixer = CadenceMixer(cfg)
        self.norm2 = RMSNorm(cfg.d_model)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, loop, state=None):
        mo, state = self.mixer(self.norm1(x), loop, state)
        x = x + mo
        x = x + self.mlp(self.norm2(x), loop)
        return x, state


# ------------------------------------------------------------------------------- LM


class CadenceLM(nn.Module):
    arch = "cadence"

    def __init__(self, cfg: CadenceConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        nn.init.normal_(self.embed.weight, std=0.02)
        self.blocks = nn.ModuleList([CadenceBlock(cfg) for _ in range(cfg.n_blocks)])
        self.norm_f = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False) if not cfg.tie_embeddings else None
        if cfg.tie_embeddings:
            self.head = None
        # residual-path scaling so that depth 6 starts at a sane variance
        n_app = cfg.depth
        for b in self.blocks:
            with torch.no_grad():
                b.mixer.out.weight.mul_(1.0 / math.sqrt(2 * n_app))
                b.mlp.down.weight.mul_(1.0 / math.sqrt(2 * n_app))

    def forward(self, idx: torch.Tensor, state=None, return_state: bool = False):
        x = self.embed(idx)
        states = []
        for l in range(self.cfg.depth):
            blk = self.blocks[l % self.cfg.n_blocks]
            s = None if state is None else state[l]
            x, s = blk(x, l // self.cfg.n_blocks, s)
            states.append(s)
        logits = F.linear(self.norm_f(x), self.embed.weight)
        if return_state:
            return logits, states
        return logits

    def n_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        return n - self.embed.weight.numel() if non_embedding else n

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, n_new: int, temperature=0.8, top_k=None):
        """Incremental decoding: one token in, one token out, state carried across calls."""
        self.eval()
        for m in self.modules():
            if hasattr(m, "force_token"):
                m.force_token = True
        state = [None] * self.cfg.depth
        logits = None
        for t in range(idx.shape[1]):
            logits, state = self(idx[:, t:t + 1], state, return_state=True)
        out = [idx]
        for _ in range(n_new):
            lg = logits[:, -1] / max(temperature, 1e-6)
            if top_k:
                v, _ = torch.topk(lg, min(top_k, lg.size(-1)))
                lg[lg < v[:, [-1]]] = -float("inf")
            nxt = torch.multinomial(F.softmax(lg, dim=-1), 1)
            out.append(nxt)
            logits, state = self(nxt, state, return_state=True)
        return torch.cat(out, dim=1)


# ------------------------------------------------------------ transformer baseline


@dataclass
class TransformerConfig:
    vocab_size: int = 4096
    d_model: int = 192
    n_heads: int = 4
    n_layers: int = 6
    mlp_hidden: int = 512
    seq_len: int = 256
    rope: bool = False
    tie_embeddings: bool = True

    @property
    def d_head(self) -> int:
        return self.d_model // self.n_heads

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TransformerConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)

    def forward(self, x):
        B, T, d = x.shape
        H, dh = self.cfg.n_heads, self.cfg.d_head
        q, k, v = self.qkv(x).split(d, dim=-1)
        q, k, v = (t.view(B, T, H, dh).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).reshape(B, T, d))


class TransformerBlock(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = RMSNorm(cfg.d_model)
        d, hh = cfg.d_model, cfg.mlp_hidden
        self.gate = nn.Linear(d, hh, bias=False)
        self.up = nn.Linear(d, hh, bias=False)
        self.down = nn.Linear(hh, d, bias=False)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        h = self.norm2(x)
        x = x + self.down(F.silu(self.gate(h)) * self.up(h))
        return x


class TransformerLM(nn.Module):
    arch = "transformer"

    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        nn.init.normal_(self.embed.weight, std=0.02)
        self.pos = nn.Embedding(cfg.seq_len, cfg.d_model)
        nn.init.normal_(self.pos.weight, std=0.01)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm_f = RMSNorm(cfg.d_model)
        self.head = None if cfg.tie_embeddings else nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        n_app = cfg.n_layers
        for b in self.blocks:
            with torch.no_grad():
                b.attn.proj.weight.mul_(1.0 / math.sqrt(2 * n_app))
                b.down.weight.mul_(1.0 / math.sqrt(2 * n_app))

    def forward(self, idx, state=None, return_state=False):
        B, T = idx.shape
        x = self.embed(idx) + self.pos.weight[:T]
        for b in self.blocks:
            x = b(x)
        logits = F.linear(self.norm_f(x), self.embed.weight)
        return (logits, None) if return_state else logits

    def n_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters() if p is not self.pos.weight)
        return n - self.embed.weight.numel() if non_embedding else n

    @torch.no_grad()
    def generate(self, idx, n_new, temperature=0.8, top_k=None):
        for _ in range(n_new):
            logits = self(idx[:, -self.cfg.seq_len:])[:, -1] / max(temperature, 1e-6)
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            idx = torch.cat([idx, torch.multinomial(F.softmax(logits, -1), 1)], dim=1)
        return idx


def build_model(arch: str, cfg_dict: dict) -> nn.Module:
    if arch == "cadence":
        return CadenceLM(CadenceConfig.from_dict(cfg_dict))
    if arch == "transformer":
        return TransformerLM(TransformerConfig.from_dict(cfg_dict))
    raise ValueError(arch)
