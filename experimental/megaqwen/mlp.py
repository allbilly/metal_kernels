"""MegaQwen MLP residual block.

Pre-norm residual block:

    h  = x + attn_out
    xn = rmsnorm(h)
    act = silu(gate @ xn) * (up @ xn)
    out = h + down @ act

Prefill uses the fused add+RMSNorm kernel (also returns h). Decode uses the
fused down GEMV + residual kernel for the last projection.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from .kernels import down_residual, fused_add_rmsnorm


class MegaQwenMLPBlock:
    def __init__(self, hidden: int = 1024, inter: int = 3584, eps: float = 1e-6):
        self.hidden = hidden
        self.inter = inter
        self.eps = eps
        self.add_norm = fused_add_rmsnorm(hidden)
        self.down = down_residual(inter, hidden)

    def __call__(
        self,
        x: mx.array,
        attn_out: mx.array,
        norm_weight: mx.array,
        gate_weight: mx.array,
        up_weight: mx.array,
        down_weight: mx.array,
    ) -> mx.array:
        if x.shape[-2] == 1:
            h = x + attn_out
            xn = mx.fast.rms_norm(h, norm_weight, self.eps)
            act = nn.silu(xn @ gate_weight.T) * (xn @ up_weight.T)
            return self.down(act[0], down_weight, h[0])[None, :]

        xn, h = self.add_norm(x, attn_out, norm_weight)
        act = nn.silu(xn @ gate_weight.T) * (xn @ up_weight.T)
        return h + act @ down_weight.T
