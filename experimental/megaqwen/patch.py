"""Patch mlx_lm's Qwen3.5 decoder layer with MegaQwen fusions.

The layer is:

    r  = mixer(input_layernorm(x))
    h  = x + r
    out = h + mlp(post_attention_layernorm(h))

MegaQwen fuses the post-attention residual add + RMSNorm, and at decode fuses
the down projection + residual. Attention/GDN stays on MLX.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .kernels import bm4_gemv, fused_add_rmsnorm


def patch_qwen_decoder(mode: str = "both", eps: float = 1e-6) -> Tuple[Any, Any]:
    """Return (original_call, restore_call). mode: both, addnorm, down."""
    from mlx_lm.models.qwen3_5 import DecoderLayer

    original = DecoderLayer.__call__
    add_norm_cache = {}
    down_cache = {}
    gate_cache = {}
    bm4_down_cache = {}

    def mega_call(self, x: mx.array, mask: Optional[mx.array] = None, cache: Any = None):
        mlp = self.mlp
        if not hasattr(mlp, "gate_proj"):
            return original(self, x, mask=mask, cache=cache)
        if mode in ("down", "bm4") and x.shape[-2] != 1:
            return original(self, x, mask=mask, cache=cache)

        norm = self.input_layernorm(x)
        if self.is_linear:
            r = self.linear_attn(norm, mask, cache)
        else:
            r = self.self_attn(norm, mask, cache)

        hidden = x.shape[-1]
        if mode in ("both", "addnorm"):
            if hidden not in add_norm_cache:
                add_norm_cache[hidden] = fused_add_rmsnorm(hidden)
            xn, h = add_norm_cache[hidden](x, r, self.post_attention_layernorm.weight)
            if mode == "addnorm":
                return h + mlp(xn)
        else:
            h = x + r
            xn = self.post_attention_layernorm(h)

        inter = mlp.gate_proj.weight.shape[0]
        from mlx_lm.models.activations import swiglu

        if mode == "bm4":
            if (hidden, inter) not in gate_cache:
                gate_cache[(hidden, inter)] = (
                    bm4_gemv(hidden, inter, 4, 4, 4),
                    bm4_gemv(hidden, inter, 4, 4, 4),
                )
            if (inter, hidden) not in bm4_down_cache:
                bm4_down_cache[(inter, hidden)] = bm4_gemv(inter, hidden, 4, 16, 4)
            gate_fn, up_fn = gate_cache[(hidden, inter)]
            gate = gate_fn(xn[0, 0], mlp.gate_proj.weight)
            up = up_fn(xn[0, 0], mlp.up_proj.weight)
            act = swiglu(gate[None, None, :], up[None, None, :])
            out = bm4_down_cache[(inter, hidden)](
                act[0, 0], mlp.down_proj.weight
            )
            return h + out.reshape(x.shape)

        act = swiglu(mlp.gate_proj(xn), mlp.up_proj(xn))
        if mode == "mlp":
            return h + mlp.down_proj(act)
        if mode in ("both", "down") and x.shape[-2] == 1:
            if (inter, hidden) not in down_cache:
                down_cache[(inter, hidden)] = bm4_gemv(inter, hidden, 4, 16, 4)
            out = down_cache[(inter, hidden)](act[0, 0], mlp.down_proj.weight)
            return h + out.reshape(x.shape)
        return h + mlp.down_proj(act)

    DecoderLayer.__call__ = mega_call

    def restore() -> None:
        DecoderLayer.__call__ = original

    return original, restore
