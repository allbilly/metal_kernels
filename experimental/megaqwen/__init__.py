"""MegaQwen: fused Metal kernels for Qwen3.5-0.8B on Apple silicon."""

from .kernels import bm4_gemv, down, down_residual, fused_add_rmsnorm, lm_head
from .mlp import MegaQwenMLPBlock
from .patch import patch_qwen_decoder

__all__ = [
    "fused_add_rmsnorm",
    "down",
    "down_residual",
    "bm4_gemv",
    "lm_head",
    "MegaQwenMLPBlock",
    "patch_qwen_decoder",
]
