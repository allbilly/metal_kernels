#!/usr/bin/env python3
"""Rotated-buffer GEMV chain for Qwen3.5-0.8B decode.

Loads every linear weight and streams all of them once per pass, so the
working set (~1.6 GB) exceeds SLC. The isolated `gemv.py` sweep reuses one
matrix and is cache-inflated; use this for the real roofline.

usage: python benchmarks/gemv_chain.py [--passes N] [--warmup N]
"""

from __future__ import annotations

import argparse
import statistics
import time
from typing import Dict, List, Tuple

import mlx.core as mx
from huggingface_hub import snapshot_download


def load_weights(model: str) -> Tuple[List[Tuple[str, mx.array]], int]:
    path = snapshot_download(model)
    tensors = mx.load(path + "/model.safetensors")
    weights: List[Tuple[str, mx.array]] = []
    for name, array in tensors.items():
        if array.ndim != 2:
            continue
        if not name.startswith("language_model."):
            continue
        if "vision_tower" in name or "embed_tokens" in name:
            continue
        weights.append((name, array))
    if not any("lm_head" in name for name, _ in weights):
        weights.append(("lm_head", tensors["language_model.model.embed_tokens.weight"]))
    total = sum(array.size * array.itemsize for _, array in weights)
    return weights, total


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="mlx-community/Qwen3.5-0.8B-bf16")
    ap.add_argument("--passes", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=2)
    args = ap.parse_args()

    weights, total = load_weights(args.model)
    xs: Dict[int, mx.array] = {}
    for _, array in weights:
        k = array.shape[1]
        if k not in xs:
            xs[k] = mx.random.normal((1, k)).astype(mx.bfloat16)
            mx.eval(xs[k])
    print("%d tensors, %.2f GB" % (len(weights), total / 1e9))

    def one_pass():
        outputs = [xs[array.shape[1]] @ array.T for _, array in weights]
        mx.eval(outputs)

    for _ in range(args.warmup):
        one_pass()
    mx.synchronize()

    samples = []
    for _ in range(args.passes):
        start = time.perf_counter()
        one_pass()
        mx.synchronize()
        samples.append((time.perf_counter() - start) * 1e3)

    med = statistics.median(samples)
    print("median %.2f ms/pass, %.1f GB/s" % (med, total / (med * 1e6)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
