#!/usr/bin/env python3
"""RMSNorm kernel A/B for Qwen3.5-0.8B (H=1024).

Compares mx.fast.rms_norm, unfused MLX ops, mx.compile, and two custom
mx.fast.metal_kernel variants at decode (S=1) and prefill (S=2048) shapes.

Run from the repo root with .venv active:
  python experimental/experimental/benchmarks/rmsnorm.py
  python experimental/benchmarks/rmsnorm.py --e2e
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

ROOT = Path(__file__).resolve().parents[1]
DTYPES = {"bfloat16": mx.bfloat16, "float16": mx.float16, "float32": mx.float32}
KERNEL_SOURCE = r"""
    uint row = threadgroup_position_in_grid.x;
    uint lane = thread_position_in_threadgroup.x;
    uint nthreads = threads_per_threadgroup.x;
    threadgroup float partial[NT];

    float acc = 0.0f;
    for (uint i = lane; i < H; i += nthreads) {
        float v = float(x[row * H + i]);
        acc += v * v;
    }
    partial[lane] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = nthreads / 2; stride > 0; stride >>= 1) {
        if (lane < stride) {
            partial[lane] += partial[lane + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float scale = metal::rsqrt(partial[0] * (1.0f / float(H)) + 1e-6f);
    for (uint i = lane; i < H; i += nthreads) {
        out[row * H + i] = T(float(x[row * H + i]) * scale * float(w[i]));
    }
"""


def make_custom(hidden: int, threads: int):
    kernel = mx.fast.metal_kernel(
        name="rmsnorm_%d_%d" % (hidden, threads),
        input_names=["x", "w"],
        output_names=["out"],
        source=KERNEL_SOURCE,
    )

    def custom(x: mx.array, w: mx.array) -> mx.array:
        rows = x.size // hidden
        return kernel(
            inputs=[x, w],
            template=[("T", x.dtype), ("H", hidden), ("NT", threads)],
            grid=(rows * threads, 1, 1),
            threadgroup=(threads, 1, 1),
            output_shapes=[x.shape],
            output_dtypes=[x.dtype],
        )[0]

    return custom


def make_variants(hidden: int, eps: float) -> Dict[str, object]:
    def mlx_fast(x, w):
        return mx.fast.rms_norm(x, w, eps)

    def unfused(x, w):
        return x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + eps) * w

    return {
        "mlx_fast": mlx_fast,
        "unfused": unfused,
        "compiled": mx.compile(unfused, shapeless=True),
        "custom64": make_custom(hidden, 64),
        "custom256": make_custom(hidden, 256),
    }


def tolerance(dtype: mx.Dtype) -> Tuple[float, float]:
    if dtype == mx.float32:
        return 1e-4, 1e-4
    if dtype == mx.float16:
        return 1e-2, 1e-2
    return 2e-2, 2e-2


def queued_microseconds(fn, x: mx.array, w: mx.array, iters: int, runs: int) -> List[float]:
    """Queue `iters` calls, sync once; returns microseconds per call per run."""
    for _ in range(3):
        y = fn(x, w)
    mx.eval(y)
    mx.synchronize()
    samples = []
    for _ in range(runs):
        start = time.perf_counter()
        outputs = [fn(x, w) for _ in range(iters)]
        mx.eval(outputs)
        mx.synchronize()
        samples.append((time.perf_counter() - start) / iters * 1e6)
    return samples


def benchmark_shape(
    hidden: int,
    rows: int,
    dtype: mx.Dtype,
    runs: int,
    variants: Dict[str, object],
) -> List[Dict]:
    x = mx.random.normal((1, rows, hidden)).astype(dtype)
    w = mx.random.normal((hidden,)).astype(dtype)
    mx.eval(x, w)
    reference = mx.fast.rms_norm(x, w, 1e-6)
    mx.eval(reference)
    atol, rtol = tolerance(dtype)
    itemsize = 4 if dtype == mx.float32 else 2
    moved = 2 * rows * hidden * itemsize + hidden * itemsize

    iters = 2000 if rows == 1 else (50 if rows <= 2048 else 20)
    out = []
    for name, fn in variants.items():
        y = fn(x, w)
        mx.eval(y)
        maxdiff = float(
            mx.max(mx.abs(y.astype(mx.float32) - reference.astype(mx.float32)))
        )
        ok = bool(mx.allclose(y, reference, atol=atol, rtol=rtol))
        samples = queued_microseconds(fn, x, w, iters, runs)
        median = statistics.median(samples)
        smallest = min(samples)
        out.append(
            {
                "variant": name,
                "rows": rows,
                "iters": iters,
                "min_us": smallest,
                "median_us": median,
                "max_us": max(samples),
                "gbps": moved / (smallest * 1e3),
                "maxdiff": maxdiff,
                "ok": ok,
            }
        )
    return out


def fmt_table(rows: List[Dict]) -> str:
    header = "%-10s %8s %8s %8s %8s %8s" % (
        "variant",
        "min",
        "median",
        "max",
        "GB/s",
        "maxdiff",
    )
    lines = [header, "-" * len(header)]
    for item in rows:
        lines.append(
            "%-10s %8.2f %8.2f %8.2f %8.1f %8.3f%s"
            % (
                item["variant"],
                item["min_us"],
                item["median_us"],
                item["max_us"],
                item["gbps"],
                item["maxdiff"],
                "" if item["ok"] else "  FAIL",
            )
        )
    return "\n".join(lines)


def patch_model(model, hidden: int, threads: int):
    kernel = make_custom(hidden, threads)

    class CustomRMSNorm(nn.Module):
        def __init__(self, weight: mx.array, eps: float):
            super().__init__()
            self.weight = weight
            self.eps = eps

        def __call__(self, x: mx.array) -> mx.array:
            return kernel(x, self.weight)

    patched = 0
    restore = []
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.RMSNorm):
            continue
        if module.weight.shape[-1] != hidden:
            continue
        parent = model
        parts = name.split(".")
        for part in parts[:-1]:
            parent = parent[int(part)] if isinstance(parent, (list, tuple)) else getattr(parent, part)
        restore.append((parent, parts[-1], module))
        _set_child(parent, parts[-1], CustomRMSNorm(module.weight, module.eps))
        patched += 1
    return patched, restore


def _set_child(parent, key, value) -> None:
    if isinstance(parent, (list, tuple)):
        parent[int(key)] = value
    else:
        setattr(parent, key, value)


def measure_e2e(model, tokenizer, prompt: str, runs: int, max_tokens: int) -> Tuple[List[float], str]:
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler

    sampler = make_sampler(temp=0)
    # Warm up compilation and the KV cache path.
    for _ in range(2):
        last = None
        for response in stream_generate(model, tokenizer, prompt, max_tokens=2, sampler=sampler):
            last = response
    mx.synchronize()

    samples: List[float] = []
    text = ""
    for _ in range(runs):
        last = None
        for response in stream_generate(
            model, tokenizer, prompt, max_tokens=max_tokens, sampler=sampler
        ):
            last = response
        samples.append(last.generation_tps)
        text = last.text
    return samples, text


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hidden", type=int, default=1024, help="Qwen3.5-0.8B hidden size")
    ap.add_argument("--prefill-rows", type=int, default=2048)
    ap.add_argument("--dtype", choices=sorted(DTYPES), default="bfloat16")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--e2e", action="store_true", help="also measure mlx_lm decode tok/s")
    ap.add_argument("--model", default="~/.cache/models/Qwen3.5-0.8B-bf16")
    ap.add_argument("--prompt-file", default=str(ROOT / "benchmark_prompt_10k.txt"))
    ap.add_argument("--e2e-runs", type=int, default=5)
    ap.add_argument("--e2e-tokens", type=int, default=16)
    args = ap.parse_args()

    dtype = DTYPES[args.dtype]
    variants = make_variants(args.hidden, 1e-6)
    report: Dict[str, object] = {"hidden": args.hidden, "dtype": args.dtype, "results": []}

    for rows in (1, args.prefill_rows):
        items = benchmark_shape(args.hidden, rows, dtype, args.runs, variants)
        report["results"].extend(items)
        if not args.json:
            print()
            print("B=1, S=%d, H=%d, %s (median of %d runs)" % (rows, args.hidden, args.dtype, args.runs))
            print(fmt_table(items))

    if args.e2e:
        from mlx_lm import load

        prompt = Path(args.prompt_file).read_text()
        model_path = os.path.expanduser(args.model)
        if not args.json:
            print()
            print("loading %s" % model_path)
        model, tokenizer = load(model_path)
        baseline_a, text_a = measure_e2e(
            model, tokenizer, prompt, args.e2e_runs, args.e2e_tokens
        )
        patched_count, restore = patch_model(model, args.hidden, 64)
        mx.eval(model.parameters())
        mx.synchronize()
        custom, custom_text = measure_e2e(
            model, tokenizer, prompt, args.e2e_runs, args.e2e_tokens
        )
        for parent, key, module in restore:
            _set_child(parent, key, module)
        mx.eval(model.parameters())
        mx.synchronize()
        baseline_b, text_b = measure_e2e(
            model, tokenizer, prompt, args.e2e_runs, args.e2e_tokens
        )
        base_a = statistics.median(baseline_a)
        base_b = statistics.median(baseline_b)
        base_med = statistics.median([base_a, base_b])
        e2e = {
            "patched_norms": patched_count,
            "baseline_a_tps": baseline_a,
            "baseline_b_tps": baseline_b,
            "custom_tps": custom,
            "baseline_a_median": base_a,
            "baseline_b_median": base_b,
            "custom_median": statistics.median(custom),
            "drift_pct": 100.0 * (base_b / base_a - 1.0),
            "delta_pct": 100.0 * (statistics.median(custom) / base_med - 1.0),
            "text_match": text_a == custom_text == text_b,
        }
        report["e2e"] = e2e
        if not args.json:
            print()
            print(
                "end-to-end decode, %d tokens, temp 0, %d runs, %d RMSNorm sites patched"
                % (args.e2e_tokens, args.e2e_runs, patched_count)
            )
            print(
                "  baseline A %.2f tok/s  custom %.2f tok/s  baseline B %.2f tok/s"
                % (e2e["baseline_a_median"], e2e["custom_median"], e2e["baseline_b_median"])
            )
            print(
                "  custom vs baseline mean: %+.1f%%  (baseline drift %+.1f%%)"
                % (e2e["delta_pct"], e2e["drift_pct"])
            )
            print("  output text matches: %s" % ("yes" if e2e["text_match"] else "no"))

    if args.json:
        print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
