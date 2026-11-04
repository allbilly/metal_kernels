#!/usr/bin/env python3
"""Attention/GDN output-projection A/B for MegaQwen.

The mixer output projection is the same shape in every Qwen3.5-0.8B layer:
K=2048, N=1024. This checks whether a custom GEMV can replace MLX's tuned
`o_proj` / `out_proj` without changing a single bf16 output.

Run from the repo root with .venv active:
  python experimental/benchmarks/mixer.py
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from typing import Dict, List, Tuple

import mlx.core as mx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from megaqwen import bm4_gemv, down, down_residual


def timeit(fn, args, iters=200, runs=7) -> Tuple[float, float]:
    for _ in range(5):
        mx.eval(fn(*args))
    mx.synchronize()
    samples = []
    for _ in range(runs):
        start = time.perf_counter()
        outputs = [fn(*args) for _ in range(iters)]
        mx.eval(outputs)
        mx.synchronize()
        samples.append((time.perf_counter() - start) / iters * 1e6)
    return min(samples), statistics.median(samples)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="~/.cache/models/Qwen3.5-0.8B-bf16")
    ap.add_argument("--rows", type=int, default=8)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    from mlx_lm import load
    import os

    model, _ = load(os.path.expanduser(args.model))
    layers = model.language_model.model.layers
    gdn = next(layer for layer in layers if layer.is_linear).linear_attn.out_proj
    attn = next(layer for layer in layers if not layer.is_linear).self_attn.o_proj

    report: List[Dict] = []
    for name, module in (("gdn_out_proj", gdn), ("attn_o_proj", attn)):
        w = module.weight
        k, n = w.shape[1], w.shape[0]
        x = (mx.random.normal((1, 1, k)) * 0.02).astype(mx.bfloat16)
        h = (mx.random.normal((1, 1, n)) * 0.02).astype(mx.bfloat16)
        mx.eval(x, h)

        mlx_proj = lambda a, w: module(a)[0, 0]
        custom = down(k, n, args.rows, 4)
        ref = mlx_proj(x, w)
        got = custom(x[0, 0], w)
        mx.eval(ref, got)
        exact = bool(mx.array_equal(ref, got))

        ref_res = ref + h[0, 0]
        got_res = down_residual(k, n, args.rows, 4)(x[0, 0], w, h[0, 0])
        mx.eval(ref_res, got_res)
        exact_res = bool(mx.array_equal(ref_res, got_res))

        bm4 = bm4_gemv(k, n, 4, 16, 4)
        got_bm4 = bm4(x[0, 0], w)
        mx.eval(got_bm4)
        exact_bm4 = bool(mx.array_equal(ref, got_bm4))

        cases = (
            (
                "proj",
                lambda a, w: custom(a[0, 0], w),
                lambda a, w: module(a)[0, 0],
                (x, w),
                exact,
            ),
            (
                "proj+res",
                lambda a, w, r: down_residual(k, n, args.rows, 4)(a[0, 0], w, r[0, 0]),
                lambda a, w, r: module(a)[0, 0] + r[0, 0],
                (x, w, h),
                exact_res,
            ),
            (
                "proj/bm4",
                lambda a, w: bm4(a[0, 0], w),
                lambda a, w: module(a)[0, 0],
                (x, w),
                exact_bm4,
            ),
        )
        for suffix, fn, mlx_fn, call_args, ok in cases:
            lo, med = timeit(fn, call_args)
            mlo, mmed = timeit(mlx_fn, call_args)
            report.append(
                {
                    "path": name + "/" + suffix,
                    "mlx_min_us": mlo,
                    "mlx_median_us": mmed,
                    "mega_min_us": lo,
                    "mega_median_us": med,
                    "speedup": mmed / med,
                    "bitwise": ok,
                }
            )

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        header = "%-22s %10s %10s %8s %8s" % (
            "path", "mlx min", "mega min", "speedup", "bitwise",
        )
        print(header)
        print("-" * len(header))
        for item in report:
            print(
                "%-22s %10.1f %10.1f %8.2fx %8s"
                % (
                    item["path"],
                    item["mlx_min_us"],
                    item["mega_min_us"],
                    item["speedup"],
                    "yes" if item["bitwise"] else "NO",
                )
            )
    return 0 if all(item["bitwise"] for item in report) else 1


if __name__ == "__main__":
    raise SystemExit(main())
