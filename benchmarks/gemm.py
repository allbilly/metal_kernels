#!/usr/bin/env python3
"""Prefill GEMM for Qwen3.5-0.8B shapes.

M > 1, so MLX dispatches steel_gemm, not gemv_bfloat16_bm*.
Reports ms, prefill tok/s, TFLOP/s, and GB/s per shape.

usage: python benchmarks/gemm.py [--rows 512] [--runs 5]
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from typing import List, Tuple

import mlx.core as mx

SHAPES: List[Tuple[int, int]] = [
    (1024, 248320),
    (1024, 6144),
    (1024, 4096),
    (1024, 3584),
    (1024, 2048),
    (3584, 1024),
    (2048, 1024),
    (1024, 512),
    (1024, 16),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", type=int, nargs="+", default=[512], help="prefill M")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    report = []
    for m in args.rows:
        for k, n in SHAPES:
            x = mx.random.normal((m, k)).astype(mx.bfloat16)
            w = mx.random.normal((n, k)).astype(mx.bfloat16)
            mx.eval(x, w)
            for _ in range(2):
                y = x @ w.T
                mx.eval(y)
            mx.synchronize()

            samples = []
            for _ in range(args.runs):
                start = time.perf_counter()
                y = x @ w.T
                mx.eval(y)
                mx.synchronize()
                samples.append(time.perf_counter() - start)
                del y

            med = statistics.median(samples)
            flops = 2.0 * m * n * k
            moved = (m * k + n * k + m * n) * 2
            report.append(
                {
                    "m": m,
                    "k": k,
                    "n": n,
                    "ms": med * 1e3,
                    "tok_s": m / med,
                    "tflops": flops / med / 1e12,
                    "gbps": moved / med / 1e9,
                }
            )

    if args.json:
        print(json.dumps(report, indent=2))
        return 0

    for m in args.rows:
        print()
        print("prefill M=%d, bf16" % m)
        header = "%-12s %8s %9s %9s %8s" % ("N×K", "ms", "tok/s", "TFLOP/s", "GB/s")
        print(header)
        print("-" * len(header))
        for r in report:
            if r["m"] != m:
                continue
            print(
                "%-12s %8.2f %9.0f %9.2f %8.1f"
                % (
                    "%d×%d" % (r["n"], r["k"]),
                    r["ms"],
                    r["tok_s"],
                    r["tflops"],
                    r["gbps"],
                )
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
