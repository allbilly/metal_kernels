#!/usr/bin/env python3
"""Streaming read bandwidth of one 512 MB MLX buffer.

Reads the whole buffer with a sum reduction, which touches every byte once,
and reports GB/s as bytes / wall time, median of --runs.

usage: python scripts/mem_bw.py
       python scripts/mem_bw.py --mb 1024 --runs 10
"""

from __future__ import annotations

import argparse
import statistics
import time

import mlx.core as mx

NOMINAL_GBPS = 68.3  # Apple M1 LPDDR4X rating


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mb", type=int, default=512, help="buffer size (default: 512)")
    ap.add_argument("--runs", type=int, default=5, help="timed runs (default: 5)")
    args = ap.parse_args()

    n = args.mb * 1024 * 1024 // 2          # bf16 elements
    x = mx.ones((n,), dtype=mx.bfloat16)
    mx.eval(x)
    for _ in range(2):
        mx.eval(mx.sum(x))
    mx.synchronize()

    samples = []
    for _ in range(args.runs):
        start = time.perf_counter()
        mx.eval(mx.sum(x))
        mx.synchronize()
        samples.append(time.perf_counter() - start)

    sec = statistics.median(samples)
    gbps = args.mb / 1024 / sec
    print("%d MB read: median %.3f ms, %.1f GB/s (%.0f%% of %.1f nominal)"
          % (args.mb, sec * 1e3, gbps, 100 * gbps / NOMINAL_GBPS, NOMINAL_GBPS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
