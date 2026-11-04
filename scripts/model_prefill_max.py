#!/usr/bin/env python3
"""Shape-aware ceiling for Qwen3.5-0.8B BF16 prefill.

Takes the eight GEMM shapes and counts from README 1.3, plus their measured
ms at M=2048, and models each GEMM as

    t = max( 2*M*N*K / (P * tile_eff * wave_eff) , bytes / BW )

where tile_eff is the waste from padding M/N/K to the Metal tile, and
wave_eff is the occupancy of the last GPU wave.  Summing over the per-token
counts gives a prefill estimate under explicit peak and concurrency
assumptions.  This is a model, not a measurement; no GPU is used.

usage: python scripts/model_prefill_max.py
       python scripts/model_prefill_max.py --peak 2.0 --capacity 64
"""

from __future__ import annotations

import argparse
import math

# (M, K, N, count, measured_ms) per token, steel_gemm nt tile 64x32x32
SHAPES = [
    (2048, 1024, 6144, 18, 16.35),
    (2048, 1024, 4096, 6, 10.95),
    (2048, 1024, 3584, 48, 9.65),
    (2048, 3584, 1024, 24, 9.76),
    (2048, 1024, 2048, 18, 5.70),
    (2048, 2048, 1024, 24, 5.74),
    (2048, 1024, 512, 12, 1.73),
    (2048, 1024, 16, 36, 0.43),
]

BM, BN, BK = 64, 32, 32      # steel_gemm nt tile
BW = 59.2e9                  # measured streaming read, GB/s
NON_GEMM_MS = 0.2475         # 23.6% of 1.048 ms/token
TOKENS = 10_000
MEASURED_BASE_MS = 0.656     # 10k mix, see README 1.3


def shape_model(m: int, k: int, n: int, peak: float, capacity: int):
    """Return (tiles, tile_eff, wave_eff, ms) for one dispatch."""
    tm, tn, tk = math.ceil(m / BM), math.ceil(n / BN), math.ceil(k / BK)
    tiles = tm * tn
    waves = math.ceil(tiles / capacity)
    tile_eff = (m * n * k) / (tm * BM * tn * BN * tk * BK)
    wave_eff = tiles / (waves * capacity)
    flops = 2.0 * m * n * k
    t_compute = flops / (peak * 1e12 * tile_eff * wave_eff)
    t_memory = ((m * k + k * n + m * n) * 2) / BW
    return tiles, tile_eff, wave_eff, max(t_compute, t_memory) * 1e3


def prefill_ms(peak: float, capacity: int) -> float:
    gemm = 0.0
    m = SHAPES[0][0]
    for m, k, n, count, _ in SHAPES:
        gemm += count * shape_model(m, k, n, peak, capacity)[3]
    return gemm / m + NON_GEMM_MS


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--peak", type=float, nargs="+", default=[2.62, 2.0, 1.5],
                    help="assumed BF16 TFLOP/s (default: 2.62 2.0 1.5)")
    ap.add_argument("--capacity", type=int, nargs="+", default=[8, 16, 32, 64],
                    help="concurrent output tiles per GPU (default: 8 16 32 64)")
    args = ap.parse_args()

    print("isolated GEMM-only: %.3f ms/token -> %.0f tok/s (not end-to-end)"
          % (MEASURED_BASE_MS, 1000 / MEASURED_BASE_MS))
    print("non-GEMM floor:    %.4f ms/token -> %.0f tok/s"
          % (NON_GEMM_MS, 1000 / NON_GEMM_MS))
    best = 0.996 / 1.517 + NON_GEMM_MS
    print("best kernel:       1.517 TFLOP/s -> %.0f tok/s" % (1000 / best))
    print("measured current:  954 tok/s")
    print()

    cap = args.capacity[0]
    print("per shape at peak %.2f TFLOP/s, capacity %d:" % (args.peak[0], cap))
    print("%-18s %5s %7s %8s %8s %9s %9s"
          % ("M x K x N", "count", "tiles", "tile_eff", "wave_eff", "measured", "model"))
    for m, k, n, count, ms in SHAPES:
        tiles, te, we, t = shape_model(m, k, n, args.peak[0], cap)
        print("%-18s %5d %7d %8.2f %8.2f %9.2f %9.2f"
              % ("%dx%dx%d" % (m, k, n), count, tiles, te, we, ms, t))
    print()

    print("conditional shape-aware prefill roofline tok/s (rows: peak TFLOP/s, cols: tile capacity)")
    print("%6s" % "peak", "".join("%9d" % c for c in args.capacity))
    for p in args.peak:
        row = "".join("%9.0f" % (1000 / prefill_ms(p, c)) for c in args.capacity)
        print("%6.2f%s" % (p, row))
    print()
    print("model only: not a measured result and not an absolute hardware maximum.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
