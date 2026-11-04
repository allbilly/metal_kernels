#!/usr/bin/env python3
"""Paired MLX bf16 vs fp16 A/B (x @ w.T) on Qwen model shapes.

5-round ABBA per (M, shape) = 10 pairs, median per dtype, FLOP-weighted ratio
per M. Writes <out>/ab.csv and holds one dtype's buffers at a time.

Run from the repo root with the repo venv active:
  source .venv/bin/activate
  python experimental/benchmarks/ab_dtype.py
"""

from __future__ import annotations

import argparse
import csv
import statistics
import time
from pathlib import Path

import mlx.core as mx

SHAPES = [
    (248320, 1024),
    (6144, 1024),
    (4096, 1024),
    (3584, 1024),
    (2048, 1024),
    (1024, 3584),
    (1024, 2048),
    (512, 1024),
    (16, 1024),
]
MS = [1, 128, 1011, 4096]
DTYPES = [("bf16", mx.bfloat16), ("fp16", mx.float16)]
MAX_BUFFER_BYTES = 3 << 30
ROUNDS = 5


def timed_once(x: mx.array, w: mx.array) -> float:
    start = time.perf_counter()
    y = x @ w.T
    mx.eval(y)
    mx.synchronize()
    elapsed = time.perf_counter() - start
    del y
    return elapsed * 1e3


def run_segment(m: int, n: int, k: int, dtype, warm: bool) -> float:
    x = mx.random.normal((m, k), dtype=dtype)
    w = mx.random.normal((n, k), dtype=dtype)
    mx.eval(x, w)
    mx.synchronize()
    if warm:
        for _ in range(2):
            timed_once(x, w)
    ms = timed_once(x, w)
    del x, w
    mx.clear_cache()
    return ms


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="traces/dtype-ab", help="output directory")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    itemsize = mx.array(0, dtype=mx.float16).itemsize
    print(
        "MLX %s, %d shapes x %d M, %d-round ABBA, fp16/bf16"
        % (mx.__version__, len(SHAPES), len(MS), ROUNDS)
    )

    rows = []
    started = time.perf_counter()
    for m in MS:
        per_m_ratios = []
        per_m_wins = 0
        per_m_pairs = 0
        weighted_fp16 = 0.0
        weighted_bf16 = 0.0
        for n, k in SHAPES:
            largest = max(m * k, n * k, m * n) * itemsize
            if largest > MAX_BUFFER_BYTES:
                print(
                    "SKIP M=%d N=%d K=%d: buffer %.2f GB > 3 GB"
                    % (m, n, k, largest / 2**30),
                    flush=True,
                )
                continue

            samples = {"bf16": [], "fp16": []}
            for _ in range(ROUNDS):
                for name, dtype in (DTYPES[0], DTYPES[1], DTYPES[1], DTYPES[0]):
                    warm = not samples[name]
                    samples[name].append(run_segment(m, n, k, dtype, warm))

            bf16_ms = statistics.median(samples["bf16"])
            fp16_ms = statistics.median(samples["fp16"])
            ratio = fp16_ms / bf16_ms
            wins = 0
            for r in range(ROUNDS):
                if samples["fp16"][2 * r] < samples["bf16"][2 * r]:
                    wins += 1
                if samples["fp16"][2 * r + 1] < samples["bf16"][2 * r + 1]:
                    wins += 1
            flops = 2.0 * m * n * k
            weighted_fp16 += flops * fp16_ms
            weighted_bf16 += flops * bf16_ms
            per_m_ratios.append(ratio)
            per_m_wins += wins
            per_m_pairs += 2 * ROUNDS
            rows.append((m, k, n, bf16_ms, fp16_ms, ratio))
            print(
                "M=%-5d N=%-6d K=%-5d bf16 %8.3f ms  fp16 %8.3f ms  ratio %.3f  wins %2d/%d"
                % (m, n, k, bf16_ms, fp16_ms, ratio, wins, 2 * ROUNDS),
                flush=True,
            )

        if per_m_ratios:
            print(
                "M=%d summary: shapes %d, ratio min %.3f median %.3f max %.3f, "
                "fp16 wins %d/%d, FLOP-weighted ratio %.3f"
                % (
                    m,
                    len(per_m_ratios),
                    min(per_m_ratios),
                    statistics.median(per_m_ratios),
                    max(per_m_ratios),
                    per_m_wins,
                    per_m_pairs,
                    weighted_fp16 / weighted_bf16,
                ),
                flush=True,
            )

    csv_path = out / "ab.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["m", "k", "n", "bf16_ms", "fp16_ms", "ratio"])
        for m, k, n, bf16_ms, fp16_ms, ratio in rows:
            writer.writerow(
                [m, k, n, "%.4f" % bf16_ms, "%.4f" % fp16_ms, "%.4f" % ratio]
            )

    print()
    print("points %d, elapsed %.1f s" % (len(rows), time.perf_counter() - started))
    print("saved %s" % csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
