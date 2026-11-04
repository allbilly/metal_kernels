#!/usr/bin/env python3
"""Sweep MLX bf16/fp16 GEMM (x @ w.T) over an M x N grid at fixed K.

Saves <out>/gemm.csv and a 3D TFLOP/s surface <out>/gemm.png.

Run from the repo root with .venv active:
  python experimental/benchmarks/sweep_gemm.py --quick
  python experimental/benchmarks/sweep_gemm.py --k 1024 --runs 5
"""

from __future__ import annotations

import argparse
import csv
import statistics
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlx.core as mx
import numpy as np

M_FULL = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
N_FULL = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
M_QUICK = [2, 8, 32]
N_QUICK = [64, 1024, 8192]
DTYPES = {"bf16": mx.bfloat16, "fp16": mx.float16}
MAX_OUT_BYTES = 1 << 30


def bench(x: mx.array, w: mx.array, runs: int) -> float:
    for _ in range(2):
        y = x @ w.T
        mx.eval(y)
    mx.synchronize()
    samples = []
    for _ in range(runs):
        start = time.perf_counter()
        y = x @ w.T
        mx.eval(y)
        mx.synchronize()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples)


def plot(rows, ms, ns, k, dtype_name, path: Path) -> None:
    xs = np.log2(ns)
    ys = np.log2(ms)
    grid = np.full((len(ms), len(ns)), np.nan)
    for m, n, _, tflops in rows:
        grid[ms.index(m), ns.index(n)] = tflops
    x, y = np.meshgrid(xs, ys)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(
        x, y, grid, cmap="viridis", rstride=1, cstride=1, linewidth=0.1
    )
    ax.set_xticks(xs, labels=[str(n) for n in ns])
    ax.set_yticks(ys, labels=[str(m) for m in ms])
    ax.set_xticklabels([str(n) for n in ns], rotation=45, ha="right")
    ax.set_yticklabels([str(m) for m in ms], rotation=-45, ha="left")
    ax.set_xlabel("N (log2)")
    ax.set_ylabel("M (log2)")
    ax.set_zlabel("TFLOP/s")
    ax.set_title("MLX %s GEMM x @ w.T, K=%d" % (dtype_name, k))
    fig.colorbar(surf, ax=ax, shrink=0.6, aspect=12, label="TFLOP/s")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="traces/sweep-gemm", help="output directory")
    ap.add_argument("--k", type=int, default=1024, help="contraction width")
    ap.add_argument("--runs", type=int, default=5, help="timed runs per shape")
    ap.add_argument("--quick", action="store_true", help="small M/N grid")
    ap.add_argument(
        "--dtype", choices=sorted(DTYPES), default="bf16", help="matmul dtype"
    )
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dtype = DTYPES[args.dtype]
    itemsize = mx.array(0, dtype=dtype).itemsize
    ms = M_QUICK if args.quick else M_FULL
    ns = N_QUICK if args.quick else N_FULL

    rows = []
    started = time.perf_counter()
    for m in ms:
        for n in ns:
            x_bytes = m * args.k * itemsize
            w_bytes = n * args.k * itemsize
            out_bytes = m * n * itemsize
            largest = max(x_bytes, w_bytes, out_bytes)
            if largest > MAX_OUT_BYTES:
                print(
                    "SKIP M=%d N=%d: buffer %.1f GB > 1 GB"
                    % (m, n, largest / 2**30)
                )
                continue
            x = mx.random.normal((m, args.k)).astype(dtype)
            w = mx.random.normal((n, args.k)).astype(dtype)
            mx.eval(x, w)
            seconds = bench(x, w, args.runs)
            tflops = 2.0 * m * n * args.k / seconds / 1e12
            rows.append((m, n, seconds * 1e3, tflops))
            print(
                "M=%-5d N=%-5d %-2d %8.3f ms %8.1f TFLOP/s"
                % (m, n, args.k, seconds * 1e3, tflops),
                flush=True,
            )
            del x, w
            mx.clear_cache()

    csv_path = out / "gemm.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["m", "k", "n", "ms", "tflops"])
        for m, n, millis, tflops in rows:
            writer.writerow([m, args.k, n, "%.4f" % millis, "%.3f" % tflops])

    png_path = out / "gemm.png"
    if rows:
        plot(rows, ms, ns, args.k, args.dtype, png_path)

    values = [r[3] for r in rows]
    print()
    print(
        "points %d, min %.1f, median %.1f, max %.1f TFLOP/s, %.1f s"
        % (
            len(values),
            min(values),
            statistics.median(values),
            max(values),
            time.perf_counter() - started,
        )
    )
    print("saved %s" % csv_path)
    if rows:
        print("saved %s" % png_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
