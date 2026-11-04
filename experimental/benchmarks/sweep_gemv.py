#!/usr/bin/env python3
"""Sweep MLX GEMV bandwidth over a K x N grid at M=1.

Run from the repo root with the repo venv active:
  source .venv/bin/activate
  python experimental/benchmarks/sweep_gemv.py
  python experimental/benchmarks/sweep_gemv.py --quick --out traces/sweep-gemv-quick

Writes gemv.csv and gemv.png to the output directory.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import statistics
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlx.core as mx
import numpy as np

DEFAULT_K = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
DEFAULT_N = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
QUICK_K = [64, 1024, 8192]
QUICK_N = [64, 1024, 8192]
DTYPES = {"bf16": mx.bfloat16, "fp16": mx.float16}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="traces/sweep-gemv", help="output directory")
    ap.add_argument("--runs", type=int, default=10, help="timed runs per shape")
    ap.add_argument("--quick", action="store_true", help="use the 3 x 3 quick grid")
    ap.add_argument("--dtype", choices=tuple(DTYPES), default="bf16")
    args = ap.parse_args()
    if args.runs < 1:
        ap.error("--runs must be at least 1")

    started = time.perf_counter()
    dtype = DTYPES[args.dtype]
    probe = mx.array(0, dtype=dtype)
    itemsize = probe.itemsize
    del probe

    ks = QUICK_K if args.quick else DEFAULT_K
    ns = QUICK_N if args.quick else DEFAULT_N
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "gemv.csv"
    png_path = out_dir / "gemv.png"

    total = len(ks) * len(ns)
    print(
        "MLX %s GEMV sweep: K=%s N=%s, %d shapes, %d runs"
        % (args.dtype, ks, ns, total, args.runs)
    )

    rows = []
    skipped = 0
    for index, (k, n) in enumerate(itertools.product(ks, ns), 1):
        weight_bytes = n * k * itemsize
        if weight_bytes > 1_000_000_000:
            skipped += 1
            print(
                "[%d/%d] SKIP K=%d N=%d: weight %.2f GB > 1.00 GB"
                % (index, total, k, n, weight_bytes / 1e9)
            )
            mx.clear_cache()
            continue

        x = mx.random.normal((1, k)).astype(dtype)
        w = mx.random.normal((n, k)).astype(dtype)
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

        seconds = statistics.median(samples)
        ms = seconds * 1e3
        moved = (k * n + n + k) * itemsize
        gbs = moved / seconds / 1e9
        rows.append({"k": k, "n": n, "ms": ms, "gbs": gbs})
        print(
            "[%d/%d] K=%d N=%d: %.3f ms, %.2f GB/s"
            % (index, total, k, n, ms, gbs)
        )

        del y, x, w
        mx.clear_cache()

    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["k", "n", "ms", "gbs"])
        for row in rows:
            writer.writerow(
                [row["k"], row["n"], "%.6f" % row["ms"], "%.3f" % row["gbs"]]
            )

    if not rows:
        print("no benchmark points; skipped plot")
        return 1

    k_index = {k: i for i, k in enumerate(ks)}
    n_index = {n: i for i, n in enumerate(ns)}
    z = np.full((len(ns), len(ks)), np.nan)
    for row in rows:
        z[n_index[row["n"]], k_index[row["k"]]] = row["gbs"]

    x_ticks = np.log2(np.asarray(ks, dtype=np.float64))
    y_ticks = np.log2(np.asarray(ns, dtype=np.float64))
    xx, yy = np.meshgrid(x_ticks, y_ticks)

    fig = plt.figure(figsize=(10, 7.5))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(
        xx, yy, z, cmap="viridis", linewidth=0.2, antialiased=True
    )
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([str(k) for k in ks], fontsize=8)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([str(n) for n in ns], fontsize=8)
    ax.set_xlabel("log2(K)")
    ax.set_ylabel("log2(N)")
    ax.set_zlabel("achieved GB/s")
    ax.set_title("MLX %s GEMV achieved bandwidth (M=1)" % args.dtype)
    ax.view_init(elev=28, azim=-135)
    fig.colorbar(surf, ax=ax, shrink=0.65, aspect=15, pad=0.08, label="GB/s")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    gbs_values = [row["gbs"] for row in rows]
    elapsed = time.perf_counter() - started
    print()
    print(
        "summary: %d points, GB/s min %.2f median %.2f max %.2f, "
        "elapsed %.2f s%s"
        % (
            len(rows),
            min(gbs_values),
            statistics.median(gbs_values),
            max(gbs_values),
            elapsed,
            ", %d skipped" % skipped if skipped else "",
        )
    )
    print("csv:  %s" % csv_path)
    print("plot: %s" % png_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
