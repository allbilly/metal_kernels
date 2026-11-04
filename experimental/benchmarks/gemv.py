#!/usr/bin/env python3
"""GEMV A/B for Qwen3.5-0.8B decode shapes.

Compares MLX's x @ w.T (gemv_bfloat16_bm* on batch 1) with a custom
mx.fast.metal_kernel: one threadgroup per output row, 64 threads, tree sum.

Run from the repo root with .venv active:
  python experimental/experimental/benchmarks/gemv.py
  python experimental/benchmarks/gemv.py --n 3584 --json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from typing import Dict, List

import mlx.core as mx

KERNEL_SOURCE = r"""
    uint row = threadgroup_position_in_grid.x;
    uint lane = thread_position_in_threadgroup.x;
    uint nthreads = threads_per_threadgroup.x;
    threadgroup float partial[NT];

    float acc = 0.0f;
    for (uint i = lane; i < K; i += nthreads) {
        acc += float(x[i]) * float(w[row * K + i]);
    }
    partial[lane] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = nthreads / 2; stride > 0; stride >>= 1) {
        if (lane < stride) {
            partial[lane] += partial[lane + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) {
        out[row] = T(partial[0]);
    }
"""


def make_custom(k: int, n: int, threads: int):
    kernel = mx.fast.metal_kernel(
        name="gemv_%d_%d_%d" % (k, n, threads),
        input_names=["x", "w"],
        output_names=["out"],
        source=KERNEL_SOURCE,
    )

    def custom(x: mx.array, w: mx.array) -> mx.array:
        return kernel(
            inputs=[x, w],
            template=[("T", x.dtype), ("K", k), ("NT", threads)],
            grid=(n * threads, 1, 1),
            threadgroup=(threads, 1, 1),
            output_shapes=[(1, n)],
            output_dtypes=[x.dtype],
        )[0]

    return custom


def warm(variants, x: mx.array, w: mx.array, seconds: float) -> None:
    outputs = [fn(x, w) for fn in variants.values()]
    mx.eval(outputs)
    if seconds > 0:
        start = time.perf_counter()
        while time.perf_counter() - start < seconds:
            outputs = [fn(x, w) for fn in variants.values()]
            mx.eval(outputs)
    mx.synchronize()


def queued(
    fn, x: mx.array, w: mx.array, iters: int, runs: int, warmup_seconds: float
) -> List[float]:
    warm({"k": fn}, x, w, warmup_seconds)
    samples = []
    for _ in range(runs):
        start = time.perf_counter()
        outputs = [fn(x, w) for _ in range(iters)]
        mx.eval(outputs)
        mx.synchronize()
        samples.append((time.perf_counter() - start) / iters * 1e6)
    return samples


def paired(variants, x: mx.array, w: mx.array, rounds: int, warmup_seconds: float):
    warm(variants, x, w, warmup_seconds)
    samples: Dict[str, List[float]] = {name: [] for name in variants}
    round_times = []
    order = list(variants)
    for r in range(rounds):
        seq = order + order[::-1] if r % 2 == 0 else order[::-1] + order
        times = {}
        for name in seq:
            start = time.perf_counter()
            y = variants[name](x, w)
            mx.eval(y)
            mx.synchronize()
            times[name] = (time.perf_counter() - start) * 1e6
            samples[name].append(times[name])
        round_times.append(times)
    return samples, round_times


QWEN_SHAPES = [
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
    ap.add_argument("--k", type=int, default=1024, help="input width")
    ap.add_argument("--n", type=int, nargs="+", default=[1024, 3584], help="output rows")
    ap.add_argument(
        "--qwen",
        action="store_true",
        help="Qwen3.5-0.8B decode shapes (hidden 1024, MLP 3584, vocab 248320)",
    )
    ap.add_argument("--threads", type=int, default=64)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--runs", type=int, default=7)
    ap.add_argument(
        "--paired",
        action="store_true",
        help="alternate variants ABBA instead of timing blocks",
    )
    ap.add_argument("--rounds", type=int, default=6, help="paired rounds")
    ap.add_argument(
        "--warmup-seconds",
        type=float,
        default=2.0,
        help="time-based warmup before sampling (DVFS)",
    )
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    ok_all = True
    report = []
    shapes = QWEN_SHAPES if args.qwen else [(args.k, n) for n in args.n]
    for k, n in shapes:
        x = mx.random.normal((1, k)).astype(mx.bfloat16)
        w = mx.random.normal((n, k)).astype(mx.bfloat16)
        mx.eval(x, w)
        reference = x @ w.T
        mx.eval(reference)

        custom = make_custom(k, n, args.threads)
        variants = {
            "mlx_gemv": lambda a, b: a @ b.T,
            "custom%d" % args.threads: custom,
        }
        round_times = []
        ratios = {}
        if args.paired:
            sample_map, round_times = paired(
                variants, x, w, args.rounds, args.warmup_seconds
            )
            base = next(iter(variants))
            ratios = {}
            for name in variants:
                if name == base:
                    continue
                rel = [t[name] / t[base] for t in round_times]
                wins = sum(t[name] < t[base] for t in round_times)
                ratios[name] = (statistics.median(rel), wins, len(rel))
        else:
            sample_map = {
                name: queued(fn, x, w, args.iters, args.runs, args.warmup_seconds)
                for name, fn in variants.items()
            }

        items = []
        for name, fn in variants.items():
            y = fn(x, w)
            mx.eval(y)
            maxdiff = float(
                mx.max(mx.abs(y.astype(mx.float32) - reference.astype(mx.float32)))
            )
            ok = bool(mx.allclose(y, reference, atol=2e-2, rtol=2e-2))
            ok_all = ok_all and ok
            samples = sample_map[name]
            smallest = min(samples)
            moved = n * k * 2 + k * 2 + n * 2
            items.append(
                {
                    "variant": name,
                    "n": n,
                    "k": k,
                    "min_us": smallest,
                    "median_us": statistics.median(samples),
                    "gbps": moved / (smallest * 1e3),
                    "maxdiff": maxdiff,
                    "ok": ok,
                    "ratio_median": ratios.get(name, (None, 0, 0))[0],
                    "wins": ratios.get(name, (None, 0, 0))[1],
                }
            )
        report.extend(items)
        if not args.json:
            print()
            mode = "paired samples" if args.paired else "queued runs"
            count = args.rounds * 2 if args.paired else args.runs
            print("B=1, K=%d, N=%d, bf16 (min of %d %s)" % (k, n, count, mode))
            header = "%-9s %8s %8s %8s %8s" % ("variant", "min µs", "median", "GB/s", "maxdiff")
            print(header)
            print("-" * len(header))
            for item in items:
                print(
                    "%-9s %8.1f %8.1f %8.1f %8.3f%s"
                    % (
                        item["variant"],
                        item["min_us"],
                        item["median_us"],
                        item["gbps"],
                        item["maxdiff"],
                        "" if item["ok"] else "  FAIL",
                    )
                )
            if args.paired:
                base = next(iter(variants))
                for name, (ratio, wins, nrounds) in ratios.items():
                    decisive = abs(wins - nrounds / 2) > nrounds**0.5
                    print(
                        "  %s / %s: ratio %.3f, wins %d/%d%s"
                        % (
                            name,
                            base,
                            ratio,
                            wins,
                            nrounds,
                            "" if decisive else " (noise)",
                        )
                    )

    if args.json:
        print(json.dumps(report, indent=2))
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
