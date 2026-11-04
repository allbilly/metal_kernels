#!/usr/bin/env python3
"""bm4 gate/up and down GEMV A/B for the Qwen3.5-0.8B MLP.

Run from the repo root with .venv active:
  python experimental/benchmarks/mlp_bm4.py
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from typing import Dict, List

import mlx.core as mx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from megaqwen import bm4_gemv


def timeit(fn, args, iters=100, runs=7):
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
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    from mlx_lm import load

    model, _ = load(os.path.expanduser(args.model))
    mlp = model.language_model.model.layers[0].mlp
    inter = mlp.gate_proj.weight.shape[0]
    hidden = mlp.gate_proj.weight.shape[1]
    x = (mx.random.normal((1, 1, hidden)) * 0.02).astype(mx.bfloat16)
    act = (mx.random.normal((1, 1, inter)) * 0.02).astype(mx.bfloat16)
    mx.eval(x, act)

    gate = bm4_gemv(hidden, inter, 4, 4, 4)
    up = bm4_gemv(hidden, inter, 4, 4, 4)
    down = bm4_gemv(inter, hidden, 4, 16, 4)

    cases = (
        ("gate_proj", lambda a: mlp.gate_proj(a)[0, 0], lambda a: gate(a[0, 0], mlp.gate_proj.weight), (x,)),
        ("up_proj", lambda a: mlp.up_proj(a)[0, 0], lambda a: up(a[0, 0], mlp.up_proj.weight), (x,)),
        ("down_proj", lambda a: mlp.down_proj(a)[0, 0], lambda a: down(a[0, 0], mlp.down_proj.weight), (act,)),
    )

    report: List[Dict] = []
    for name, mlx_fn, mega_fn, call_args in cases:
        ref = mlx_fn(*call_args)
        got = mega_fn(*call_args)
        mx.eval(ref, got)
        exact = bool(mx.array_equal(ref, got))
        mlo, mmed = timeit(mlx_fn, call_args)
        lo, med = timeit(mega_fn, call_args)
        report.append(
            {
                "path": name,
                "mlx_min_us": mlo,
                "mlx_median_us": mmed,
                "mega_min_us": lo,
                "mega_median_us": med,
                "speedup": mmed / med,
                "bitwise": exact,
            }
        )

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        header = "%-12s %10s %10s %8s %8s" % (
            "path", "mlx min", "mega min", "speedup", "bitwise",
        )
        print(header)
        print("-" * len(header))
        for item in report:
            print(
                "%-12s %10.1f %10.1f %8.2fx %8s"
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
