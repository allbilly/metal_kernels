#!/usr/bin/env python3
"""MegaQwen A/B: fused Metal kernels vs the equivalent MLX ops.

Run from the repo root with .venv active:
  python experimental/benchmarks/megaqwen.py
  python experimental/benchmarks/megaqwen.py --json
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
import mlx.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from megaqwen import MegaQwenMLPBlock, down_residual, fused_add_rmsnorm
from megaqwen.patch import patch_qwen_decoder


def timeit(fn, args, iters, runs=7):
    for _ in range(3):
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


def add_norm_case(rows: int, hidden: int, iters: int) -> List[Dict]:
    x = mx.random.normal((1, rows, hidden)).astype(mx.bfloat16)
    res = mx.random.normal((1, rows, hidden)).astype(mx.bfloat16)
    w = mx.random.normal((hidden,)).astype(mx.bfloat16)
    mx.eval(x, res, w)
    custom = fused_add_rmsnorm(hidden)
    mlx_norm = lambda a, b, c: (
        mx.fast.rms_norm(a + b, c, 1e-6),
        a + b,
    )
    ref_xn, ref_h = mlx_norm(x, res, w)
    got_xn, got_h = custom(x, res, w)
    mx.eval(ref_xn, ref_h, got_xn, got_h)
    ok = bool(mx.allclose(got_xn, ref_xn, atol=2e-2, rtol=2e-2)) and bool(
        mx.allclose(got_h, ref_h, atol=2e-2, rtol=2e-2)
    )
    items = []
    for name, fn in (("mlx_add_norm", mlx_norm), ("mega_add_norm", custom)):
        lo, med = timeit(fn, (x, res, w), iters)
        items.append({"case": "add_norm", "rows": rows, "variant": name,
                      "min_us": lo, "median_us": med, "ok": ok})
    return items


def down_case(inter: int, hidden: int, iters: int) -> List[Dict]:
    act = mx.random.normal((inter,)).astype(mx.bfloat16)
    w = mx.random.normal((hidden, inter)).astype(mx.bfloat16)
    res = mx.random.normal((hidden,)).astype(mx.bfloat16)
    mx.eval(act, w, res)
    custom = down_residual(inter, hidden)
    mlx_down = lambda a, b, c: b @ a + c
    ref = mlx_down(act, w, res)
    got = custom(act, w, res)
    mx.eval(ref, got)
    ok = bool(mx.allclose(got, ref, atol=2e-2, rtol=2e-2))
    items = []
    for name, fn in (("mlx_down", mlx_down), ("mega_down", custom)):
        lo, med = timeit(fn, (act, w, res), iters)
        items.append({"case": "down", "inter": inter, "hidden": hidden,
                      "variant": name, "min_us": lo, "median_us": med, "ok": ok})
    return items


def block_case(rows: int, hidden: int, inter: int, iters: int) -> List[Dict]:
    x = mx.random.normal((1, rows, hidden)).astype(mx.bfloat16) * 0.02
    attn = mx.random.normal((1, rows, hidden)).astype(mx.bfloat16) * 0.02
    wn = mx.ones((hidden,)).astype(mx.bfloat16)
    wg = mx.random.normal((inter, hidden)).astype(mx.bfloat16) * 0.02
    wu = mx.random.normal((inter, hidden)).astype(mx.bfloat16) * 0.02
    wd = mx.random.normal((hidden, inter)).astype(mx.bfloat16) * 0.02
    mx.eval(x, attn, wn, wg, wu, wd)
    block = MegaQwenMLPBlock(hidden, inter)

    def mlx_block(a, b, c, d, e, f):
        h = a + b
        xn = mx.fast.rms_norm(h, c, 1e-6)
        act = nn.silu(xn @ d.T) * (xn @ e.T)
        return h + act @ f.T

    ref = mlx_block(x, attn, wn, wg, wu, wd)
    got = block(x, attn, wn, wg, wu, wd)
    mx.eval(ref, got)
    ok = bool(mx.allclose(got, ref, atol=3e-2, rtol=3e-2))
    items = []
    for name, fn in (("mlx_block", mlx_block), ("mega_block", block)):
        lo, med = timeit(fn, (x, attn, wn, wg, wu, wd), iters)
        items.append({"case": "block", "rows": rows, "variant": name,
                      "min_us": lo, "median_us": med, "ok": ok})
    return items


def print_group(title: str, items: List[Dict]) -> None:
    print()
    print(title)
    header = "%-14s %10s %10s %8s %6s" % ("variant", "min µs", "median", "speedup", "ok")
    print(header)
    print("-" * len(header))
    base = items[0]["median_us"]
    for item in items:
        print(
            "%-14s %10.2f %10.2f %8.2fx %6s"
            % (
                item["variant"],
                item["min_us"],
                item["median_us"],
                base / item["median_us"],
                "yes" if item["ok"] else "NO",
            )
        )


def e2e_case(
    model_path: str, prompt_file: str, runs: int, tokens: int, mode: str = "down"
) -> Dict:
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler

    prompt = open(prompt_file).read()
    model, tokenizer = load(os.path.expanduser(model_path))
    sampler = make_sampler(temp=0)

    def one():
        last = None
        for response in stream_generate(
            model, tokenizer, prompt, max_tokens=tokens, sampler=sampler
        ):
            last = response
        return last.generation_tps, last.prompt_tps, last.text

    # Warm both paths.
    one()
    _, restore = patch_qwen_decoder(mode)
    one()
    restore()

    baseline, custom = [], []
    pre_base, pre_custom = [], []
    texts = []
    for i in range(runs):
        if i % 2 == 0:
            b = one()
            _, restore = patch_qwen_decoder(mode)
            c = one()
            restore()
        else:
            _, restore = patch_qwen_decoder(mode)
            c = one()
            restore()
            b = one()
        baseline.append(b[0])
        custom.append(c[0])
        pre_base.append(b[1])
        pre_custom.append(c[1])
        texts.append(b[2] == c[2])

    paired = [100.0 * (c / b - 1.0) for b, c in zip(baseline, custom)]
    paired_pre = [100.0 * (c / b - 1.0) for b, c in zip(pre_base, pre_custom)]
    return {
        "baseline": baseline,
        "custom": custom,
        "prefill_baseline": pre_base,
        "prefill_custom": pre_custom,
        "baseline_median": statistics.median(baseline),
        "custom_median": statistics.median(custom),
        "paired_delta_pct": statistics.median(paired),
        "prefill_baseline_median": statistics.median(pre_base),
        "prefill_custom_median": statistics.median(pre_custom),
        "prefill_paired_delta_pct": statistics.median(paired_pre),
        "text_match": all(texts),
        "mode": mode,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--e2e", action="store_true", help="patch a Qwen layer and measure mlx_lm")
    ap.add_argument("--e2e-mode", choices=["addnorm", "down", "both", "mlp", "bm4"], default="down")
    ap.add_argument("--e2e-runs", type=int, default=5)
    ap.add_argument("--e2e-tokens", type=int, default=16)
    ap.add_argument("--model", default="~/.cache/models/Qwen3.5-0.8B-bf16")
    ap.add_argument(
        "--prompt-file",
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "benchmark_prompt_10k.txt",
        ),
    )
    args = ap.parse_args()

    report = []
    for rows, iters in ((1, 1000), (2048, 50)):
        items = add_norm_case(rows, 1024, iters)
        report.extend(items)
        if not args.json:
            print_group("fused add+RMSNorm, S=%d, H=1024" % rows, items)

    items = down_case(3584, 1024, 200)
    report.extend(items)
    if not args.json:
        print_group("fused down+residual, K=3584, N=1024", items)

    for rows, iters in ((1, 50), (2048, 4)):
        items = block_case(rows, 1024, 3584, iters)
        report.extend(items)
        if not args.json:
            print_group("MegaQwen MLP block, S=%d" % rows, items)

    if args.e2e:
        e2e = e2e_case(
            args.model, args.prompt_file, args.e2e_runs, args.e2e_tokens, args.e2e_mode
        )
        summary = {
            "case": "e2e",
            "mode": args.e2e_mode,
            "baseline_median_tps": e2e["baseline_median"],
            "custom_median_tps": e2e["custom_median"],
            "paired_delta_pct": e2e["paired_delta_pct"],
            "prefill_baseline_median": e2e["prefill_baseline_median"],
            "prefill_custom_median": e2e["prefill_custom_median"],
            "prefill_paired_delta_pct": e2e["prefill_paired_delta_pct"],
            "text_match": e2e["text_match"],
        }
        report.append(summary)
        if not args.json:
            print()
            print(
                "end-to-end, %d paired runs x %d tokens, temp 0, mode %s"
                % (args.e2e_runs, args.e2e_tokens, args.e2e_mode)
            )
            print(
                "  decode: baseline %.2f  custom %.2f tok/s  paired delta %+.2f%%"
                % (
                    e2e["baseline_median"],
                    e2e["custom_median"],
                    e2e["paired_delta_pct"],
                )
            )
            print(
                "  prefill: baseline %.1f  custom %.1f tok/s  paired delta %+.2f%%"
                % (
                    e2e["prefill_baseline_median"],
                    e2e["prefill_custom_median"],
                    e2e["prefill_paired_delta_pct"],
                )
            )
            print("  text match: %s" % ("yes" if e2e["text_match"] else "no"))

    if args.json:
        print(json.dumps(report, indent=2))
    return 0 if all(item["ok"] for item in report if "ok" in item) else 1


if __name__ == "__main__":
    raise SystemExit(main())
