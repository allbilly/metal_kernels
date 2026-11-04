"""Torch MPS GEMM/GEMV on the Qwen3.5-0.8B shapes, for comparison with MLX."""

import argparse
import statistics
import time

import torch
import torch.nn.functional as F

SHAPES = [
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
DTYPES = [("bf16", torch.bfloat16), ("fp16", torch.float16), ("fp32", torch.float32)]


def bench(m, k, n, dtype, runs):
    x = torch.randn(m, k, device="mps", dtype=dtype)
    w = torch.randn(n, k, device="mps", dtype=dtype)
    for _ in range(3):
        F.linear(x, w)
    torch.mps.synchronize()
    samples = []
    for _ in range(runs):
        t0 = time.perf_counter()
        y = F.linear(x, w)
        torch.mps.synchronize()
        samples.append(time.perf_counter() - t0)
        del y
    return statistics.median(samples)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", type=int, default=7)
    args = ap.parse_args()
    for name, dtype in DTYPES:
        print("dtype", name)
        for m, runs in ((1, 20), (1011, args.runs)):
            print("M", m)
            for k, n in SHAPES:
                sec = bench(m, k, n, dtype, runs)
                flops = 2.0 * m * n * k
                moved = (m * k + n * k + m * n) * torch.finfo(dtype).bits // 8
                print(
                    "%-10s %7.2f ms %8.2f TFLOP/s %8.1f GB/s"
                    % ("%dx%d" % (n, k), sec * 1e3, flops / sec / 1e12, moved / sec / 1e9)
                )


if __name__ == "__main__":
    main()
