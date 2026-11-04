#!/usr/bin/env python3
"""Native fp16/bf16 vs fp32 ALU and simdgroup-matrix throughput on this GPU.

MLX's GEMM converts both dtypes to fp32, so this times the units directly:
a register FMA loop and a simdgroup_matrix 8x8 MMA loop.

Run from the repo root with .venv active:
  python experimental/benchmarks/fp16_units.py
"""

from __future__ import annotations

import argparse
import statistics
import time

import mlx.core as mx

ALU = r"""
    uint tid = thread_position_in_threadgroup.x;
    uint gid = threadgroup_position_in_grid.x;
    uint idx = gid * threads_per_threadgroup.x + tid;
    vec<T, 4> a = vec<T, 4>(x[idx]);
    vec<T, 4> b = a * T(0.5);
    vec<T, 4> acc0 = a;
    vec<T, 4> acc1 = a;
    vec<T, 4> acc2 = a;
    vec<T, 4> acc3 = a;
    #pragma unroll 1
    for (int i = 0; i < ITER; i++) {
        acc0 = acc0 * a + b;
        acc1 = acc1 * a + b;
        acc2 = acc2 * a + b;
        acc3 = acc3 * a + b;
    }
    out[idx] = acc0.x + acc1.y + acc2.z + acc3.w;
"""

MMA = r"""
    uint sg = simdgroup_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;
    uint gid = threadgroup_position_in_grid.x;
    uint nwarps = threads_per_threadgroup.x >> 5;
    uint warp = gid * nwarps + sg;
    T v = x[warp];
    simdgroup_matrix<T, 8, 8> A;
    simdgroup_matrix<T, 8, 8> B;
    simdgroup_matrix<T, 8, 8> C0;
    simdgroup_matrix<T, 8, 8> C1;
    vec<T, 2> av = vec<T, 2>(v, v);
    vec<T, 2> bv = vec<T, 2>(v, v);
    vec<T, 2> cv = vec<T, 2>(T(0), T(0));
    reinterpret_cast<thread vec<T, 2>&>(A.thread_elements()) = av;
    reinterpret_cast<thread vec<T, 2>&>(B.thread_elements()) = bv;
    reinterpret_cast<thread vec<T, 2>&>(C0.thread_elements()) = cv;
    reinterpret_cast<thread vec<T, 2>&>(C1.thread_elements()) = cv;
    #pragma unroll
    for (int i = 0; i < ITER; i++) {
        simdgroup_multiply_accumulate(C0, A, B, C0);
        simdgroup_multiply_accumulate(C1, A, B, C1);
    }
    vec<T, 2> c0 = reinterpret_cast<thread vec<T, 2>&>(C0.thread_elements());
    vec<T, 2> c1 = reinterpret_cast<thread vec<T, 2>&>(C1.thread_elements());
    out[warp * 32 + lane] = c0[0] + c0[1] + c1[0] + c1[1];
"""


def timeit(kernel, dtype, n, iters, runs, inner=5):
    x = mx.full((n,), 0.25, dtype=dtype)
    kw = dict(
        inputs=[x],
        template=[("T", dtype), ("ITER", iters)],
        grid=(n, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(n,)],
        output_dtypes=[dtype],
    )
    for _ in range(3):
        out = kernel(**kw)[0]
        mx.eval(out)
    mx.synchronize()
    samples = []
    for _ in range(runs):
        start = time.perf_counter()
        outs = [kernel(**kw)[0] for _ in range(inner)]
        mx.eval(outs)
        mx.synchronize()
        samples.append((time.perf_counter() - start) / inner)
    assert bool(mx.all(mx.isfinite(out))), "non-finite output"
    return statistics.median(samples), float(out[0])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--threads", type=int, default=14 * 1024)
    args = ap.parse_args()

    threads = args.threads
    alu_iter = 65536
    mma_iter = 1024  # fully unrolled: the bf16 simdgroup loop fails to compile
    warps = threads // 32

    alu = mx.fast.metal_kernel(
        name="fp16_units_alu",
        input_names=["x"],
        output_names=["out"],
        source=ALU,
    )
    mma = mx.fast.metal_kernel(
        name="fp16_units_mma",
        input_names=["x"],
        output_names=["out"],
        source=MMA,
    )

    print("MLX %s, %d threads, median of %d" % (mx.__version__, threads, args.runs))
    results = {}
    for name, dtype in (("fp32", mx.float32), ("fp16", mx.float16), ("bf16", mx.bfloat16)):
        alu_flops = threads * alu_iter * 16 * 2
        mma_flops = warps * mma_iter * 2 * 8 * 8 * 8 * 2
        alu_sec, alu_val = timeit(alu, dtype, threads, alu_iter, args.runs)
        mma_sec, mma_val = timeit(mma, dtype, threads, mma_iter, args.runs, inner=20)
        results[name] = (alu_flops / alu_sec / 1e12, mma_flops / mma_sec / 1e12)
        print(
            "%-4s ALU %5.2f TFLOP/s (%.3f)   MMA %5.2f TFLOP/s (%.1f)"
            % (name, results[name][0], alu_val, results[name][1], mma_val)
        )
    for name in ("fp16", "bf16"):
        for i, unit in enumerate(("ALU", "MMA")):
            ratio = results[name][i] / results["fp32"][i]
            print("%s/fp32 %s %.2fx" % (name, unit, ratio))


if __name__ == "__main__":
    main()
