# rmsnorm kernel a/b

`experimental/benchmarks/rmsnorm.py` compares four ways to run Qwen3.5-0.8B RMSNorm (`H=1024`):

- `mx.fast.rms_norm` — MLX's fused primitive
- unfused MLX ops — `x * rsqrt(mean(x*x) + eps) * w`
- `mx.compile(unfused, shapeless=True)`
- `mx.fast.metal_kernel` — one threadgroup per row, 64 or 256 threads

The point is not that the custom kernel wins. It is to measure kernel latency,
check correctness, and then check whether the kernel delta survives end-to-end.

## method

M1 Air 8 GB, macOS 27.2, MLX 0.32.4, bf16, `B=1`, `H=1024`.
Kernel timings queue 2000 calls at `S=1` and 50 calls at `S=2048`, sync once,
repeat 7 times, and report the minimum. Per-call `mx.eval` is excluded because
it adds ~150 µs of Python/sync overhead and hides the GPU cost.

Correctness is `mx.allclose` against `mx.fast.rms_norm` with `atol=rtol=2e-2`
for bf16. `maxdiff` is the largest absolute difference.

End-to-end is mlx_lm 0.32.2, 16 tokens, `temp 0`, the 1012-token
`benchmark_prompt_10k.txt`, 5 or 7 runs as noted. It measures baseline, patches all
49 `nn.RMSNorm` sites with the 64-thread kernel, measures again, restores the
original modules, and measures baseline a second time (A/B/A).

## kernel results

| variant     | S=1 min µs | S=2048 min µs | S=2048 GB/s | maxdiff |
|-------------|------------|---------------|-------------|---------|
| `mlx_fast`  |       2.33 |         99.16 |        84.6 |   0.000 |
| `unfused`   |      22.87 |        444.68 |        18.9 |   0.062 |
| `compiled`  |      11.38 |        359.45 |        23.3 |   0.062 |
| `custom64`  |       4.65 |        110.75 |        75.8 |   0.062 |
| `custom256` |       4.69 |        131.69 |        63.7 |   0.062 |

`mx.fast.rms_norm` wins both shapes. The custom kernel is 2.0× slower at
`S=1` (launch-bound) and 1.12× slower at `S=2048`. `mx.compile` improves the
unfused graph 2.0× at `S=1` and 1.24× at `S=2048`, but it is still far behind
the fused primitive. Going from 64 to 256 threads makes prefill 19% slower:
more threads and more reduction barriers cost more than the extra parallelism.

`GB/s` is op bytes divided by time. At `S=2048` it exceeds the 68.3 GB/s DRAM
peak because the 4 MB input is reused across the 50 queued calls and hits cache;
do not read it as a DRAM roofline.

## end-to-end

| session | runs | baseline A tok/s | custom tok/s | baseline B tok/s | custom vs mean | baseline drift |
|---------|------|------------------|--------------|------------------|----------------|----------------|
| 1       | 5    |            34.89 |        34.39 |            34.29 |          -0.6% |          -1.7% |
| 2       | 7    |            35.65 |        34.21 |            35.24 |          -3.5% |          -1.1% |
| 3       | 7    |            35.57 |        35.70 |            35.28 |          +0.8% |          -0.8% |

The generated text matched in every session. The custom kernel ranges
34.2–35.7 tok/s and the baseline 34.3–35.7 tok/s; the delta flips sign
(-3.5% to +0.8%) and the baseline drift is the same order. A 2× kernel
regression does not become a 2× model regression: the end-to-end delta stays
within a few percent, and the restoring baseline shows the drift is comparable
to the effect.

## gemv a/b

`experimental/benchmarks/gemv.py` compares `x @ w.T` (MLX's `gemv_bfloat16_bm4...` at batch 1)
with a custom `mx.fast.metal_kernel`: one threadgroup per output row, 64
threads, tree sum. `K=1024`, bf16, min of 7 queued runs.

| variant   |  N=1024 min µs |  N=3584 min µs |
|-----------|----------------|----------------|
| mlx_gemv  |           19.9 |           89.1 |
| custom64  |           24.7 |           96.4 |

MLX is 1.24× faster at `N=1024` and 1.08× at `N=3584`. The custom kernel is
correct but loses at both Qwen shapes, so it was not wired into mlx_lm.
`GB/s` uses weight + input + output bytes and includes weight reuse across
queued calls; at `N=1024` it exceeds the 68.3 GB/s DRAM peak because the 2 MB
weight matrix is cache-resident.

## per-kernel counters

`scripts/counters.py` joins `gpu-counter-value` samples to Shader Timeline
intervals by timestamp. Run one kernel at a time so the join is valid:

```bash
scripts/profile.sh --attach PID --seconds 3 --counters --keep --out traces/gemv
scripts/counters.py traces/gemv
```

Separate 3 s traces, same `K=1024, N=3584` shape:

| kernel                   |      n |   alu% |   buf% |   llc% |   occ% |  read GB/s |
|--------------------------|--------|--------|--------|--------|--------|------------|
| custom GEMV 64t          |   4246 |   63.6 |   14.7 |   13.7 |   71.9 |       25.9 |
| MLX gemv_bfloat16_bm4    |   1338 |   19.4 |   74.4 |   28.0 |   39.6 |       28.7 |

The custom kernel is ALU-limited (63.6%) while MLX's GEMV is buffer-read
limited (74.4%); the custom kernel has higher occupancy (71.9% vs 39.6%) but
lower read bandwidth (25.9 vs 28.7 GB/s). Scalar loads and the tree reduction
do more ALU work than MLX's tuned GEMV. The next variant should use vectorized
loads and a SIMD-group reduction, then re-run the same counters and timing.
Counter values are averages over samples inside each shader interval, not
Xcode's exact per-kernel aggregation.

## megaqwen

`experimental/megaqwen/` packages the fused residual+RMSNorm and down+residual kernels and
a Qwen MLP block. The fused add+RMSNorm is 2.18× faster than the MLX op at
`S=2048` and 1.07× at `S=1`; the block is consistently ~1% faster at prefill
and within drift at decode. After matching MLX's `simd_shuffle_down` GEMV
reduction, the integrated `down` path is token-identical at 64 and 128 tokens;
the paired 64-token A/B measured +0.12% decode and the 128-token run -0.35%,
so the effect is not stable. `experimental/benchmarks/mixer.py` shows the
attention/GDN output projections are bitwise-identical but slower than MLX
(the bm4 layout is 0.96x for GDN and 0.98x for attention), so that fusion is
not integrated. `experimental/benchmarks/mlp_bm4.py` shows the tuned bm4 down projection is
1.11x faster than `nn.Linear` while remaining bitwise-equal; the integrated
`bm4` mode measured +1.22% decode at 64 tokens and +0.49% at 128, text
identical. The down kernel tuning came from metal-profiler's static AGX
analysis (56% SCIB/control, 9 loads before wait; hoisting row pointers cut
102 to 79.5 µs). The full table, the end-to-end patch results, the one-kernel
caveat, and the Apple-silicon references are in `experimental/megaqwen/README.md`.

## what this changes

Both RMSNorm and this GEMV rewrite lose to the built-in. For GEMV the counters
point at ALU work in the custom variant, not bandwidth, so the next attempt is
vectorized loads plus a SIMD-group reduction; only wire it if it wins the same
A/B. A custom kernel has to beat `mx.fast.rms_norm` or MLX's tuned GEMV, not an
unfused graph. If neither wins, profile the linear-attention path next instead
of rewriting another op that is already near its roofline.

Before a kernel rewrite, profile it with `scripts/profile.sh` and
`scripts/timeline.py`; after it, use the same A/B/A shape here so a fast
microbenchmark does not hide a slower model.

## run

From the repo root with `.venv` active (`source .venv/bin/activate`):

```bash
python experimental/benchmarks/rmsnorm.py --runs 7
python experimental/benchmarks/rmsnorm.py --e2e --e2e-runs 7
python experimental/benchmarks/gemv.py --runs 7
python experimental/benchmarks/megaqwen.py
python experimental/benchmarks/mixer.py
python experimental/benchmarks/mlp_bm4.py
scripts/profile.sh --attach PID --seconds 3 --counters --keep --out traces/gemv
scripts/counters.py traces/gemv
```

## reading

- [MLX custom Metal kernels](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html)
- [Kernel fusion in MLX](https://nipunbatra.github.io/blog/posts/2026-04-25-mlx-kernel-fusion.html)
- [Apple: optimizing GPU performance](https://developer.apple.com/documentation/xcode/optimizing-gpu-performance)
