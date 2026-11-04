# MegaQwen

MegaQwen is a scoped Metal megakernel prototype for Qwen3.5-0.8B on Apple silicon.
It implements the residual + RMSNorm and MLP stages as fused whole-GPU
dispatches, not as one never-returning kernel.

## why not one kernel

Apple GPUs have no cooperative grid launch and no cross-threadgroup execution
barrier. A dispatch boundary costs ~1.6-2.7 µs of GPU time; an in-kernel
barrier costs ~2.1-2.6 µs with static slices (3.1-3.9 µs fenced), and a long
kernel is not preemptible. The published Apple-silicon designs therefore keep
the "static megakernel" (a pre-encoded program of fused whole-GPU dispatches)
and reject the one-kernel form:

- [lithos-metal](https://github.com/lithos-ai/lithos-metal) — layer-wise mixer megakernels, ICB replay.
- [mpk-apple](https://github.com/jiazhihao/mpk-apple/blob/main/docs/design/design.md) — 5 fused dispatches per Qwen layer, ~330 per token.
- [AGXForge](https://github.com/allbilly/AGXForge) — 7 dispatches per layer against mlx_lm's 16, fused projections and residuals. 1.14x at 1,792 tokens on M5, 0.91-1.02x at 196, and 1.32x when the same design is compiled by Apple. Archival research release, not maintained.
- [Hazy Research megakernel](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles) — the CUDA counter/barrier design that does not transfer.

## design

Qwen3.5-0.8B MLP residual block:

```text
h  = x + attn_out
xn = rmsnorm(h, w)
act = silu(gate @ xn) * (up @ xn)
out = h + down @ act
```

MegaQwen kernels:

- `fused_add_rmsnorm` — computes `h` and `xn` in one dispatch. It writes both
  because `h` is needed again for the final residual; a kernel that wrote only
  `xn` would not save the block-level traffic.
- `down_residual` — `down @ act + h` in one dispatch, one warp per output row,
  32 rows per threadgroup, vectorized bf16 loads.
- `lm_head` — optional tied-embedding GEMV. It wins in queued throughput but
  ties MLX on a single per-token call; see the note below.

The block uses the fused add+RMSNorm at prefill and the fused down+residual at
decode. Gate/up and the prefill GEMMs stay on MLX: the custom gate/up variants
were not faster.

## measured

M1 Air 8 GB, macOS 27.2, MLX 0.32.4, bf16, medians of 7 queued runs.

| kernel                    | MLX       | MegaQwen  | speedup |
|---------------------------|-----------|-----------|---------|
| add+RMSNorm, S=1          | 7.52 µs   | 7.05 µs   | 1.07x   |
| add+RMSNorm, S=2048       | 591.5 µs  | 271.1 µs  | 2.18x   |
| down+residual, K=3584     | 101.5 µs  | 96.2 µs   | 1.05x   |
| MLP block, S=1            | 329.1 µs  | 335.0 µs  | 0.98x   |
| MLP block, S=2048         | 31.44 ms  | 31.03 ms  | 1.01x   |

The decode block is within run-to-run drift (0.94-1.04x across sessions): the
fused kernels win individually, but the block is dominated by the gate/up/down
GEMMs. Prefill is consistently ~1% faster. The honest conclusion is that
elementwise fusion is not where decode time is; the full layer needs the
attention/GDN mixer and a faster GEMV path before the block win is material.

The `lm_head` kernel is 8.12 ms vs 8.56 ms queued at `V=248320`, but 8.37 ms
vs 8.35 ms on a single per-token call. It is included as a kernel-result, not
as an end-to-end win.

## end-to-end

`experimental/megaqwen/patch.py` replaces `DecoderLayer.__call__` in mlx_lm. The default
`down` mode keeps attention/GDN, gate/up, and prefill on MLX; `bm4` also
replaces gate/up at decode with the optimized bm4 kernels.

| mode      | decode Δ | prefill Δ | tokens |
|-----------|---------:|----------:|--------|
| `addnorm` |   -0.02% |    -0.26% | match  |
| `down`    |   +0.12% |    -0.29% | match  |
| `both`    |   -0.12% |    -0.26% | match  |
| `bm4`     |   +1.22% |    -0.18% | match  |

5 paired runs per arm, 64 tokens, temp 0, 1012-token prompt, M1 Air. The paired
A/B alternates baseline and patched runs so thermal drift does not favor either
side. Absolute tok/s moved between sessions (29-37), so only the paired deltas
are comparable. At 128 tokens, `down` measured -0.35% decode / +0.29% prefill
and `bm4` +0.49% decode / -0.25% prefill; both token-identical.

Numerics: MLX's GEMV reduces with `simd_shuffle_down` from 16 down to 1, not
`simd_sum`. Replacing the custom reduction with the same tree made the down
GEMV bitwise identical to `mlp.down_proj`, and both `down` and `bm4` produce
token-identical text at 64 and 128 tokens.

The honest result: the custom kernels are real wins in isolation (2.18x at
prefill, 1.07x at decode), and the integrated path is token-identical, but the
full-layer end-to-end effect is -0.1% to +1.2% and within drift. Decode streams
1.6 GB of weights per token; at 30-37 tok/s that is 48-59 GB/s of the 68 GB/s
DRAM peak, so MLX's GEMV is already near the memory roofline.

## bm4 MLP integration

`megaqwen.kernels.bm4_gemv` implements the bm4 layout for the MLP shapes:
gate/up `K=1024, N=3584` and down `K=3584, N=1024`. The integrated `bm4` mode
is token-identical at 64 and 128 tokens:

| path      | MLX `nn.Linear` | bm4     | speedup | bitwise |
|-----------|-----------------|---------|---------|---------|
| gate_proj | 90.9 µs         | 91.1 µs | 1.02x   | yes     |
| up_proj   | 95.4 µs         | 101.8 µs| 0.95x   | yes     |
| down_proj | 94.7 µs         | 79.5 µs | 1.11x   | yes     |

Min values; speedup from medians. `experimental/benchmarks/mlp_bm4.py` reproduces this.
The down projection is the real win (11-16% faster); gate is a tie and up is
slower, so the `bm4` mode uses custom gate/up only because the combined path
still measured positive.

The down kernel was tuned with
[metal-profiler](https://github.com/imperatormk/metal-profiler), which
disassembles the AGX binary statically. It reported 56% SCIB/control
instructions and 9 loads before the wait; hoisting the row pointers removed
the per-iteration `(row0+t)*K` address math and cut down from ~102 µs to
79.5 µs. End-to-end `bm4`: +1.22% decode at 64 tokens, +0.49% at 128, text
match.

## attention/GDN mixer

All 24 layers use an output projection with `K=2048, N=1024` (`o_proj` for the
6 full-attention layers, `out_proj` for the 18 GDN layers). Replacing it is
token-identical, but slower:

| path                          |       MLX |  MegaQwen |  speedup |
|-------------------------------|-----------|-----------|----------|
| GDN `out_proj`                |   39.1 µs |   41.6 µs |    0.95x |
| GDN `out_proj` + residual     |   42.3 µs |   55.8 µs |    0.73x |
| GDN `out_proj`, bm4           |   40.2 µs |   41.9 µs |    0.96x |
| attention `o_proj`            |   41.8 µs |   44.5 µs |    0.93x |
| attention `o_proj` + residual |   43.0 µs |   58.0 µs |    0.70x |
| attention `o_proj`, bm4       |   42.8 µs |   42.8 µs |    0.98x |

`experimental/benchmarks/mixer.py` reproduces this; every output is bitwise identical.
`bm4_gemv` implements MLX's bm4 layout (4 simdgroups, 4 rows each, 4 columns
per lane) and is still 4% slower for GDN and a tie for attention. The custom
GEMV loses 3-4 µs per layer, and fusing the residual makes it worse.
Integrating it would regress the model, so the mixer projection stays on MLX.
The internal attention/GDN elementwise stages are smaller than that loss, so
they cannot recover a material win either.

This is the current blocker: a material token-identical bf16 win is not
available from more fusion on this M1 Air. The remaining levers change the
problem: 4-bit/8-bit weights (less traffic, different numerics/quality) or
speculative decoding (a drafter and extra verify pass).

## run

From the repo root with `.venv` active (`source .venv/bin/activate`):

```bash
python experimental/benchmarks/megaqwen.py
python experimental/benchmarks/megaqwen.py --e2e --e2e-mode down --e2e-runs 7
python experimental/benchmarks/mixer.py
python experimental/benchmarks/mlp_bm4.py
python experimental/benchmarks/megaqwen.py --json
```

## next

1. A material token-identical bf16 win from more fusion is not available:
   `experimental/benchmarks/mixer.py` shows the attention/GDN output projection is
   bitwise-identical but 4-30% slower than MLX; the bm4 layout is also slower
   (0.96x GDN, 0.98x attention), and `experimental/benchmarks/mlp_bm4.py` shows the MLP
   bm4 kernels are 0.92-0.97x against the actual `nn.Linear`.
2. Remaining levers are quantized GEMV (different numerics/quality) or
   speculative decoding (a drafter plus verify pass); both need a product
   decision before implementation.
3. Keep the ICB/pre-encoded dispatch idea for host overhead, but it does not
   change the GPU work measured here.
