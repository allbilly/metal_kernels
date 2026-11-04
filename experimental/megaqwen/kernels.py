"""Fused Metal kernels for a Qwen3.5-0.8B decode/prefill block."""

from __future__ import annotations

import mlx.core as mx

ADD_RMSNORM = r"""
    uint row = threadgroup_position_in_grid.x;
    uint lane = thread_position_in_threadgroup.x;
    uint nt = threads_per_threadgroup.x;
    threadgroup float partial[NT];
    float vals[H / NT];
    float acc = 0.0f;
    #pragma unroll
    for (int k = 0; k < H / NT; k++) {
        uint i = lane + k * nt;
        float v = float(x[row * H + i]) + float(res[row * H + i]);
        vals[k] = v;
        acc += v * v;
        hout[row * H + i] = T(v);
    }
    partial[lane] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = nt / 2; s > 0; s >>= 1) {
        if (lane < s) partial[lane] += partial[lane + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float scale = metal::rsqrt(partial[0] * (1.0f / float(H)) + EPS);
    #pragma unroll
    for (int k = 0; k < H / NT; k++) {
        uint i = lane + k * nt;
        out[row * H + i] = T(vals[k] * scale * float(w[i]));
    }
"""

DOWN_RESIDUAL = r"""
    uint gid = threadgroup_position_in_grid.x;
    uint tid = thread_position_in_threadgroup.x;
    uint warp = tid >> 5;
    uint lane = tid & 31;
    uint row = gid * ROWS + warp;
    device const vec<T, VEC>* xv = (device const vec<T, VEC>*)x;
    device const vec<T, VEC>* wv = (device const vec<T, VEC>*)(w + row * K);
    float acc = 0.0f;
    for (int k = lane; k < K / VEC; k += 32) {
        vec<T, VEC> a = xv[k];
        vec<T, VEC> b = wv[k];
        #pragma unroll
        for (int j = 0; j < VEC; j++) acc += float(a[j]) * float(b[j]);
    }
    for (ushort sn = 16; sn >= 1; sn >>= 1) acc += simd_shuffle_down(acc, sn);
    if (lane == 0) {
        T rounded = T(acc);
        out[row] = T(float(rounded) + float(res[row]));
    }
"""

LM_HEAD = r"""
    uint gid = threadgroup_position_in_grid.x;
    uint tid = thread_position_in_threadgroup.x;
    uint warp = tid >> 5;
    uint lane = tid & 31;
    uint row = gid * ROWS + warp;
    device const vec<T, VEC>* xv = (device const vec<T, VEC>*)x;
    device const vec<T, VEC>* wv = (device const vec<T, VEC>*)(w + row * K);
    float acc = 0.0f;
    for (int k = lane; k < K / VEC; k += 32) {
        vec<T, VEC> a = xv[k];
        vec<T, VEC> b = wv[k];
        #pragma unroll
        for (int j = 0; j < VEC; j++) acc += float(a[j]) * float(b[j]);
    }
    for (ushort sn = 16; sn >= 1; sn >>= 1) acc += simd_shuffle_down(acc, sn);
    if (lane == 0) out[row] = T(acc);
"""


def fused_add_rmsnorm(hidden: int, threads: int = 256, eps: float = 1e-6):
    """Return (x, residual, weight) -> (normalized, summed_residual)."""
    assert hidden % threads == 0
    source = ADD_RMSNORM.replace("EPS", "%.9gf" % eps)
    kernel = mx.fast.metal_kernel(
        name="megaqwen_add_rmsnorm_%d_%d" % (hidden, threads),
        input_names=["x", "res", "w"],
        output_names=["out", "hout"],
        source=source,
    )

    def call(x: mx.array, res: mx.array, w: mx.array):
        rows = x.size // hidden
        return kernel(
            inputs=[x, res, w],
            template=[("T", x.dtype), ("H", hidden), ("NT", threads)],
            grid=(rows * threads, 1, 1),
            threadgroup=(threads, 1, 1),
            output_shapes=[x.shape, x.shape],
            output_dtypes=[x.dtype, x.dtype],
        )

    return call


DOWN = r"""
    uint gid = threadgroup_position_in_grid.x;
    uint tid = thread_position_in_threadgroup.x;
    uint warp = tid >> 5;
    uint lane = tid & 31;
    uint row = gid * ROWS + warp;
    device const vec<T, VEC>* xv = (device const vec<T, VEC>*)x;
    device const vec<T, VEC>* wv = (device const vec<T, VEC>*)(w + row * K);
    float acc = 0.0f;
    for (int k = lane; k < K / VEC; k += 32) {
        vec<T, VEC> a = xv[k];
        vec<T, VEC> b = wv[k];
        #pragma unroll
        for (int j = 0; j < VEC; j++) acc += float(a[j]) * float(b[j]);
    }
    for (ushort sn = 16; sn >= 1; sn >>= 1) acc += simd_shuffle_down(acc, sn);
    if (lane == 0) out[row] = T(acc);
"""


def down(inter: int, hidden: int, rows: int = 32, vec: int = 4):
    """Return (activation, down_weight) -> output (no residual)."""
    assert hidden % rows == 0 and inter % vec == 0
    kernel = mx.fast.metal_kernel(
        name="megaqwen_down_only_%d_%d_%d_%d" % (inter, hidden, rows, vec),
        input_names=["x", "w"],
        output_names=["out"],
        source=DOWN,
    )

    def call(act: mx.array, w: mx.array):
        return kernel(
            inputs=[act.reshape(-1), w],
            template=[("T", act.dtype), ("K", inter), ("ROWS", rows), ("VEC", vec)],
            grid=((hidden // rows) * rows * 32, 1, 1),
            threadgroup=(rows * 32, 1, 1),
            output_shapes=[(hidden,)],
            output_dtypes=[act.dtype],
        )[0]

    return call


BM4 = r"""
    uint gid = threadgroup_position_in_grid.x;
    uint sg = simdgroup_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;
    uint row0 = gid * ROWS + sg * TM;
    device const vec<T, TN>* xp = (device const vec<T, TN>*)x + lane;
    device const vec<T, TN>* wp = (device const vec<T, TN>*)(w + row0 * K) + lane;
    float acc[TM];
    #pragma unroll
    for (int t = 0; t < TM; t++) acc[t] = 0.0f;
    for (int i = 0; i < K / (32 * TN); i++) {
        int off = i * 32;
        vec<T, TN> xa = xp[off];
        #pragma unroll
        for (int t = 0; t < TM; t++) {
            vec<T, TN> wb = wp[t * (K / TN) + off];
            #pragma unroll
            for (int j = 0; j < TN; j++) acc[t] += float(xa[j]) * float(wb[j]);
        }
    }
    #pragma unroll
    for (int t = 0; t < TM; t++) {
        for (ushort sn = 16; sn >= 1; sn >>= 1) acc[t] += simd_shuffle_down(acc[t], sn);
        if (lane == 0) out[row0 + t] = T(acc[t]);
    }
"""


def bm4_gemv(inter: int, hidden: int, bm: int = 4, tm: int = 4, tn: int = 4):
    """MLX bm4-style layout: BM simdgroups, TM rows each, TN cols per lane."""
    rows = bm * tm
    assert hidden % rows == 0 and inter % (32 * tn) == 0
    kernel = mx.fast.metal_kernel(
        name="megaqwen_bm4_%d_%d_%d_%d_%d" % (inter, hidden, bm, tm, tn),
        input_names=["x", "w"],
        output_names=["out"],
        source=BM4,
    )

    def call(x: mx.array, w: mx.array):
        return kernel(
            inputs=[x.reshape(-1), w],
            template=[
                ("T", x.dtype),
                ("K", inter),
                ("ROWS", rows),
                ("TM", tm),
                ("TN", tn),
            ],
            grid=((hidden // rows) * bm * 32, 1, 1),
            threadgroup=(bm * 32, 1, 1),
            output_shapes=[(hidden,)],
            output_dtypes=[x.dtype],
        )[0]

    return call


def down_residual(inter: int, hidden: int, rows: int = 32, vec: int = 4):
    """Return (activation, down_weight, residual) -> output."""
    assert hidden % rows == 0 and inter % vec == 0
    kernel = mx.fast.metal_kernel(
        name="megaqwen_down_%d_%d_%d_%d" % (inter, hidden, rows, vec),
        input_names=["x", "w", "res"],
        output_names=["out"],
        source=DOWN_RESIDUAL,
    )

    def call(act: mx.array, w: mx.array, res: mx.array):
        return kernel(
            inputs=[act.reshape(-1), w, res.reshape(-1)],
            template=[("T", act.dtype), ("K", inter), ("ROWS", rows), ("VEC", vec)],
            grid=((hidden // rows) * rows * 32, 1, 1),
            threadgroup=(rows * 32, 1, 1),
            output_shapes=[(hidden,)],
            output_dtypes=[act.dtype],
        )[0]

    return call


def lm_head(hidden: int, vocab: int, rows: int = 8, vec: int = 4):
    """Return (x, embedding_weight) -> logits (1, vocab)."""
    assert vocab % rows == 0 and hidden % vec == 0
    kernel = mx.fast.metal_kernel(
        name="megaqwen_lm_head_%d_%d_%d_%d" % (hidden, vocab, rows, vec),
        input_names=["x", "w"],
        output_names=["out"],
        source=LM_HEAD,
    )

    def call(x: mx.array, w: mx.array):
        flat = x.reshape(-1, hidden)
        return kernel(
            inputs=[flat, w],
            template=[("T", x.dtype), ("K", hidden), ("ROWS", rows), ("VEC", vec)],
            grid=((vocab // rows) * rows * 32, 1, 1),
            threadgroup=(rows * 32, 1, 1),
            output_shapes=[(flat.shape[0], vocab)],
            output_dtypes=[x.dtype],
        )[0]

    return call
