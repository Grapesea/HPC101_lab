"""TileLang implementation of the chunked Gated Delta Network prefill.

The forward is split in the same way as TileLang's GDN examples:

1. Compute the chunk-local W and U matrices in parallel.
2. Walk the chunks recurrently to produce V_new, the state entering every
   chunk, and the final state.
3. Compute every chunk's output in parallel from its entering state.

Unlike the upstream examples, the kernels below also handle grouped-value
attention (Hq != Hv), a non-multiple-of-64 sequence length, an FP32 initial
state, and dynamic batch/sequence dimensions.
"""

import torch
import tilelang
import tilelang.language as T


CHUNK_SIZE = 64
HEAD_DIM_K = 128
HEAD_DIM_V = 128
BLOCK_DIM = 64
LOG2E = 1.4426950408889634
SCALE = HEAD_DIM_K**-0.5


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    }
)
def _tilelang_recompute_w_u(
    qk_dtype,
    gate_dtype,
    accum_dtype,
):
    """Compute W=A@(beta*exp(g)*K) and U=A@(beta*V)."""

    batch_size = T.dynamic("batch_size")
    num_tokens = T.dynamic("num_tokens")
    H = T.dynamic("num_heads_v")
    Hg = T.dynamic("num_heads_qk")

    k_shape = (batch_size, num_tokens, Hg, HEAD_DIM_K)
    v_shape = (batch_size, num_tokens, H, HEAD_DIM_V)
    gate_shape = (batch_size, num_tokens, H)
    a_shape = (batch_size, num_tokens, H, CHUNK_SIZE)
    wu_shape = (batch_size, num_tokens, H, HEAD_DIM_V)

    @T.prim_func
    def kernel(
        k: T.Tensor(k_shape, dtype=qk_dtype),
        v: T.Tensor(v_shape, dtype=qk_dtype),
        g: T.Tensor(gate_shape, dtype=gate_dtype),
        beta: T.Tensor(gate_shape, dtype=gate_dtype),
        a: T.Tensor(a_shape, dtype=qk_dtype),
        w: T.Tensor(wu_shape, dtype=qk_dtype),
        u: T.Tensor(wu_shape, dtype=qk_dtype),
        total_chunks: T.int32,
    ):
        with T.Kernel(total_chunks * H, threads=128) as (block,):
            bh = block % H
            chunk_linear = block // H
            bb = chunk_linear % batch_size
            chunk = chunk_linear // batch_size
            bhg = bh // (H // Hg)
            left = chunk * CHUNK_SIZE

            beta_shared = T.alloc_shared((CHUNK_SIZE,), dtype=gate_dtype, scope="shared")
            gate_shared = T.alloc_shared((CHUNK_SIZE,), dtype=gate_dtype, scope="shared")
            a_shared = T.alloc_shared((CHUNK_SIZE, CHUNK_SIZE), dtype=qk_dtype)

            k_shared = T.alloc_shared((CHUNK_SIZE, BLOCK_DIM), dtype=qk_dtype)
            v_shared = T.alloc_shared((CHUNK_SIZE, BLOCK_DIM), dtype=qk_dtype)
            k_scaled_shared = T.alloc_shared((CHUNK_SIZE, BLOCK_DIM), dtype=qk_dtype)
            v_scaled_shared = T.alloc_shared((CHUNK_SIZE, BLOCK_DIM), dtype=qk_dtype)

            result_fragment = T.alloc_fragment((CHUNK_SIZE, BLOCK_DIM), dtype=accum_dtype)
            result_shared = T.alloc_shared((CHUNK_SIZE, BLOCK_DIM), dtype=qk_dtype)

            T.annotate_layout(
                {
                    k_shared: tilelang.layout.make_swizzled_layout(k_shared),
                    v_shared: tilelang.layout.make_swizzled_layout(v_shared),
                }
            )
            T.disable_warp_group_reg_alloc()

            # Padding invalid rows with zero lets all GEMMs keep their fixed
            # 64-token shape, including the final short chunk.
            for row in T.Parallel(CHUNK_SIZE):
                if left + row < num_tokens:
                    beta_shared[row] = beta[bb, left + row, bh]
                    gate_shared[row] = T.exp2(g[bb, left + row, bh] * LOG2E)
                else:
                    beta_shared[row] = 0
                    gate_shared[row] = 0

            for row, col in T.Parallel(CHUNK_SIZE, CHUNK_SIZE):
                if left + row < num_tokens:
                    a_shared[row, col] = a[bb, left + row, bh, col]
                else:
                    a_shared[row, col] = 0

            # U = A @ (beta * V).
            for tile in T.serial(HEAD_DIM_V // BLOCK_DIM):
                for row, col in T.Parallel(CHUNK_SIZE, BLOCK_DIM):
                    if left + row < num_tokens:
                        v_shared[row, col] = v[bb, left + row, bh, tile * BLOCK_DIM + col]
                    else:
                        v_shared[row, col] = 0
                for row, col in T.Parallel(CHUNK_SIZE, BLOCK_DIM):
                    v_scaled_shared[row, col] = v_shared[row, col] * beta_shared[row]
                T.gemm(a_shared, v_scaled_shared, result_fragment, clear_accum=True)
                T.copy(result_fragment, result_shared)
                for row, col in T.Parallel(CHUNK_SIZE, BLOCK_DIM):
                    if left + row < num_tokens:
                        u[bb, left + row, bh, tile * BLOCK_DIM + col] = result_shared[row, col]

            # W = A @ (beta * exp(g) * K).
            for tile in T.serial(HEAD_DIM_K // BLOCK_DIM):
                for row, col in T.Parallel(CHUNK_SIZE, BLOCK_DIM):
                    if left + row < num_tokens:
                        k_shared[row, col] = k[bb, left + row, bhg, tile * BLOCK_DIM + col]
                    else:
                        k_shared[row, col] = 0
                for row, col in T.Parallel(CHUNK_SIZE, BLOCK_DIM):
                    k_scaled_shared[row, col] = (
                        k_shared[row, col] * beta_shared[row] * gate_shared[row]
                    )
                T.gemm(a_shared, k_scaled_shared, result_fragment, clear_accum=True)
                T.copy(result_fragment, result_shared)
                for row, col in T.Parallel(CHUNK_SIZE, BLOCK_DIM):
                    if left + row < num_tokens:
                        w[bb, left + row, bh, tile * BLOCK_DIM + col] = result_shared[row, col]

    return kernel


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    }
)
def _tilelang_chunk_state(
    qk_dtype,
    gate_dtype,
    state_dtype,
    accum_dtype,
    use_initial_state,
):
    """Run the inter-chunk recurrence and materialize each entering state."""

    batch_size = T.dynamic("batch_size")
    num_tokens = T.dynamic("num_tokens")
    H = T.dynamic("num_heads_v")
    Hg = T.dynamic("num_heads_qk")
    num_chunks_dynamic = T.ceildiv(num_tokens, CHUNK_SIZE)

    k_shape = (batch_size, num_tokens, Hg, HEAD_DIM_K)
    wu_shape = (batch_size, num_tokens, H, HEAD_DIM_V)
    gate_shape = (batch_size, num_tokens, H)
    state_shape = (batch_size, H, HEAD_DIM_K, HEAD_DIM_V)
    hidden_shape = (
        batch_size,
        num_chunks_dynamic,
        H,
        HEAD_DIM_K,
        HEAD_DIM_V,
    )

    @T.prim_func
    def kernel(
        k: T.Tensor(k_shape, dtype=qk_dtype),
        w: T.Tensor(wu_shape, dtype=qk_dtype),
        u: T.Tensor(wu_shape, dtype=qk_dtype),
        g: T.Tensor(gate_shape, dtype=gate_dtype),
        initial_state: T.Tensor(state_shape, dtype=state_dtype),
        hidden: T.Tensor(hidden_shape, dtype=qk_dtype),
        final_state: T.Tensor(state_shape, dtype=state_dtype),
        v_new: T.Tensor(wu_shape, dtype=qk_dtype),
        num_chunks: T.int32,
    ):
        with T.Kernel(
            HEAD_DIM_V // BLOCK_DIM,
            batch_size * H,
            threads=128,
        ) as (bv, bbh):
            bb = bbh // H
            bh = bbh % H
            bhg = bh // (H // Hg)

            state_shared = T.alloc_shared((HEAD_DIM_K, BLOCK_DIM), dtype=qk_dtype)
            state_fragment = T.alloc_fragment((HEAD_DIM_K, BLOCK_DIM), dtype=accum_dtype)

            w_shared = T.alloc_shared((CHUNK_SIZE, HEAD_DIM_K), dtype=qk_dtype)
            u_shared = T.alloc_shared((CHUNK_SIZE, BLOCK_DIM), dtype=qk_dtype)
            u_fragment = T.alloc_fragment((CHUNK_SIZE, BLOCK_DIM), dtype=accum_dtype)
            value_fragment = T.alloc_fragment((CHUNK_SIZE, BLOCK_DIM), dtype=accum_dtype)
            value_shared = T.alloc_shared((CHUNK_SIZE, BLOCK_DIM), dtype=qk_dtype)
            k_shared = T.alloc_shared((CHUNK_SIZE, HEAD_DIM_K), dtype=qk_dtype)
            gate_shared = T.alloc_shared((CHUNK_SIZE,), dtype=gate_dtype, scope="shared")
            gate_last = T.alloc_var(T.float32)

            T.annotate_layout(
                {
                    u_shared: tilelang.layout.make_swizzled_layout(u_shared),
                    value_shared: tilelang.layout.make_swizzled_layout(value_shared),
                }
            )
            T.use_swizzle(10)

            if use_initial_state:
                for dim_k, dim_v in T.Parallel(HEAD_DIM_K, BLOCK_DIM):
                    state_fragment[dim_k, dim_v] = initial_state[
                        bb, bh, dim_k, bv * BLOCK_DIM + dim_v
                    ]
            else:
                T.clear(state_fragment)
            T.copy(state_fragment, state_shared)

            # This loop is intentionally serial: state from chunk c is an
            # input to chunk c+1.
            for chunk in T.serial(num_chunks):
                left = chunk * CHUNK_SIZE
                right = left + CHUNK_SIZE

                # Save S_c for the parallel output kernel.
                T.copy(
                    state_shared,
                    hidden[
                        bb,
                        chunk,
                        bh,
                        0:HEAD_DIM_K,
                        bv * BLOCK_DIM : (bv + 1) * BLOCK_DIM,
                    ],
                )

                for row, dim_k in T.Parallel(CHUNK_SIZE, HEAD_DIM_K):
                    if left + row < num_tokens:
                        w_shared[row, dim_k] = w[bb, left + row, bh, dim_k]
                    else:
                        w_shared[row, dim_k] = 0
                T.gemm(w_shared, state_shared, value_fragment, clear_accum=True)

                for row, dim_v in T.Parallel(CHUNK_SIZE, BLOCK_DIM):
                    if left + row < num_tokens:
                        u_shared[row, dim_v] = u[
                            bb, left + row, bh, bv * BLOCK_DIM + dim_v
                        ]
                    else:
                        u_shared[row, dim_v] = 0
                T.copy(u_shared, u_fragment)
                for row, dim_v in T.Parallel(CHUNK_SIZE, BLOCK_DIM):
                    value_fragment[row, dim_v] = (
                        u_fragment[row, dim_v] - value_fragment[row, dim_v]
                    )

                # Save the unscaled V_new; state update below uses a version
                # decayed to the end of the chunk.
                T.copy(value_fragment, value_shared)
                for row, dim_v in T.Parallel(CHUNK_SIZE, BLOCK_DIM):
                    if left + row < num_tokens:
                        v_new[
                            bb, left + row, bh, bv * BLOCK_DIM + dim_v
                        ] = value_shared[row, dim_v]

                if right <= num_tokens:
                    gate_last = g[bb, right - 1, bh]
                else:
                    gate_last = g[bb, num_tokens - 1, bh]

                for row in T.Parallel(CHUNK_SIZE):
                    if left + row < num_tokens:
                        gate_shared[row] = g[bb, left + row, bh]
                    else:
                        gate_shared[row] = gate_last

                for row, dim_v in T.Parallel(CHUNK_SIZE, BLOCK_DIM):
                    if left + row < num_tokens:
                        value_fragment[row, dim_v] *= T.exp2(
                            (gate_last - gate_shared[row]) * LOG2E
                        )
                    else:
                        value_fragment[row, dim_v] = 0

                gate_last = T.exp2(gate_last * LOG2E)
                for dim_k, dim_v in T.Parallel(HEAD_DIM_K, BLOCK_DIM):
                    state_fragment[dim_k, dim_v] *= gate_last

                for row, dim_k in T.Parallel(CHUNK_SIZE, HEAD_DIM_K):
                    if left + row < num_tokens:
                        k_shared[row, dim_k] = k[bb, left + row, bhg, dim_k]
                    else:
                        k_shared[row, dim_k] = 0

                T.copy(value_fragment, value_shared)
                T.gemm(k_shared, value_shared, state_fragment, transpose_A=True)
                T.copy(state_fragment, state_shared)

            T.copy(
                state_fragment,
                final_state[
                    bb,
                    bh,
                    0:HEAD_DIM_K,
                    bv * BLOCK_DIM : (bv + 1) * BLOCK_DIM,
                ],
            )

    return kernel


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    }
)
def _tilelang_chunk_output(
    qk_dtype,
    gate_dtype,
    accum_dtype,
):
    """Compute output chunks in parallel from V_new and entering states."""

    batch_size = T.dynamic("batch_size")
    num_tokens = T.dynamic("num_tokens")
    H = T.dynamic("num_heads_v")
    Hg = T.dynamic("num_heads_qk")
    num_chunks_dynamic = T.ceildiv(num_tokens, CHUNK_SIZE)

    qk_shape = (batch_size, num_tokens, Hg, HEAD_DIM_K)
    value_shape = (batch_size, num_tokens, H, HEAD_DIM_V)
    gate_shape = (batch_size, num_tokens, H)
    hidden_shape = (
        batch_size,
        num_chunks_dynamic,
        H,
        HEAD_DIM_K,
        HEAD_DIM_V,
    )

    @T.prim_func
    def kernel(
        q: T.Tensor(qk_shape, dtype=qk_dtype),
        k: T.Tensor(qk_shape, dtype=qk_dtype),
        v_new: T.Tensor(value_shape, dtype=qk_dtype),
        hidden: T.Tensor(hidden_shape, dtype=qk_dtype),
        g: T.Tensor(gate_shape, dtype=gate_dtype),
        output: T.Tensor(value_shape, dtype=qk_dtype),
        total_chunks: T.int32,
    ):
        with T.Kernel(
            HEAD_DIM_V // BLOCK_DIM,
            total_chunks * H,
            threads=128,
        ) as (bv, block):
            bh = block % H
            chunk_linear = block // H
            bb = chunk_linear % batch_size
            chunk = chunk_linear // batch_size
            bhg = bh // (H // Hg)
            left = chunk * CHUNK_SIZE

            q_shared = T.alloc_shared((CHUNK_SIZE, BLOCK_DIM), dtype=qk_dtype)
            k_shared = T.alloc_shared((CHUNK_SIZE, BLOCK_DIM), dtype=qk_dtype)
            value_shared = T.alloc_shared((CHUNK_SIZE, BLOCK_DIM), dtype=qk_dtype)
            hidden_shared = T.alloc_shared((BLOCK_DIM, BLOCK_DIM), dtype=qk_dtype)
            scores_shared = T.alloc_shared((CHUNK_SIZE, CHUNK_SIZE), dtype=qk_dtype)
            output_shared = T.alloc_shared((CHUNK_SIZE, BLOCK_DIM), dtype=qk_dtype)

            scores_fragment = T.alloc_fragment((CHUNK_SIZE, CHUNK_SIZE), dtype=accum_dtype)
            output_fragment = T.alloc_fragment((CHUNK_SIZE, BLOCK_DIM), dtype=accum_dtype)
            gate_shared = T.alloc_shared((CHUNK_SIZE,), dtype=gate_dtype, scope="shared")

            T.annotate_layout(
                {
                    q_shared: tilelang.layout.make_swizzled_layout(q_shared),
                    k_shared: tilelang.layout.make_swizzled_layout(k_shared),
                    value_shared: tilelang.layout.make_swizzled_layout(value_shared),
                }
            )
            T.disable_warp_group_reg_alloc()
            T.clear(scores_fragment)
            T.clear(output_fragment)

            for tile_k in T.serial(HEAD_DIM_K // BLOCK_DIM):
                for row, dim_k in T.Parallel(CHUNK_SIZE, BLOCK_DIM):
                    if left + row < num_tokens:
                        q_shared[row, dim_k] = q[
                            bb, left + row, bhg, tile_k * BLOCK_DIM + dim_k
                        ]
                        k_shared[row, dim_k] = k[
                            bb, left + row, bhg, tile_k * BLOCK_DIM + dim_k
                        ]
                    else:
                        q_shared[row, dim_k] = 0
                        k_shared[row, dim_k] = 0

                T.copy(
                    hidden[
                        bb,
                        chunk,
                        bh,
                        tile_k * BLOCK_DIM : (tile_k + 1) * BLOCK_DIM,
                        bv * BLOCK_DIM : (bv + 1) * BLOCK_DIM,
                    ],
                    hidden_shared,
                )
                T.gemm(q_shared, hidden_shared, output_fragment)
                T.gemm(q_shared, k_shared, scores_fragment, transpose_B=True)

            for row in T.Parallel(CHUNK_SIZE):
                if left + row < num_tokens:
                    gate_shared[row] = g[bb, left + row, bh]
                else:
                    gate_shared[row] = 0

            for row, dim_v in T.Parallel(CHUNK_SIZE, BLOCK_DIM):
                if left + row < num_tokens:
                    output_fragment[row, dim_v] *= T.exp2(
                        gate_shared[row] * LOG2E
                    )
                else:
                    output_fragment[row, dim_v] = 0

            for row, col in T.Parallel(CHUNK_SIZE, CHUNK_SIZE):
                if left + row < num_tokens and left + col < num_tokens:
                    if row >= col:
                        scores_fragment[row, col] *= T.exp2(
                            (gate_shared[row] - gate_shared[col]) * LOG2E
                        )
                    else:
                        scores_fragment[row, col] = 0
                else:
                    scores_fragment[row, col] = 0

            for row, dim_v in T.Parallel(CHUNK_SIZE, BLOCK_DIM):
                if left + row < num_tokens:
                    value_shared[row, dim_v] = v_new[
                        bb, left + row, bh, bv * BLOCK_DIM + dim_v
                    ]
                else:
                    value_shared[row, dim_v] = 0

            T.copy(scores_fragment, scores_shared)
            T.gemm(scores_shared, value_shared, output_fragment)

            for row, dim_v in T.Parallel(CHUNK_SIZE, BLOCK_DIM):
                output_fragment[row, dim_v] *= SCALE

            T.copy(output_fragment, output_shared)
            for row, dim_v in T.Parallel(CHUNK_SIZE, BLOCK_DIM):
                if left + row < num_tokens:
                    output[
                        bb, left + row, bh, bv * BLOCK_DIM + dim_v
                    ] = output_shared[row, dim_v]

    return kernel


# q/k: [B, T, Hq, 128] BF16
# v: [B, T, Hv, 128] BF16
# g_cumsum/beta: [B, T, Hv] FP32
# A: [B, T, Hv, 64] BF16
# initial_state/final_state: [B, Hv, 128, 128] FP32
def gdn_prefill_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_cumsum: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    initial_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run GDN prefill using three TileLang kernels."""

    batch_size, num_tokens, num_heads_qk, head_dim_k = q.shape
    _, _, num_heads_v, head_dim_v = v.shape

    if head_dim_k != HEAD_DIM_K or head_dim_v != HEAD_DIM_V:
        raise ValueError("GDN kernels require key/value head dimensions of 128")
    if num_tokens <= 0:
        raise ValueError("GDN kernels require a non-empty sequence")
    if num_heads_v % num_heads_qk != 0:
        raise ValueError("the number of value heads must be divisible by Q/K heads")
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16 or v.dtype != torch.bfloat16:
        raise TypeError("q, k, and v must be bfloat16")
    if g_cumsum.dtype != torch.float32 or beta.dtype != torch.float32:
        raise TypeError("g_cumsum and beta must be float32")
    if A.dtype != torch.bfloat16:
        raise TypeError("A must be bfloat16")
    if initial_state is not None and initial_state.dtype != torch.float32:
        raise TypeError("initial_state must be float32")

    num_chunks = tilelang.cdiv(num_tokens, CHUNK_SIZE)
    total_chunks = batch_size * num_chunks

    # Intermediate tensors deliberately use BF16, matching the Tensor Core
    # data path in the upstream TileLang GDN implementation. Accumulation and
    # the recurrent state remain FP32 inside the kernels.
    w = torch.empty_like(v)
    u = torch.empty_like(v)
    v_new = torch.empty_like(v)
    hidden = torch.empty(
        (
            batch_size,
            num_chunks,
            num_heads_v,
            HEAD_DIM_K,
            HEAD_DIM_V,
        ),
        dtype=q.dtype,
        device=q.device,
    )
    output = torch.empty_like(v)
    final_state = torch.empty(
        (batch_size, num_heads_v, HEAD_DIM_K, HEAD_DIM_V),
        dtype=torch.float32,
        device=q.device,
    )

    recompute_w_u = _tilelang_recompute_w_u(
        qk_dtype=q.dtype,
        gate_dtype=g_cumsum.dtype,
        accum_dtype="float32",
    )
    recompute_w_u(k, v, g_cumsum, beta, A, w, u, total_chunks)

    use_initial_state = initial_state is not None
    chunk_state = _tilelang_chunk_state(
        qk_dtype=q.dtype,
        gate_dtype=g_cumsum.dtype,
        state_dtype="float32",
        accum_dtype="float32",
        use_initial_state=use_initial_state,
    )
    # In the zero-state specialization the input is dead. Reusing the output
    # buffer avoids allocating a dummy 128x128 state tensor.
    state_source = initial_state if initial_state is not None else final_state
    chunk_state(
        k,
        w,
        u,
        g_cumsum,
        state_source,
        hidden,
        final_state,
        v_new,
        num_chunks,
    )

    chunk_output = _tilelang_chunk_output(
        qk_dtype=q.dtype,
        gate_dtype=g_cumsum.dtype,
        accum_dtype="float32",
    )
    chunk_output(
        q,
        k,
        v_new,
        hidden,
        g_cumsum,
        output,
        total_chunks,
    )

    return output, final_state
