"""TileLang implementation of the chunked Gated Delta Network prefill.

The forward is fused into one persistent residual-state-output kernel:

1. Walk the chunks recurrently and directly compute
   V_new=A@(beta*(V-exp(g)*K@state)).
2. Compute each chunk's output from its entering state and V_new.
3. Advance the state without materializing chunk intermediates in HBM.

Unlike the upstream examples, the kernel below also handles grouped-value
attention (Hq != Hv), a non-multiple-of-64 sequence length, an FP32 initial
state, and dynamic batch/sequence dimensions.
"""

import torch
import tilelang
import tilelang.language as T


CHUNK_SIZE = 64
HEAD_DIM_K = 128
HEAD_DIM_V = 128
BLOCK_DIM = 128
LOG2E = 1.4426950408889634
SCALE = HEAD_DIM_K**-0.5


# Adapted from QwenLM/FlashQLA's MIT-licensed Hopper fused forward kernel.
# The course-provided A already contains the gate similarity transform.  Beta
# is applied row-wise to the residual before A, avoiding an A conversion.
@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    }
)
def _tilelang_warp_specialized_forward(
    qk_dtype,
    gate_dtype,
    state_dtype,
    accum_dtype,
    use_initial_state,
    block_dv,
):
    batch_size = T.dynamic("batch_size")
    num_tokens = T.dynamic("num_tokens")
    H = T.dynamic("num_heads_v")
    Hg = T.dynamic("num_heads_qk")
    qk_shape = (batch_size, num_tokens, Hg, HEAD_DIM_K)
    value_shape = (batch_size, num_tokens, H, HEAD_DIM_V)
    gate_shape = (batch_size, num_tokens, H)
    a_shape = (batch_size, num_tokens, H, CHUNK_SIZE)
    state_shape = (batch_size, H, HEAD_DIM_K, HEAD_DIM_V)

    @T.prim_func
    def kernel(
        q: T.Tensor(qk_shape, dtype=qk_dtype),
        k: T.Tensor(qk_shape, dtype=qk_dtype),
        v: T.Tensor(value_shape, dtype=qk_dtype),
        g: T.Tensor(gate_shape, dtype=gate_dtype),
        beta: T.Tensor(gate_shape, dtype=gate_dtype),
        a: T.Tensor(a_shape, dtype=qk_dtype),
        initial_state: T.Tensor(state_shape, dtype=state_dtype),
        output: T.Tensor(value_shape, dtype=qk_dtype),
        final_state: T.Tensor(state_shape, dtype=state_dtype),
        num_chunks: T.int32,
    ):
        with T.Kernel(
            T.ceildiv(HEAD_DIM_V, block_dv) * batch_size * H,
            threads=512,
        ) as (block,):
            bbh = block // T.ceildiv(HEAD_DIM_V, block_dv)
            bv = block % T.ceildiv(HEAD_DIM_V, block_dv)
            bb = bbh // H
            bh = bbh % H
            bhg = bh // (H // Hg)
            dv_start = bv * block_dv
            dv_end = (bv + 1) * block_dv
            num_full_chunks = num_tokens // CHUNK_SIZE

            q_shared = T.alloc_shared((2, CHUNK_SIZE, HEAD_DIM_K), dtype=qk_dtype)
            k_shared = T.alloc_shared((2, CHUNK_SIZE, HEAD_DIM_K), dtype=qk_dtype)
            v_shared = T.alloc_shared((2, CHUNK_SIZE, block_dv), dtype=qk_dtype)
            a_shared = T.alloc_shared((2, CHUNK_SIZE, CHUNK_SIZE), dtype=qk_dtype)
            g_shared = T.alloc_shared((2, CHUNK_SIZE), dtype=accum_dtype, scope="shared")
            b_shared = T.alloc_shared((2, CHUNK_SIZE), dtype=accum_dtype, scope="shared")

            output_shared = T.alloc_shared((CHUNK_SIZE, block_dv), dtype=qk_dtype)
            state_shared = T.alloc_shared((HEAD_DIM_K, block_dv), dtype=qk_dtype)
            value_shared = T.alloc_shared((CHUNK_SIZE, block_dv), dtype=qk_dtype)
            value_decay_shared = T.alloc_shared((CHUNK_SIZE, block_dv), dtype=qk_dtype)
            scores_shared = T.alloc_shared((CHUNK_SIZE, CHUNK_SIZE), dtype=qk_dtype)
            gate_exp_shared = T.alloc_shared((CHUNK_SIZE), dtype=accum_dtype, scope="shared")
            gate_decay_shared = T.alloc_shared((CHUNK_SIZE), dtype=accum_dtype, scope="shared")

            state_fragment = T.alloc_fragment((HEAD_DIM_K, block_dv), dtype=accum_dtype)
            output_fragment = T.alloc_fragment((CHUNK_SIZE, block_dv), dtype=accum_dtype)
            value_fragment = T.alloc_fragment((CHUNK_SIZE, block_dv), dtype=accum_dtype)
            prediction_fragment = T.alloc_fragment((CHUNK_SIZE, block_dv), dtype=accum_dtype)
            scores_fragment = T.alloc_fragment((CHUNK_SIZE, CHUNK_SIZE), dtype=accum_dtype)
            gate_fragment = T.alloc_fragment((CHUNK_SIZE, CHUNK_SIZE), dtype=accum_dtype)
            gate_last_local = T.alloc_local((1), dtype=accum_dtype)

            data_ready = T.alloc_barrier(arrive_count=[96] * 2)
            data_free = T.alloc_barrier(arrive_count=[384] * 2)
            bar_output = T.alloc_barrier(arrive_count=128)
            bar_start = T.alloc_barrier(arrive_count=416)
            bar_state = T.alloc_barrier(arrive_count=256)
            bar_value_ready = T.alloc_barrier(arrive_count=128)
            bar_decay_ready = T.alloc_barrier(arrive_count=128)
            bar_finish = T.alloc_barrier(arrive_count=288)

            T.use_swizzle(10)
            tx = T.get_thread_binding()

            if tx < 128:
                T.set_max_nreg(160, 1)
                if use_initial_state:
                    T.copy(
                        initial_state[bb, bh, 0:HEAD_DIM_K, dv_start:dv_end],
                        state_fragment,
                    )
                else:
                    T.clear(state_fragment)

                for chunk in T.serial(num_chunks):
                    T.barrier_wait(data_ready[chunk % 2], (chunk // 2) % 2)
                    T.barrier_arrive(bar_start)
                    T.barrier_wait(bar_start, chunk % 2)
                    T.copy(state_fragment, state_shared)
                    T.barrier_arrive(bar_state)
                    T.barrier_wait(bar_state, chunk % 2)
                    gate_last_local[0] = gate_exp_shared[CHUNK_SIZE - 1]
                    for dim_k, dim_v in T.Parallel(HEAD_DIM_K, block_dv):
                        state_fragment[dim_k, dim_v] *= gate_last_local[0]
                    T.barrier_wait(bar_decay_ready, chunk % 2)
                    T.gemm(
                        k_shared[chunk % 2, :, :],
                        value_decay_shared,
                        state_fragment,
                        transpose_A=True,
                        clear_accum=False,
                    )
                    T.barrier_arrive(data_free[chunk % 2])

                T.copy(
                    state_fragment,
                    final_state[bb, bh, 0:HEAD_DIM_K, dv_start:dv_end],
                )

            elif tx < 256:
                T.set_max_nreg(128, 1)
                for chunk in T.serial(num_chunks):
                    T.barrier_wait(data_ready[chunk % 2], (chunk // 2) % 2)
                    T.barrier_arrive(bar_start)
                    T.barrier_wait(bar_start, chunk % 2)
                    for row in T.Parallel(CHUNK_SIZE):
                        gate_exp_shared[row] = T.exp2(
                            g_shared[chunk % 2, row] * LOG2E
                        )
                    for row in T.Parallel(CHUNK_SIZE):
                        gate_decay_shared[row] = T.if_then_else(
                            chunk * CHUNK_SIZE + row < num_tokens,
                            T.exp2(
                                (
                                    g_shared[chunk % 2, CHUNK_SIZE - 1]
                                    - g_shared[chunk % 2, row]
                                )
                                * LOG2E
                            ),
                            0.0,
                        )
                    T.barrier_arrive(bar_state)
                    T.barrier_wait(bar_state, chunk % 2)
                    T.gemm(
                        k_shared[chunk % 2, :, :],
                        state_shared,
                        prediction_fragment,
                        clear_accum=True,
                    )
                    for row, dim_v in T.Parallel(CHUNK_SIZE, block_dv):
                        prediction_fragment[row, dim_v] *= -gate_exp_shared[row]
                    for row, dim_v in T.Parallel(CHUNK_SIZE, block_dv):
                        prediction_fragment[row, dim_v] += v_shared[
                            chunk % 2, row, dim_v
                        ]
                    for row, dim_v in T.Parallel(CHUNK_SIZE, block_dv):
                        prediction_fragment[row, dim_v] *= b_shared[
                            chunk % 2, row
                        ]
                    for row, dim_v in T.Parallel(CHUNK_SIZE, block_dv):
                        v_shared[chunk % 2, row, dim_v] = prediction_fragment[
                            row, dim_v
                        ]
                    T.gemm(
                        a_shared[chunk % 2, :, :],
                        v_shared[chunk % 2, :, :],
                        value_fragment,
                        clear_accum=True,
                    )
                    T.copy(value_fragment, value_shared)
                    T.barrier_arrive(bar_value_ready)
                    for row, dim_v in T.Parallel(CHUNK_SIZE, block_dv):
                        value_fragment[row, dim_v] *= gate_decay_shared[row]
                    T.copy(value_fragment, value_decay_shared)
                    T.barrier_arrive(bar_decay_ready)
                    T.barrier_arrive(bar_finish)
                    T.barrier_wait(bar_finish, chunk % 2)
                    T.barrier_arrive(data_free[chunk % 2])

            elif tx < 384:
                T.set_max_nreg(128, 1)
                for chunk in T.serial(num_chunks):
                    T.barrier_wait(data_ready[chunk % 2], (chunk // 2) % 2)
                    T.barrier_arrive(bar_start)
                    T.barrier_wait(bar_start, chunk % 2)
                    T.gemm(
                        q_shared[chunk % 2, :, :],
                        k_shared[chunk % 2, :, :],
                        scores_fragment,
                        transpose_B=True,
                        clear_accum=True,
                    )
                    for row, col in T.Parallel(CHUNK_SIZE, CHUNK_SIZE):
                        gate_fragment[row, col] = (
                            g_shared[chunk % 2, row] - g_shared[chunk % 2, col]
                        )
                    for row, col in T.Parallel(CHUNK_SIZE, CHUNK_SIZE):
                        if row >= col:
                            gate_fragment[row, col] = T.exp2(
                                gate_fragment[row, col] * LOG2E
                            )
                        else:
                            gate_fragment[row, col] = 0

                    T.barrier_wait(bar_state, chunk % 2)
                    T.gemm(
                        q_shared[chunk % 2, :, :],
                        state_shared,
                        output_fragment,
                        clear_accum=True,
                    )
                    for row, col in T.Parallel(CHUNK_SIZE, CHUNK_SIZE):
                        scores_fragment[row, col] *= SCALE * gate_fragment[row, col]
                    T.copy(scores_fragment, scores_shared)
                    for row, dim_v in T.Parallel(CHUNK_SIZE, block_dv):
                        output_fragment[row, dim_v] *= (
                            SCALE * gate_exp_shared[row]
                        )
                    T.barrier_wait(bar_value_ready, chunk % 2)
                    T.gemm(
                        scores_shared,
                        value_shared,
                        output_fragment,
                        clear_accum=False,
                    )
                    T.barrier_arrive(bar_finish)
                    T.barrier_wait(bar_finish, chunk % 2)
                    T.copy(output_fragment, output_shared)
                    T.barrier_arrive(data_free[chunk % 2])
                T.barrier_arrive(bar_output)

            else:
                T.set_max_nreg(32, 0)
                if tx < 416:
                    for chunk in T.serial(num_chunks):
                        T.barrier_wait(data_free[chunk % 2], (chunk // 2 + 1) % 2)
                        left = chunk * CHUNK_SIZE
                        right = left + CHUNK_SIZE
                        if right <= num_tokens:
                            T.tma_copy(
                                q[bb, left:right, bhg, 0:HEAD_DIM_K],
                                q_shared[chunk % 2, :, :],
                                barrier=data_ready[chunk % 2],
                            )
                            T.tma_copy(
                                k[bb, left:right, bhg, 0:HEAD_DIM_K],
                                k_shared[chunk % 2, :, :],
                                barrier=data_ready[chunk % 2],
                            )
                        else:
                            for row, dim_k in T.Parallel(CHUNK_SIZE, HEAD_DIM_K):
                                if left + row < num_tokens:
                                    q_shared[chunk % 2, row, dim_k] = q[
                                        bb, left + row, bhg, dim_k
                                    ]
                                    k_shared[chunk % 2, row, dim_k] = k[
                                        bb, left + row, bhg, dim_k
                                    ]
                                else:
                                    q_shared[chunk % 2, row, dim_k] = 0
                                    k_shared[chunk % 2, row, dim_k] = 0
                        T.barrier_arrive(data_ready[chunk % 2])

                elif tx < 448:
                    for chunk in T.serial(num_chunks):
                        T.barrier_wait(data_free[chunk % 2], (chunk // 2 + 1) % 2)
                        left = chunk * CHUNK_SIZE
                        right = left + CHUNK_SIZE
                        if right <= num_tokens:
                            T.tma_copy(
                                v[bb, left:right, bh, dv_start:dv_end],
                                v_shared[chunk % 2, :, :],
                                barrier=data_ready[chunk % 2],
                            )
                            for row in T.Parallel(CHUNK_SIZE):
                                b_shared[chunk % 2, row] = beta[bb, left + row, bh]
                        else:
                            for row, dim_v in T.Parallel(CHUNK_SIZE, block_dv):
                                if left + row < num_tokens:
                                    v_shared[chunk % 2, row, dim_v] = v[
                                        bb, left + row, bh, dv_start + dim_v
                                    ]
                                else:
                                    v_shared[chunk % 2, row, dim_v] = 0
                            for row in T.Parallel(CHUNK_SIZE):
                                if left + row < num_tokens:
                                    b_shared[chunk % 2, row] = beta[
                                        bb, left + row, bh
                                    ]
                                else:
                                    b_shared[chunk % 2, row] = 0
                        T.barrier_arrive(data_ready[chunk % 2])

                elif tx < 480:
                    for chunk in T.serial(num_chunks):
                        T.barrier_wait(data_free[chunk % 2], (chunk // 2 + 1) % 2)
                        left = chunk * CHUNK_SIZE
                        right = left + CHUNK_SIZE
                        if right <= num_tokens:
                            T.tma_copy(
                                a[bb, left:right, bh, 0:CHUNK_SIZE],
                                a_shared[chunk % 2, :, :],
                                barrier=data_ready[chunk % 2],
                            )
                            for row in T.Parallel(CHUNK_SIZE):
                                g_shared[chunk % 2, row] = g[bb, left + row, bh]
                        else:
                            for row, col in T.Parallel(CHUNK_SIZE, CHUNK_SIZE):
                                if left + row < num_tokens:
                                    a_shared[chunk % 2, row, col] = a[
                                        bb, left + row, bh, col
                                    ]
                                else:
                                    a_shared[chunk % 2, row, col] = 0
                            for row in T.Parallel(CHUNK_SIZE):
                                if left + row < num_tokens:
                                    g_shared[chunk % 2, row] = g[bb, left + row, bh]
                                else:
                                    g_shared[chunk % 2, row] = g[
                                        bb, num_tokens - 1, bh
                                    ]
                        T.barrier_arrive(data_ready[chunk % 2])

                else:
                    for chunk in T.serial(num_full_chunks):
                        right = chunk * CHUNK_SIZE
                        left = right - CHUNK_SIZE
                        T.barrier_arrive(bar_start)
                        T.barrier_wait(bar_start, chunk % 2)
                        if chunk > 0:
                            T.copy(
                                output_shared,
                                output[bb, left:right, bh, dv_start:dv_end],
                            )
                        T.barrier_arrive(bar_finish)
                        T.barrier_wait(bar_state, chunk % 2)

                    if num_full_chunks < num_chunks:
                        right = num_full_chunks * CHUNK_SIZE
                        left = right - CHUNK_SIZE
                        T.barrier_arrive(bar_start)
                        T.barrier_wait(bar_start, num_full_chunks % 2)
                        if num_full_chunks > 0:
                            T.copy(
                                output_shared,
                                output[bb, left:right, bh, dv_start:dv_end],
                            )
                        T.barrier_arrive(bar_finish)
                        T.barrier_wait(bar_state, num_full_chunks % 2)

                    last_left = (num_chunks - 1) * CHUNK_SIZE
                    T.barrier_wait(bar_output, 0)
                    for row, dim_v in T.Parallel(CHUNK_SIZE, block_dv):
                        if last_left + row < num_tokens:
                            output[bb, last_left + row, bh, dv_start + dim_v] = (
                                output_shared[row, dim_v]
                            )

    return kernel


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    }
)
def _tilelang_fused_forward(
    qk_dtype,
    gate_dtype,
    state_dtype,
    accum_dtype,
    use_initial_state,
):
    """Compute residuals, outputs, and recurrent states in one kernel."""

    batch_size = T.dynamic("batch_size")
    num_tokens = T.dynamic("num_tokens")
    H = T.dynamic("num_heads_v")
    Hg = T.dynamic("num_heads_qk")
    k_shape = (batch_size, num_tokens, Hg, HEAD_DIM_K)
    value_shape = (batch_size, num_tokens, H, HEAD_DIM_V)
    gate_shape = (batch_size, num_tokens, H)
    a_shape = (batch_size, num_tokens, H, CHUNK_SIZE)
    state_shape = (batch_size, H, HEAD_DIM_K, HEAD_DIM_V)

    @T.prim_func
    def kernel(
        q: T.Tensor(k_shape, dtype=qk_dtype),
        k: T.Tensor(k_shape, dtype=qk_dtype),
        v: T.Tensor(value_shape, dtype=qk_dtype),
        g: T.Tensor(gate_shape, dtype=gate_dtype),
        beta: T.Tensor(gate_shape, dtype=gate_dtype),
        a: T.Tensor(a_shape, dtype=qk_dtype),
        initial_state: T.Tensor(state_shape, dtype=state_dtype),
        output: T.Tensor(value_shape, dtype=qk_dtype),
        final_state: T.Tensor(state_shape, dtype=state_dtype),
        num_chunks: T.int32,
    ):
        with T.Kernel(
            HEAD_DIM_V // BLOCK_DIM,
            batch_size * H,
            threads=512,
        ) as (bv, bbh):
            bb = bbh // H
            bh = bbh % H
            bhg = bh // (H // Hg)

            state_shared = T.alloc_shared((HEAD_DIM_K, BLOCK_DIM), dtype=qk_dtype)
            state_fragment = T.alloc_fragment((HEAD_DIM_K, BLOCK_DIM), dtype=accum_dtype)

            a_shared = T.alloc_shared((CHUNK_SIZE, CHUNK_SIZE), dtype=qk_dtype)
            residual_shared = T.alloc_shared((CHUNK_SIZE, BLOCK_DIM), dtype=qk_dtype)
            scores_fragment = T.alloc_fragment(
                (CHUNK_SIZE, CHUNK_SIZE), dtype=accum_dtype
            )
            value_fragment = T.alloc_fragment((CHUNK_SIZE, BLOCK_DIM), dtype=accum_dtype)
            output_fragment = T.alloc_fragment((CHUNK_SIZE, BLOCK_DIM), dtype=accum_dtype)
            k_shared = T.alloc_shared((CHUNK_SIZE, HEAD_DIM_K), dtype=qk_dtype)
            q_shared = T.alloc_shared((CHUNK_SIZE, HEAD_DIM_K), dtype=qk_dtype)
            gate_shared = T.alloc_shared((CHUNK_SIZE,), dtype=gate_dtype, scope="shared")
            gate_exp_shared = T.alloc_shared((CHUNK_SIZE,), dtype=gate_dtype, scope="shared")
            decay_shared = T.alloc_shared((CHUNK_SIZE,), dtype=gate_dtype, scope="shared")
            beta_shared = T.alloc_shared((CHUNK_SIZE,), dtype=gate_dtype, scope="shared")
            gate_last = T.alloc_var(T.float32)

            T.annotate_layout(
                {
                    residual_shared: tilelang.layout.make_swizzled_layout(residual_shared),
                    q_shared: tilelang.layout.make_swizzled_layout(q_shared),
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

                # The old implementation formed W=A@(beta*exp(g)*K) and
                # U=A@(beta*V) in a separate kernel, then evaluated U-W@S.
                # Reassociate the same expression around the prediction
                # residual instead:
                #   V_new = A @ (beta * (V - exp(g) * K @ S)).
                for row, dim_k in T.Parallel(CHUNK_SIZE, HEAD_DIM_K):
                    if left + row < num_tokens:
                        k_shared[row, dim_k] = k[bb, left + row, bhg, dim_k]
                        q_shared[row, dim_k] = q[bb, left + row, bhg, dim_k]
                    else:
                        k_shared[row, dim_k] = 0
                        q_shared[row, dim_k] = 0

                T.gemm(k_shared, state_shared, value_fragment, clear_accum=True)

                for row in T.Parallel(CHUNK_SIZE):
                    if left + row < num_tokens:
                        gate_shared[row] = g[bb, left + row, bh]
                        beta_shared[row] = beta[bb, left + row, bh]
                    else:
                        gate_shared[row] = 0
                        beta_shared[row] = 0

                # Read g_cumsum from HBM once, then derive gamma from its
                # shared-memory copy for all downstream consumers.
                for row in T.Parallel(CHUNK_SIZE):
                    gate_exp_shared[row] = T.exp2(gate_shared[row] * LOG2E)

                for row, dim_v in T.Parallel(CHUNK_SIZE, BLOCK_DIM):
                    if left + row < num_tokens:
                        residual_shared[row, dim_v] = v[
                            bb, left + row, bh, bv * BLOCK_DIM + dim_v
                        ]
                    else:
                        residual_shared[row, dim_v] = 0
                for row, dim_v in T.Parallel(CHUNK_SIZE, BLOCK_DIM):
                    value_fragment[row, dim_v] = beta_shared[row] * (
                        residual_shared[row, dim_v]
                        - gate_exp_shared[row] * value_fragment[row, dim_v]
                    )
                T.copy(value_fragment, residual_shared)

                for row, col in T.Parallel(CHUNK_SIZE, CHUNK_SIZE):
                    if left + row < num_tokens:
                        a_shared[row, col] = a[bb, left + row, bh, col]
                    else:
                        a_shared[row, col] = 0
                T.gemm(a_shared, residual_shared, value_fragment, clear_accum=True)

                # Keep W_hat in the former residual buffer. It is consumed by
                # both the chunk-local output and the state update.
                T.copy(value_fragment, residual_shared)

                T.gemm(
                    q_shared,
                    k_shared,
                    scores_fragment,
                    transpose_B=True,
                    clear_accum=True,
                )
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

                # A is dead after A @ residual, so its 64x64 shared buffer
                # can hold Gamma Lower(QK^T) Gamma^-1.
                T.copy(scores_fragment, a_shared)

                # O_[c] must use the entering state S_[c]. Delay this GEMM
                # until the score fragment is dead so output_fragment does
                # not stay live across the residual and QK computations.
                T.gemm(
                    q_shared,
                    state_shared,
                    output_fragment,
                    clear_accum=True,
                )
                for row, dim_v in T.Parallel(CHUNK_SIZE, BLOCK_DIM):
                    if left + row < num_tokens:
                        output_fragment[row, dim_v] *= gate_exp_shared[row]
                    else:
                        output_fragment[row, dim_v] = 0

                T.gemm(a_shared, residual_shared, output_fragment)
                for row, dim_v in T.Parallel(CHUNK_SIZE, BLOCK_DIM):
                    output_fragment[row, dim_v] *= SCALE

                if right <= num_tokens:
                    gate_last = g[bb, right - 1, bh]
                else:
                    gate_last = g[bb, num_tokens - 1, bh]

                # The end-of-chunk decay is row-wise. Compute its exponent
                # once per token instead of once per value element.
                for row in T.Parallel(CHUNK_SIZE):
                    if left + row < num_tokens:
                        decay_shared[row] = T.exp2(
                            (gate_last - gate_shared[row]) * LOG2E
                        )
                    else:
                        decay_shared[row] = 0

                for row, dim_v in T.Parallel(CHUNK_SIZE, BLOCK_DIM):
                    value_fragment[row, dim_v] *= decay_shared[row]

                gate_last = T.exp2(gate_last * LOG2E)
                for dim_k, dim_v in T.Parallel(HEAD_DIM_K, BLOCK_DIM):
                    state_fragment[dim_k, dim_v] *= gate_last

                T.copy(value_fragment, residual_shared)
                T.gemm(k_shared, residual_shared, state_fragment, transpose_A=True)
                T.copy(state_fragment, state_shared)

                # W_hat is dead after the state update. Reuse its shared
                # buffer to convert/store the BF16 output.
                T.copy(output_fragment, residual_shared)
                for row, dim_v in T.Parallel(CHUNK_SIZE, BLOCK_DIM):
                    if left + row < num_tokens:
                        output[
                            bb, left + row, bh, bv * BLOCK_DIM + dim_v
                        ] = residual_shared[row, dim_v]

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
    """Run GDN prefill using one persistent fused TileLang kernel."""

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
    # Chunk intermediates remain inside the persistent kernel. Only output
    # and the final FP32 state are materialized.
    output = torch.empty_like(v)
    final_state = torch.empty(
        (batch_size, num_heads_v, HEAD_DIM_K, HEAD_DIM_V),
        dtype=torch.float32,
        device=q.device,
    )

    use_initial_state = initial_state is not None
    # A four-CTA grid severely underfills the 14-SM grading partition. Split
    # DV once in that case; larger grids retain the full tile because the
    # duplicated Q/K/A traffic outweighs the extra CTA-level parallelism.
    grid_size = batch_size * num_heads_v
    block_dv = 64 if grid_size <= 4 else 128
    fused_forward = _tilelang_warp_specialized_forward(
        qk_dtype=q.dtype,
        gate_dtype=g_cumsum.dtype,
        state_dtype="float32",
        accum_dtype="float32",
        use_initial_state=use_initial_state,
        block_dv=block_dv,
    )
    # In the zero-state specialization the input is dead. Reusing the output
    # buffer avoids allocating a dummy 128x128 state tensor.
    state_source = initial_state if initial_state is not None else final_state
    fused_forward(
        q,
        k,
        v,
        g_cumsum,
        beta,
        A,
        state_source,
        output,
        final_state,
        num_chunks,
    )

    return output, final_state
