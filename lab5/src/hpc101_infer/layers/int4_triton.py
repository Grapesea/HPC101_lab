"""Fused group-wise INT4 dequantization and matrix multiplication."""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # Reference-only environments need not install Triton.
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _int4_matmul_kernel(
        input_ptr,
        qweight_ptr,
        scales_ptr,
        zeros_ptr,
        output_ptr,
        M,
        N,
        K,
        GROUP_SIZE: tl.constexpr,
        stride_am,
        stride_ak,
        stride_qn,
        stride_qk,
        stride_sn,
        stride_sg,
        stride_om,
        stride_on,
        SYMMETRIC: tl.constexpr,
        INPUT_BF16: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offsets_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k_start in range(0, K, BLOCK_K):
            offsets_k = k_start + tl.arange(0, BLOCK_K)
            input_values = tl.load(
                input_ptr
                + offsets_m[:, None] * stride_am
                + offsets_k[None, :] * stride_ak,
                mask=(offsets_m[:, None] < M) & (offsets_k[None, :] < K),
                other=0.0,
            )

            packed = tl.load(
                qweight_ptr
                + offsets_n[None, :] * stride_qn
                + (offsets_k[:, None] // 2) * stride_qk,
                mask=(offsets_n[None, :] < N) & (offsets_k[:, None] < K),
                other=0,
            )
            shifts = (offsets_k[:, None] & 1) * 4
            codes = (packed >> shifts) & 0x0F
            # Decode tiles are aligned to a complete quantization group.  A
            # scale belongs to an output channel and a whole K group, so load
            # it once per N lane and broadcast inside the tile instead of
            # issuing BLOCK_K identical scale loads.
            group_offset = k_start // GROUP_SIZE
            scales = tl.load(
                scales_ptr
                + offsets_n[None, :] * stride_sn
                + group_offset * stride_sg,
                mask=offsets_n[None, :] < N,
                other=0.0,
            )
            if SYMMETRIC:
                weights = (codes.to(tl.float32) - 8.0) * scales
            else:
                zero_points = tl.load(
                    zeros_ptr
                    + offsets_n[None, :] * stride_sn
                    + group_offset * stride_sg,
                    mask=offsets_n[None, :] < N,
                    other=0,
                )
                weights = (
                    codes.to(tl.float32) - zero_points.to(tl.float32)
                ) * scales

            if INPUT_BF16:
                weights = weights.to(tl.bfloat16)
            else:
                weights = weights.to(tl.float16)
            accumulator += tl.dot(input_values, weights)

        tl.store(
            output_ptr
            + offsets_m[:, None] * stride_om
            + offsets_n[None, :] * stride_on,
            accumulator,
            mask=(offsets_m[:, None] < M) & (offsets_n[None, :] < N),
        )


def int4_matmul(
    inputs: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor | None,
    *,
    group_size: int,
    out_features: int,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute ``inputs @ dequantize(qweight).T`` without materializing weights."""
    if triton is None:
        raise RuntimeError("the int4_triton backend requires Triton")
    if not inputs.is_cuda:
        raise ValueError("the int4_triton backend requires CUDA inputs")
    if inputs.dtype not in {torch.float16, torch.bfloat16}:
        raise TypeError("the int4_triton backend requires float16 or bfloat16 inputs")
    if inputs.shape[-1] > qweight.shape[-1] * 2:
        raise ValueError("input width exceeds packed weight width")

    original_shape = inputs.shape[:-1]
    matrix = inputs.reshape(-1, inputs.shape[-1]).contiguous()
    rows, inner = matrix.shape
    output = torch.empty(
        (rows, out_features), device=inputs.device, dtype=inputs.dtype
    )
    # Decode is launch/bandwidth dominated; prefill benefits from a larger M tile.
    if rows <= 4:
        block_m, block_n, block_k, num_warps = 16, 128, 64, 4
    else:
        block_m, block_n, block_k, num_warps = 32, 64, 32, 4
    dummy_zeros = qweight if zeros is None else zeros
    grid = (triton.cdiv(rows, block_m), triton.cdiv(out_features, block_n))
    _int4_matmul_kernel[grid](
        matrix,
        qweight,
        scales,
        dummy_zeros,
        output,
        rows,
        out_features,
        inner,
        group_size,
        matrix.stride(0),
        matrix.stride(1),
        qweight.stride(0),
        qweight.stride(1),
        scales.stride(0),
        scales.stride(1),
        output.stride(0),
        output.stride(1),
        SYMMETRIC=zeros is None,
        INPUT_BF16=inputs.dtype == torch.bfloat16,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=num_warps,
        num_stages=3,
    )
    if bias is not None:
        output.add_(bias)
    return output.reshape(*original_shape, out_features)
