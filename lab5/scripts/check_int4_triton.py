"""Fast CUDA correctness check for the fused INT4 linear backend."""

from __future__ import annotations

from _bootstrap import add_local_src_to_path

add_local_src_to_path()

import torch

from hpc101_infer.layers.int4_triton import int4_matmul
from hpc101_infer.quantization.methods.rtn import quantize_weight_rtn
from hpc101_infer.quantization.packing import dequantize_weight


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(0)
    device = torch.device("cuda")
    weight = torch.randn(257, 256, device=device, dtype=torch.bfloat16) * 0.05
    quantized = quantize_weight_rtn(weight, 128, scale_dtype=torch.float16)
    reference_weight = dequantize_weight(quantized, dtype=torch.bfloat16)

    for rows in (1, 17):
        inputs = torch.randn(rows, 256, device=device, dtype=torch.bfloat16)
        expected = torch.nn.functional.linear(inputs, reference_weight)
        actual = int4_matmul(
            inputs,
            quantized.qweight,
            quantized.scales,
            quantized.zeros,
            group_size=quantized.group_size,
            out_features=257,
        )
        torch.cuda.synchronize()
        difference = (actual.float() - expected.float()).abs()
        torch.testing.assert_close(actual, expected, atol=0.08, rtol=0.02)
        print(
            f"int4_triton rows={rows}: max_abs={difference.max().item():.6f}, "
            f"mean_abs={difference.mean().item():.6f}"
        )


if __name__ == "__main__":
    main()
