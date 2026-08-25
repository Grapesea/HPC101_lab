from __future__ import annotations

import pytest
import torch

from hpc101_infer.quantization.methods.gptq import quantize_weight_gptq
from hpc101_infer.quantization.methods.rtn import quantize_weight_rtn
from hpc101_infer.quantization.packing import dequantize_weight, unpack_int4


@pytest.mark.parametrize("symmetric", [True, False])
@pytest.mark.parametrize("block_size", [31, 64, 128])
def test_gptq_shapes_packing_and_reconstruction(
    symmetric: bool, block_size: int
) -> None:
    generator = torch.Generator().manual_seed(7)
    weight = torch.randn(7, 130, generator=generator)
    activations = torch.randn(257, 130, generator=generator)

    quantized, metadata = quantize_weight_gptq(
        weight,
        activations,
        64,
        block_size=block_size,
        symmetric=symmetric,
    )
    reconstructed = dequantize_weight(quantized, dtype=torch.float32)

    assert reconstructed.shape == weight.shape
    assert torch.isfinite(reconstructed).all()
    assert quantized.qweight.shape == (7, 96)
    assert quantized.scales.shape == (7, 3)
    assert (quantized.zeros is None) is symmetric
    assert metadata["activation_tokens"] == 257
    assert metadata["dead_columns"] == 0
    assert metadata["predicted_loss"] >= 0

    # 补齐位置必须反量化为零。
    codes = unpack_int4(quantized.qweight, quantized.padded_shape[1])
    if symmetric:
        assert torch.equal(codes[:, 130:], torch.full_like(codes[:, 130:], 8))
    else:
        assert quantized.zeros is not None
        assert torch.equal(
            codes[:, 130:], quantized.zeros[:, -1:].expand_as(codes[:, 130:])
        )


def test_gptq_reduces_calibration_reconstruction_error() -> None:
    # 强相关输入使二阶误差补偿相对 RTN 有稳定、明显的优势。
    generator = torch.Generator().manual_seed(11)
    latent = torch.randn(512, 8, generator=generator)
    mixing = torch.randn(8, 64, generator=generator)
    activations = latent @ mixing + 0.01 * torch.randn(
        512, 64, generator=generator
    )
    weight = torch.randn(12, 64, generator=generator)

    gptq, _ = quantize_weight_gptq(weight, activations, 64, block_size=32)
    rtn = quantize_weight_rtn(weight, 64)
    gptq_weight = dequantize_weight(gptq, dtype=torch.float32)
    rtn_weight = dequantize_weight(rtn, dtype=torch.float32)
    gptq_error = (activations @ (weight - gptq_weight).T).square().mean()
    rtn_error = (activations @ (weight - rtn_weight).T).square().mean()

    assert gptq_error < rtn_error


def test_gptq_handles_dead_columns_without_mutating_inputs() -> None:
    generator = torch.Generator().manual_seed(13)
    weight = torch.randn(4, 64, generator=generator)
    activations = torch.randn(32, 64, generator=generator)
    activations[:, 9] = 0
    original_weight = weight.clone()
    original_activations = activations.clone()

    quantized, metadata = quantize_weight_gptq(weight, activations, 64)

    assert metadata["dead_columns"] == 1
    assert torch.equal(weight, original_weight)
    assert torch.equal(activations, original_activations)
    assert torch.isfinite(dequantize_weight(quantized, dtype=torch.float32)).all()


@pytest.mark.parametrize(
    ("weight", "activations", "message"),
    [
        (torch.empty(0, 64), torch.empty(2, 64), "non-zero"),
        (torch.empty(2, 0), torch.empty(2, 0), "non-zero"),
        (torch.empty(2, 64), torch.empty(0, 64), "not be empty"),
    ],
)
def test_gptq_rejects_empty_inputs(
    weight: torch.Tensor, activations: torch.Tensor, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        quantize_weight_gptq(weight, activations, 64)
