"""BF16 Linear 与持久化 INT4 权重的参考 Linear 实现。"""

from __future__ import annotations

from typing import Mapping, Protocol

import torch
from torch import nn
from torch.nn import functional as F

from hpc101_infer.quantization.packing import dequantize_weight, pack_int4
from hpc101_infer.quantization.types import QuantizedModuleManifest, QuantizedWeight


class LinearFactory(Protocol):
    def create(
        self,
        module_name: str,
        in_features: int,
        out_features: int,
        bias: bool,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> nn.Module: ...


class BF16LinearFactory:
    def create(
        self,
        module_name: str,
        in_features: int,
        out_features: int,
        bias: bool,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> nn.Linear:
        del module_name
        return nn.Linear(
            in_features,
            out_features,
            bias=bias,
            device=device,
            dtype=dtype,
        )


class QuantizedLinear(nn.Module):
    """保存 packed INT4 权重，并在 forward 中临时反量化。

    ``qweight`` 使用一个 uint8 保存两个 4-bit code；``scales`` 和可选的
    ``zeros`` 按 ``[out_features, num_groups]`` 存储。输入维度会补齐到
    group size 的整数倍，但矩阵乘前仍只接收原始 ``in_features``。
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        group_size: int,
        *,
        symmetric: bool = True,
        padded_in_features: int | None = None,
        bias: bool = False,
        device: torch.device | str | None = None,
        scale_dtype: torch.dtype = torch.float16,
        backend: str = "int4_reference",
    ) -> None:
        super().__init__()
        if padded_in_features is None:
            padded_in_features = (
                (in_features + group_size - 1) // group_size * group_size
            )
        if padded_in_features < in_features or padded_in_features % group_size:
            raise ValueError("invalid padded input size")
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.symmetric = symmetric
        self.padded_in_features = padded_in_features
        if backend not in {"int4_reference", "int4_triton"}:
            raise ValueError(f"unsupported quantized linear backend: {backend}")
        self.backend = backend
        self.register_buffer(
            "qweight",
            torch.empty(
                out_features,
                padded_in_features // 2,
                dtype=torch.uint8,
                device=device,
            ),
        )
        self.register_buffer(
            "scales",
            torch.empty(
                out_features,
                padded_in_features // group_size,
                dtype=scale_dtype,
                device=device,
            ),
        )
        self.register_buffer(
            "zeros",
            (
                None
                if symmetric
                else torch.empty(
                    out_features,
                    padded_in_features // group_size,
                    dtype=torch.uint8,
                    device=device,
                )
            ),
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, device=device))
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_quantized_weight(
        cls, quantized: QuantizedWeight, bias: torch.Tensor | None = None
    ) -> "QuantizedLinear":
        out_features, in_features = quantized.original_shape
        module = cls(
            in_features,
            out_features,
            quantized.group_size,
            symmetric=quantized.symmetric,
            padded_in_features=quantized.padded_shape[1],
            bias=bias is not None,
            device=quantized.qweight.device,
            scale_dtype=quantized.scales.dtype,
        )
        module.qweight.copy_(quantized.qweight)
        module.scales.copy_(quantized.scales)
        if module.zeros is not None and quantized.zeros is not None:
            module.zeros.copy_(quantized.zeros)
        if bias is not None:
            module.bias.data.copy_(bias)
        return module

    def quantized_weight(self) -> QuantizedWeight:
        return QuantizedWeight(
            qweight=self.qweight,
            scales=self.scales,
            zeros=self.zeros,
            original_shape=(self.out_features, self.in_features),
            padded_shape=(self.out_features, self.padded_in_features),
            bits=4,
            group_size=self.group_size,
            symmetric=self.symmetric,
            packing="uint8_little_nibble",
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        rows = inputs.numel() // inputs.shape[-1]
        if self.backend == "int4_triton" and rows <= 64:
            from hpc101_infer.layers.int4_triton import int4_matmul

            return int4_matmul(
                inputs,
                self.qweight,
                self.scales,
                self.zeros,
                group_size=self.group_size,
                out_features=self.out_features,
                bias=self.bias,
            )
        # 大 M 的 prefill 能够用高效 GEMM 摊薄一次反量化；小 M decode
        # 则走上面的融合 kernel，避免反复物化完整权重。
        weight = dequantize_weight(self.quantized_weight(), dtype=inputs.dtype)
        return F.linear(inputs, weight, self.bias)


class QuantizedLinearFactory:
    def __init__(
        self,
        manifest: Mapping[str, QuantizedModuleManifest],
        *,
        scale_dtype: torch.dtype = torch.float16,
        backend: str = "int4_reference",
    ) -> None:
        if scale_dtype not in {torch.float16, torch.bfloat16}:
            raise ValueError("scale_dtype must be float16 or bfloat16")
        self.manifest = dict(manifest)
        self.scale_dtype = scale_dtype
        self.backend = backend
        self._fallback = BF16LinearFactory()

    def create(
        self,
        module_name: str,
        in_features: int,
        out_features: int,
        bias: bool,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> nn.Module:
        entry = self.manifest.get(module_name)
        if entry is None:
            return self._fallback.create(
                module_name, in_features, out_features, bias, device, dtype
            )
        if entry.original_shape != (out_features, in_features):
            raise RuntimeError(
                f"manifest shape mismatch for {module_name}: "
                f"manifest={entry.original_shape}, model={(out_features, in_features)}"
            )
        scale_dtype = (
            dtype if dtype in {torch.float16, torch.bfloat16} else self.scale_dtype
        )
        return QuantizedLinear(
            in_features,
            out_features,
            entry.group_size,
            symmetric=entry.symmetric,
            padded_in_features=entry.padded_shape[1],
            bias=bias,
            device=device,
            scale_dtype=scale_dtype,
            backend=self.backend,
        )


class QuantizedTiedEmbedding(nn.Module):
    """INT4 storage shared by token embedding lookup and the output LM head."""

    def __init__(
        self,
        qweight: torch.Tensor,
        scales: torch.Tensor,
        embedding_dim: int,
        group_size: int,
        output_dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.num_embeddings = qweight.shape[0]
        self.embedding_dim = embedding_dim
        self.group_size = group_size
        self.output_dtype = output_dtype
        self.register_buffer("qweight", qweight)
        self.register_buffer("scales", scales)

    @classmethod
    def from_float_weight(
        cls,
        weight: torch.Tensor,
        device: torch.device,
        *,
        group_size: int = 128,
        chunk_rows: int = 1024,
    ) -> "QuantizedTiedEmbedding":
        rows, width = weight.shape
        if width % group_size or width % 2:
            raise ValueError("embedding width must align to INT4 group size")
        packed = torch.empty((rows, width // 2), dtype=torch.uint8)
        scales = torch.empty((rows, width // group_size), dtype=torch.float16)
        epsilon = torch.finfo(torch.float32).eps
        for start in range(0, rows, chunk_rows):
            end = min(start + chunk_rows, rows)
            groups = weight[start:end].float().view(-1, width // group_size, group_size)
            chunk_scales = groups.abs().amax(dim=-1).div(7.0).clamp_min(epsilon)
            codes = torch.round(groups / chunk_scales.unsqueeze(-1)).clamp(-8, 7)
            codes = (codes.to(torch.int16) + 8).to(torch.uint8)
            packed[start:end].copy_(pack_int4(codes.reshape(end - start, width)))
            scales[start:end].copy_(chunk_scales.to(torch.float16))
        return cls(
            packed.to(device),
            scales.to(device),
            width,
            group_size,
            weight.dtype,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        packed = F.embedding(input_ids, self.qweight)
        low = packed & 0x0F
        high = packed >> 4
        codes = torch.stack((low, high), dim=-1).flatten(-2)
        group_scales = F.embedding(input_ids, self.scales)
        selected_scales = torch.repeat_interleave(
            group_scales, self.group_size, dim=-1
        ).to(self.output_dtype)
        return (codes.to(self.output_dtype) - 8.0) * selected_scales

    def linear(self, hidden_states: torch.Tensor) -> torch.Tensor:
        from hpc101_infer.layers.int4_triton import int4_matmul

        return int4_matmul(
            hidden_states,
            self.qweight,
            self.scales,
            None,
            group_size=self.group_size,
            out_features=self.num_embeddings,
        )
