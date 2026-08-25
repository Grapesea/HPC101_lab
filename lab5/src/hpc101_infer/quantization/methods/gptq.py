from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import nn

from hpc101_infer.quantization.packing import pack_int4
from hpc101_infer.quantization.types import (
    LayerContext,
    LayerQuantizationResult,
    QuantizedWeight,
    SCALE_DTYPES,
)


@dataclass(frozen=True)
class GPTQOptions:
    block_size: int
    damp_percent: float


@dataclass(frozen=True)
class GPTQModuleState:
    activations: torch.Tensor
    activation_tokens: int


@dataclass(frozen=True)
class GPTQLayerState:
    modules: Mapping[str, GPTQModuleState]
    options: GPTQOptions


def _calibration_option(
    calibration: Mapping[str, Any],
    name: str,
    default: int | float,
) -> int | float:
    gptq = calibration.get("gptq", calibration)
    if not isinstance(gptq, Mapping):
        raise TypeError("config.calibration['gptq'] must be a mapping")
    return gptq.get(name, default)


def _parse_gptq_options(calibration: Mapping[str, Any]) -> GPTQOptions:
    block_size = _calibration_option(calibration, "block_size", 128)
    damp_percent = _calibration_option(calibration, "damp_percent", 0.01)

    if (
        not isinstance(block_size, int)
        or isinstance(block_size, bool)
        or block_size <= 0
    ):
        raise ValueError("GPTQ block_size must be a positive integer")
    if (
        not isinstance(damp_percent, (int, float))
        or isinstance(damp_percent, bool)
        or not 0.0 < float(damp_percent) < 1.0
    ):
        raise ValueError("GPTQ damp_percent must be in (0, 1)")

    gptq = calibration.get("gptq", calibration)
    if isinstance(gptq, Mapping) and gptq.get("desc_act", False):
        raise ValueError("GPTQ desc_act is not supported by this checkpoint format")
    return GPTQOptions(
        block_size=block_size,
        damp_percent=float(damp_percent),
    )


def quantize_weight_gptq(
    weight: torch.Tensor,
    activations: torch.Tensor,
    group_size: int,
    *,
    block_size: int = 128,
    damp_percent: float = 0.01,
    symmetric: bool = True,
    scale_dtype: torch.dtype = torch.float16,
) -> tuple[QuantizedWeight, dict[str, float | int]]:
    """
    使用 GPTQ 算法将权重量化为 INT4。

    参数：
        weight: 待量化的权重张量，形状为 (out_features, in_features)。
        activations: 校准数据集的输入激活，形状为 (calibration_tokens, in_features)。
        group_size: 量化粒度，即每个 group 中的列数。
        block_size: 分块计算时每个 block 中的列数。
        damp_percent: 阻尼比例，用于改善 Hessian 的数值稳定性。
        symmetric: 是否使用对称量化。
        scale_dtype: 缩放因子的 dtype。

    返回：
        quantized_weight: 量化后的权重对象，具体详见 QuantizedWeight 类的定义。
        metadatas: 量化过程中的统计信息，不影响评测，用于分析和调试。
    """

    if weight.ndim != 2 or not weight.is_floating_point():
        raise ValueError("weight must be a floating-point matrix")
    if activations.ndim != 2 or not activations.is_floating_point():
        raise ValueError("activations must be a floating-point matrix")
    if activations.shape[1] != weight.shape[1]:
        raise ValueError("activation width does not match weight input size")
    if weight.shape[0] == 0 or weight.shape[1] == 0:
        raise ValueError("weight dimensions must be non-zero")
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if not 0.0 < damp_percent < 1.0:
        raise ValueError("damp_percent must be in (0, 1)")
    if scale_dtype not in SCALE_DTYPES.values():
        raise ValueError("unsupported scale dtype")
    if activations.shape[0] == 0:
        raise ValueError("activations must not be empty")
    if not torch.isfinite(weight).all():
        raise ValueError("weight must contain only finite values")
    if not torch.isfinite(activations).all():
        raise ValueError("activations must contain only finite values")

    # GPTQ 的更新和分解使用 FP32；输入权重本身不被修改。
    work = weight.detach().float().clone()
    # ActivationCapture 为节省显存会把校准激活保存在 CPU，因此在这里
    # 显式搬到当前权重所在设备。
    inputs = activations.detach().to(device=weight.device, dtype=torch.float32)
    out_features, in_features = work.shape
    activation_tokens = inputs.shape[0]

    # 对层重构误差 ||X(W-Q)^T||_F^2 求 Hessian。常数 2/N 不改变
    # 最优更新，但保留它可使 predicted_loss 的含义与推导一致。
    hessian = inputs.T @ inputs
    hessian.mul_(2.0 / activation_tokens)
    diagonal = torch.diagonal(hessian)
    dead = diagonal == 0
    dead_columns = int(dead.sum().item())
    if dead_columns:
        # 校准集中恒为零的输入通道不影响层输出；将其权重清零，并给
        # Hessian 设置单位对角元，使后续分解保持正定。
        hessian[dead, dead] = 1.0
        work[:, dead] = 0.0

    damp = damp_percent * torch.diagonal(hessian).mean()
    indices = torch.arange(in_features, device=hessian.device)
    hessian[indices, indices] += damp

    # U^T U = H^{-1}。用两次 Cholesky 而不是显式通用矩阵求逆，
    # 与 GPTQ 的逐列误差传播公式相匹配，也有更好的数值稳定性。
    try:
        hessian_cholesky = torch.linalg.cholesky(hessian)
        inverse_hessian = torch.cholesky_inverse(hessian_cholesky)
        inverse_factor = torch.linalg.cholesky(inverse_hessian, upper=True)
    except RuntimeError as error:
        raise RuntimeError(
            "Hessian Cholesky factorization failed; try increasing damp_percent"
        ) from error
    del inputs, hessian, hessian_cholesky, inverse_hessian

    padded_in_features = (
        (in_features + group_size - 1) // group_size * group_size
    )
    num_groups = padded_in_features // group_size
    encoded = torch.empty(
        (out_features, padded_in_features), dtype=torch.uint8, device=work.device
    )
    scales = torch.empty(
        (out_features, num_groups), dtype=torch.float32, device=work.device
    )
    zeros = (
        None
        if symmetric
        else torch.empty(
            (out_features, num_groups), dtype=torch.uint8, device=work.device
        )
    )

    predicted_loss = torch.zeros((), dtype=torch.float64, device=work.device)
    eps = torch.finfo(torch.float32).eps

    # 每个 block 内逐列量化；block 完成后，把累计误差一次传播到右侧，
    # 避免每一列都对完整剩余矩阵执行一次更新。
    for block_start in range(0, in_features, block_size):
        block_end = min(block_start + block_size, in_features)
        block_width = block_end - block_start
        block = work[:, block_start:block_end].clone()
        block_errors = torch.zeros_like(block)

        for local_index in range(block_width):
            column_index = block_start + local_index
            group_index = column_index // group_size
            group_start = group_index * group_size

            if column_index == group_start:
                group_end = min(group_start + group_size, in_features)
                # 当前 block 已经接受了前面各列的局部补偿，而 work 中只含
                # 已完成 blocks 的补偿。group 跨 block 时需拼接两者。
                local_group_end = min(group_end, block_end)
                group = block[:, local_index : local_group_end - block_start]
                if group_end > block_end:
                    group = torch.cat((group, work[:, block_end:group_end]), dim=1)
                if group_end - group_start < group_size:
                    group = torch.nn.functional.pad(
                        group, (0, group_size - (group_end - group_start))
                    )
                if symmetric:
                    group_scale = group.abs().amax(dim=1).div(7.0).clamp_min(eps)
                    group_zero = None
                else:
                    zero_value = torch.zeros_like(group[:, 0])
                    minimum = torch.minimum(group.amin(dim=1), zero_value)
                    maximum = torch.maximum(group.amax(dim=1), zero_value)
                    group_scale = ((maximum - minimum) / 15.0).clamp_min(eps)
                    group_zero = torch.round(-minimum / group_scale).clamp(0, 15)
                scales[:, group_index] = group_scale
                if zeros is not None and group_zero is not None:
                    zeros[:, group_index] = group_zero.to(torch.uint8)

            column = block[:, local_index]
            scale = scales[:, group_index]
            if symmetric:
                code = torch.round(column / scale).clamp(-8, 7) + 8
                reconstructed = (code - 8) * scale
            else:
                assert zeros is not None
                zero_point = zeros[:, group_index].float()
                code = torch.round(column / scale + zero_point).clamp(0, 15)
                reconstructed = (code - zero_point) * scale
            encoded[:, column_index] = code.to(torch.uint8)

            divisor = inverse_factor[column_index, column_index]
            error = (column - reconstructed) / divisor
            predicted_loss += (
                (column - reconstructed).double().square() / divisor.double().square()
            ).sum() * 0.5
            block_errors[:, local_index] = error

            # 先在当前 block 内传播，后续列量化时便看到修正后的权重。
            block[:, local_index:block_width] -= error.unsqueeze(1) * (
                inverse_factor[
                    column_index, column_index:block_end
                ].unsqueeze(0)
            )

        # 将该 block 的所有量化误差集中传播到尚未处理的 blocks。
        if block_end < in_features:
            work[:, block_end:] -= block_errors @ inverse_factor[
                block_start:block_end, block_end:
            ]

    # padding 代表不存在的输入，编码成反量化后严格为零的 code。
    if padded_in_features > in_features:
        tail_groups = torch.arange(
            in_features, padded_in_features, device=work.device
        ) // group_size
        if symmetric:
            encoded[:, in_features:] = 8
        else:
            assert zeros is not None
            encoded[:, in_features:] = zeros[:, tail_groups]

    quantized = QuantizedWeight(
        qweight=pack_int4(encoded),
        scales=scales.to(scale_dtype),
        zeros=zeros,
        original_shape=(out_features, in_features),
        padded_shape=(out_features, padded_in_features),
        bits=4,
        group_size=group_size,
        symmetric=symmetric,
        packing="uint8_little_nibble",
    )
    metadata: dict[str, float | int] = {
        "activation_tokens": activation_tokens,
        "block_size": block_size,
        "damp_percent": damp_percent,
        "dead_columns": dead_columns,
        "predicted_loss": float(predicted_loss.item()),
    }
    return quantized, metadata


class GPTQQuantizationMethod:
    name = "gptq"
    version = "1"

    def calibrate_layer(self, context: LayerContext) -> GPTQLayerState:
        if context.activations is None:
            raise ValueError("GPTQ requires calibration activations")

        options = _parse_gptq_options(context.config.calibration or {})
        modules = dict(context.layer.named_modules())
        states: dict[str, GPTQModuleState] = {}
        for name in context.target_modules:
            module = modules.get(name)
            if not isinstance(module, nn.Linear):
                raise TypeError(f"target is not a Linear module: {name}")
            if name not in context.activations:
                raise ValueError(f"missing calibration activations for {name}")

            activations = context.activations[name]
            if activations.ndim != 2 or activations.shape[1] != module.in_features:
                raise ValueError(
                    f"invalid calibration activation shape for {name}: "
                    f"expected [tokens, {module.in_features}], "
                    f"got {tuple(activations.shape)}"
                )
            if activations.shape[0] == 0:
                raise ValueError(f"calibration activations for {name} are empty")
            if not activations.is_floating_point():
                raise TypeError(f"calibration activations for {name} must be floating")
            if not torch.isfinite(activations).all():
                raise ValueError(f"calibration activations for {name} are not finite")

            states[name] = GPTQModuleState(
                activations=activations.detach(),
                activation_tokens=activations.shape[0],
            )

        return GPTQLayerState(modules=states, options=options)

    def quantize_layer(
        self, context: LayerContext, state: GPTQLayerState
    ) -> LayerQuantizationResult:
        if not isinstance(state, GPTQLayerState):
            raise TypeError("state must be a GPTQLayerState")

        scale_dtype = SCALE_DTYPES[context.config.scale_dtype]
        modules = dict(context.layer.named_modules())
        weights: dict[str, QuantizedWeight] = {}
        module_metadata: dict[str, dict[str, float | int]] = {}

        for name in context.target_modules:
            module = modules.get(name)
            if not isinstance(module, nn.Linear):
                raise TypeError(f"target is not a Linear module: {name}")
            module_state = state.modules.get(name)
            if module_state is None:
                raise ValueError(f"missing GPTQ calibration state for {name}")

            try:
                weights[name], module_metadata[name] = quantize_weight_gptq(
                    module.weight,
                    module_state.activations,
                    context.config.group_size,
                    block_size=state.options.block_size,
                    damp_percent=state.options.damp_percent,
                    symmetric=context.config.symmetric,
                    scale_dtype=scale_dtype,
                )
            except RuntimeError as error:
                raise RuntimeError(
                    f"GPTQ failed for layer {context.layer_index} module {name} "
                    f"with {module_state.activation_tokens} calibration tokens: {error}"
                ) from error

        return LayerQuantizationResult(
            weights=weights,
            metadata={
                "gptq": {
                    "block_size": state.options.block_size,
                    "damp_percent": state.options.damp_percent,
                    "modules": module_metadata,
                }
            },
        )
