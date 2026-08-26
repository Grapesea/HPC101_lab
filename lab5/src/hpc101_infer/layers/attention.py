"""Gemma 4 的全局与滑动窗口 attention 参考实现。"""

from __future__ import annotations

import torch
from torch import nn

from hpc101_infer.layers.linear import BF16LinearFactory, LinearFactory
from hpc101_infer.layers.norm import RMSNorm
from hpc101_infer.layers.rotary import RotaryEmbedding
from hpc101_infer.models.config import RotaryConfig
from hpc101_infer.runtime.kv_cache import KVCache


def make_causal_mask(
    sequence_length: int,
    layer_type: str,
    sliding_window: int,
    device: torch.device,
) -> torch.Tensor:
    """构造 ``[1, 1, query_length, key_length]`` 的加性 causal mask。"""
    query = torch.arange(sequence_length, device=device).unsqueeze(1)
    key = torch.arange(sequence_length, device=device).unsqueeze(0)
    allowed = key <= query
    if layer_type == "sliding_attention":
        allowed &= key > query - sliding_window
    mask = torch.zeros(
        (sequence_length, sequence_length), device=device, dtype=torch.float32
    )
    return mask.masked_fill(~allowed, float("-inf")).unsqueeze(0).unsqueeze(0)


def make_attention_mask(
    positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    key_length: int,
    layer_type: str,
    dtype: torch.dtype,
    sliding_window: int = -1,
    key_position_offset: int = 0,
    key_positions: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """同时屏蔽未来 token、padding token 和滑动窗口外的历史 token。"""
    if key_positions is None:
        key_positions = torch.arange(
            key_position_offset,
            key_position_offset + key_length,
            device=positions.device,
        ).unsqueeze(0)
    query_valid = positions < sequence_lengths[:, None]
    allowed = key_positions[:, None, :] <= positions[:, :, None]
    allowed &= key_positions[:, None, :] < sequence_lengths[:, None, None]
    if layer_type == "sliding_attention":
        allowed &= key_positions[:, None, :] > (
            positions[:, :, None] - sliding_window
        )
    allowed &= query_valid[:, :, None]
    # 全 padding 的 query 若整行都是 -inf，softmax 会产生 NaN。这里临时
    # 放行一个 key，随后再通过 query_valid 将该 query 的输出清零。
    safe_allowed = allowed.clone()
    safe_allowed[..., 0] |= ~query_valid
    mask = torch.zeros(allowed.shape, device=positions.device, dtype=dtype)
    mask.masked_fill_(~safe_allowed, float("-inf"))
    return mask.unsqueeze(1), query_valid


def repeat_kv(hidden_states: torch.Tensor, repeats: int) -> torch.Tensor:
    """将较少的 KV heads 逻辑扩展到 query heads，实现 grouped-query attention。"""
    if repeats == 1:
        return hidden_states
    batch, heads, seq_len, head_dim = hidden_states.shape
    expanded = hidden_states[:, :, None].expand(
        batch, heads, repeats, seq_len, head_dim
    )
    return expanded.reshape(batch, heads * repeats, seq_len, head_dim)


def chunked_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    layer_type: str,
    *,
    sliding_window: int = -1,
    key_position_offset: int = 0,
    key_positions: torch.Tensor | None = None,
    query_chunk_size: int = 512,
) -> torch.Tensor:
    outputs = []
    key_transposed = key.transpose(2, 3)
    for start in range(0, query.shape[2], query_chunk_size):
        end = min(start + query_chunk_size, query.shape[2])
        scores = torch.matmul(query[:, :, start:end], key_transposed)
        mask, query_valid = make_attention_mask(
            positions[:, start:end],
            sequence_lengths,
            key.shape[2],
            layer_type,
            scores.dtype,
            sliding_window,
            key_position_offset,
            key_positions,
        )
        scores.add_(mask)
        probability = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
        chunk = torch.matmul(probability, value)
        chunk.mul_(query_valid[:, None, :, None])
        outputs.append(chunk)
    return torch.cat(outputs, dim=2)


class AttentionLayer(nn.Module):
    def __init__(
        self,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        hidden_size: int,
        rotary_config: RotaryConfig,
        rms_norm_eps: float,
        max_position_embeddings: int,
        attention_k_eq_v: bool = False,
        linear_factory: LinearFactory | None = None,
        module_prefix: str = "self_attn",
    ):
        super().__init__()
        factory = linear_factory or BF16LinearFactory()
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads
        self.num_kv_groups = num_qo_heads // self.num_kv_heads
        self.value_is_key = attention_k_eq_v
        self.q_proj = factory.create(
            f"{module_prefix}.q_proj", hidden_size, num_qo_heads * self.head_dim, False
        )
        self.k_proj = factory.create(
            f"{module_prefix}.k_proj",
            hidden_size,
            self.num_kv_heads * self.head_dim,
            False,
        )
        self.v_proj = (
            None
            if attention_k_eq_v
            else factory.create(
                f"{module_prefix}.v_proj",
                hidden_size,
                self.num_kv_heads * self.head_dim,
                False,
            )
        )
        self.o_proj = factory.create(
            f"{module_prefix}.o_proj",
            num_qo_heads * self.head_dim,
            hidden_size,
            False,
        )
        self.q_norm = RMSNorm(self.head_dim, rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, rms_norm_eps)
        self.v_norm = RMSNorm(self.head_dim, rms_norm_eps, with_scale=False)

        self.rotary = RotaryEmbedding(
            rotary_config,
            head_dim,
            max_position_embeddings,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        sequence_lengths: torch.Tensor,
        max_seq_len: int,
        layer_id: int,
        kv_cache: KVCache | None = None,
    ) -> torch.Tensor:
        batch, seq_len, _ = hidden_states.shape
        # Projection 后先保持 [batch, sequence, heads, head_dim]，应用 RoPE
        # 后再转为 attention 计算使用的 [batch, heads, sequence, head_dim]。
        query = self.q_proj(hidden_states).view(batch, seq_len, -1, self.head_dim)
        key = self.k_proj(hidden_states).view(
            batch, seq_len, self.num_kv_heads, self.head_dim
        )
        value = (
            key
            if self.value_is_key
            else self.v_proj(hidden_states).view(
                batch, seq_len, self.num_kv_heads, self.head_dim
            )
        )
        query = self.q_norm(query)
        key = self.k_norm(key)
        query, key = self.rotary(position_ids, query, key)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = self.v_norm(value).transpose(1, 2)
        if kv_cache is not None:
            kv_cache.write(layer_id, position_ids, key, value, sequence_lengths)
            cached = kv_cache.view(layer_id, max_seq_len)
            key, value = cached.key, cached.value
        key = repeat_kv(key, self.num_kv_groups)
        value = repeat_kv(value, self.num_kv_groups)
        output = chunked_attention(
            query, key, value, position_ids, sequence_lengths, "full_attention"
        ).transpose(1, 2).reshape(batch, seq_len, -1)
        return self.o_proj(output)


class SlidingAttentionLayer(nn.Module):
    def __init__(
        self,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        hidden_size: int,
        rotary_config: RotaryConfig,
        rms_norm_eps: float,
        max_position_embeddings: int,
        sliding_window: int,
        linear_factory: LinearFactory | None = None,
        module_prefix: str = "self_attn",
    ):
        super().__init__()
        factory = linear_factory or BF16LinearFactory()
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads
        self.num_kv_groups = num_qo_heads // self.num_kv_heads
        self.sliding_window = sliding_window
        self.q_proj = factory.create(
            f"{module_prefix}.q_proj", hidden_size, num_qo_heads * self.head_dim, False
        )
        self.k_proj = factory.create(
            f"{module_prefix}.k_proj",
            hidden_size,
            self.num_kv_heads * self.head_dim,
            False,
        )
        self.v_proj = factory.create(
            f"{module_prefix}.v_proj",
            hidden_size,
            self.num_kv_heads * self.head_dim,
            False,
        )
        self.o_proj = factory.create(
            f"{module_prefix}.o_proj",
            num_qo_heads * self.head_dim,
            hidden_size,
            False,
        )
        self.q_norm = RMSNorm(self.head_dim, rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, rms_norm_eps)
        self.v_norm = RMSNorm(self.head_dim, rms_norm_eps, with_scale=False)

        self.rotary = RotaryEmbedding(
            rotary_config,
            head_dim,
            max_position_embeddings,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        sequence_lengths: torch.Tensor,
        max_seq_len: int,
        layer_id: int,
        kv_cache: KVCache | None = None,
    ) -> torch.Tensor:
        batch, seq_len, _ = hidden_states.shape
        # Projection 后先保持 [batch, sequence, heads, head_dim]，应用 RoPE
        # 后再转为 attention 计算使用的 [batch, heads, sequence, head_dim]。
        query = self.q_proj(hidden_states).view(batch, seq_len, -1, self.head_dim)
        key = self.k_proj(hidden_states).view(
            batch, seq_len, self.num_kv_heads, self.head_dim
        )
        value = self.v_proj(hidden_states).view(
            batch, seq_len, self.num_kv_heads, self.head_dim
        )
        query = self.q_norm(query)
        key = self.k_norm(key)
        query, key = self.rotary(position_ids, query, key)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = self.v_norm(value).transpose(1, 2)
        key_positions = None
        if kv_cache is not None:
            kv_cache.write(layer_id, position_ids, key, value, sequence_lengths)
            # A generation prefill starts at absolute position zero and already
            # has all prompt K/V in this forward, so using the current tensors
            # avoids losing early queries when the prompt exceeds the ring.
            # Chunked quality scoring also has seq_len > 1, but chunks after the
            # first start at a non-zero position and must include cached history.
            if seq_len == 1 or position_ids[0, 0].item() != 0:
                cached = kv_cache.view(layer_id, max_seq_len, sequence_lengths)
                key, value = cached.key, cached.value
                key_positions = cached.key_positions
        key = repeat_kv(key, self.num_kv_groups)
        value = repeat_kv(value, self.num_kv_groups)
        output = chunked_attention(
            query,
            key,
            value,
            position_ids,
            sequence_lengths,
            "sliding_attention",
            sliding_window=self.sliding_window,
            key_positions=key_positions,
        ).transpose(1, 2).reshape(batch, seq_len, -1)
        return self.o_proj(output)
