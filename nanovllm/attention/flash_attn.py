import torch
from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache

from nanovllm.attention.base import (
    DecoderAttentionBackend,
    EncoderAttentionBackend,
)


class FlashAttentionEncoderBackend(EncoderAttentionBackend):
    name = "flash_attn"

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        softmax_scale: float,
        causal: bool,
    ):
        return flash_attn_varlen_func(
            query,
            key,
            value,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=softmax_scale,
            causal=causal,
        )


class FlashAttentionDecoderBackend(DecoderAttentionBackend):
    name = "flash_attn"

    def prefill(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        block_tables: torch.Tensor | None,
        softmax_scale: float,
    ):
        if block_tables is not None:
            key = key_cache
            value = value_cache
        return flash_attn_varlen_func(
            query,
            key,
            value,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=softmax_scale,
            causal=True,
            block_table=block_tables,
        )

    def decode(
        self,
        query: torch.Tensor,
        *,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        context_lens: torch.Tensor,
        block_tables: torch.Tensor,
        softmax_scale: float,
    ):
        output = flash_attn_with_kvcache(
            query.unsqueeze(1),
            key_cache,
            value_cache,
            cache_seqlens=context_lens,
            block_table=block_tables,
            softmax_scale=softmax_scale,
            causal=True,
        )
        return output.squeeze(1)
