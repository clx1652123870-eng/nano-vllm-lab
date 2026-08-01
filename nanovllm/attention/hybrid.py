import torch

from nanovllm.attention.base import EncoderAttentionBackend
from nanovllm.attention.flash_attn import FlashAttentionEncoderBackend
from nanovllm.attention.triton_attn import TritonEncoderAttentionBackend


class HybridEncoderAttentionBackend(EncoderAttentionBackend):
    """Triton for short windows and FlashAttention for long sequences."""

    name = "hybrid"

    def __init__(self, triton_max_seqlen: int = 64) -> None:
        self.triton_max_seqlen = triton_max_seqlen
        self.triton = TritonEncoderAttentionBackend()
        self.flash_attn = FlashAttentionEncoderBackend()

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
    ) -> torch.Tensor:
        backend = (
            self.triton
            if max(max_seqlen_q, max_seqlen_k) <= self.triton_max_seqlen
            else self.flash_attn
        )
        return backend.forward(
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
