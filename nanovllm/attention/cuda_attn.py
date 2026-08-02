import torch

from nanovllm.attention.base import EncoderAttentionBackend
from nanovllm.attention.flash_attn import FlashAttentionEncoderBackend
from nanovllm.kernels import cuda_packed_attention


class CUDAFusedAttentionEncoderBackend(EncoderAttentionBackend):
    """BF16 packed-varlen online-softmax CUDA kernel for short windows."""

    name = "cuda_fused"

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
        validate_inputs(
            query,
            key,
            value,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            causal,
        )
        return cuda_packed_attention(
            query,
            key,
            value,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            softmax_scale,
            causal,
        )


class CUDAHybridEncoderAttentionBackend(EncoderAttentionBackend):
    """Custom CUDA for short windows and external FlashAttention otherwise."""

    name = "cuda_hybrid"

    def __init__(self, cuda_max_seqlen: int = 64) -> None:
        self.cuda_max_seqlen = cuda_max_seqlen
        self.cuda_fused = CUDAFusedAttentionEncoderBackend()
        self.flash_attn = FlashAttentionEncoderBackend()

    def forward(self, query, key, value, **kwargs):
        backend = (
            self.cuda_fused
            if max(kwargs["max_seqlen_q"], kwargs["max_seqlen_k"])
            <= self.cuda_max_seqlen
            else self.flash_attn
        )
        return backend.forward(query, key, value, **kwargs)


def validate_inputs(
    query,
    key,
    value,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    causal,
):
    if not query.is_cuda or not key.is_cuda or not value.is_cuda:
        raise ValueError("CUDA fused attention requires CUDA tensors")
    if query.dtype != torch.bfloat16:
        raise ValueError("CUDA fused attention currently requires BF16")
    if key.dtype != query.dtype or value.dtype != query.dtype:
        raise ValueError("query, key and value must have the same dtype")
    if query.ndim != 3 or key.ndim != 3 or value.ndim != 3:
        raise ValueError("query, key and value must have shape [tokens, heads, dim]")
    if key.shape != value.shape:
        raise ValueError("key and value must have identical shapes")
    if query.size(2) != key.size(2):
        raise ValueError("query and KV head dimensions must match")
    if query.size(1) % key.size(1):
        raise ValueError("query heads must be divisible by KV heads")
    if query.size(2) > 128:
        raise ValueError("CUDA fused attention supports head_dim <= 128")
    if max(max_seqlen_q, max_seqlen_k) > 64:
        raise ValueError("CUDA fused attention supports sequence lengths <= 64")
    if cu_seqlens_q.dtype != torch.int32 or cu_seqlens_k.dtype != torch.int32:
        raise ValueError("cu_seqlens must use int32")
    if not cu_seqlens_q.is_cuda or not cu_seqlens_k.is_cuda:
        raise ValueError("cu_seqlens must be CUDA tensors")
    if cu_seqlens_q.numel() != cu_seqlens_k.numel():
        raise ValueError("Q and KV must contain the same number of sequences")
    if causal and not torch.equal(cu_seqlens_q, cu_seqlens_k):
        raise ValueError(
            "CUDA fused causal attention requires matching Q/KV lengths"
        )
