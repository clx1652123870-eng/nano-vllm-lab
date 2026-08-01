import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from nanovllm.attention.base import EncoderAttentionBackend


class TorchSDPAEncoderBackend(EncoderAttentionBackend):
    """Packed-varlen adapter for PyTorch SDPA.

    This reference adapter loops over packed sequences. It is useful for
    correctness and backend comparison, not a claim of optimal varlen
    performance.
    """

    name = "torch_sdpa"
    sdpa_backend: SDPBackend | None = None

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
        del max_seqlen_q, max_seqlen_k
        q_offsets = cu_seqlens_q.detach().cpu().tolist()
        k_offsets = cu_seqlens_k.detach().cpu().tolist()
        if len(q_offsets) != len(k_offsets):
            raise ValueError("Q and KV must contain the same number of sequences")
        validate_packed_offsets(q_offsets, query.size(0), "query")
        validate_packed_offsets(k_offsets, key.size(0), "key")
        if key.size(0) != value.size(0):
            raise ValueError("key and value must contain the same number of tokens")

        output = torch.empty_like(query)
        for q_start, q_end, k_start, k_end in zip(
            q_offsets[:-1],
            q_offsets[1:],
            k_offsets[:-1],
            k_offsets[1:],
        ):
            q_seq = query[q_start:q_end]
            k_seq = key[k_start:k_end]
            v_seq = value[k_start:k_end]
            if q_seq.size(0) == 0:
                continue
            if k_seq.size(0) == 0:
                raise ValueError("non-empty query sequence requires KV tokens")
            if causal and q_seq.size(0) != k_seq.size(0):
                raise ValueError(
                    "torch_sdpa encoder reference only supports causal "
                    "attention when Q and KV lengths match"
                )
            k_seq, v_seq = expand_kv_heads(q_seq, k_seq, v_seq)
            result = self._scaled_dot_product_attention(
                q_seq,
                k_seq,
                v_seq,
                causal=causal,
                softmax_scale=softmax_scale,
            )
            output[q_start:q_end] = result.squeeze(0).transpose(0, 1)
        return output

    def _scaled_dot_product_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        causal: bool,
        softmax_scale: float,
    ) -> torch.Tensor:
        query = query.transpose(0, 1).unsqueeze(0)
        key = key.transpose(0, 1).unsqueeze(0)
        value = value.transpose(0, 1).unsqueeze(0)
        if self.sdpa_backend is None:
            return F.scaled_dot_product_attention(
                query,
                key,
                value,
                dropout_p=0.0,
                is_causal=causal,
                scale=softmax_scale,
            )
        with sdpa_kernel(self.sdpa_backend):
            return F.scaled_dot_product_attention(
                query,
                key,
                value,
                dropout_p=0.0,
                is_causal=causal,
                scale=softmax_scale,
            )


class CUDNNSDPAEncoderBackend(TorchSDPAEncoderBackend):
    """Packed-varlen adapter that forces PyTorch's cuDNN SDPA backend."""

    name = "cudnn_sdpa"
    sdpa_backend = SDPBackend.CUDNN_ATTENTION


class TorchMathSDPAEncoderBackend(TorchSDPAEncoderBackend):
    """Reference backend that forces PyTorch's unfused SDPA math path."""

    name = "torch_math"
    sdpa_backend = SDPBackend.MATH


def expand_kv_heads(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_query_heads = query.size(1)
    num_kv_heads = key.size(1)
    if num_query_heads == 0 or num_kv_heads == 0:
        raise ValueError("query and KV must each contain at least one head")
    if value.size(1) != num_kv_heads:
        raise ValueError("key and value must have the same number of heads")
    if num_query_heads == num_kv_heads:
        return key, value
    if num_query_heads % num_kv_heads:
        raise ValueError("query heads must be divisible by KV heads")
    repeats = num_query_heads // num_kv_heads
    return (
        key.repeat_interleave(repeats, dim=1),
        value.repeat_interleave(repeats, dim=1),
    )


def validate_packed_offsets(
    offsets: list[int],
    total_tokens: int,
    name: str,
) -> None:
    if not offsets or offsets[0] != 0 or offsets[-1] != total_tokens:
        raise ValueError(
            f"{name} cu_seqlens must start at 0 and end at {total_tokens}"
        )
    if any(start > end for start, end in zip(offsets[:-1], offsets[1:])):
        raise ValueError(f"{name} cu_seqlens must be non-decreasing")
