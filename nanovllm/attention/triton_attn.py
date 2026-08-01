import torch
import triton
import triton.language as tl

from nanovllm.attention.base import EncoderAttentionBackend


@triton.jit
def _packed_attention_forward_kernel(
    query_ptr,
    key_ptr,
    value_ptr,
    output_ptr,
    cu_seqlens_q_ptr,
    cu_seqlens_k_ptr,
    stride_qt: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qd: tl.constexpr,
    stride_kt: tl.constexpr,
    stride_kh: tl.constexpr,
    stride_kd: tl.constexpr,
    stride_vt: tl.constexpr,
    stride_vh: tl.constexpr,
    stride_vd: tl.constexpr,
    stride_ot: tl.constexpr,
    stride_oh: tl.constexpr,
    stride_od: tl.constexpr,
    NUM_QUERY_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PADDED_HEAD_DIM: tl.constexpr,
    SOFTMAX_SCALE: tl.constexpr,
    CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    query_block = tl.program_id(0)
    query_head = tl.program_id(1)
    sequence = tl.program_id(2)

    q_start = tl.load(cu_seqlens_q_ptr + sequence)
    q_end = tl.load(cu_seqlens_q_ptr + sequence + 1)
    k_start = tl.load(cu_seqlens_k_ptr + sequence)
    k_end = tl.load(cu_seqlens_k_ptr + sequence + 1)
    q_length = q_end - q_start
    k_length = k_end - k_start

    kv_group_size = NUM_QUERY_HEADS // NUM_KV_HEADS
    kv_head = query_head // kv_group_size
    local_q_offsets = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
    dim_offsets = tl.arange(0, PADDED_HEAD_DIM)
    q_valid = local_q_offsets < q_length
    dim_valid = dim_offsets < HEAD_DIM

    query_offsets = (
        (q_start + local_q_offsets)[:, None] * stride_qt
        + query_head * stride_qh
        + dim_offsets[None, :] * stride_qd
    )
    query = tl.load(
        query_ptr + query_offsets,
        mask=q_valid[:, None] & dim_valid[None, :],
        other=0.0,
    )

    row_max = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    row_sum = tl.zeros((BLOCK_M,), tl.float32)
    accumulator = tl.zeros((BLOCK_M, PADDED_HEAD_DIM), tl.float32)
    log2_scale = SOFTMAX_SCALE * 1.4426950408889634

    for key_block_start in tl.range(0, k_length, BLOCK_N, num_stages=2):
        local_k_offsets = key_block_start + tl.arange(0, BLOCK_N)
        k_valid = local_k_offsets < k_length
        key_offsets = (
            (k_start + local_k_offsets)[None, :] * stride_kt
            + kv_head * stride_kh
            + dim_offsets[:, None] * stride_kd
        )
        key = tl.load(
            key_ptr + key_offsets,
            mask=dim_valid[:, None] & k_valid[None, :],
            other=0.0,
        )
        scores = tl.dot(query, key) * log2_scale
        attention_mask = q_valid[:, None] & k_valid[None, :]
        if CAUSAL:
            attention_mask &= (
                local_q_offsets[:, None] >= local_k_offsets[None, :]
            )
        scores = tl.where(attention_mask, scores, -float("inf"))

        block_max = tl.maximum(row_max, tl.max(scores, axis=1))
        correction = tl.math.exp2(row_max - block_max)
        probabilities = tl.math.exp2(scores - block_max[:, None])
        block_sum = tl.sum(probabilities, axis=1)
        accumulator *= correction[:, None]

        value_offsets = (
            (k_start + local_k_offsets)[:, None] * stride_vt
            + kv_head * stride_vh
            + dim_offsets[None, :] * stride_vd
        )
        value = tl.load(
            value_ptr + value_offsets,
            mask=k_valid[:, None] & dim_valid[None, :],
            other=0.0,
        )
        accumulator += tl.dot(probabilities.to(tl.bfloat16), value)
        row_sum = row_sum * correction + block_sum
        row_max = block_max

    output = accumulator / row_sum[:, None]
    output_offsets = (
        (q_start + local_q_offsets)[:, None] * stride_ot
        + query_head * stride_oh
        + dim_offsets[None, :] * stride_od
    )
    tl.store(
        output_ptr + output_offsets,
        output,
        mask=q_valid[:, None] & dim_valid[None, :],
    )


class TritonEncoderAttentionBackend(EncoderAttentionBackend):
    """Experimental BF16 packed-varlen forward Attention kernel."""

    name = "triton"

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
        num_sequences = cu_seqlens_q.numel() - 1
        num_query_heads = query.size(1)
        num_kv_heads = key.size(1)
        head_dim = query.size(2)
        padded_head_dim = triton.next_power_of_2(head_dim)
        if max_seqlen_q <= 64:
            block_m, block_n, num_warps = 32, 32, 4
        else:
            block_m, block_n, num_warps = 128, 64, 8
        output = torch.empty(
            query.shape,
            dtype=query.dtype,
            device=query.device,
        )
        grid = (
            triton.cdiv(max_seqlen_q, block_m),
            num_query_heads,
            num_sequences,
        )
        with torch.cuda.device(query.device):
            _packed_attention_forward_kernel[grid](
                query,
                key,
                value,
                output,
                cu_seqlens_q,
                cu_seqlens_k,
                *query.stride(),
                *key.stride(),
                *value.stride(),
                *output.stride(),
                NUM_QUERY_HEADS=num_query_heads,
                NUM_KV_HEADS=num_kv_heads,
                HEAD_DIM=head_dim,
                PADDED_HEAD_DIM=padded_head_dim,
                SOFTMAX_SCALE=softmax_scale,
                CAUSAL=causal,
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                num_warps=num_warps,
            )
        return output


def validate_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    causal: bool,
) -> None:
    if not query.is_cuda or not key.is_cuda or not value.is_cuda:
        raise ValueError("Triton attention requires CUDA tensors")
    if query.dtype != torch.bfloat16:
        raise ValueError("experimental Triton attention currently requires BF16")
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
    if query.size(2) > 256:
        raise ValueError("experimental Triton attention supports head_dim <= 256")
    if cu_seqlens_q.dtype != torch.int32 or cu_seqlens_k.dtype != torch.int32:
        raise ValueError("cu_seqlens must use int32")
    if not cu_seqlens_q.is_cuda or not cu_seqlens_k.is_cuda:
        raise ValueError("cu_seqlens must be CUDA tensors")
    if cu_seqlens_q.numel() != cu_seqlens_k.numel():
        raise ValueError("Q and KV must contain the same number of sequences")
    if max_seqlen_q < 1 or max_seqlen_k < 1:
        raise ValueError("maximum sequence lengths must be positive")
    if causal and not torch.equal(cu_seqlens_q, cu_seqlens_k):
        raise ValueError(
            "experimental Triton causal attention requires matching Q/KV lengths"
        )
