import os

import torch
import triton
import triton.language as tl


@triton.jit
def _awq_shift(output_offsets):
    nibble = output_offsets % 8
    return ((nibble % 2) * 4 + nibble // 2) * 4


@triton.jit
def _awq_dequantize_kernel(
    qweight_ptr,
    qzeros_ptr,
    scales_ptr,
    output_ptr,
    num_elements,
    N: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements
    input_offsets = offsets // N
    output_offsets = offsets % N
    packed_columns = N // 8
    packed_offsets = input_offsets * packed_columns + output_offsets // 8
    shifts = _awq_shift(output_offsets)
    quantized = (
        tl.load(qweight_ptr + packed_offsets, mask=mask, other=0) >> shifts
    ) & 0xF
    group_offsets = input_offsets // GROUP_SIZE
    zero_offsets = group_offsets * packed_columns + output_offsets // 8
    zeros = (
        tl.load(qzeros_ptr + zero_offsets, mask=mask, other=0) >> shifts
    ) & 0xF
    scales = tl.load(
        scales_ptr + group_offsets * N + output_offsets,
        mask=mask,
        other=0.0,
    )
    tl.store(output_ptr + offsets, (quantized - zeros) * scales, mask=mask)


def awq_dequantize(
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    _validate_awq_tensors(qweight, qzeros, scales, group_size)
    K = qweight.shape[0]
    N = qweight.shape[1] * 8
    output = torch.empty((K, N), dtype=dtype, device=qweight.device)
    num_elements = K * N
    _awq_dequantize_kernel[(triton.cdiv(num_elements, 256),)](
        qweight,
        qzeros,
        scales,
        output,
        num_elements,
        N=N,
        GROUP_SIZE=group_size,
        BLOCK_SIZE=256,
        num_warps=4,
    )
    return output


@triton.jit
def _awq_gemm_kernel(
    input_ptr,
    qweight_ptr,
    qzeros_ptr,
    scales_ptr,
    output_ptr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am,
    stride_ak,
    stride_qk,
    stride_qn,
    stride_om,
    stride_on,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    offsets_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets_k = tl.arange(0, BLOCK_K)
    shifts = _awq_shift(offsets_n)
    packed_n = offsets_n // 8
    packed_columns = N // 8
    input_ptrs = (
        input_ptr
        + offsets_m[:, None] * stride_am
        + offsets_k[None, :] * stride_ak
    )
    qweight_ptrs = (
        qweight_ptr
        + offsets_k[:, None] * stride_qk
        + packed_n[None, :] * stride_qn
    )
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

    for k_start in range(0, tl.cdiv(K, BLOCK_K)):
        logical_k = k_start * BLOCK_K + offsets_k
        k_mask = logical_k < K
        activations = tl.load(
            input_ptrs,
            mask=(offsets_m[:, None] < M) & k_mask[None, :],
            other=0.0,
        )
        packed_weight = tl.load(
            qweight_ptrs,
            mask=k_mask[:, None] & (offsets_n[None, :] < N),
            other=0,
        )
        quantized = (packed_weight >> shifts[None, :]) & 0xF
        groups = logical_k // GROUP_SIZE
        packed_zero_offsets = (
            groups[:, None] * packed_columns + packed_n[None, :]
        )
        packed_zeros = tl.load(
            qzeros_ptr + packed_zero_offsets,
            mask=k_mask[:, None] & (offsets_n[None, :] < N),
            other=0,
        )
        zeros = (packed_zeros >> shifts[None, :]) & 0xF
        scale_offsets = groups[:, None] * N + offsets_n[None, :]
        scales = tl.load(
            scales_ptr + scale_offsets,
            mask=k_mask[:, None] & (offsets_n[None, :] < N),
            other=0.0,
        )
        weights = ((quantized - zeros) * scales).to(activations.dtype)
        accumulator = tl.dot(activations, weights, accumulator)
        input_ptrs += BLOCK_K * stride_ak
        qweight_ptrs += BLOCK_K * stride_qk

    output_ptrs = (
        output_ptr
        + offsets_m[:, None] * stride_om
        + offsets_n[None, :] * stride_on
    )
    tl.store(
        output_ptrs,
        accumulator,
        mask=(offsets_m[:, None] < M) & (offsets_n[None, :] < N),
    )


def awq_gemm(
    x: torch.Tensor,
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    _validate_awq_tensors(qweight, qzeros, scales, group_size)
    if not x.is_cuda or not x.is_contiguous() or x.ndim != 2:
        raise ValueError("AWQ GEMM requires a contiguous CUDA matrix")
    M, K = x.shape
    N = qweight.shape[1] * 8
    if K != qweight.shape[0]:
        raise ValueError("activation and qweight K dimensions do not match")
    if x.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("AWQ GEMM supports FP16 and BF16 activations")
    if M <= 16:
        block_m, block_n, block_k, num_warps = 16, 64, 32, 4
    elif M <= 64:
        block_m, block_n, block_k, num_warps = 32, 64, 32, 4
    else:
        block_m, block_n, block_k, num_warps = 64, 64, 32, 4
    output = torch.empty((M, N), dtype=x.dtype, device=x.device)
    grid = (triton.cdiv(M, block_m) * triton.cdiv(N, block_n),)
    _awq_gemm_kernel[grid](
        x,
        qweight,
        qzeros,
        scales,
        output,
        M,
        N,
        K,
        *x.stride(),
        *qweight.stride(),
        *output.stride(),
        GROUP_SIZE=group_size,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=num_warps,
        num_stages=3,
    )
    return output


def awq_linear(
    x: torch.Tensor,
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    scales: torch.Tensor,
    bias: torch.Tensor | None,
    group_size: int,
) -> torch.Tensor:
    input_shape = x.shape
    flattened = x.reshape(-1, input_shape[-1]).contiguous()
    kernel = os.getenv("NANOVLLM_AWQ_KERNEL", "dequantize")
    if kernel == "dequantize":
        weight = awq_dequantize(
            qweight,
            qzeros,
            scales,
            group_size,
            flattened.dtype,
        )
        output = flattened @ weight
    elif kernel == "triton_fused":
        output = awq_gemm(
            flattened,
            qweight,
            qzeros,
            scales,
            group_size,
        )
    else:
        raise ValueError(
            "NANOVLLM_AWQ_KERNEL must be 'dequantize' or 'triton_fused'"
        )
    if bias is not None:
        output.add_(bias)
    return output.reshape(input_shape[:-1] + (qweight.shape[1] * 8,))


def _validate_awq_tensors(
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
) -> None:
    if group_size not in (32, 64, 128):
        raise ValueError("AWQ group_size must be one of 32, 64 or 128")
    if not qweight.is_cuda or not qzeros.is_cuda or not scales.is_cuda:
        raise ValueError("AWQ tensors must be on CUDA")
    if (
        not qweight.is_contiguous()
        or not qzeros.is_contiguous()
        or not scales.is_contiguous()
    ):
        raise ValueError("AWQ tensors must be contiguous")
    if qweight.dtype != torch.int32 or qzeros.dtype != torch.int32:
        raise ValueError("qweight and qzeros must be int32")
    K, packed_n = qweight.shape
    if K % group_size:
        raise ValueError("input size must be divisible by group_size")
    if qzeros.shape != (K // group_size, packed_n):
        raise ValueError("qzeros shape does not match qweight")
    if scales.shape != (K // group_size, packed_n * 8):
        raise ValueError("scales shape does not match qweight")
