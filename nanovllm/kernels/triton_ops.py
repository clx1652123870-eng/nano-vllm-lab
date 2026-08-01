import torch
import triton
import triton.language as tl


@triton.jit
def _softmax_kernel(
    input_ptr,
    output_ptr,
    num_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_cols
    values = tl.load(input_ptr + row * num_cols + offsets, mask=mask, other=-float("inf"))
    values = values.to(tl.float32)
    values -= tl.max(values, axis=0)
    numerator = tl.exp(values)
    output = numerator / tl.sum(numerator, axis=0)
    tl.store(output_ptr + row * num_cols + offsets, output, mask=mask)


def triton_softmax(x: torch.Tensor) -> torch.Tensor:
    if not x.is_cuda or x.ndim < 1 or not x.is_contiguous():
        raise ValueError("Triton softmax requires a contiguous CUDA tensor")
    num_cols = x.shape[-1]
    if num_cols < 1 or num_cols > 65536:
        raise ValueError("Triton softmax supports 1 <= num_cols <= 65536")
    output = torch.empty_like(x)
    num_rows = x.numel() // num_cols
    block_size = triton.next_power_of_2(num_cols)
    num_warps = 4 if block_size <= 2048 else 8
    _softmax_kernel[(num_rows,)](
        x,
        output,
        num_cols,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    return output


@triton.jit
def _rms_norm_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    num_cols,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_cols
    values = tl.load(input_ptr + row * num_cols + offsets, mask=mask, other=0.0)
    values_fp32 = values.to(tl.float32)
    variance = tl.sum(values_fp32 * values_fp32, axis=0) / num_cols
    normalized = values_fp32 * tl.rsqrt(variance + eps)
    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0)
    tl.store(
        output_ptr + row * num_cols + offsets,
        normalized * weight,
        mask=mask,
    )


def triton_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    if not x.is_cuda or not x.is_contiguous():
        raise ValueError("Triton RMSNorm requires a contiguous CUDA tensor")
    if weight.ndim != 1 or weight.numel() != x.shape[-1]:
        raise ValueError("weight must match the last input dimension")
    if not weight.is_cuda or not weight.is_contiguous():
        raise ValueError("weight must be a contiguous CUDA tensor")
    num_cols = x.shape[-1]
    block_size = triton.next_power_of_2(num_cols)
    if block_size > 65536:
        raise ValueError("Triton RMSNorm supports hidden sizes up to 65536")
    output = torch.empty_like(x)
    num_rows = x.numel() // num_cols
    num_warps = 4 if block_size <= 2048 else 8
    _rms_norm_kernel[(num_rows,)](
        x,
        weight,
        output,
        num_cols,
        eps=eps,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    return output


@triton.jit
def _silu_and_mul_kernel(
    input_ptr,
    output_ptr,
    num_elements,
    hidden_size,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements
    row = offsets // hidden_size
    col = offsets % hidden_size
    gate = tl.load(
        input_ptr + row * hidden_size * 2 + col,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    up = tl.load(
        input_ptr + row * hidden_size * 2 + hidden_size + col,
        mask=mask,
        other=0.0,
    )
    output = gate * tl.sigmoid(gate) * up
    tl.store(output_ptr + offsets, output, mask=mask)


def triton_silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    if not x.is_cuda or not x.is_contiguous() or x.shape[-1] % 2:
        raise ValueError("Triton SiLU-and-mul requires contiguous [..., 2 * hidden]")
    hidden_size = x.shape[-1] // 2
    output_shape = x.shape[:-1] + (hidden_size,)
    output = torch.empty(output_shape, dtype=x.dtype, device=x.device)
    num_elements = output.numel()
    _silu_and_mul_kernel[(triton.cdiv(num_elements, 256),)](
        x,
        output,
        num_elements,
        hidden_size,
        BLOCK_SIZE=256,
        num_warps=4,
    )
    return output


@triton.jit
def _matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
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
    a_ptrs = a_ptr + offsets_m[:, None] * stride_am + offsets_k[None, :] * stride_ak
    b_ptrs = b_ptr + offsets_k[:, None] * stride_bk + offsets_n[None, :] * stride_bn
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

    for k_start in range(0, tl.cdiv(K, BLOCK_K)):
        k_mask = k_start * BLOCK_K + offsets_k < K
        a = tl.load(
            a_ptrs,
            mask=(offsets_m[:, None] < M) & k_mask[None, :],
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=k_mask[:, None] & (offsets_n[None, :] < N),
            other=0.0,
        )
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = c_ptr + offsets_m[:, None] * stride_cm + offsets_n[None, :] * stride_cn
    tl.store(
        c_ptrs,
        accumulator,
        mask=(offsets_m[:, None] < M) & (offsets_n[None, :] < N),
    )


def triton_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if not a.is_cuda or not b.is_cuda or a.ndim != 2 or b.ndim != 2:
        raise ValueError("Triton matmul requires two CUDA matrices")
    if not a.is_contiguous() or not b.is_contiguous():
        raise ValueError("Triton matmul requires contiguous inputs")
    if a.shape[1] != b.shape[0] or a.dtype != b.dtype:
        raise ValueError("incompatible matrix shapes or dtypes")
    M, K = a.shape
    N = b.shape[1]
    if M <= 16:
        block_m, block_n, block_k, num_warps = 16, 64, 32, 4
    elif M <= 64:
        block_m, block_n, block_k, num_warps = 32, 64, 32, 4
    else:
        block_m, block_n, block_k, num_warps = 64, 64, 32, 4
    output = torch.empty((M, N), dtype=a.dtype, device=a.device)
    grid = (triton.cdiv(M, block_m) * triton.cdiv(N, block_n),)
    _matmul_kernel[grid](
        a,
        b,
        output,
        M,
        N,
        K,
        *a.stride(),
        *b.stride(),
        *output.stride(),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=num_warps,
        num_stages=3,
    )
    return output
