import os
from functools import lru_cache
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


@lru_cache(maxsize=1)
def _load_extension():
    source_dir = Path(__file__).resolve().parent / "csrc"
    return load(
        name="nanovllm_cuda_ops_v2",
        sources=[
            str(source_dir / "bindings.cpp"),
            str(source_dir / "kernels.cu"),
        ],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=os.getenv("NANOVLLM_CUDA_BUILD_VERBOSE") == "1",
    )


def cuda_softmax(x: torch.Tensor) -> torch.Tensor:
    return _load_extension().softmax(x)


def cuda_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    return _load_extension().rms_norm(x, weight, eps)


def cuda_silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    return _load_extension().silu_and_mul(x)


def cuda_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return _load_extension().matmul(a, b)


def cuda_packed_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float,
    causal: bool,
) -> torch.Tensor:
    return _load_extension().packed_attention(
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
