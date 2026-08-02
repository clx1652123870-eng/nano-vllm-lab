from nanovllm.attention.base import (
    DecoderAttentionBackend,
    EncoderAttentionBackend,
)


_ALIASES = {
    "cuda": "cuda_fused",
    "cuda_flash": "cuda_fused",
    "custom_cuda": "cuda_fused",
    "cudnn": "cudnn_sdpa",
    "flash": "flash_attn",
    "flash_attention": "flash_attn",
    "flash_triton": "hybrid",
    "math": "torch_math",
    "pytorch_math": "torch_math",
    "sdpa": "torch_sdpa",
    "torch": "torch_sdpa",
}

_ENCODER_BACKENDS = (
    "cuda_fused",
    "cuda_hybrid",
    "cudnn_sdpa",
    "flash_attn",
    "hybrid",
    "torch_math",
    "torch_sdpa",
    "triton",
)
_DECODER_BACKENDS = ("flash_attn",)


def normalize_attention_backend_name(name: str) -> str:
    normalized = name.strip().lower().replace("-", "_")
    return _ALIASES.get(normalized, normalized)


def create_encoder_attention_backend(name: str) -> EncoderAttentionBackend:
    normalized = normalize_attention_backend_name(name)
    if normalized not in _ENCODER_BACKENDS:
        supported = ", ".join(supported_encoder_attention_backends())
        raise ValueError(
            f"unsupported encoder attention backend {name!r}; "
            f"supported backends: {supported}"
        )
    if normalized == "flash_attn":
        from nanovllm.attention.flash_attn import FlashAttentionEncoderBackend

        return FlashAttentionEncoderBackend()
    if normalized == "cuda_fused":
        from nanovllm.attention.cuda_attn import CUDAFusedAttentionEncoderBackend

        return CUDAFusedAttentionEncoderBackend()
    if normalized == "cuda_hybrid":
        from nanovllm.attention.cuda_attn import CUDAHybridEncoderAttentionBackend

        return CUDAHybridEncoderAttentionBackend()
    if normalized == "cudnn_sdpa":
        from nanovllm.attention.torch_sdpa import CUDNNSDPAEncoderBackend

        return CUDNNSDPAEncoderBackend()
    if normalized == "torch_math":
        from nanovllm.attention.torch_sdpa import TorchMathSDPAEncoderBackend

        return TorchMathSDPAEncoderBackend()
    if normalized == "triton":
        from nanovllm.attention.triton_attn import TritonEncoderAttentionBackend

        return TritonEncoderAttentionBackend()
    if normalized == "hybrid":
        from nanovllm.attention.hybrid import HybridEncoderAttentionBackend

        return HybridEncoderAttentionBackend()

    from nanovllm.attention.torch_sdpa import TorchSDPAEncoderBackend

    return TorchSDPAEncoderBackend()


def create_decoder_attention_backend(name: str) -> DecoderAttentionBackend:
    normalized = normalize_attention_backend_name(name)
    if normalized not in _DECODER_BACKENDS:
        supported = ", ".join(supported_decoder_attention_backends())
        raise ValueError(
            f"unsupported decoder attention backend {name!r}; "
            f"supported backends: {supported}"
        )
    from nanovllm.attention.flash_attn import FlashAttentionDecoderBackend

    return FlashAttentionDecoderBackend()


def supported_encoder_attention_backends() -> tuple[str, ...]:
    return tuple(sorted(_ENCODER_BACKENDS))


def supported_decoder_attention_backends() -> tuple[str, ...]:
    return tuple(sorted(_DECODER_BACKENDS))
