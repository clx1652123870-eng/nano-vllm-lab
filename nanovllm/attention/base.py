from abc import ABC, abstractmethod

import torch


class EncoderAttentionBackend(ABC):
    """Attention over packed Q/K/V without a persistent KV cache."""

    name: str

    @abstractmethod
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
        raise NotImplementedError


class DecoderAttentionBackend(ABC):
    """Causal prefill and paged-KV decode attention."""

    name: str

    @abstractmethod
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
    ) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def decode(
        self,
        query: torch.Tensor,
        *,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        context_lens: torch.Tensor,
        block_tables: torch.Tensor,
        softmax_scale: float,
    ) -> torch.Tensor:
        raise NotImplementedError
