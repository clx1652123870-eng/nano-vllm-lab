import os
from dataclasses import dataclass
from transformers import AutoConfig

from nanovllm.attention import (
    normalize_attention_backend_name,
    supported_decoder_attention_backends,
    supported_encoder_attention_backends,
)


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    distributed_init_port: int = 0
    attention_backend: str = "flash_attn"
    vision_attention_backend: str | None = None

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        assert self.max_num_seqs >= 1
        assert self.max_num_batched_tokens >= 1
        self.hf_config = AutoConfig.from_pretrained(self.model)
        quantization_config = getattr(
            self.hf_config,
            "quantization_config",
            None,
        )
        if (
            quantization_config
            and quantization_config.get("quant_method") == "awq"
            and self.tensor_parallel_size != 1
        ):
            raise ValueError(
                "the first AWQ implementation requires tensor_parallel_size=1"
            )
        self.max_model_len = min(
            self.max_model_len,
            self.text_config.max_position_embeddings,
        )
        self.attention_backend = normalize_attention_backend_name(
            self.attention_backend
        )
        if self.attention_backend not in supported_decoder_attention_backends():
            supported = ", ".join(supported_decoder_attention_backends())
            raise ValueError(
                f"unsupported decoder attention backend "
                f"{self.attention_backend!r}; supported backends: {supported}"
            )

        vision_backend = self.vision_attention_backend or self.attention_backend
        self.vision_attention_backend = normalize_attention_backend_name(
            vision_backend
        )
        if (
            hasattr(self.hf_config, "vision_config")
            and self.vision_attention_backend
            not in supported_encoder_attention_backends()
        ):
            supported = ", ".join(supported_encoder_attention_backends())
            raise ValueError(
                f"unsupported vision attention backend "
                f"{self.vision_attention_backend!r}; "
                f"supported backends: {supported}"
            )

    @property
    def text_config(self):
        return getattr(self.hf_config, "text_config", self.hf_config)
