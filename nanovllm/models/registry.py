from torch import nn
from transformers import PretrainedConfig

from nanovllm.models.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from nanovllm.models.qwen3 import Qwen3ForCausalLM


MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "Qwen3ForCausalLM": Qwen3ForCausalLM,
    "Qwen2_5_VLForConditionalGeneration": Qwen2_5_VLForConditionalGeneration,
}


def get_model_class(config: PretrainedConfig) -> type[nn.Module]:
    for architecture in config.architectures or []:
        if model_class := MODEL_REGISTRY.get(architecture):
            return model_class
    raise ValueError(
        f"unsupported model architectures: {getattr(config, 'architectures', None)}"
    )
