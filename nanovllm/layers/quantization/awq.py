from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import nn

from nanovllm.layers.quantization.awq_kernels import awq_linear


@dataclass(frozen=True)
class AWQConfig:
    bits: int
    group_size: int
    zero_point: bool
    version: str
    modules_to_not_convert: tuple[str, ...]

    @classmethod
    def from_hf_config(cls, hf_config) -> "AWQConfig | None":
        raw = getattr(hf_config, "quantization_config", None)
        if not raw:
            return None
        if raw.get("quant_method") != "awq":
            raise ValueError(
                f"unsupported quantization method: {raw.get('quant_method')!r}"
            )
        config = cls(
            bits=int(raw["bits"]),
            group_size=int(raw["group_size"]),
            zero_point=bool(raw.get("zero_point", True)),
            version=str(raw.get("version", "gemm")).lower(),
            modules_to_not_convert=tuple(raw.get("modules_to_not_convert", ())),
        )
        if config.bits != 4 or config.group_size != 128:
            raise ValueError("nano-vllm currently supports AWQ INT4 group_size=128")
        if not config.zero_point or config.version != "gemm":
            raise ValueError("nano-vllm currently supports asymmetric AWQ GEMM")
        return config


class AWQLinearBase(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        quant_config: AWQConfig,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if dist.get_world_size() != 1:
            raise ValueError("the first AWQ implementation supports tensor_parallel_size=1")
        if input_size % quant_config.group_size or output_size % 8:
            raise ValueError("AWQ linear dimensions are incompatible with packing")
        self.input_size = input_size
        self.output_size = output_size
        self.quant_config = quant_config
        self.qweight = self._parameter(
            (input_size, output_size // 8),
            torch.int32,
            "qweight",
        )
        self.qzeros = self._parameter(
            (input_size // quant_config.group_size, output_size // 8),
            torch.int32,
            "qzeros",
        )
        self.scales = self._parameter(
            (input_size // quant_config.group_size, output_size),
            torch.float16,
            "scales",
        )
        if bias:
            self.bias = self._parameter((output_size,), torch.get_default_dtype(), "bias")
        else:
            self.register_parameter("bias", None)

    def _parameter(self, shape, dtype, component):
        parameter = nn.Parameter(
            torch.empty(shape, dtype=dtype),
            requires_grad=False,
        )
        parameter.awq_component = component
        parameter.weight_loader = self.weight_loader
        return parameter

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
    ) -> None:
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return awq_linear(
            x,
            self.qweight,
            self.qzeros,
            self.scales,
            self.bias,
            self.quant_config.group_size,
        )


class AWQColumnParallelLinear(AWQLinearBase):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        quant_config: AWQConfig,
        bias: bool = False,
    ) -> None:
        super().__init__(input_size, output_size, quant_config, bias)


class AWQMergedColumnParallelLinear(AWQColumnParallelLinear):
    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        quant_config: AWQConfig,
        bias: bool = False,
    ) -> None:
        self.output_sizes = output_sizes
        super().__init__(
            input_size,
            sum(output_sizes),
            quant_config,
            bias,
        )

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: int,
    ) -> None:
        component = param.awq_component
        output_offset = sum(self.output_sizes[:loaded_shard_id])
        output_size = self.output_sizes[loaded_shard_id]
        if component in {"qweight", "qzeros"}:
            output_offset //= 8
            output_size //= 8
            target = param.data.narrow(1, output_offset, output_size)
        elif component == "scales":
            target = param.data.narrow(1, output_offset, output_size)
        else:
            target = param.data.narrow(0, output_offset, output_size)
        target.copy_(loaded_weight)


class AWQQKVParallelLinear(AWQMergedColumnParallelLinear):
    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int,
        quant_config: AWQConfig,
        bias: bool = False,
    ) -> None:
        self.num_heads = total_num_heads
        self.num_kv_heads = total_num_kv_heads
        output_sizes = [
            total_num_heads * head_size,
            total_num_kv_heads * head_size,
            total_num_kv_heads * head_size,
        ]
        super().__init__(
            hidden_size,
            output_sizes,
            quant_config,
            bias,
        )

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: str,
    ) -> None:
        shard_index = {"q": 0, "k": 1, "v": 2}[loaded_shard_id]
        super().weight_loader(param, loaded_weight, shard_index)


class AWQRowParallelLinear(AWQLinearBase):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        quant_config: AWQConfig,
        bias: bool = False,
    ) -> None:
        super().__init__(input_size, output_size, quant_config, bias)
