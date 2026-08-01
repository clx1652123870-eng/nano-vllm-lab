import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import (
    Qwen2_5_VLConfig,
    Qwen2_5_VLTextConfig,
    Qwen2_5_VLVisionConfig,
)

from nanovllm.attention import create_encoder_attention_backend
from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.embed_head import ParallelLMHead, VocabParallelEmbedding
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from nanovllm.layers.quantization import (
    AWQConfig,
    AWQMergedColumnParallelLinear,
    AWQQKVParallelLinear,
    AWQRowParallelLinear,
)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class Qwen2_5_VisionRMSNorm(nn.Module):

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states.to(input_dtype)


class Qwen2_5_VLRotaryEmbedding(nn.Module):
    """Text-side multimodal RoPE for temporal, height and width positions."""

    def __init__(
        self,
        head_dim: int,
        rope_theta: float,
        mrope_section: list[int],
    ) -> None:
        super().__init__()
        section_sizes = mrope_section * 2
        if sum(section_sizes) != head_dim:
            raise ValueError(
                f"mrope sections {mrope_section} do not cover head_dim={head_dim}"
            )
        self.section_sizes = section_sizes
        inv_freq = 1.0 / (
            rope_theta
            ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if positions.ndim == 1:
            positions = positions.unsqueeze(0).expand(3, -1)
        if positions.ndim != 2 or positions.size(0) != 3:
            raise ValueError(
                "Qwen2.5-VL positions must have shape [tokens] or [3, tokens]"
            )

        freqs = positions.float().unsqueeze(-1) * self.inv_freq.float()
        emb = torch.cat((freqs, freqs), dim=-1)
        cos_by_axis = emb.cos().split(self.section_sizes, dim=-1)
        sin_by_axis = emb.sin().split(self.section_sizes, dim=-1)
        cos = torch.cat(
            [part[axis % 3] for axis, part in enumerate(cos_by_axis)], -1
        )
        sin = torch.cat(
            [part[axis % 3] for axis, part in enumerate(sin_by_axis)], -1
        )
        cos = cos.unsqueeze(1).to(query.dtype)
        sin = sin.unsqueeze(1).to(query.dtype)
        query = query * cos + rotate_half(query) * sin
        key = key * cos.to(key.dtype) + rotate_half(key) * sin.to(key.dtype)
        return query, key


class Qwen2_5_VisionPatchEmbed(nn.Module):

    def __init__(self, config: Qwen2_5_VLVisionConfig) -> None:
        super().__init__()
        self.in_channels = config.in_channels
        self.temporal_patch_size = config.temporal_patch_size
        self.patch_size = config.patch_size
        self.hidden_size = config.hidden_size
        kernel_size = (
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        )
        self.proj = nn.Conv3d(
            self.in_channels,
            self.hidden_size,
            kernel_size=kernel_size,
            stride=kernel_size,
            bias=False,
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        patch_features = (
            self.in_channels * self.temporal_patch_size * self.patch_size**2
        )
        if pixel_values.ndim != 2 or pixel_values.size(-1) != patch_features:
            raise ValueError(
                "pixel_values must have shape "
                f"[num_patches, {patch_features}], got {tuple(pixel_values.shape)}"
            )
        pixel_values = pixel_values.view(
            -1,
            self.in_channels,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        )
        return self.proj(pixel_values.to(self.proj.weight.dtype)).flatten(1)


class Qwen2_5_VisionRotaryEmbedding(nn.Module):

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, position_ids: torch.Tensor) -> torch.Tensor:
        return (position_ids.unsqueeze(-1).float() * self.inv_freq).flatten(1)


def apply_vision_rotary_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    query_dtype = query.dtype
    key_dtype = key.dtype
    cos = cos.unsqueeze(1).float()
    sin = sin.unsqueeze(1).float()
    query = query.float()
    key = key.float()
    query = query * cos + rotate_half(query) * sin
    key = key * cos + rotate_half(key) * sin
    return query.to(query_dtype), key.to(key_dtype)


class Qwen2_5_VisionAttention(nn.Module):

    def __init__(
        self,
        config: Qwen2_5_VLVisionConfig,
        attention_backend: str = "flash_attn",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.scale = self.head_dim**-0.5
        self.backend = create_encoder_attention_backend(attention_backend)
        self.backend_name = self.backend.name
        # The vision tower is replicated across TP ranks in this first version.
        self.qkv = nn.Linear(self.hidden_size, self.hidden_size * 3, bias=True)
        self.proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        num_tokens = hidden_states.size(0)
        qkv = self.qkv(hidden_states).view(
            num_tokens, 3, self.num_heads, self.head_dim
        )
        query, key, value = qkv.unbind(dim=1)
        query, key = apply_vision_rotary_emb(query, key, cos, sin)
        max_seqlen = int((cu_seqlens[1:] - cu_seqlens[:-1]).max().item())
        output = self.backend.forward(
            query,
            key,
            value,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            softmax_scale=self.scale,
            causal=False,
        )
        return self.proj(output.flatten(1))


class Qwen2_5_VisionMLP(nn.Module):

    def __init__(self, config: Qwen2_5_VLVisionConfig) -> None:
        super().__init__()
        if config.hidden_act != "silu":
            raise ValueError(f"unsupported vision activation: {config.hidden_act}")
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=True
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=True
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=True
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = F.silu(self.gate_proj(hidden_states)) * self.up_proj(
            hidden_states
        )
        return self.down_proj(hidden_states)


class Qwen2_5_VisionBlock(nn.Module):

    def __init__(
        self,
        config: Qwen2_5_VLVisionConfig,
        attention_backend: str = "flash_attn",
    ) -> None:
        super().__init__()
        self.norm1 = Qwen2_5_VisionRMSNorm(config.hidden_size, eps=1e-6)
        self.norm2 = Qwen2_5_VisionRMSNorm(config.hidden_size, eps=1e-6)
        self.attn = Qwen2_5_VisionAttention(config, attention_backend)
        self.mlp = Qwen2_5_VisionMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states), cu_seqlens, cos, sin
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class Qwen2_5_VisionPatchMerger(nn.Module):

    def __init__(self, config: Qwen2_5_VLVisionConfig) -> None:
        super().__init__()
        self.merged_hidden_size = config.hidden_size * config.spatial_merge_size**2
        self.ln_q = Qwen2_5_VisionRMSNorm(config.hidden_size, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(self.merged_hidden_size, self.merged_hidden_size, bias=True),
            nn.GELU(),
            nn.Linear(self.merged_hidden_size, config.out_hidden_size, bias=True),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.ln_q(hidden_states)
        return self.mlp(hidden_states.view(-1, self.merged_hidden_size))


def get_vision_position_ids(
    grid_thw: torch.Tensor,
    spatial_merge_size: int,
) -> torch.Tensor:
    position_ids = []
    device = grid_thw.device
    for grid_t, grid_h, grid_w in grid_thw.tolist():
        grid_t, grid_h, grid_w = int(grid_t), int(grid_h), int(grid_w)
        h_ids = torch.arange(grid_h, device=device).view(-1, 1).expand(-1, grid_w)
        w_ids = torch.arange(grid_w, device=device).view(1, -1).expand(grid_h, -1)
        shape = (
            grid_h // spatial_merge_size,
            spatial_merge_size,
            grid_w // spatial_merge_size,
            spatial_merge_size,
        )
        h_ids = h_ids.reshape(shape).transpose(1, 2).flatten()
        w_ids = w_ids.reshape(shape).transpose(1, 2).flatten()
        position_ids.append(torch.stack((h_ids, w_ids), dim=-1).repeat(grid_t, 1))
    return torch.cat(position_ids)


def get_vision_cu_seqlens(grid_thw: torch.Tensor) -> torch.Tensor:
    lengths = torch.repeat_interleave(
        grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
    )
    return F.pad(lengths.cumsum(0, dtype=torch.int32), (1, 0), value=0)


def get_vision_window_index(
    grid_thw: torch.Tensor,
    spatial_merge_size: int,
    window_size: int,
    patch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = grid_thw.device
    merger_window_size = window_size // spatial_merge_size // patch_size
    spatial_merge_unit = spatial_merge_size**2
    window_indices = []
    cu_window_seqlens = [0]
    index_offset = 0

    for grid_t, grid_h, grid_w in grid_thw.tolist():
        grid_t, grid_h, grid_w = int(grid_t), int(grid_h), int(grid_w)
        llm_grid_h = grid_h // spatial_merge_size
        llm_grid_w = grid_w // spatial_merge_size
        index = torch.arange(
            grid_t * llm_grid_h * llm_grid_w, device=device
        ).reshape(grid_t, llm_grid_h, llm_grid_w)
        pad_h = (merger_window_size - llm_grid_h % merger_window_size) % merger_window_size
        pad_w = (merger_window_size - llm_grid_w % merger_window_size) % merger_window_size
        num_windows_h = (llm_grid_h + pad_h) // merger_window_size
        num_windows_w = (llm_grid_w + pad_w) // merger_window_size
        index = F.pad(index, (0, pad_w, 0, pad_h), value=-100)
        index = index.reshape(
            grid_t,
            num_windows_h,
            merger_window_size,
            num_windows_w,
            merger_window_size,
        )
        index = index.permute(0, 1, 3, 2, 4).reshape(
            grid_t,
            num_windows_h * num_windows_w,
            merger_window_size,
            merger_window_size,
        )
        window_lengths = (index != -100).sum((2, 3)).flatten()
        index = index.flatten()
        window_indices.append(index[index != -100] + index_offset)
        cumulative = (
            window_lengths.cumsum(0) * spatial_merge_unit + cu_window_seqlens[-1]
        )
        cu_window_seqlens.extend(cumulative.tolist())
        index_offset += grid_t * llm_grid_h * llm_grid_w

    window_index = torch.cat(window_indices)
    cu_seqlens = torch.tensor(
        cu_window_seqlens, dtype=torch.int32, device=device
    ).unique_consecutive()
    return window_index, cu_seqlens


class Qwen2_5_VisionTransformer(nn.Module):

    def __init__(
        self,
        config: Qwen2_5_VLVisionConfig,
        attention_backend: str = "flash_attn",
    ) -> None:
        super().__init__()
        self.attention_backend = attention_backend
        self.spatial_merge_size = config.spatial_merge_size
        self.spatial_merge_unit = self.spatial_merge_size**2
        self.patch_size = config.patch_size
        self.window_size = config.window_size
        self.fullatt_block_indexes = set(config.fullatt_block_indexes)
        self.patch_embed = Qwen2_5_VisionPatchEmbed(config)
        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = Qwen2_5_VisionRotaryEmbedding(head_dim // 2)
        self.blocks = nn.ModuleList(
            Qwen2_5_VisionBlock(config, attention_backend)
            for _ in range(config.depth)
        )
        self.merger = Qwen2_5_VisionPatchMerger(config)

    @property
    def dtype(self) -> torch.dtype:
        return self.patch_embed.proj.weight.dtype

    def forward(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        grid_thw = grid_thw.to(device=pixel_values.device)
        position_ids = get_vision_position_ids(grid_thw, self.spatial_merge_size)
        cu_seqlens = get_vision_cu_seqlens(grid_thw)
        window_index, cu_window_seqlens = get_vision_window_index(
            grid_thw,
            self.spatial_merge_size,
            self.window_size,
            self.patch_size,
        )

        hidden_states = self.patch_embed(pixel_values)
        num_tokens = hidden_states.size(0)
        if num_tokens != int(grid_thw.prod(-1).sum().item()):
            raise ValueError("pixel patch count does not match image_grid_thw")

        hidden_states = hidden_states.view(
            num_tokens // self.spatial_merge_unit,
            self.spatial_merge_unit,
            -1,
        )
        hidden_states = hidden_states[window_index].reshape(num_tokens, -1)

        rotary = self.rotary_pos_emb(position_ids)
        rotary = rotary.view(
            num_tokens // self.spatial_merge_unit,
            self.spatial_merge_unit,
            -1,
        )
        rotary = rotary[window_index].reshape(num_tokens, -1)
        rotary = torch.cat((rotary, rotary), dim=-1)
        cos, sin = rotary.cos(), rotary.sin()

        for layer_idx, block in enumerate(self.blocks):
            layer_cu_seqlens = (
                cu_seqlens
                if layer_idx in self.fullatt_block_indexes
                else cu_window_seqlens
            )
            hidden_states = block(hidden_states, layer_cu_seqlens, cos, sin)

        hidden_states = self.merger(hidden_states)
        reverse_indices = torch.argsort(window_index)
        return hidden_states[reverse_indices]


class Qwen2_5_VLAttention(nn.Module):

    def __init__(
        self,
        config: Qwen2_5_VLTextConfig,
        attention_backend: str = "flash_attn",
        quant_config: AWQConfig | None = None,
    ) -> None:
        super().__init__()
        tp_size = dist.get_world_size()
        self.total_num_heads = config.num_attention_heads
        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_heads % tp_size or self.total_num_kv_heads % tp_size:
            raise ValueError("attention heads must be divisible by tensor parallel size")
        self.num_heads = self.total_num_heads // tp_size
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = config.hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        if quant_config is None:
            self.qkv_proj = QKVParallelLinear(
                config.hidden_size,
                self.head_dim,
                self.total_num_heads,
                self.total_num_kv_heads,
                bias=True,
            )
            self.o_proj = RowParallelLinear(
                self.total_num_heads * self.head_dim,
                config.hidden_size,
                bias=False,
            )
        else:
            self.qkv_proj = AWQQKVParallelLinear(
                config.hidden_size,
                self.head_dim,
                self.total_num_heads,
                self.total_num_kv_heads,
                quant_config,
                bias=True,
            )
            self.o_proj = AWQRowParallelLinear(
                self.total_num_heads * self.head_dim,
                config.hidden_size,
                quant_config,
                bias=False,
            )
        rope_parameters = config.rope_parameters
        self.rotary_emb = Qwen2_5_VLRotaryEmbedding(
            self.head_dim,
            rope_theta=rope_parameters.get("rope_theta", 1000000.0),
            mrope_section=rope_parameters["mrope_section"],
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
            backend=attention_backend,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        query, key, value = qkv.split(
            (self.q_size, self.kv_size, self.kv_size), dim=-1
        )
        query = query.view(-1, self.num_heads, self.head_dim)
        key = key.view(-1, self.num_kv_heads, self.head_dim)
        value = value.view(-1, self.num_kv_heads, self.head_dim)
        query, key = self.rotary_emb(positions, query, key)
        output = self.attn(query, key, value)
        return self.o_proj(output.flatten(1))


class Qwen2_5_VLMLP(nn.Module):

    def __init__(
        self,
        config: Qwen2_5_VLTextConfig,
        quant_config: AWQConfig | None = None,
    ) -> None:
        super().__init__()
        if config.hidden_act != "silu":
            raise ValueError(f"unsupported text activation: {config.hidden_act}")
        if quant_config is None:
            self.gate_up_proj = MergedColumnParallelLinear(
                config.hidden_size,
                [config.intermediate_size] * 2,
                bias=False,
            )
            self.down_proj = RowParallelLinear(
                config.intermediate_size,
                config.hidden_size,
                bias=False,
            )
        else:
            self.gate_up_proj = AWQMergedColumnParallelLinear(
                config.hidden_size,
                [config.intermediate_size] * 2,
                quant_config,
                bias=False,
            )
            self.down_proj = AWQRowParallelLinear(
                config.intermediate_size,
                config.hidden_size,
                quant_config,
                bias=False,
            )
        self.act_fn = SiluAndMul()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_up_proj(hidden_states)))


class Qwen2_5_VLDecoderLayer(nn.Module):

    def __init__(
        self,
        config: Qwen2_5_VLTextConfig,
        attention_backend: str = "flash_attn",
        quant_config: AWQConfig | None = None,
    ) -> None:
        super().__init__()
        self.self_attn = Qwen2_5_VLAttention(
            config,
            attention_backend,
            quant_config,
        )
        self.mlp = Qwen2_5_VLMLP(config, quant_config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual
        )
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen2_5_VLTextModel(nn.Module):

    def __init__(
        self,
        config: Qwen2_5_VLTextConfig,
        attention_backend: str = "flash_attn",
        quant_config: AWQConfig | None = None,
    ) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            Qwen2_5_VLDecoderLayer(config, attention_backend, quant_config)
            for _ in range(config.num_hidden_layers)
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("input_ids or inputs_embeds must be provided")
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = inputs_embeds
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen2_5_VLForConditionalGeneration(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
    }

    def __init__(
        self,
        config: Qwen2_5_VLConfig,
        attention_backend: str = "flash_attn",
        vision_attention_backend: str | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.quant_config = AWQConfig.from_hf_config(config)
        if vision_attention_backend is None:
            vision_attention_backend = attention_backend
        self.attention_backend = attention_backend
        self.vision_attention_backend = vision_attention_backend
        self.packed_modules_mapping = dict(type(self).packed_modules_mapping)
        for layer_idx in range(config.text_config.num_hidden_layers):
            prefix = f"model.layers.{layer_idx}.mlp"
            self.packed_modules_mapping[f"{prefix}.gate_proj"] = (
                f"{prefix}.gate_up_proj",
                0,
            )
            self.packed_modules_mapping[f"{prefix}.up_proj"] = (
                f"{prefix}.gate_up_proj",
                1,
            )
        self.visual = Qwen2_5_VisionTransformer(
            config.vision_config,
            vision_attention_backend,
        )
        self.model = Qwen2_5_VLTextModel(
            config.text_config,
            attention_backend,
            self.quant_config,
        )
        self.lm_head = ParallelLMHead(
            config.text_config.vocab_size,
            config.text_config.hidden_size,
        )
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def encode_images(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        return self.visual(pixel_values, image_grid_thw)

    def _merge_vision_embeddings(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        vision_embeddings: torch.Tensor,
        token_id: int,
    ) -> torch.Tensor:
        mask = input_ids == token_id
        if int(mask.sum().item()) != vision_embeddings.size(0):
            raise ValueError(
                f"vision token count {int(mask.sum().item())} does not match "
                f"vision embedding count {vision_embeddings.size(0)}"
            )
        output = inputs_embeds.clone()
        output[mask] = vision_embeddings.to(output.dtype)
        return output

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
    ) -> torch.Tensor:
        inputs_embeds = self.model.embed_tokens(input_ids)
        if pixel_values is not None:
            if image_grid_thw is None:
                raise ValueError("image_grid_thw is required with pixel_values")
            image_embeddings = self.visual(pixel_values, image_grid_thw)
            inputs_embeds = self._merge_vision_embeddings(
                input_ids,
                inputs_embeds,
                image_embeddings,
                self.config.image_token_id,
            )
        if pixel_values_videos is not None:
            if video_grid_thw is None:
                raise ValueError("video_grid_thw is required with pixel_values_videos")
            video_embeddings = self.visual(pixel_values_videos, video_grid_thw)
            inputs_embeds = self._merge_vision_embeddings(
                input_ids,
                inputs_embeds,
                video_embeddings,
                self.config.video_token_id,
            )
        return self.model(None, positions, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)
