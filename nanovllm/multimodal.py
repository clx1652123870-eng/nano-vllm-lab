from dataclasses import dataclass
from itertools import groupby
from typing import Any

import torch


@dataclass(slots=True)
class MultiModalPrompt:
    input_ids: list[int]
    mm_token_type_ids: list[int] | torch.Tensor
    pixel_values: torch.Tensor | None = None
    image_grid_thw: torch.Tensor | None = None
    attention_mask: list[int] | torch.Tensor | None = None

    @classmethod
    def from_processor(cls, inputs: Any, index: int = 0) -> "MultiModalPrompt":
        input_ids = _select_1d(inputs["input_ids"], index).tolist()
        mm_token_type_ids = _select_1d(inputs["mm_token_type_ids"], index)
        attention_mask = (
            _select_1d(inputs["attention_mask"], index)
            if "attention_mask" in inputs
            else None
        )

        pixel_values = inputs.get("pixel_values")
        image_grid_thw = inputs.get("image_grid_thw")
        if pixel_values is not None or image_grid_thw is not None:
            batch_size = _batch_size(inputs["input_ids"])
            if batch_size != 1 or index != 0:
                raise ValueError("first multimodal version only supports batch size 1")
            if pixel_values is None or image_grid_thw is None:
                raise ValueError("pixel_values and image_grid_thw must be provided together")

        return cls(
            input_ids=input_ids,
            mm_token_type_ids=mm_token_type_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            attention_mask=attention_mask,
        )


def compute_qwen2_5_vl_mrope_positions(
    input_ids: list[int] | torch.Tensor,
    mm_token_type_ids: list[int] | torch.Tensor,
    image_grid_thw: torch.Tensor | None,
    spatial_merge_size: int,
    attention_mask: list[int] | torch.Tensor | None = None,
) -> tuple[torch.Tensor, int]:
    input_ids = _to_1d_long(input_ids)
    mm_token_type_ids = _to_1d_long(mm_token_type_ids)
    if input_ids.numel() != mm_token_type_ids.numel():
        raise ValueError("input_ids and mm_token_type_ids must have the same length")

    if attention_mask is None:
        active_mask = torch.ones_like(input_ids, dtype=torch.bool)
    else:
        active_mask = _to_1d_long(attention_mask).bool()
        if active_mask.numel() != input_ids.numel():
            raise ValueError("attention_mask must have the same length as input_ids")

    active_token_types = mm_token_type_ids[active_mask]
    image_iter = iter(image_grid_thw.long()) if image_grid_thw is not None else None

    current_pos = 0
    chunks: list[torch.Tensor] = []
    for modality_type, group in groupby(
        enumerate(active_token_types.tolist()), lambda item: item[1]
    ):
        group = list(group)
        length = len(group)
        if modality_type == 0:
            chunks.append(torch.arange(length, dtype=torch.long).view(1, -1).expand(3, -1) + current_pos)
            current_pos += length
        elif modality_type == 1:
            if image_iter is None:
                raise ValueError("image_grid_thw is required for image tokens")
            grid_thw = next(image_iter)
            chunks.append(
                _get_vision_position_ids(
                    current_pos,
                    grid_thw,
                    spatial_merge_size,
                )
            )
            current_pos += max(int(grid_thw[1]), int(grid_thw[2])) // spatial_merge_size
        else:
            raise ValueError(
                "first multimodal version supports text/image token types only"
            )

    if not chunks:
        positions = torch.zeros(3, input_ids.numel(), dtype=torch.long)
        return positions, 0

    active_positions = torch.cat(chunks, dim=1)
    if active_positions.size(1) != int(active_mask.sum().item()):
        raise ValueError(
            "computed MRoPE length does not match active input token count"
        )

    positions = torch.zeros(3, input_ids.numel(), dtype=torch.long)
    positions[:, active_mask] = active_positions
    mrope_position_delta = int(active_positions.max().item() + 1 - active_positions.size(1))
    return positions, mrope_position_delta


def _get_vision_position_ids(
    start_position: int,
    grid_thw: torch.Tensor,
    spatial_merge_size: int,
) -> torch.Tensor:
    llm_grid_t = int(grid_thw[0])
    llm_grid_h = int(grid_thw[1]) // spatial_merge_size
    llm_grid_w = int(grid_thw[2]) // spatial_merge_size

    position_temporal = torch.arange(llm_grid_t, dtype=torch.long)
    position_height = torch.arange(llm_grid_h, dtype=torch.long) + start_position
    position_width = torch.arange(llm_grid_w, dtype=torch.long) + start_position

    position_width = position_width.repeat(llm_grid_h * llm_grid_t)
    position_height = position_height.repeat_interleave(llm_grid_w).repeat(llm_grid_t)
    position_temporal = position_temporal.repeat_interleave(llm_grid_h * llm_grid_w) + start_position
    return torch.stack([position_temporal, position_height, position_width], dim=0)


def _select_1d(value: torch.Tensor, index: int) -> torch.Tensor:
    if value.ndim == 1:
        if index != 0:
            raise ValueError("cannot select a non-zero index from an unbatched tensor")
        return value.detach().cpu().long()
    if value.ndim == 2:
        return value[index].detach().cpu().long()
    raise ValueError(f"expected a 1D or 2D tensor, got shape {tuple(value.shape)}")


def _to_1d_long(value: list[int] | torch.Tensor) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().long()
    else:
        value = torch.tensor(value, dtype=torch.long)
    if value.ndim != 1:
        raise ValueError(f"expected a 1D value, got shape {tuple(value.shape)}")
    return value


def _batch_size(value: torch.Tensor) -> int:
    return 1 if value.ndim == 1 else int(value.size(0))
