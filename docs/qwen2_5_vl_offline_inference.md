# Qwen2.5-VL 单图离线推理教程

本文说明当前 nano-vllm 中 Qwen2.5-VL 单图离线推理的使用方式、代码链路和面试时容易被问到的设计点。

## 当前支持范围

已支持：

- Qwen2.5-VL-3B-Instruct。
- Qwen2.5-VL-3B-Instruct-AWQ 的 W4A16 文本权重。
- 单 GPU，`tensor_parallel_size=1`。
- 单请求单图片。
- `AutoProcessor` 产出的 `input_ids`、`mm_token_type_ids`、`pixel_values`、`image_grid_thw`。
- Prefill 阶段运行视觉塔，并把 image placeholder token 的 embedding 替换为视觉 embedding。
- Decode 阶段复用 KV Cache，不重复运行视觉塔。
- Qwen2.5-VL 文本侧 3D MRoPE position 和 `mrope_position_delta`。
- `temperature=0.0` 的 greedy 采样，便于和 Transformers 对齐。

暂不支持：

- 多图、多视频、多模态 batching。
- 多模态 prefix cache。
- 多模态 chunked prefill。
- `tensor_parallel_size>1` 的视觉路径。
- CUDA Graph 下的动态视觉输入。

## 运行示例

默认使用仓库里的 `assets/logo.png`：

```bash
/home/agua/anaconda3/envs/yolo26/bin/python examples/qwen2_5_vl_offline.py \
  --engine nano \
  --max-new-tokens 8 \
  --no-tqdm \
  --gpu-memory-utilization 0.72 \
  --max-model-len 1024 \
  --max-num-batched-tokens 1024
```

运行 Transformers 参考：

```bash
/home/agua/anaconda3/envs/yolo26/bin/python examples/qwen2_5_vl_offline.py \
  --engine transformers \
  --max-new-tokens 8 \
  --no-tqdm
```

运行完整对齐基线：

```bash
/home/agua/anaconda3/envs/yolo26/bin/python examples/qwen2_5_vl_alignment.py \
  --max-new-tokens 8 \
  --max-model-len 4096 \
  --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.72
```

这个脚本会检查三层：

- `input_ids`：nano-vllm 使用的 prompt token 必须和 `AutoProcessor` 输出一致。
- MRoPE：nano-vllm 计算的 3D position ids 和 `mrope_position_delta` 必须和 Transformers helper 一致。
- Greedy token：Transformers 和 nano-vllm 生成的前 N 个 token 必须一致。

当前验证结果，两边前 8 个 token 一致：

```text
[108893, 45930, 101987, 104059, 101599, 2073, 83819, 8273]
```

解码文本：

```text
这张图片展示了一个名为“Nano-v
```

Attention 后端现在可以单独配置。文本 Decoder 当前使用 `flash_attn`，视觉
Encoder 可选择 `flash_attn`、`torch_sdpa`、`torch_math`、`cudnn_sdpa`、
`triton` 或 `hybrid`：

```bash
/home/agua/anaconda3/envs/yolo26/bin/python examples/qwen2_5_vl_offline.py \
  --engine nano \
  --max-new-tokens 8 \
  --no-tqdm \
  --max-model-len 4096 \
  --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.72 \
  --attention-backend flash_attn \
  --vision-attention-backend torch_sdpa
```

接口设计、能力边界和 benchmark 方法见：

```text
docs/attention_backends.md
```

## AWQ W4A16 离线推理

本地 AWQ checkpoint：

```text
/home/agua/models/Qwen2.5-VL-3B-Instruct-AWQ
```

运行命令：

```bash
/home/agua/anaconda3/envs/yolo26/bin/python examples/qwen2_5_vl_offline.py \
  --engine nano \
  --model /home/agua/models/Qwen2.5-VL-3B-Instruct-AWQ \
  --image assets/dog.png \
  --max-new-tokens 32 \
  --no-tqdm \
  --gpu-memory-utilization 0.72 \
  --max-model-len 4096 \
  --max-num-batched-tokens 4096
```

第一版支持 `bits=4`、`group_size=128`、asymmetric zero point、单 GPU。视觉塔
仍是 BF16，文本 Decoder 的 Linear 从 packed INT4 checkpoint 加载。详细格式、
kernel 和 BF16/AWQ 性能对比见：

```text
docs/qwen2_5_vl_awq.md
```

## 数据链路

整体链路：

```text
PIL Image + text
    -> AutoProcessor
    -> MultiModalPrompt
    -> Sequence
    -> Scheduler
    -> ModelRunner.prepare_prefill
    -> Qwen2_5_VLForConditionalGeneration.forward
    -> Vision Transformer
    -> image placeholder embedding 替换
    -> Text Decoder + KV Cache
    -> Sampler
```

关键文件：

- `examples/qwen2_5_vl_offline.py`：离线运行入口。
- `examples/qwen2_5_vl_alignment.py`：Transformers 与 nano-vllm 的回归对齐脚本。
- `nanovllm/multimodal.py`：`MultiModalPrompt` 和 Qwen2.5-VL 3D MRoPE position 计算。
- `nanovllm/engine/sequence.py`：保存图片 tensor、grid 和 MRoPE delta。
- `nanovllm/engine/scheduler.py`：限制多模态 prefill 必须一次完成。
- `nanovllm/engine/block_manager.py`：多模态请求禁用 token-only prefix cache。
- `nanovllm/engine/model_runner.py`：prefill 传入 `pixel_values/image_grid_thw`，decode 使用 `mrope_position_delta`。
- `nanovllm/models/qwen2_5_vl.py`：视觉塔、文本 Decoder、embedding 替换和 logits。

## 为什么需要 mm_token_type_ids

Qwen2.5-VL 的输入序列里不仅有 `<|image_pad|>`，还有 `<|vision_start|>`、`<|vision_end|>` 和普通文本 token。当前 Transformers 版本用 `mm_token_type_ids` 标记每个 token 属于：

```text
0: text
1: image
2: video
```

计算 MRoPE 时按连续 modality group 处理，而不是简单扫描 token id。这样可以正确处理：

- 图片前后的文本位置。
- 图片 placeholder 对应的 3D position。
- 图片后文本 position 从视觉区域压缩后的最大位置继续增长。

## Qwen2.5-VL 的 3D MRoPE

文本 token：

```text
temporal == height == width
```

也就是退化成普通 1D RoPE。

图片 token：

```text
temporal: 图像时间轴，静态图通常为 1
height:   patch merge 后的行坐标
width:    patch merge 后的列坐标
```

对于示例图：

```text
image_grid_thw = [1, 36, 74]
spatial_merge_size = 2
image token 数 = 1 * 36 * 74 / 4 = 666
```

但是图片在文本位置上不是占用 666 个连续 1D 位置，而是按：

```text
max(grid_h, grid_w) / spatial_merge_size
```

推进。因此需要保存：

```text
mrope_position_delta = max_position + 1 - input_token_count
```

Decode 阶段新 token 的位置为：

```text
position = 当前序列长度 - 1 + mrope_position_delta
```

然后复制到 temporal、height、width 三个轴。

## 为什么图片只在 Prefill 阶段传入

Prefill 阶段需要处理完整 prompt，其中包含 image placeholder token。模型流程是：

```text
pixel_values -> Vision Transformer -> image_embeddings
input_ids -> token_embeddings
token_embeddings[image_token_mask] = image_embeddings
```

这一步完成后，文本 Decoder 会把视觉信息写入 KV Cache。Decode 阶段每次只输入上一步生成的一个文本 token，直接读取 KV Cache，不需要也不应该重复运行视觉塔。

## 为什么先禁用 prefix cache

原 nano-vllm 的 prefix cache hash 只基于 token ids。多模态请求中，不同图片可能产生完全相同数量的 `<|image_pad|>` token，token ids 相同但视觉 embedding 不同。

如果继续使用 token-only prefix cache，就可能把 A 图片的 KV Cache 复用给 B 图片，结果会错。因此当前实现中：

```text
seq.has_multimodal -> 不查 prefix cache，不写 prefix cache hash
```

后续如果要支持多模态 prefix cache，hash 必须包含图片内容 hash、processor 输出 hash 或视觉 embedding 标识。

## 为什么先禁用 chunked prefill

当前模型内 `_merge_vision_embeddings()` 会检查：

```text
当前 input_ids 里的 image token 数 == vision embedding 数
```

如果 chunked prefill 把图片 placeholder 切成多段，某一段只包含部分 image token，但视觉塔仍输出完整图片 embedding，数量就对不上。第一版为了正确性，要求带图片 prompt 一次性 prefill 完成。

## 面试常见追问

### 1. 你这个和直接调用官方 vLLM 有什么区别？

这里不是包装官方 vLLM，而是在 nano-vllm 里自己接了模型结构、权重映射、视觉塔、MRoPE、KV Cache 调度和采样链路。重点是理解 VLM 推理系统内部数据如何从 processor 进入 prefill，再进入 decode。

### 2. 为什么 VLM 的 position ids 是 3 维？

因为图片 token 不是普通一维文本序列。Qwen2.5-VL 对视觉 token 使用 temporal、height、width 三个坐标，让 RoPE 能表达图像二维空间和视频时间信息。纯文本 token 三个轴相同，所以兼容普通文本 RoPE。

### 3. 为什么 image token 数是 `grid_t * grid_h * grid_w / 4`？

Qwen2.5-VL 的视觉塔使用 `spatial_merge_size=2`，也就是每 `2x2` 个视觉 patch merge 成一个 LLM 侧视觉 token，所以除以 `2^2=4`。

### 4. Decode 时为什么不再传图片？

图片信息已经在 prefill 中通过视觉 embedding 注入文本 Decoder，并写入 KV Cache。Decode 每步只生成一个新文本 token，attention 会读之前的 KV Cache，因此重复跑视觉塔既浪费也会破坏标准增量生成流程。

### 5. 为什么多模态 prefix cache 有风险？

因为 token 序列只包含 `<|image_pad|>`，不同图片的 token ids 可能完全一样。如果 hash 只看 token，就无法区分图片内容。正确做法是把图片内容或视觉特征也纳入 prefix hash。

### 6. 你怎么验证正确性？

先做分层验证：

- processor 输出字段和 token 数。
- MRoPE positions 与 Transformers 完全一致。
- image placeholder 数量与 vision embedding 数量一致。
- nano-vllm 与 Transformers 的 greedy token 对齐。

当前示例中，前 8 个生成 token 已与 Transformers 完全一致。

### 7. 下一步性能优化应该做什么？

先保持这个 BF16 正确性基线，再做：

- Attention backend 抽象和 benchmark。
- 区分 vision attention、text prefill attention、decode paged attention。
- AWQ W4A16 权重加载与 Linear。
- 使用 Nsight 找真实热点后再做 1 到 2 个 kernel 融合。
