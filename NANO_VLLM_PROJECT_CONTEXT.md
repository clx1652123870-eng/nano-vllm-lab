# Nano-vLLM 项目上下文与后续开发交接

> 更新时间：2026-07-28
> 用途：在新的 Codex 对话中快速恢复项目背景、代码现状、技术决策和后续计划。
> 工作目录：`/home/agua/tensorrtlearning/nano-vllm`

## 1. 项目当前状态摘要

这个项目以轻量级推理框架 nano-vllm 为基础，目标不是简单调用官方 vLLM，而是通过亲自实现模型、调度、Attention、量化和算子优化来理解大模型/VLM推理系统。

当前已经完成：

- Qwen2.5-VL-3B-Instruct 的模型层结构。
- Qwen2.5-VL 视觉编码器。
- 视觉 Window/Full FlashAttention。
- 文本 Decoder、GQA、KV Cache Attention。
- 文本侧 3D MRoPE。
- 视觉 embedding 替换 image/video placeholder token 的模型内逻辑。
- Qwen3/Qwen2.5-VL 模型注册表。
- Qwen2.5-VL checkpoint 权重名称映射。
- `Config` 和 `ModelRunner` 对 VLM `text_config` 的适配。
- 模型结构、权重映射、MRoPE 和视觉 FlashAttention 数值验证。
- Processor 到 `Sequence/ModelRunner` 的单图多模态数据通路。
- Qwen2.5-VL 3D MRoPE position 和 Decode delta 管理。
- Qwen2.5-VL 单图离线推理。
- Transformers 与 nano-vllm 的 greedy token 对齐基线。
- 离线 TTFT、TPOT、吞吐和显存 profiling。
- FastAPI 单图非流式在线接口。
- `AsyncLLMEngine` 请求队列、Future 和专用 engine thread。
- Qwen2.5-VL 单图 prefill、批量 decode 的 continuous batching。
- 在线 C1/CN profiling 和 JSON 报告。
- SSE 逐 token 流式输出。
- 客户端断开/超时后的请求取消和 KV Cache 回收。
- HTTP 最大并发限制和 429 backpressure。
- OpenAI 兼容的 `/v1/chat/completions` 非流式/流式接口。
- C1/C2/C4 greedy token、decode batch 和系统吞吐回归脚本。

当前尚未完成：

- 同一个 prefill batch 中处理多张图片。
- API 进程与 EngineCore 进程拆分。
- AWQ INT4 权重加载和量化 Linear。
- Attention 后端统一抽象与系统 benchmark。
- CUDA/Triton 关键算子融合。

因此当前准确状态是：

```text
Qwen2.5-VL 模型层与加载基础：已完成
Qwen2.5-VL 文本路径基础：已接入
Qwen2.5-VL 图片端到端离线推理：已完成
Transformers/nano-vllm 离线对齐：已完成
nano-vllm 在线非流式推理：已完成
Async request queue/Future：已完成
单图 prefill + continuous decode batching：已完成
SSE/OpenAI API/取消/超时/并发保护：已实现
C1/C2/C4 自动回归：脚本和真实 Qwen2.5-VL 回归均已完成
EngineCore 进程拆分：未实现
AWQ/算子优化：未开始正式编码
```

## 2. 最终项目目标

项目最终聚焦四部分。

### 2.1 Qwen2.5-VL 端到端支持

目标模型：

```text
/home/agua/models/Qwen2.5-VL-3B-Instruct
```

计划完成：

- 使用 `AutoProcessor` 处理文本和图片。
- 支持单请求单图片，后续再扩展多图片/视频。
- 将 `pixel_values`、`image_grid_thw` 和视觉 token 元数据传入引擎。
- Prefill 阶段运行视觉编码器。
- 将视觉 embedding 替换 `<|image_pad|>` token embedding。
- 为文本、图片和视频 token 构造 Qwen2.5-VL 3D MRoPE position。
- 保存 `mrope_position_delta`，供 Decode 阶段继续生成位置。
- Decode 阶段不重复运行视觉编码器，只使用文本 Decoder 和 KV Cache。
- 与 Transformers 参考输出进行 logits/token 级正确性验证。
- 完成单图离线生成。

### 2.2 Attention 后端对比与优化

计划支持或对比：

- PyTorch SDPA。
- FlashAttention 2。
- cuDNN SDPA。

重点分析：

- Q/K/V 输入 layout。
- GQA 的 Q head 与 KV head 数量关系。
- Packed varlen 输入。
- `cu_seqlens` 的含义。
- causal mask 和 non-causal mask。
- Bottom-right causal 对齐。
- Vision Window Attention。
- LLM Paged KV Cache。
- Prefill 与 Decode 的不同输入格式。

性能维度：

- 不同 batch size。
- 不同 prompt/sequence length。
- Prefill latency 和 token throughput。
- Decode latency、TPOT 和 token throughput。
- GPU 显存占用。
- 数值误差。

### 2.3 AWQ INT4 量化推理

计划优先支持官方已经量化好的 checkpoint，而不是先实现完整 AWQ 量化工具。

推荐目标：

```text
Qwen/Qwen2.5-VL-3B-Instruct-AWQ
```

标准 AWQ 路径：

```text
BF16/FP16 原始模型
    -> 离线校准和 AWQ 权重量化
    -> qweight/qzeros/scales checkpoint
    -> nano-vllm 加载
    -> W4A16 推理
```

第一阶段计划：

- 读取 `quantization_config`。
- 识别 `quant_method="awq"`。
- 加载 packed INT4 `qweight`。
- 加载 `qzeros` 和 `scales`。
- 支持 `group_size=128`。
- 实现正确性基线：即时反量化后调用 Linear。
- 再实现融合的 AWQ Linear CUDA/Triton kernel。

AWQ W4A16 含义：

- Weight：INT4。
- Activation：FP16/BF16。
- Accumulator：通常 FP32。
- 输出：FP16/BF16。
- 标准 W4A16 不需要 activation scale。
- 权重需要 group-wise weight scale，非对称量化还需要 zero point。

推理时不是生成完整 FP32 权重矩阵，而是在 kernel 内按块即时反量化权重，并与 FP16/BF16 activation 完成矩阵乘。

### 2.4 Profiling 与算子融合

工具：

- Nsight Systems：观察端到端时间线、CPU/GPU 间隙和 kernel launch。
- Nsight Compute：观察单 kernel 的访存、occupancy、Tensor Core 和瓶颈。
- PyTorch Profiler：快速定位 Python/PyTorch 路径热点。

候选融合点：

- RMSNorm + QKV Projection。
- QKV Projection + RoPE。
- Gate Projection + Up Projection + SiLU + Mul。
- INT4 反量化 + GEMM。
- Attention 前后的 layout 转换。
- RoPE + KV Cache 写入。
- KV Cache 写入相关小 kernel。

三周范围内不追求大量 kernel。目标是：

```text
定位 1~2 个真实热点
    -> 实现融合版本
    -> 替换原路径
    -> 测量 Prefill/Decode
    -> 测量端到端 VLM 收益
```

## 3. 建议的三周范围

硬件只有一张 16GB GPU，因此需要控制范围。

### 必做范围

- 单 GPU。
- `tensor_parallel_size=1` 先打通。
- 单请求单图片。
- Qwen2.5-VL-3B。
- BF16 正确性基线。
- 离线推理优先。
- 一个可复现的 Attention benchmark。
- 一个 AWQ W4A16 推理路径。
- 一个或两个融合/优化 kernel。
- 端到端性能结果。

### 暂不作为第一目标

- 多机。
- 完整 DP/PP/TP/PCP 支持。
- 多图片复杂 batching。
- 完整视频服务。
- 完整 OpenAI API 兼容。
- 同时实现多种 INT8/INT4/FP8 格式。
- 自己从零实现 AWQ 校准器。
- 大量模型注册。

### 可选扩展

- FastAPI 在线接口。
- OpenAI Chat Completions 兼容。
- 多图片或视频。
- 视觉塔 Tensor Parallel。
- CUDA Graph。
- Prefix Cache 对多模态请求的支持。

## 4. 当前环境

### 4.1 路径

```text
项目：
/home/agua/tensorrtlearning/nano-vllm

Qwen2.5-VL 模型：
/home/agua/models/Qwen2.5-VL-3B-Instruct

Qwen3 模型：
/home/agua/models/Qwen3-1.7B

Python：
/home/agua/anaconda3/envs/yolo26/bin/python
```

### 4.2 软件版本

```text
Python        3.10.20
PyTorch       2.11.0+cu130
Transformers  5.12.1
Triton        3.6.0
flash-attn    2.8.3.post1
PyTorch CUDA  13.0
CUDA Toolkit  13.1
```

### 4.3 GPU

此前已验证的设备：

```text
NVIDIA RTX 5080
显存约 16GB
Compute Capability 12.0
架构 sm_120
```

### 4.4 flash-attn 安装背景

直接执行：

```bash
MAX_JOBS=4 python -m pip install flash-attn --no-build-isolation
```

最初失败，因为安装脚本猜测了一个不存在的预编译 wheel，并且网络连接关闭。

最终使用源码针对 `sm_120` 编译成功。核心环境变量：

```bash
CUDA_HOME=/usr/local/cuda-13.1 \
FLASH_ATTENTION_FORCE_BUILD=TRUE \
FLASH_ATTN_CUDA_ARCHS=120 \
MAX_JOBS=2 \
NVCC_THREADS=2 \
python -m pip install \
  /tmp/flash-attn-src/flash_attn-2.8.3.post1.tar.gz \
  --no-build-isolation -v
```

已验证以下接口可以在 GPU 上运行：

```python
from flash_attn import flash_attn_varlen_func
from flash_attn import flash_attn_with_kvcache
```

还验证过 PyTorch 的 cuDNN SDPA 和 GQA 路径可以在 RTX 5080 上运行。

## 5. Git 仓库状态

远程仓库：

```text
git@github.com:clx1652123870-eng/nano-vllm-lab.git
```

当前分支：

```text
main
```

当前已提交基线：

```text
f85cf92 test
```

当前 AsyncLLM、continuous batching 和在线文档修改尚未 commit。提交前检查：

```text
git status --short --branch
git diff --check
python -m unittest discover -s tests -p 'test_async_llm_engine.py' -v
```

不要自动提交 `profiles/` 中的临时 benchmark，除非明确需要保留为性能基线。

## 6. 原始 nano-vllm 架构

原项目是一个轻量级、同步、离线推理框架，主要支持 Qwen3 和 Tensor Parallel。

主要文件：

```text
nanovllm/llm.py
    对外 LLM 类型

nanovllm/engine/llm_engine.py
    请求进入、tokenizer、主调度循环

nanovllm/engine/sequence.py
    单个文本序列状态

nanovllm/engine/scheduler.py
    Prefill/Decode 调度

nanovllm/engine/block_manager.py
    KV Cache block 分配和 prefix cache

nanovllm/engine/model_runner.py
    模型构造、权重加载、输入准备、CUDA Graph、执行

nanovllm/utils/context.py
    当前 batch 的 Attention 元数据

nanovllm/layers/attention.py
    Decoder Attention、KV Cache 写入、FlashAttention

nanovllm/models/qwen3.py
    原始唯一模型实现
```

原始离线推理流程：

```text
LLM.generate()
    -> LLMEngine.add_request()
    -> tokenizer.encode()
    -> Sequence
    -> Scheduler.schedule()
    -> ModelRunner.run()
    -> prepare_prefill() 或 prepare_decode()
    -> set_context()
    -> model.forward()
    -> Attention
    -> compute_logits()
    -> Sampler
    -> Sequence.append_token()
```

原项目不包含：

- HTTP server。
- OpenAI API。
- AsyncLLM。
- 多模态 Processor。
- 图片字段。
- 模型注册表。
- 量化注册表。

## 7. 原始 Decoder Attention 路径

文件：

```text
nanovllm/layers/attention.py
```

当前 Q/K/V layout：

```text
Q: [total_tokens, num_q_heads, head_dim]
K: [total_tokens, num_kv_heads, head_dim]
V: [total_tokens, num_kv_heads, head_dim]
```

KV Cache layout：

```text
[num_blocks, block_size, num_kv_heads, head_dim]
```

整个 KV Cache：

```text
[2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]
```

其中第一维的 `2` 分别表示 K 和 V。

Decoder Prefill：

```python
flash_attn_varlen_func(
    q,
    k,
    v,
    cu_seqlens_q=...,
    cu_seqlens_k=...,
    causal=True,
    block_table=...,
)
```

Decoder Decode：

```python
flash_attn_with_kvcache(
    q.unsqueeze(1),
    k_cache,
    v_cache,
    cache_seqlens=...,
    block_table=...,
    causal=True,
)
```

KV Cache 写入使用 Triton：

```text
store_kvcache_kernel
```

## 8. Qwen2.5-VL 模型配置

本地 config 解析后：

### 8.1 文本模型

```text
hidden_size             2048
num_hidden_layers       36
num_attention_heads     16
num_key_value_heads     2
intermediate_size       11008
max_position_embeddings 128000
dtype                   bfloat16
```

这是 GQA：

```text
16 个 Q heads
2 个 KV heads
每 8 个 Q heads 共享一组 K/V
head_dim = 2048 / 16 = 128
```

文本 MRoPE：

```text
rope_theta    1000000
mrope_section [16, 24, 24]
```

### 8.2 视觉模型

```text
hidden_size          1280
depth                32
num_heads            16
intermediate_size    3420
patch_size           14
temporal_patch_size  2
spatial_merge_size   2
window_size          112
out_hidden_size      2048
fullatt_block_indexes [7, 15, 23, 31]
```

视觉 head dim：

```text
1280 / 16 = 80
```

### 8.3 特殊 token

```text
vision_start_token_id 151652
vision_end_token_id   151653
image_token_id        151655
video_token_id        151656
```

### 8.4 高层计算流程

```text
原始图片
    -> AutoProcessor
    -> pixel_values + image_grid_thw
    -> 3D Conv PatchEmbed
    -> 32 层 Vision Transformer
    -> Window/Full non-causal Attention
    -> Patch Merger
    -> [num_visual_tokens, 2048]
    -> 替换 image placeholder embedding
    -> 36 层文本 Decoder
    -> LM Head
```

## 9. 已完成的代码修改

### 9.1 新增 Qwen2.5-VL 模型

文件：

```text
nanovllm/models/qwen2_5_vl.py
```

注意：

```text
qwen2.5vl.py
```

不是合适的 Python 模块名，因为 `.` 会被当作包层级分隔。文件已使用：

```text
qwen2_5_vl.py
```

实现的主要类：

```text
Qwen2_5_VisionRMSNorm
Qwen2_5_VLRotaryEmbedding
Qwen2_5_VisionPatchEmbed
Qwen2_5_VisionRotaryEmbedding
Qwen2_5_VisionAttention
Qwen2_5_VisionMLP
Qwen2_5_VisionBlock
Qwen2_5_VisionPatchMerger
Qwen2_5_VisionTransformer
Qwen2_5_VLAttention
Qwen2_5_VLMLP
Qwen2_5_VLDecoderLayer
Qwen2_5_VLTextModel
Qwen2_5_VLForConditionalGeneration
```

实现的视觉辅助逻辑：

```text
get_vision_position_ids
get_vision_cu_seqlens
get_vision_window_index
apply_vision_rotary_emb
```

模型 `forward()` 已经接受：

```python
pixel_values
image_grid_thw
pixel_values_videos
video_grid_thw
```

也已经实现：

```text
视觉编码
    -> 检查视觉 token 数量
    -> 替换 input_ids 中 image/video token 对应 embedding
    -> 进入文本模型
```

但是现有 `ModelRunner` 尚未向该 `forward()` 传递这些字段。

### 9.2 新增模型注册表

文件：

```text
nanovllm/models/registry.py
```

当前注册：

```python
MODEL_REGISTRY = {
    "Qwen3ForCausalLM": Qwen3ForCausalLM,
    "Qwen2_5_VLForConditionalGeneration":
        Qwen2_5_VLForConditionalGeneration,
}
```

加载流程：

```text
config.json architectures
    -> AutoConfig
    -> get_model_class()
    -> 模型类
```

原始 nano-vllm 没有模型注册表，因为它直接硬编码：

```python
self.model = Qwen3ForCausalLM(hf_config)
```

模型注册表是本项目为了支持多个模型新增的，不是原仓库已有功能。

### 9.3 修改 Config

文件：

```text
nanovllm/config.py
```

原因：

Qwen3 的参数直接位于：

```python
hf_config.hidden_size
hf_config.num_hidden_layers
```

Qwen2.5-VL 的文本配置位于：

```python
hf_config.text_config.hidden_size
hf_config.text_config.num_hidden_layers
```

新增：

```python
@property
def text_config(self):
    return getattr(self.hf_config, "text_config", self.hf_config)
```

这样 Qwen3 和 VLM 都可以通过统一的 `config.text_config` 读取文本 Decoder 配置。

### 9.4 修改 ModelRunner

文件：

```text
nanovllm/engine/model_runner.py
```

修改内容：

- 使用模型注册表选择模型。
- 不再硬编码 `Qwen3ForCausalLM`。
- 使用 `config.text_config` 分配 KV Cache。
- 使用 `config.text_config.hidden_size` 分配 CUDA Graph 输出。
- VLM 顶层配置仍用于构造完整模型。

核心变化：

```python
self.model = get_model_class(hf_config)(hf_config)
```

### 9.5 依赖

项目 `pyproject.toml` 已声明 `xxhash`，但当前环境最初没有安装。

已经执行：

```bash
python -m pip install xxhash
```

安装版本：

```text
xxhash 3.8.1
```

## 10. 权重加载设计

官方 checkpoint 主要文本权重名称：

```text
model.layers.N.self_attn.q_proj.weight
model.layers.N.self_attn.k_proj.weight
model.layers.N.self_attn.v_proj.weight
model.layers.N.mlp.gate_proj.weight
model.layers.N.mlp.up_proj.weight
```

nano-vllm 文本模型使用融合参数：

```text
qkv_proj
gate_up_proj
```

因此进行了映射：

```text
q_proj -> qkv_proj shard q
k_proj -> qkv_proj shard k
v_proj -> qkv_proj shard v

gate_proj -> gate_up_proj shard 0
up_proj   -> gate_up_proj shard 1
```

视觉 checkpoint 本身已经有：

```text
visual.blocks.N.attn.qkv
```

视觉 MLP 当前保持：

```text
gate_proj
up_proj
down_proj
```

没有在正确性基线里融合视觉 `gate_proj/up_proj`。原因是融合 GEMM 可能改变 BF16 kernel 选择和舍入顺序。视觉融合留到后续优化阶段，并且必须重新进行端到端数值验证。

为了避免全局 `gate_proj` 映射误伤视觉 MLP，文本 MLP 映射被限制到：

```text
model.layers.N.mlp.*
```

## 11. MRoPE 关键细节

Qwen2.5-VL 使用 temporal、height、width 三个位置维度。

纯文本 token：

```text
temporal position == height position == width position
```

因此纯文本退化为普通 1D RoPE。

视觉 token：

```text
temporal position
height position
width position
```

分别进入 head dimension 的不同区间。

一个重要实现细节：

```python
mrope_section * 2
```

在参考实现中是 Python 列表重复：

```python
[16, 24, 24] * 2
==
[16, 24, 24, 16, 24, 24]
```

不是：

```python
[32, 48, 48]
```

这个问题在验证时已经发现并修正。修正后，当前文本 MRoPE 与 Transformers 参考实现逐元素完全一致。

## 12. 为什么视觉 Attention 没有直接复用 Decoder Attention

文本模型已经复用：

```text
nanovllm/layers/attention.py::Attention
```

视觉模型目前在：

```text
nanovllm/models/qwen2_5_vl.py::Qwen2_5_VisionAttention
```

中直接调用：

```python
flash_attn_varlen_func
```

原因是现有公共 `Attention` 是 Decoder 专用抽象，它依赖：

- 全局请求 `Context`。
- `slot_mapping`。
- Paged KV Cache。
- Prefix Cache。
- Prefill/Decode 分支。
- `block_table`。
- `causal=True`。

而视觉 Attention：

- `causal=False`。
- 不需要 KV Cache。
- 没有逐 token Decode。
- 每层可能使用 Window 或 Full Attention。
- 使用视觉自己的 `cu_seqlens`。

因此不能直接复用现有 Decoder `Attention`。

后续重构建议：

```text
layers/attention.py
    DecoderAttention
        Prefill
        Decode
        Paged KV Cache

    EncoderAttention
        non-causal varlen
        Window Attention
        Full Attention
```

再将 PyTorch、FlashAttention、cuDNN 作为可切换 backend。

## 13. 已完成的验证

### 13.1 Checkpoint 参数映射

本地 checkpoint：

```text
/home/agua/models/Qwen2.5-VL-3B-Instruct
```

验证结果：

```text
checkpoint tensor 数量 824
经过映射后缺失参数数量 0
```

### 13.2 模型参数量

Meta device 构造结果：

```text
4,065,787,904 parameters
```

这里包含：

- 文本模型。
- 视觉编码器。
- Patch Merger。
- LM Head/tied embedding。

### 13.3 文本 MRoPE

随机 Q/K 和 3D position 与 Transformers 对比：

```text
Q max diff 0
K max diff 0
exact Q True
exact K True
```

### 13.4 视觉 FlashAttention

使用相同前两层官方权重，并让 Transformers 和 nano-vllm 都使用 FlashAttention 2：

```text
max_abs_diff  0
mean_abs_diff 0
cosine        1.0
```

这验证了：

- QKV layout。
- 视觉 RoPE。
- Window reorder。
- `cu_seqlens`。
- FlashAttention 调用。
- MLP。
- Patch Merger 相关结构。

使用不同 backend 比较完整 32 层 ViT 时，SDPA 与 FlashAttention 的单层 BF16 微小差异会被深层网络放大。因此跨 backend 的最终 embedding 不应要求逐元素相等；正确性对比必须区分：

```text
同后端结构验证
跨后端误差评估
端到端生成一致性
```

### 13.5 静态检查

已执行：

```bash
python -m py_compile $(rg --files nanovllm -g '*.py')
git diff --check
```

均通过。

### 13.6 模型注册表

验证结果：

```text
Qwen2.5-VL config
    -> Qwen2_5_VLForConditionalGeneration

Qwen3 config
    -> Qwen3ForCausalLM
```

## 14. 历史记录：当时尚未打通的多模态链路

> 本节记录实现前的缺口，当前这些链路已经打通。实际实现以
> `nanovllm/multimodal.py`、`Sequence`、`Scheduler`、`ModelRunner` 和
> `docs/qwen2_5_vl_offline_inference.md` 为准。

现有 `LLMEngine.add_request()` 只接受：

```python
prompt: str | list[int]
```

现有 `Sequence` 只保存：

- token IDs。
- sampling params。
- KV Cache block table。
- scheduled token 数量。
- cached token 数量。

它没有保存：

- `pixel_values`。
- `image_grid_thw`。
- `video_grid_thw`。
- multimodal placeholder range。
- 3D positions。
- `mrope_position_delta`。
- 预计算视觉 embedding。

现有 `ModelRunner.prepare_prefill()` 只返回：

```text
input_ids
1D positions
```

现有模型调用：

```python
self.model(input_ids, positions)
```

并没有传入图片。因此当前不能认为已经支持图片推理。

## 15. 历史实施方案：单图离线推理（已完成）

建议按照以下顺序改造，先不做 HTTP。

### 15.1 定义多模态请求类型

可以新增类似：

```python
@dataclass
class MultiModalPrompt:
    input_ids: list[int]
    pixel_values: torch.Tensor | None
    image_grid_thw: torch.Tensor | None
    positions: torch.Tensor
    mrope_position_delta: int
```

也可以先把字段直接加入 `Sequence`，但数据类型单独建模会更清楚。

### 15.2 使用 AutoProcessor

在引擎外层加载：

```python
from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained(model_path)
```

输入消息示例：

```python
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "描述这张图片"},
        ],
    }
]
```

Processor 应产生：

```text
input_ids
pixel_values
image_grid_thw
```

需要验证当前 Transformers 5.12.1 返回的额外字段，例如：

```text
attention_mask
mm_token_type_ids
```

不要假设字段，先打印真实输出。

### 15.3 扩展 Sequence

至少保存：

```text
pixel_values
image_grid_thw
mrope_positions
mrope_position_delta
vision_processed
```

注意多进程/共享内存时 Tensor 的序列化和设备位置。

初版建议：

- Processor 输出保存在 CPU。
- ModelRunner Prefill 时搬到 GPU。
- 图片只在 Prefill 使用一次。
- Decode 不重复发送大 Tensor。

### 15.4 计算 Prefill 3D positions

Prefill position 不能继续只使用：

```python
range(start, end)
```

混合文本和图片时要按 Qwen2.5-VL MRoPE 规则生成：

```text
[3, total_tokens]
```

需要记录：

```text
mrope_position_delta
```

### 15.5 Decode positions

生成新文本 token 时，位置不是简单使用原始 token 数量。

需要类似：

```text
decode_position = sequence_length + mrope_position_delta
```

并复制到 temporal/height/width 三个轴：

```text
[3, batch_size]
```

必须与 Transformers 的 generation 逻辑验证。

### 15.6 扩展 ModelRunner

`prepare_prefill()` 需要准备：

```text
input_ids
positions
pixel_values
image_grid_thw
视觉 token 与请求的对应关系
```

`run_model()` 在 Prefill 时调用：

```python
self.model(
    input_ids,
    positions,
    pixel_values=pixel_values,
    image_grid_thw=image_grid_thw,
)
```

Decode 时仍然调用：

```python
self.model(input_ids, positions)
```

### 15.7 多请求 batching

当前实现限制：

```text
每个 Prefill step 只允许一个带图片请求
多个已完成 Prefill 的请求允许一起 Decode
```

后续再处理：

- 多请求图片拼接。
- `image_grid_thw` 拼接。
- 每个请求 placeholder 范围。
- 多图片视觉 embedding 拆分。
- Chunked Prefill。
- Prefix Cache。

### 15.8 正确性验证

固定：

- 同一图片。
- 同一 prompt。
- greedy decoding。
- 相同 dtype。
- 相同最大 token 数。

对比：

```text
Transformers
nano-vllm
```

分层验证：

```text
Processor input_ids
视觉 embedding
Prefill hidden states
最后 token logits
首个生成 token
完整生成 token 序列
```

## 16. 在线服务现状

当前已经实现：

- FastAPI `POST /generate`。
- FastAPI `POST /generate_stream`。
- OpenAI `POST /v1/chat/completions`，支持非流式和 SSE。
- `image_path` 和 `image_base64`。
- 单进程、单 GPU、单图。
- 模型和 Processor 启动时加载一次。
- `AsyncLLMEngine` 专用 engine thread。
- 线程安全 request queue。
- `Future + asyncio.wrap_future()` 返回结果。
- `AsyncEngineStreamEvent` 和 per-request `asyncio.Queue` 返回 token。
- `loop.call_soon_threadsafe()` 跨线程投递 stream event。
- `max_num_seqs` 活跃请求上限。
- `max_concurrent_requests` HTTP in-flight 上限和 429。
- 覆盖预处理、排队和生成的总请求 timeout。
- 客户端断开、timeout 和显式 cancel。
- `Scheduler.abort(seq_id)` 从 waiting/running 删除 sequence。
- `BlockManager.deallocate()` 归还已取消 sequence 的 KV Cache blocks。
- 每步单图 prefill。
- 多请求 decode continuous batching。
- TTFT、TPOT、batch size、requests/s 和 output tokens/s profiling。
- JSON 原始结果和聚合报告。
- C1/C2/C4 exact greedy token、decode batch 和吞吐矩阵报告。

当前架构：

```text
HTTP/FastAPI event loop
    -> asyncio.to_thread(Processor)
    -> ConcurrencyLimiter
    -> AsyncLLMEngine 请求队列
    -> 非流式 Future / 流式 asyncio.Queue

nanovllm-engine thread
    -> 接纳最多 max_num_seqs 个请求
    -> LLMEngine.add_request()
    -> LLMEngine.step_with_metadata()
    -> Scheduler
    -> 单图 Prefill / 多请求 Decode
    -> step metadata(seq_id/token_id/finish_reason)
    -> 按 seq_id 完成 Future 或发送 token event

取消路径
    -> AsyncLLMEngine.cancel(request_id)
    -> LLMEngine.abort_request(seq_id)
    -> Scheduler.abort(seq_id)
    -> BlockManager.deallocate(seq)
```

Qwen2.5-VL 当前 batching 约束：

```text
Prefill:
    每个 step 只允许一个带图片请求

Decode:
    已完成 prefill 的多个请求可以组成 batch
```

`max_num_seqs` 同时作为 engine admission capacity，超过容量的请求继续保留在外部
queue，避免 scheduler 内部请求无界增长和 decode 饥饿。

性能判断不能只看平均请求延迟。C1/CN 对比必须保持相同 prompt、图片、
`max_new_tokens`、warmup 和 measure 次数，并同时比较：

```text
client/server E2E
queue wait
request/engine TTFT
decode TPOT
mean/max decode batch size
整轮 requests/s
整轮 output tokens/s
```

详细说明：

```text
docs/qwen2_5_vl_online_server.md
```

在线服务代码、无 GPU 协议/生命周期测试和真实 Qwen2.5-VL C1/C2/C4 回归已经
完成。在线阶段暂时封板，下一阶段进入 Attention backend，不继续扩展外围 API。

## 17. Attention 后端重构建议

不要直接在各模型文件中散落 backend 调用。建议定义接口：

```python
class EncoderAttentionBackend:
    def forward(
        self,
        q,
        k,
        v,
        cu_seqlens,
        max_seqlen,
        causal=False,
    ):
        ...
```

实现：

```text
TorchSDPABackend
FlashAttentionBackend
CuDNNAttentionBackend
```

Decoder 单独定义：

```text
DecoderPrefillBackend
DecoderKVCacheBackend
```

原因：

- PyTorch SDPA 通常接收 dense `[B, H, S, D]`。
- FlashAttention varlen 接收 packed `[T, H, D]` 和 `cu_seqlens`。
- Decoder FlashAttention 还要接收 paged KV cache 和 block table。
- cuDNN 高层 SDPA 不直接接收 nano-vllm 的 block table。

视觉 backend 初步 benchmark 可以从相同长度的单图开始，不必先处理 paged cache。

## 18. cuBLAS、cuDNN 和 FlashAttention 的结论

### cuBLAS

cuBLAS 提供 GEMM。

可以用：

```text
QK^T GEMM
softmax
PV GEMM
```

拼出普通 Attention，但会显式产生 score 矩阵，不等于 FlashAttention。

cuBLAS 本身没有一个直接叫 FlashAttention 的 API。

### flash-attn

当前安装的是 Dao-AILab FlashAttention CUDA 扩展。

它不是 cuBLAS。

它内部使用自定义 CUDA/CUTLASS 等实现，并通过 Python extension 暴露：

```text
flash_attn_varlen_func
flash_attn_with_kvcache
```

### cuDNN

NVIDIA 官方的 fused SDPA/FlashAttention 类路径主要在 cuDNN。

当前 PyTorch 可以通过：

```python
from torch.nn.attention import SDPBackend, sdpa_kernel
```

强制选择 cuDNN Attention。

但高层 SDPA 与 nano-vllm 的 paged KV cache/block table 之间还需要适配。

## 19. AWQ 概念结论

AWQ 的 `Activation-aware` 表示：

```text
离线量化权重时观察 activation
```

不表示标准 AWQ 推理时 activation 也被量化。

标准 W4A16：

```text
W: INT4
A: FP16/BF16
Accumulator: FP32
```

权重近似：

```text
W_fp ~= (W_int4 - zero_point) * weight_scale
```

BF16 转 FP32 累加不是“反量化”，只是浮点精度扩展，不需要 scale。

FP32 累加后转 BF16 是浮点舍入，也不是带 scale 的整数量化。

只有 W4A8/W8A8 等 activation 也为整数的方案才需要 activation scale。

AWQ 推理第一阶段应实现：

```python
output = awq_gemm(
    activation,  # BF16/FP16
    qweight,     # packed INT4
    scales,
    qzeros,
    group_size=128,
)
```

## 20. Profiling 与 benchmark 设计

### 20.1 必须分开 Prefill 和 Decode

Prefill：

- 大量 token 并行。
- 更偏计算密集。
- Attention 和 GEMM 占比高。

Decode：

- 每个请求每步一个 token。
- 大量读取权重和 KV Cache。
- 更偏 memory bandwidth 和 kernel launch。

### 20.2 建议指标

```text
TTFT
Prefill latency
Prefill tokens/s
TPOT
Decode tokens/s
Request throughput
Peak GPU memory
Model weight memory
KV Cache memory
视觉编码器 latency
```

### 20.3 Attention benchmark 维度

建议组合：

```text
batch_size:      1, 2, 4, 8
sequence_length: 128, 512, 1024, 2048, 4096
dtype:           BF16
backend:         PyTorch, FlashAttention, cuDNN
phase:           vision/prefill/decode
```

注意：

- 每个 shape 先 warmup。
- cuDNN/torch.compile 首次编译不计入稳态延迟。
- 使用 CUDA Event 测 GPU 时间。
- 每组多次运行并报告中位数/P90。
- 同时验证 finite、max error、mean error。

## 21. 风险与约束

### 21.1 16GB 显存

当前完整模型参数约 40.66 亿。

BF16 权重理论体积已经约：

```text
4.066B * 2 bytes ~= 8.1GB
```

还需要：

- KV Cache。
- Vision 激活。
- Text Prefill 激活。
- FlashAttention workspace。
- CUDA Graph。
- PyTorch allocator。

因此初次端到端测试建议：

```text
tensor_parallel_size=1
enforce_eager=True
max_model_len 较小
max_num_batched_tokens 较小
max_num_seqs 较小
```

不要一开始就用原默认：

```text
max_num_batched_tokens=16384
max_num_seqs=512
```

### 21.2 Transformers 版本差异

模型 config 文件标记的版本较旧，但当前环境是 Transformers 5.12.1。

当前 `AutoConfig` 把文本字段放在：

```python
config.text_config
```

后续参考官方实现时要以当前实际 Python 对象和 checkpoint 权重名为准，不要只参考旧博客。

### 21.3 TP

当前视觉塔在 TP 多 rank 下是复制的，不是切分的。

文本 Decoder 使用 nano-vllm 原有 TP Linear。

第一阶段明确限制 TP=1。TP>1 的视觉内存、权重加载和 embedding 一致性需要单独验证。

### 21.4 Prefix Cache

图片 token 与视觉 embedding 绑定。

Prefix cache 不能只按 token ID 认为不同图片相同，因为不同图片可能有相同数量的 `<|image_pad|>` token。

多模态 prefix hash 后续必须包含图片内容 hash 或处理后的多模态特征标识。

第一版可以暂时禁用多模态 prefix cache。

### 21.5 CUDA Graph

图片分辨率导致视觉 token 数动态变化。

初版使用：

```text
enforce_eager=True
```

先避开视觉 CUDA Graph 的动态 shape 复杂度。

## 22. 建议里程碑

### 第一周：VLM 正确性

```text
Day 1-2
Processor 输出检查
多模态请求数据结构

Day 3-4
Sequence/ModelRunner 传递图片
Prefill 视觉 embedding 注入

Day 5
3D MRoPE 和 Decode delta

Day 6-7
Transformers 对齐
单图离线生成
```

### 第二周：Attention 和 Profiling

```text
Day 8-9
Attention backend 抽象

Day 10-11
PyTorch/Flash/cuDNN correctness benchmark

Day 12
Prefill/Decode benchmark

Day 13-14
Nsight Systems/Compute 定位热点
选择融合目标
```

### 第三周：AWQ 和算子融合

```text
Day 15-17
AWQ checkpoint 解析
AWQ Linear 正确性基线

Day 18-19
AWQ 或选定热点的 Triton/CUDA kernel

Day 20
端到端替换和回归

Day 21
性能表格、显存结果、README 和演示
```

## 23. 项目最终交付物

代码：

- Qwen2.5-VL 单图离线推理。
- Attention backend 接口。
- PyTorch/Flash/cuDNN backend 或 benchmark。
- AWQ W4A16 推理路径。
- 至少一个融合/优化 kernel。
- 可重复运行的 benchmark。

正确性结果：

- Vision embedding 误差。
- Prefill logits 误差。
- 首 token 对齐。
- 生成 token 对比。
- AWQ 与 BF16 输出质量对比。

性能结果：

- 视觉编码器耗时。
- Prefill latency/throughput。
- Decode TPOT/throughput。
- BF16/AWQ 显存。
- 优化前后 kernel 时间。
- 优化前后端到端时间。

文档：

- 架构图。
- 输入 layout。
- MRoPE 说明。
- KV Cache layout。
- Attention backend 适用条件。
- AWQ 数据格式。
- Profiling 截图/表格。
- 已知限制。

## 24. 新对话建议开场提示

在新的 Codex 对话中可以直接发送：

```text
请先阅读：
/home/agua/tensorrtlearning/nano-vllm/NANO_VLLM_PROJECT_CONTEXT.md

然后检查当前 git status 和相关源码。这个项目正在为 nano-vllm
增加 Qwen2.5-VL、Attention 后端、AWQ 和算子融合支持。
当前已经完成单图离线推理、异步 HTTP 服务和 continuous decode batching。

当前已经实现 SSE、请求取消、超时/并发保护、OpenAI Chat Completions，
并通过真实 Qwen2.5-VL C1/C2/C4 正确性与吞吐回归。下一步开始
Attention backend 抽象和 benchmark。不要直接跳到 ZMQ/EngineCore 多进程。
```

## 25. 新对话开始时建议先执行

```bash
cd /home/agua/tensorrtlearning/nano-vllm

git status --short --branch
git diff --check

sed -n '1,220p' NANO_VLLM_PROJECT_CONTEXT.md
sed -n '1,280p' nanovllm/engine/async_llm_engine.py
sed -n '1,180p' nanovllm/engine/llm_engine.py
sed -n '1,220p' examples/qwen2_5_vl_server.py
```

然后重点阅读：

```text
nanovllm/engine/llm_engine.py
nanovllm/engine/async_llm_engine.py
nanovllm/engine/sequence.py
nanovllm/engine/scheduler.py
nanovllm/engine/model_runner.py
nanovllm/utils/context.py
nanovllm/layers/attention.py
```

## 26. 继续开发时的原则

1. 先正确性，后性能。
2. 先单图 Prefill 和 Decode batching，后多图片 Prefill batching。
3. 先离线推理，后在线服务。
4. 先 BF16，后 AWQ。
5. 先使用参考 backend 验证，再融合 kernel。
6. 每个优化必须有独立 kernel benchmark 和端到端 benchmark。
7. 不把首次编译时间混入稳态性能。
8. 不因为单 kernel 更快就宣称端到端更快。
9. 不把跨 Attention backend 的 BF16 差异直接判断为结构错误。
10. 不在没有图片数据通路时宣称已经支持 Qwen2.5-VL 推理。

## 27. 在线服务 V4 实现记录

本轮新增：

```text
LLMEngine.step_with_metadata
    -> token_id
    -> finish_reason

AsyncLLMEngine
    -> stream_generate()
    -> AsyncEngineStreamEvent(token/done/error)
    -> cancel(request_id)
    -> terminal request accounting

Scheduler
    -> abort(seq_id)
    -> waiting/running 删除
    -> KV Cache deallocate

FastAPI
    -> /generate_stream
    -> /v1/chat/completions
    -> max_concurrent_requests
    -> request_timeout_seconds

Regression
    -> C1/C2/C4 exact greedy token comparison
    -> decode batch assertion
    -> requests/s 和 output_tokens/s speedup
```

取消在 engine step 边界执行，不尝试从另一个线程中断正在运行的 CUDA kernel。
这是为了保证 Scheduler 和 KV Cache 所有权仍由 engine thread 独占。

无 GPU 测试命令：

```bash
/home/agua/anaconda3/envs/yolo26/bin/python \
  -m unittest discover -s tests -p 'test_*.py' -v
```

真实 VLM 回归命令：

```bash
/home/agua/anaconda3/envs/yolo26/bin/python \
  examples/qwen2_5_vl_online_regression.py \
  --url http://127.0.0.1:8000/generate \
  --image assets/dog.png \
  --image-input base64 \
  --max-new-tokens 8 \
  --warmup-iters 2 \
  --measure-iters 12 \
  --concurrencies 1,2,4 \
  --output-json profiles/qwen2_5_vl_online_regression.json
```

详细协议、架构和面试问答见：

```text
docs/qwen2_5_vl_online_server.md
```

真实回归结果（2026-07-28）：

```text
Greedy token correctness: passed, 0 mismatches
Max decode batch: C1=1, C2=2, C4=4
Output tokens/s: C1=15.380, C2=17.561, C4=19.526
Throughput speedup vs C1: C2=1.142x, C4=1.270x
Mean client E2E: C1=510.02ms, C2=891.38ms, C4=1615.06ms
```

C4 吞吐没有随 decode batch 线性提升，是因为 2049-token 的多模态 prefill
仍逐请求串行执行，而当前只生成 8 个 token。这个结果证明 decode continuous
batching 已生效，同时说明下一项端到端吞吐瓶颈在 multimodal prefill。
