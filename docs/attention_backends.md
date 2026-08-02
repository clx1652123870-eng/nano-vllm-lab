# Nano-vLLM Attention Backend 抽象

本文记录 nano-vllm 第一版 Attention backend 抽象的设计、实现、验证方法和当前限制。

## 1. 当前实现结论

模型层不再直接调用具体 Attention 算子，而是通过两类接口：

```text
EncoderAttentionBackend
    用于视觉 Encoder
    packed varlen Q/K/V
    不持久化 KV Cache
    支持 non-causal，也保留 causal 参数

DecoderAttentionBackend
    用于文本 Decoder
    causal prefill
    paged KV Cache decode
```

当前支持矩阵：

| 场景 | flash_attn | torch_sdpa/math/cuDNN | triton | cuda_fused | hybrid | cuda_hybrid |
| --- | --- | --- | --- | --- | --- | --- |
| Vision Encoder packed varlen | 支持 | 支持 | 支持，实验 kernel | 支持，短序列 | Triton/Flash 分派 | CUDA/Flash 分派 |
| Text Decoder causal prefill | 支持 | 不支持 | 不支持 | 不支持 | 不支持 | 不支持 |
| Text Decoder paged-KV decode | 支持 | 不支持 | 不支持 | 不支持 | 不支持 | 不支持 |

默认配置仍然是：

```text
attention_backend=flash_attn
vision_attention_backend=flash_attn
```

因此不传新参数时，原来的 FlashAttention 路径和 checkpoint 参数名都不变。

## 2. 为什么必须拆成 Encoder 和 Decoder 两类接口

视觉 Encoder 与文本 Decoder 虽然都计算：

```text
Attention(Q, K, V) = softmax(QK^T * scale + mask)V
```

但运行时数据结构不同，不能只用一个简单的 `forward(q, k, v)` 覆盖。

视觉 Encoder 的输入是 packed varlen：

```text
Q: [total_tokens, num_q_heads, head_dim]
K: [total_tokens, num_kv_heads, head_dim]
V: [total_tokens, num_kv_heads, head_dim]
cu_seqlens: [num_sequences + 1]
```

Qwen2.5-VL 的 Window Attention 会把多个窗口压到同一个 token buffer，再通过
`cu_seqlens` 表示每个窗口的边界。Full Attention 则通常对应一个或少量长序列。
视觉侧不需要跨生成 step 保留 KV Cache。

文本 Decoder 分两个阶段：

```text
Prefill:
    一次处理多个 prompt token
    写入 KV Cache
    执行 causal attention

Decode:
    每个请求通常只产生一个新 query token
    从 paged KV Cache 读取历史 K/V
    通过 block_tables 找到逻辑序列对应的物理 cache block
```

普通 PyTorch SDPA 接收 dense Q/K/V Tensor，并不理解 nano-vllm 的
`block_tables`。因此实现视觉 SDPA 不等于已经实现 Decoder 的 Paged SDPA。

## 3. 代码结构

```text
nanovllm/attention/base.py
    EncoderAttentionBackend
    DecoderAttentionBackend

nanovllm/attention/factory.py
    名称规范化
    backend 能力检查
    具体实现的延迟导入

nanovllm/attention/flash_attn.py
    FlashAttentionEncoderBackend
    FlashAttentionDecoderBackend

nanovllm/attention/torch_sdpa.py
    TorchSDPAEncoderBackend
    TorchMathSDPAEncoderBackend
    CUDNNSDPAEncoderBackend

nanovllm/attention/triton_attn.py
    TritonEncoderAttentionBackend
    packed-varlen Triton forward kernel

nanovllm/attention/hybrid.py
    短窗口选择 Triton
    长序列选择外部 FlashAttention

nanovllm/attention/cuda_attn.py
    cuda_fused: 项目内短序列 fused CUDA kernel
    cuda_hybrid: 短序列选择 CUDA，长序列选择外部 FlashAttention

nanovllm/layers/attention.py
    文本 Decoder Attention
    KV Cache 写入
    调用 DecoderAttentionBackend

nanovllm/models/qwen2_5_vl.py
    视觉和文本 backend 接线

nanovllm/models/qwen3.py
    文本 backend 接线
```

运行时配置链路：

```text
CLI / LLM kwargs
    -> Config
    -> ModelRunner
    -> model constructor
    -> Qwen2.5-VL Vision/Text Attention
    -> backend factory
    -> concrete backend
```

factory 使用延迟导入。选择 `torch_sdpa` 时不会仅因为导入 factory 就加载
FlashAttention Python 扩展。

## 4. 接口契约

### 4.1 EncoderAttentionBackend

核心参数：

```python
forward(
    query,
    key,
    value,
    *,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    softmax_scale,
    causal,
)
```

输入和输出约定：

```text
query:  [Tq, Hq, D]
key:    [Tk, Hkv, D]
value:  [Tk, Hkv, D]
output: [Tq, Hq, D]
```

`cu_seqlens=[0, 5, 13]` 表示 packed buffer 中有两个序列：

```text
sequence 0: token [0, 5), length 5
sequence 1: token [5, 13), length 8
```

`max_seqlen_q/max_seqlen_k` 是 FlashAttention kernel 选择和 workspace
计算需要的元数据。PyTorch 参考适配器不直接使用它们，但接口仍保留，避免模型层
感知具体后端差异。

### 4.2 DecoderAttentionBackend

Prefill 接口同时接收当前 Q/K/V 和 KV Cache：

```python
prefill(
    query,
    key,
    value,
    *,
    key_cache,
    value_cache,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    block_tables,
    softmax_scale,
)
```

没有 prefix cache 时，Attention 直接读取当前 `key/value`。有 prefix cache 时，
FlashAttention 根据 `block_tables` 从 paged cache 读取 K/V。

Decode 接口：

```python
decode(
    query,
    *,
    key_cache,
    value_cache,
    context_lens,
    block_tables,
    softmax_scale,
)
```

输入 query 为 `[batch, Hq, D]`，统一输出 `[batch, Hq, D]`。

当前 `store_kvcache` Triton kernel 仍放在 `nanovllm/layers/attention.py`。它负责
根据 `slot_mapping` 修改 KV Cache；backend 负责读取 cache 并计算 Attention。
这是第一版清晰的所有权边界。后续若要融合 RoPE、cache write 和 decode
Attention，需要扩展 backend 接口，而不是在模型文件里添加特殊分支。

## 5. 具体后端

### 5.1 FlashAttention

视觉和文本 prefill 调用：

```text
flash_attn_varlen_func
```

文本 decode 调用：

```text
flash_attn_with_kvcache
```

后者原生接收：

```text
cache_seqlens
block_table
paged K/V cache
```

所以它适合作为当前 Decoder backend。

### 5.2 PyTorch SDPA

视觉参考后端调用：

```python
torch.nn.functional.scaled_dot_product_attention
```

PyTorch SDPA 需要接近 `[B, H, S, D]` 的 dense 输入，而 nano-vllm 视觉侧是
packed varlen。当前适配器按 `cu_seqlens` 切出每个序列，再逐个调用 SDPA。

GQA 的处理方式：

```text
Hq == Hkv:
    直接计算

Hq > Hkv:
    K/V 按 group repeat 到 Hq
```

这一版的定位是：

```text
正确性参考
接口可替换性验证
端到端 smoke test
```

它还不是最优 packed-varlen 实现，原因包括：

- Python 循环会产生多次 kernel launch。
- `cu_seqlens.cpu().tolist()` 会引入 GPU 到 CPU 的同步。
- 显式展开 GQA K/V 会产生额外 Tensor。
- 没有按相同长度对窗口分组并组成 dense batch。

另外，“使用 PyTorch SDPA”描述的是前端 API。CUDA 上实际使用 math、
memory-efficient、Flash 或 cuDNN kernel，由 PyTorch dispatcher 和硬件支持决定。

### 5.3 强制 cuDNN SDPA

`CUDNNSDPAEncoderBackend` 复用同一个 packed adapter，但在每次调用外使用：

```python
sdpa_kernel(SDPBackend.CUDNN_ATTENTION)
```

它保证候选 kernel 来自 cuDNN，适合回答“默认 SDPA 到底选了什么”的问题。
它仍然承担逐 packed sequence 调用的 adapter 开销，所以测试结果不能简单推广为
cuDNN Attention 的普遍性能。

### 5.4 Triton

`TritonEncoderAttentionBackend` 直接消费 packed Q/K/V 和 CUDA
`cu_seqlens`。kernel 使用 online softmax，不物化完整 Attention matrix，并支持
GQA head 映射。当前仅支持 CUDA BF16、`head_dim <= 256` 和 forward，不支持
Paged KV Cache decode。

实现和实测细节见：

```text
docs/attention_backend_benchmark.md
```

### 5.5 项目内 CUDA fused Attention

`CUDAFusedAttentionEncoderBackend` 直接消费 packed Q/K/V。一个 CUDA block 处理
一个 `(sequence, query_head)`，在 shared memory 中保存 K/V，并在寄存器中维护
online softmax 的 `row_max`、`row_sum` 和输出累加器。它融合 QK、scale、causal
mask、softmax 和 PV，不物化完整 score matrix，并支持 GQA。

当前实现面向教学和三方对比：只支持 CUDA BF16、`head_dim <= 128`、
`max_seqlen <= 64`，内部使用 FP32 scalar FMA，没有使用 WMMA/Tensor Core。
`cuda_hybrid` 在短窗口调用该 kernel，长序列回退外部 FlashAttention。

## 6. 配置和运行

### 6.1 默认 FlashAttention

`assets/dog.png` 会产生 2049 个输入 token，因此预算要至少设为 4096：

```bash
/home/agua/anaconda3/envs/yolo26/bin/python \
  examples/qwen2_5_vl_offline.py \
  --engine nano \
  --max-new-tokens 8 \
  --no-tqdm \
  --max-model-len 4096 \
  --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.72 \
  --attention-backend flash_attn \
  --vision-attention-backend flash_attn
```

### 6.2 切换视觉后端

```bash
/home/agua/anaconda3/envs/yolo26/bin/python \
  examples/qwen2_5_vl_offline.py \
  --engine nano \
  --max-new-tokens 8 \
  --no-tqdm \
  --max-model-len 4096 \
  --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.72 \
  --attention-backend flash_attn \
  --vision-attention-backend triton
```

最后一个参数还可取 `flash_attn`、`torch_sdpa`、`torch_math`、`cudnn_sdpa`
或 `hybrid`。
这里文本 Decoder 仍使用 FlashAttention。以下配置会明确报错：

```text
--attention-backend torch_sdpa
```

原因不是参数没接通，而是 Paged SDPA Decoder 尚未实现。

### 6.3 离线 profiling

比较视觉后端时保持图片、prompt、生成长度、warmup 和测量次数一致：

```bash
/home/agua/anaconda3/envs/yolo26/bin/python \
  examples/qwen2_5_vl_profile.py \
  --max-new-tokens 8 \
  --warmup-iters 2 \
  --measure-iters 10 \
  --max-model-len 4096 \
  --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.72 \
  --vision-attention-backend flash_attn \
  --output-json profiles/attention_vision_flash.json
```

将后两个参数改为任意候选视觉后端，例如：

```text
--vision-attention-backend triton
--output-json profiles/attention_vision_triton.json
```

当前比较主要观察：

```text
prefill_latency_ms
prefill_tokens_per_s
ttft_ms
peak_memory
```

因为两个实验的文本 Decoder 都是 FlashAttention，`decode_tpot_ms` 不是视觉
backend 的核心指标。

### 6.4 在线服务

```bash
/home/agua/anaconda3/envs/yolo26/bin/python \
  examples/qwen2_5_vl_server.py \
  --host 127.0.0.1 \
  --port 8000 \
  --max-model-len 4096 \
  --max-num-batched-tokens 4096 \
  --vision-attention-backend flash_attn
```

`GET /health` 的 `limits` 会返回实际使用的文本和视觉 backend。

## 7. 验证结果

单元测试：

```bash
/home/agua/anaconda3/envs/yolo26/bin/python \
  -m unittest discover -s tests -p 'test_*.py' -v
```

当前结果：

```text
30 tests passed
```

覆盖内容：

- backend 名称、别名和能力检查。
- Decoder 拒绝尚未实现的 `torch_sdpa`。
- packed non-causal Attention 对显式参考公式。
- packed causal Attention 对显式参考公式。
- GQA head 展开。
- backend 切换不改变视觉模块的 `state_dict` 参数名。
- 原在线引擎、取消、SSE、OpenAI API 和 C1/C2/C4 测试没有回归。

CUDA synthetic 对齐：

```text
layout: [13, 4, 80], KV heads=2, BF16
sequences: [5, 8]
自动 SDPA/cuDNN/Triton/Flash/Hybrid 均成功，输出 shape 一致
```

真实 Qwen2.5-VL smoke：

```text
input tokens: 2049
image tokens: 2025
image_grid_thw: [[1, 90, 90]]

vision=flash_attn:
    [108893, 45930, 101987, 99593]

vision=torch_sdpa:
    [108893, 45930, 101987, 99593]

vision=cudnn_sdpa:
    [108893, 45930, 101987, 99593]

vision=triton:
    [108893, 45930, 101987, 99593]

vision=hybrid:
    [108893, 45930, 101987, 99593]
```

上述五种视觉后端前 4 个 greedy token 一致。严格 `torch_math` 在低分辨率图片
上完成端到端验证，在当前 8100-patch 输入上因物化 Full Attention 中间矩阵而
OOM。这个结果不替代更完整的多 shape 数值误差和性能 benchmark。

真实 shape 性能数据和误差见 `docs/attention_backend_benchmark.md`。
PyTorch Math/SDPA 基线与 Hybrid 的专项对比见
`docs/pytorch_baseline_optimization_comparison.md`。

## 8. 如何新增一个 Backend

以新增一个视觉 Encoder backend 为例：

1. 在 `nanovllm/attention/` 新增实现文件。
2. 实现 `EncoderAttentionBackend.forward`，保持输入输出 layout 不变。
3. 在 factory 中登记稳定名称。
4. 不在 Qwen2.5-VL 模型文件中新增 cuDNN 分支。
5. 增加显式 Attention 公式的数值测试。
6. 增加 FlashAttention/candidate backend 的多 shape CUDA 对齐。
7. 记录不支持的 dtype、head dimension、causal、GQA 和序列长度。
8. 最后再跑真实模型 greedy token 和 profiling 回归。

如果新增 Decoder backend，还必须定义：

- Paged KV Cache 的物理 layout。
- `block_tables` 到 K/V 地址的映射。
- prefill 与 decode 是否用同一 kernel。
- GQA 和 bottom-right causal 对齐语义。
- KV Cache 写入由谁负责。

## 9. 面试常见问题

### 为什么不能只在模型里写 `if backend == ...`？

这样会让模型语义、runtime cache 管理和 kernel API 混在一起。每增加一个后端都要
修改模型文件，容易出现 layout、mask 和 cache 行为不一致。接口让模型只描述
Q/K/V 的产生方式，让 backend 负责把统一语义适配到具体算子。

### 为什么视觉和文本不能共用一个 backend 接口？

视觉是无持久 cache 的 packed varlen Attention；文本 decode 是带
`block_tables` 的 Paged KV Attention。二者的状态、输入元数据和算子能力不同。
强行统一会得到大量可选参数，调用方无法知道哪些组合有效。

### PyTorch SDPA 是不是一个固定 CUDA kernel？

不是。它是 PyTorch 的统一前端，实际 kernel 由 dispatcher 根据设备、dtype、
shape 和上下文选择。要声称使用 cuDNN 或 Flash kernel，必须强制 backend 并用
profiler 验证，不能只看 Python API 名称。

### 为什么 Decoder 暂时没有 torch_sdpa？

普通 SDPA 不理解 paged cache 和 block table。可以先把 paged K/V gather 成
dense Tensor 再调用 SDPA，但会增加显存复制和延迟，只适合作为正确性参考，不是
最终 decode 实现。

### 抽象会不会影响模型权重加载？

不会。backend 对象不包含模型参数，也不是 checkpoint module。Q/K/V projection
和 output projection 仍在原模型模块中，测试确认切换 backend 前后
`state_dict` key 一致。

### KV Cache 写入为什么不放进 backend？

当前 Triton `store_kvcache` 是 runtime 的统一状态更新，Attention backend 只负责
读取并计算。这让第一版边界简单且保持原行为。若 profiling 证明 cache write 和
RoPE/Attention 之间的 launch 开销重要，再扩展接口做融合。

### 如何证明这次不是只做了代码重命名？

有三层证据：

```text
显式 Attention 公式 vs torch_sdpa 单元测试
候选 backend CUDA Tensor 数值对齐
五种高性能/自动 backend 的真实 Qwen2.5-VL greedy token 对齐
严格 PyTorch Math backend 的低分辨率端到端回归和高分辨率 OOM 边界
```

同时 factory 会拒绝能力不匹配的 Decoder 配置，而不是静默 fallback。

## 10. 阶段结论

抽象和第一轮 benchmark 已完成：

```text
Vision 真实 shape:
    28 x Window + 4 x Full
    Triton 在短 Window 上快于外部 FlashAttention
    Full Attention 仍是外部 FlashAttention 更快
    Hybrid 按 sequence length 选择两者
    Hybrid 与纯 Flash 的端到端 TTFT 差异接近噪声
    Torch/cuDNN packed adapter 受逐段调用限制

在线框架对比:
    nano-vllm TTFT 更低
    vLLM TPOT 和吞吐更好

AWQ:
    INT4 checkpoint 已可离线和在线推理
    当前省权重显存但 decode 慢于 BF16
```

后续三方实测已经完成。Window shape 下 P50 为 PyTorch Math `10.917 ms`、
PyTorch SDPA `2.548 ms`、项目内 CUDA fused `1.080 ms`、Triton fused
`0.119 ms`。Triton 相对三者分别为 `91.39x`、`21.33x`、`9.04x`；Full
Attention 则继续由外部 FlashAttention 更合适。完整数据、Nsight 证据和表述边界见：

```text
profiles/attention_pytorch_cuda_triton_rtx5080.json
profiles/nsight/attention_backends_summary.json
docs/nano_vllm_qwen2_5_vl_interview_guide.md
```
