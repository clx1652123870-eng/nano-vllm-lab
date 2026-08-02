# nano-vllm Qwen2.5-VL 项目总结与面试指南

本文是当前仓库的最终总结文档。它覆盖从 Qwen2.5-VL 模型接入、离线正确性、
在线 continuous batching、OpenAI/SSE 服务、AWQ W4A16、Attention backend 抽象，
到 CUDA/Triton 融合算子、vLLM 对比和 Nsight 分析的完整主线。

记录环境与数据日期：2026-08-01。

## 1. 一句话项目介绍

在教学版 nano-vllm 上完成 Qwen2.5-VL-3B 的多模态推理与在线服务扩展，实现
MRoPE、视觉 Window/Full Attention、continuous batching、SSE/OpenAI API、请求
取消和 AWQ W4A16；设计 Attention backend 抽象，并实现 PyTorch、自定义 CUDA、
Triton 和外部 FlashAttention 的 shape-aware dispatch 与系统化性能分析。

## 2. 当前完成范围

### 模型与正确性

- Qwen2.5-VL-3B-Instruct BF16 单图推理。
- Processor/chat template、多模态 token 和视觉输入传递。
- 视觉 PatchEmbed、32 层 Vision Transformer 和 PatchMerger。
- 文本侧 Qwen2.5-VL MRoPE。
- Transformers 与 nano-vllm input IDs、MRoPE、greedy token 对齐。
- Qwen2.5-VL-3B-Instruct-AWQ W4A16 文本权重加载。

### 在线服务

- 单进程、单 GPU，模型只加载一次。
- FastAPI native 非流式 `/generate`。
- native SSE `/generate_stream`。
- OpenAI Chat Completions `/v1/chat/completions`。
- 多客户端请求队列和 decode continuous batching。
- 并发上限、超时、客户端断开取消和 KV Cache 回收。
- TTFT、TPOT、吞吐、step latency、batch size 和显存 profile。
- BF16 与 AWQ 在线 C1/C2/C4 回归。

### 性能工程

- Vision Attention backend 抽象。
- PyTorch Math、PyTorch SDPA、cuDNN SDPA、外部 FlashAttention。
- 自定义 Triton packed-varlen FlashAttention-style kernel。
- 自定义 CUDA packed-varlen online-softmax fused Attention。
- Triton/FlashAttention Hybrid 与 CUDA/FlashAttention Hybrid。
- Softmax、RMSNorm、SiLU-and-Mul、Matmul 的 PyTorch/CUDA/Triton 微基准。
- nano-vllm 与 vLLM 的在线和 CUDA kernel 级对比。
- Kineto operator profile 和带 NVTX 标记的 Nsight Systems profile。

## 3. 输入数据流

测试图片统一使用：

```text
assets/dog.png
原图尺寸: 1254 x 1254
问题: 描述这张图片
```

Processor 输出：

```text
input tokens: 2049
image tokens: 2025
image_grid_thw: [[1, 90, 90]]
pixel_values: [8100, 1176], FP32
```

`8100` 是视觉 patch token 数。视觉塔通过 `spatial_merge_size=2` 将每 `2x2` 个
视觉 token 合并，因此注入文本序列的是：

```text
8100 / 4 = 2025 image tokens
```

完整数据流：

```text
PIL Image
  -> AutoProcessor
  -> pixel_values / image_grid_thw / input_ids / mm_token_type_ids
  -> MultiModalPrompt
  -> Vision Transformer
  -> 2025 个视觉 embedding
  -> 替换文本 embedding 中的 image_pad token
  -> Qwen2.5-VL Text Decoder
  -> LM Head
  -> Sampler
```

## 4. MRoPE

普通 LLM RoPE 只有一条 token position。Qwen2.5-VL 的图片 token 同时需要：

```text
Temporal position
Height position
Width position
```

因此多模态 position IDs 的逻辑形状是：

```text
[3, num_tokens]
```

文本 token 的三个轴位置相同，图片 token 按 `T/H/W` 网格展开。`mrope_section`
将 head dimension 分段，不同段选择不同轴的 cos/sin。生成第一个文本 token 后，
后续 decode 位置通过 `mrope_position_delta` 接续。

面试重点：模型能够加载不代表 MRoPE 正确。MRoPE 错误通常不会立即崩溃，而是表现
为首 token 或后续 token 与 Transformers 分叉。

代码位置：

```text
nanovllm/multimodal.py
nanovllm/models/qwen2_5_vl.py
examples/qwen2_5_vl_alignment.py
```

## 5. 离线执行架构

```text
LLM.generate
  -> LLMEngine.add_request
  -> Scheduler.waiting
  -> Scheduler.schedule
  -> ModelRunner.prepare_prefill / prepare_decode
  -> Qwen2_5_VLForConditionalGeneration.forward
  -> Attention / Linear / RMSNorm / MLP
  -> Sampler
  -> Scheduler.postprocess
  -> token IDs / text
```

`Scheduler` 有两个核心队列：

```text
waiting:
    尚未完成 prefill，或因 KV Cache 不足被 preempt 的 sequence

running:
    已经完成 prefill、持有 KV Cache、可以执行 decode 的 sequence
```

当前多模态 prefill 不支持 chunk：

```text
max_num_batched_tokens >= multimodal prompt tokens
```

本例至少需要 `2049`，正式命令使用 `4096`。如果设置为 `1024`，异常不是图片文件
太大本身，而是图片经过 Processor 后产生的视觉 token 超过单 step token budget。

## 6. KV Cache 与调度

每层 Attention 的 K/V 按 block 分配：

```text
KV Cache
  -> layer
  -> block
  -> token slot
  -> KV head
  -> head dimension
```

当前 block size 为 `256`。`BlockManager` 负责：

- 为 prefill sequence 分配 block table。
- decode 时追加 token slot。
- 显存不足时 preempt sequence。
- 请求结束、取消或超时时释放 block。
- 文本 prefix cache 的 block hash/refcount。

多模态 prefill 仍串行执行。两个请求都完成 prefill 后，Scheduler 可以将它们放入
同一个 decode batch：

```text
Request A: prefill ---------------- decode -- decode -- ...
Request B:          prefill -------- decode -- decode -- ...
                                      batch=2
```

这就是为什么两个客户端能同时在线，但两张图片不会同时做视觉编码。

## 7. 在线服务架构

当前实现没有照搬 vLLM 的 ZMQ/EngineCore 多进程架构，而是保留 nano-vllm 的单机
边界：

```text
HTTP client
  -> FastAPI event loop
  -> image preprocessing thread
  -> bounded concurrency limiter
  -> AsyncLLMEngine request Queue
  -> dedicated engine thread
  -> LLMEngine / Scheduler
  -> ModelRunner
  -> GPU
  -> per-request event Queue
  -> SSE or JSON response
```

这样做的理由：

- 同步 `LLMEngine.step()` 不阻塞 FastAPI event loop。
- 模型只在 engine thread 中初始化一次。
- 每个请求有独立状态和 stream sink，不会串 token。
- 生命周期先正确，再考虑进程隔离和 IPC。

### 请求生命周期

```text
accepted
  -> preprocess
  -> queued
  -> admitted
  -> prefill
  -> running/decode
  -> finished/cancelled/failed
  -> KV blocks released
```

取消在 engine step 边界生效，不能安全地从另一个线程强行中止正在执行的 CUDA
kernel。HTTP 断开只取消 FastAPI 协程是不够的，取消必须传递到 Scheduler。

### 并发控制

需要两层限制：

```text
max_concurrent_requests:
    限制 HTTP/预处理/排队请求总数，提供 backpressure

max_num_seqs:
    限制 Scheduler active sequences 和模型 batch 上限
```

它们不能互相替代。

## 8. SSE 与 OpenAI API

SSE 每生成一个 token 就发送一个 event：

```text
metadata
token
token
...
done
```

OpenAI streaming 使用 `chat.completion.chunk`，结束标记为：

```text
data: [DONE]
```

当前兼容范围包括单图 Chat Completions、`temperature`、固定最大 token、stream 和
usage。尚未实现工具调用、`n>1`、logprobs、远程 image URL 和自定义 stop string。

## 9. Profiling 指标

### TTFT

客户端可见 TTFT：

```text
图片读取/解码
+ Processor
+ Queue wait
+ Vision Encoder
+ Text Prefill
+ 首 token sampling
+ HTTP/SSE 发送
```

引擎 TTFT 通常只覆盖 engine 内 prefill step，因此不能与客户端 TTFT 混用。

### TPOT

```text
TPOT = decode latency / decode token count
```

它主要反映自回归 decode 的 Linear/GEMV、KV Cache Attention、RoPE、RMSNorm、
Sampler 和调度开销。

### 吞吐

```text
request throughput = completed requests / benchmark duration
output throughput  = generated tokens / benchmark duration
```

吞吐提高不表示每个请求延迟降低。continuous batching 通常用更高单请求 latency
换取更高系统吞吐。

## 10. Attention backend 抽象

统一 Encoder Attention 接口：

```text
Q/K/V: [total_tokens, heads, head_dim]
cu_seqlens_q / cu_seqlens_k: packed sequence offsets
max_seqlen_q / max_seqlen_k
softmax_scale
causal
```

当前视觉 backend：

| Backend | 实现 | 主要用途 |
| --- | --- | --- |
| `torch_math` | 强制 PyTorch Math SDPA | 严格基础基线 |
| `torch_sdpa` | PyTorch 自动 SDPA | 高性能框架基线 |
| `cudnn_sdpa` | 强制 cuDNN SDPA | 库实现对比 |
| `cuda_fused` | 项目内 CUDA online-softmax | 短窗口 CUDA 基线 |
| `triton` | 项目内 Triton FlashAttention-style | 实验 kernel |
| `flash_attn` | 外部 flash-attn CUDA/CUTLASS | 生产参考 |
| `cuda_hybrid` | 短 CUDA，长 FlashAttention | 完整模型可运行 |
| `hybrid` | 短 Triton，长 FlashAttention | 当前 shape-aware 最优 |

文本 Decoder 仍只使用外部 FlashAttention，因为它还需要 Paged KV Cache prefill 和
`flash_attn_with_kvcache` decode 语义，不能直接复用视觉 Encoder kernel。

## 11. Window 与 Full Attention

Qwen2.5-VL 视觉塔共有 32 层：

```text
28 层 Window Attention
4 层 Full Attention
```

`dog.png` shape：

```text
Window:
    8100 packed tokens
    144 sequences
    sequence lengths = 4 / 16 / 64

Full:
    8100 tokens
    1 sequence
```

Hybrid dispatch：

```text
max sequence length <= 64 -> Triton
otherwise                 -> external FlashAttention
```

CUDA Hybrid 使用相同阈值，只把短序列 backend 换为自定义 CUDA。

## 12. Triton FlashAttention-style kernel

普通 Attention：

```text
S = QK^T * scale
P = Softmax(S + mask)
O = PV
```

基础实现会把 `S x S` scores/probabilities 写入显存。Triton kernel 将以下步骤放入
同一个 `@triton.jit` kernel：

```text
加载 Q block
循环加载 K/V block
tl.dot(Q, K)
scale + mask
online softmax
tl.dot(P, V)
写回 output
```

Online Softmax 保存：

```text
row_max
row_sum
output accumulator
```

新 K block 改变最大值时，通过 correction 修正已有累积结果，所以不需要保存完整
attention matrix。Softmax 与 accumulator 使用 FP32，输入和输出为 BF16。

短窗口配置：

```text
BLOCK_M = 32
BLOCK_N = 32
num_warps = 4
```

Triton 的 `tl.dot` 可以映射 Tensor Core，这是它相较教学 CUDA kernel 的主要优势。

## 13. 自定义 CUDA fused Attention

代码位于：

```text
nanovllm/kernels/csrc/kernels.cu
nanovllm/attention/cuda_attn.py
```

支持范围：

```text
BF16
packed varlen
GQA
causal / non-causal
max sequence length <= 64
head_dim <= 128
Q/K/V 允许 split/view 产生的 strided tensor，最后一维必须连续
```

CUDA kernel 使用一个 block 处理一个 `(packed sequence, query head)`，8 个 warp
并行处理 query rows。K/V 进入 shared memory，每个 warp 在寄存器中维护：

```text
query fragment
row_max
row_sum
output accumulator
```

它同样融合 QK、scale、mask、online softmax 和 PV，不在全局显存物化 `S x S`。
当前实现使用 FP32 scalar FMA，没有 WMMA/Tensor Core，因此定位是可解释的手写 CUDA
融合基线，不是对官方 FlashAttention CUDA kernel 的替代。

长序列由 `cuda_hybrid` 回退外部 FlashAttention。端到端 smoke 已验证输出前 4 个
token：

```text
[108893, 45930, 101987, 99593]
```

## 14. PyTorch、CUDA、Triton 三方性能

环境：RTX 5080、BF16、`dog.png`、5 warmup、20 measure。以下为 P50。

### Window Attention

| Backend | P50 | 相对 Triton |
| --- | ---: | ---: |
| PyTorch Math | 10.917 ms | 91.39x |
| PyTorch SDPA | 2.548 ms | 21.33x |
| 自定义 CUDA fused | 1.080 ms | 9.04x |
| 自定义 Triton fused | **0.119 ms** | 1.00x |
| 外部 FlashAttention | 0.124 ms | 1.04x |

精度相对外部 FlashAttention：

```text
CUDA max abs error:   0.0078125
Triton max abs error: 0.0078125
```

结论：

- PyTorch Math/SDPA 的 packed adapter 需要 Python 循环 144 个窗口。
- CUDA 和 Triton 都将一次 Window Attention 融合为一个 GPU operation。
- Triton 比自定义 CUDA 快 `9.04x`，主要因为 `tl.dot` 使用 Tensor Core 和更合适
  的 query/key tile；CUDA 基线仍是 scalar FMA。
- Triton 与外部 FlashAttention 在该短窗口 shape 基本持平，不能根据一次 run 宣称
  稳定超过生产 FlashAttention。

### Full Attention

| Backend | P50 |
| --- | ---: |
| PyTorch Math | 65.291 ms |
| PyTorch SDPA | 4.451 ms |
| Triton | 5.937 ms |
| 外部 FlashAttention | **4.361 ms** |

自定义 CUDA 短窗口 kernel 明确拒绝 8100-token Full Attention。长序列需要更成熟
的 Q/K/V tiling、异步流水、Tensor Core、寄存器和 shared-memory 调度。

### 32 层加权估算

仅计算 `28 * Window + 4 * Full`：

| 路径 | Attention P50 估算 |
| --- | ---: |
| PyTorch Math | 566.85 ms |
| PyTorch SDPA | 89.15 ms |
| CUDA Hybrid | 47.69 ms |
| 纯 Triton | 27.09 ms |
| Triton Hybrid | **20.79 ms** |
| 纯外部 FlashAttention | 20.92 ms |

Triton Hybrid 对 Attention 子项相较：

```text
PyTorch Math: 27.27x
PyTorch SDPA: 4.29x
CUDA Hybrid: 2.29x
```

这是 Attention 子项估算，不是整个模型加速比。Vision Encoder 还包括 PatchEmbed、
RoPE、QKV/MLP Linear、RMSNorm、PatchMerger 和布局转换。

## 15. 基础 CUDA/Triton 算子结论

| 算子和 shape | PyTorch | CUDA | Triton | 最快 |
| --- | ---: | ---: | ---: | --- |
| RMSNorm `[8100,1280]` | 0.0269 ms | 0.0341 ms | 0.0263 ms | Triton |
| SiLU-Mul `[2049,22016]` | 0.2699 ms | 0.1702 ms | 0.1697 ms | Triton/CUDA |
| Matmul M=1 | 0.0209 ms | 0.0793 ms | 0.0260 ms | PyTorch/cuBLAS |
| Matmul M=256 | 0.0352 ms | 0.6867 ms | 0.0443 ms | PyTorch/cuBLAS |

不能写“所有 Triton 算子都比 CUDA/PyTorch 快”。正确工程结论是按算子与 shape
dispatch：复杂融合和固定短窗口适合 Triton，成熟 GEMM 保留 cuBLAS，长 Attention
保留 FlashAttention。

## 16. AWQ W4A16

AWQ checkpoint 的文本 Linear 保存：

```text
qweight: [K, N/8] int32
qzeros:  [K/128, N/8] int32
scales:  [K/128, N] FP16
```

每个 int32 打包 8 个 INT4 值，checkpoint nibble 顺序为：

```text
[0, 4, 1, 5, 2, 6, 3, 7]
```

执行 W4A16 的含义：

```text
Weight: INT4 packed
Activation: BF16/FP16
Accumulator/output: 浮点
```

视觉塔保持 BF16，因为 checkpoint 的 `modules_to_not_convert` 包含 `visual`。

### 当前 kernel

```text
NANOVLLM_AWQ_KERNEL=dequantize
    Triton 解包/反量化 -> BF16 weight -> PyTorch/cuBLAS Matmul

NANOVLLM_AWQ_KERNEL=triton_fused
    实验性 fused W4A16 GEMM
```

默认使用 `dequantize`。当前 fused kernel 没有 Marlin 风格 weight repack 和生产级
INT4 Tensor Core 调度，在真实 projection 上更慢。

### 离线容量与性能

| 指标 | BF16 | AWQ |
| --- | ---: | ---: |
| 模型 unique storage | 6.99 GB | 3.17 GB |
| 引擎初始化 | 3775.60 ms | 1953.83 ms |
| KV blocks | 118 | 563 |
| TTFT | 357.09 ms | 366.93 ms |
| Decode TPOT | 13.65 ms | 21.54 ms |

权重存储减少约 `54.7%`，但当前 decode 慢约 `1.58x`。容量优化与计算加速是两个
不同目标。

## 17. AWQ 在线服务

AWQ 与 BF16 复用同一个 `AsyncLLMEngine`、Scheduler、SSE 和 OpenAI 控制面。
服务参数：

```text
--model /home/agua/models/Qwen2.5-VL-3B-Instruct-AWQ
--awq-kernel dequantize
```

`/health` 会返回：

```text
quantization: awq
quantization_kernel: dequantize
awq_online: true
num_kvcache_blocks: 503   # 本次在线进程实测
```

native、native SSE、OpenAI 非流式和 OpenAI SSE 均已实际通过。前 4 个 token：

```text
[108893, 45930, 101987, 99593]
```

### BF16 与 AWQ 在线 C1/C2/C4

同一 OpenAI SSE 客户端、同一图片、每档 12 请求、固定 32 token：

| C | 精度 | req/s | output tok/s | TTFT | TPOT | E2E |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | BF16 | 1.20 | 38.25 | 62.69 ms | 24.96 ms | 836.42 ms |
| 1 | AWQ | 0.88 | 28.30 | 66.08 ms | 34.33 ms | 1130.41 ms |
| 2 | BF16 | 1.69 | 54.01 | 77.64 ms | 35.70 ms | 1184.33 ms |
| 2 | AWQ | 1.29 | 41.26 | 80.35 ms | 47.42 ms | 1550.27 ms |
| 4 | BF16 | 2.07 | 66.38 | 152.77 ms | 57.20 ms | 1926.02 ms |
| 4 | AWQ | 1.72 | 55.07 | 121.79 ms | 70.96 ms | 2321.70 ms |

AWQ output throughput 相对 BF16：

```text
C1: -26.0%
C2: -23.6%
C4: -17.0%
```

并发提高后容量收益让相对吞吐差距缩小，但当前 AWQ 仍没有速度收益。原因是每个
Linear 都要重新反量化权重，缺少生产级 fused W4A16 kernel。

## 18. nano-vllm 与 vLLM 在线差距

控制变量：BF16、eager、TP=1、同一图片、固定 32 token、关闭 prefix cache 和
CUDA Graph。

| C | 框架 | req/s | output tok/s | TTFT | TPOT | E2E |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | nano | 1.20 | 38.25 | 62.69 ms | 24.96 ms | 836.42 ms |
| 1 | vLLM | 1.56 | 49.79 | 354.42 ms | 9.30 ms | 642.60 ms |
| 2 | nano | 1.69 | 54.01 | 77.64 ms | 35.70 ms | 1184.33 ms |
| 2 | vLLM | 2.11 | 67.65 | 492.15 ms | 14.61 ms | 945.07 ms |
| 4 | nano | 2.07 | 66.38 | 152.77 ms | 57.20 ms | 1926.02 ms |
| 4 | vLLM | 2.69 | 86.16 | 889.13 ms | 19.12 ms | 1481.71 ms |

vLLM 吞吐高约 `25%~30%`，TPOT 明显更低。nano 的 TTFT 更低与当前服务计时、
processor/cache 和调度路径不同有关，不能概括为 nano 全面更快。

## 19. vLLM 算子级对比

分别 profile O1 和 O32：

```text
O1  = Vision + multimodal prefill + first token
O32 = O1 + 31 decode tokens
decode/token ~= (O32 - O1) / 31
```

| 阶段 | nano CUDA | vLLM CUDA | nano 差距 | launches nano/vLLM |
| --- | ---: | ---: | ---: | ---: |
| O1 | 354.42 ms | 284.24 ms | +24.7% | 2534 / 978 |
| O32 | 623.65 ms | 537.97 ms | +15.9% | 35425 / 18679 |
| decode/token | 8.685 ms | 8.185 ms | +6.1% | 1061 / 571 |

O32 核心算子：

```text
GEMM:      nano 447.886 ms, vLLM 449.296 ms
Attention: nano  42.797 ms, vLLM  42.430 ms
```

真正差距：

```text
Elementwise/Layout decode/token:
    nano 0.952 ms, 688 launches
    vLLM 0.207 ms, 121 launches

Sampling decode/token:
    nano 0.090 ms
    vLLM 0.0046 ms
```

这说明下一步不是继续重写 GEMM/Attention，而是：

1. Greedy sampler fast path，避免 temperature=0 仍计算随机分支。
2. 融合 Qwen2.5-VL MRoPE。
3. 减少 FP32/BF16 转换、`cat/copy` 和 layout kernel。
4. 降低 Python scheduler 和 kernel launch gap。

## 20. Nsight Systems 实测

项目提供带 NVTX 标记的 workload：

```text
benchmarks/nsight_attention_workload.py
```

每个 backend 和 iteration 都有 range：

```text
attention::torch_math::iteration
attention::torch_sdpa::iteration
attention::cuda_fused::iteration
attention::triton::iteration
```

Nsight Systems GPU projection：

| Backend | GPU projected time/iter | GPU operations/iter | 相对 Triton |
| --- | ---: | ---: | ---: |
| PyTorch Math | 16.553 ms | 2162 | 179.87x |
| PyTorch SDPA | 5.097 ms | 290 | 55.39x |
| CUDA fused | 1.073 ms | 1 | 11.66x |
| Triton fused | **0.092 ms** | **1** | 1.00x |

Nsight instrumentation 会改变绝对延迟，因此正式 P50 使用独立 benchmark；Nsight
数据用于看时间线结构和相对关系。

### 如何从时间线得到优化思路

#### 现象一：大量短 kernel

```text
PyTorch Math: 2162 GPU operations
PyTorch SDPA:  290 GPU operations
```

推导：瓶颈包含 Python packed adapter、Matmul/Softmax/复制等多个 launch。优化方向
是 packed kernel 和算子融合，而不是只调单个 Softmax block size。

#### 现象二：CUDA/Triton 都只有一个 operation

```text
CUDA:   1.073 ms
Triton: 0.092 ms
```

推导：launch 数已经不是两者差距，问题在 kernel 内部。CUDA 基线使用 scalar FP32
FMA，Triton `tl.dot` 使用 Tensor Core 和 block tiling，所以应进一步查看 Tensor Core
利用率、occupancy、寄存器和 shared-memory pipeline，而不是继续做外层融合。

#### 现象三：GPU kernel 之间有明显空白

推导：如果 kernel 本身时间接近但 wall time 高，检查 CPU scheduler、tensor 创建、
`.tolist()`、同步、launch latency 和 Python dispatch。nano/vLLM decode 对比正属于
这一类。

#### 现象四：Memcpy 或 dtype conversion 占比高

推导：检查热路径中的 `.cpu()`、`.cuda()`、BF16/FP32 转换、contiguous/cat/copy，
优先消除中间 tensor，再决定是否写新 kernel。

## 21. Nsight 复现

```bash
mkdir -p profiles/nsight

PYTHONPATH=. TORCH_CUDA_ARCH_LIST=12.0 \
nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --cpuctxsw=none \
  --force-overwrite=true \
  --output=profiles/nsight/attention_backends \
  /home/agua/anaconda3/envs/yolo26/bin/python \
  benchmarks/nsight_attention_workload.py \
  --backends torch_math,torch_sdpa,cuda_fused,triton \
  --warmup-iters 3 \
  --measure-iters 10

nsys stats \
  --report nvtx_gpu_proj_sum,nvtx_kern_sum,cuda_kern_exec_sum \
  --format csv \
  --output profiles/nsight/attention_backends_stats \
  profiles/nsight/attention_backends.nsys-rep

python benchmarks/summarize_nsight_attention.py
```

Nsight Compute 用于回答单 kernel 为什么慢：

```bash
ncu --set full \
  --kernel-name regex:packed_attention_bf16_kernel \
  --target-processes all \
  /home/agua/anaconda3/envs/yolo26/bin/python \
  benchmarks/nsight_attention_workload.py \
  --backends cuda_fused \
  --warmup-iters 1 \
  --measure-iters 1
```

重点观察：

```text
SM throughput
DRAM throughput
Tensor Core instruction/activity
achieved occupancy
registers per thread
shared memory per block
warp stall reasons
```

当前机器执行 NCU 时返回 `ERR_NVGPUCTRPERM`，没有权限读取硬件 performance
counters。因此文档不伪造 NCU 数值；管理员开放 profiling counters 后再采集。

## 22. 性能实验如何控制变量

至少固定：

```text
同一个 checkpoint
同一 dtype/quantization
同一图片和 Processor
同一 prompt/chat template
同一 input/output token 数
temperature=0, ignore_eos=true
同一 max_model_len/max_num_batched_tokens
TP=1
同一 eager/CUDA Graph 设置
同一 prefix cache 设置
同一并发和请求数量
warmup 与正式请求分离
```

报告 P50/P90，而不是只挑最快值。微基准、离线 profile、在线 benchmark 和 profiler
instrumented latency 必须分别标注，不能放进同一列直接比较。

## 23. 关键面试问答

### Transformers 是什么？

Transformer 是模型架构；Hugging Face Transformers 是包含模型实现、Processor、
Tokenizer、权重加载和 generate 的框架。它与 nano-vllm/vLLM 都能执行推理，但后者
更专注 KV Cache、调度、continuous batching 和服务吞吐。

### Prefill 和 decode 有什么区别？

Prefill 一次处理完整 prompt，计算量大、矩阵较大，偏 compute-bound；decode 每步
只生成一个 token，但要读取所有层权重和历史 KV，常偏 memory/launch-bound。

### 图片怎么 prefill？

图片先经过 Processor 和 Vision Encoder，生成视觉 embedding，再替换文本序列的
image placeholder，随后与文本一起进入 Decoder prefill。图片不是直接进入文本
Attention。

### FlashAttention 为什么省显存？

通过 Q/K/V 分块和 online softmax，不把完整 `S x S` scores/probabilities 写入 HBM，
减少中间显存和 HBM 流量，同时保持数学结果等价。

### Triton kernel 也是 FlashAttention 吗？

它实现 FlashAttention-style 算法，但不是官方 FlashAttention 代码。外部
`flash-attn` 是成熟 CUDA/CUTLASS 实现，本项目 Triton/CUDA 是独立实现。

### Triton 为什么能比手写 CUDA 快 9.04x？

两者都已融合为一个 operation，所以不是少 launch。Triton 的 `tl.dot` 能使用
Tensor Core，并以二维 block 处理 Q/K；CUDA 教学基线每个 warp 处理 query row，
使用 scalar FP32 FMA。Nsight 的单 operation 时间验证了差距发生在 kernel 内部。

### 为什么不把 Full Attention 也用 Triton？

8100-token Full shape 上 Triton 为 5.937 ms，外部 FlashAttention 为 4.361 ms。
成熟库在长序列 tiling、异步流水、寄存器和 shared memory 上更好，所以 Hybrid
按 shape 选择。

### AWQ 为什么显存更小但速度更慢？

INT4 减少存储，不自动减少计算开销。当前路径每次 Linear 都解包和反量化，然后
调用 BF16 GEMM；没有 Marlin 风格 fused W4A16，额外开销使 TPOT 变高。

### 为什么 C4 AWQ 相对 C1 的差距缩小？

AWQ 释放的权重显存可以容纳更多 KV blocks，并发 batch 能摊薄部分固定开销。但
当前反量化仍是每层每步执行，所以绝对吞吐仍低于 BF16。

### 为什么单算子 90x 没有变成 TTFT 90x？

Amdahl 定律。被优化的是 Vision Attention 子项，完整 TTFT 还包括 Processor、
PatchEmbed、Linear、MLP、RMSNorm、RoPE、PatchMerger、文本 prefill 和 sampling。

### 如何证明 continuous batching 生效？

同时验证并发请求 token 不混淆、`max_decode_batch_size > 1`，并观察 output tok/s
相较 C1 提升。只有 HTTP 并发不能证明 GPU 合批。

### 为什么 profile 前后要 `torch.cuda.synchronize()`？

CUDA launch 是异步的。不同步只测到 CPU 提交时间；同步后才得到可解释的 GPU step
wall latency。但同步会扰动流水，所以 profile 模式和生产 benchmark 必须分开。

## 24. 当前限制

- 单 GPU、multimodal TP=1。
- 单图，视觉 prefill 串行且不能 chunk。
- Decoder backend 仍只有外部 FlashAttention。
- 自定义 CUDA fused Attention 只支持短窗口，没有 Tensor Core。
- Triton/Hybrid 在生产 FlashAttention 基线上端到端收益接近噪声。
- AWQ 只支持 INT4、group size 128、asymmetric GEMM checkpoint。
- AWQ 缺少生产级 Marlin/W4A16 Tensor Core kernel。
- OpenAI API 是主要 schema 兼容，不是完整功能等价。
- 尚未实现 CUDA Graph 在线 decode 和多进程 EngineCore。
- NCU 硬件 counter 因权限未采集。

## 25. 下一步优化优先级

根据 nano/vLLM operator profile，而不是猜测：

1. Greedy sampler fast path。
2. Qwen2.5-VL MRoPE 单 kernel 融合。
3. 清理 BF16/FP32、cat/copy/layout 转换。
4. 优化 scheduler 和 CPU launch gap。
5. CUDA Graph decode。
6. Marlin 风格 AWQ weight repack 与 W4A16 GEMM。
7. 多模态 chunked/batched prefill。

## 26. 核心复现命令

### BF16 离线

```bash
/home/agua/anaconda3/envs/yolo26/bin/python \
  examples/qwen2_5_vl_offline.py \
  --engine nano \
  --model /home/agua/models/Qwen2.5-VL-3B-Instruct \
  --image assets/dog.png \
  --max-new-tokens 32 \
  --max-model-len 4096 \
  --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.72 \
  --vision-attention-backend hybrid \
  --no-tqdm
```

### AWQ 在线

```bash
/home/agua/anaconda3/envs/yolo26/bin/python \
  examples/qwen2_5_vl_server.py \
  --model /home/agua/models/Qwen2.5-VL-3B-Instruct-AWQ \
  --served-model-name /home/agua/models/Qwen2.5-VL-3B-Instruct-AWQ \
  --host 127.0.0.1 \
  --port 8000 \
  --max-model-len 4096 \
  --max-num-seqs 4 \
  --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.72 \
  --max-concurrent-requests 4 \
  --vision-attention-backend hybrid \
  --awq-kernel dequantize
```

### AWQ C1/C2/C4

```bash
/home/agua/anaconda3/envs/yolo26/bin/python \
  benchmarks/qwen2_5_vl_serving_benchmark.py \
  --framework-label nano-vllm-awq \
  --model /home/agua/models/Qwen2.5-VL-3B-Instruct-AWQ \
  --concurrencies 1,2,4 \
  --output-len 32 \
  --result-dir profiles/awq_online

python benchmarks/compare_awq_online_profiles.py
```

### Attention 三方对比

```bash
PYTHONPATH=. TORCH_CUDA_ARCH_LIST=12.0 \
/home/agua/anaconda3/envs/yolo26/bin/python \
  benchmarks/attention_backend_benchmark.py \
  --image assets/dog.png \
  --backends torch_math,torch_sdpa,cuda_fused,triton,flash_attn,cuda_hybrid,hybrid \
  --cases window,full \
  --warmup-iters 5 \
  --measure-iters 20 \
  --output-json profiles/attention_pytorch_cuda_triton_rtx5080.json
```

## 27. 推荐简历表述

### 项目名称

```text
nanoVLM-Infer：面向 Qwen2.5-VL 的轻量级多模态推理与算子优化引擎
```

### 三条版本

> 基于 nano-vllm 完成 Qwen2.5-VL-3B 多模态推理与在线服务扩展，实现 MRoPE、
> Vision Window/Full Attention、KV Cache continuous batching、SSE/OpenAI API、
> 请求取消和 AWQ W4A16，支持 C1/C2/C4 并发回归与 TTFT/TPOT profiling。

> 实现 packed-varlen Triton FlashAttention-style kernel，单 kernel 融合 QK、Scale、
> Mask、Online Softmax 和 PV；在 RTX 5080、8100 视觉 patch 的短窗口场景下，相较
> PyTorch Math、PyTorch SDPA 和手写 CUDA fused kernel 分别加速 91.39x、21.33x
> 和 9.04x，并通过 shape-aware dispatch 在长序列回退外部 FlashAttention。

> 使用 Kineto 与 Nsight Systems 对 nano-vllm/vLLM 做算子和 launch 级分析，定位
> nano decode 每 token kernel launches 为 1061 对 571，确认主要差距来自 MRoPE、
> Elementwise/Layout、Sampler 和 CPU launch gap，而非 GEMM/FlashAttention 本体。

### 不应写的结论

```text
整个 nano-vllm 比 PyTorch 快 91x
Triton 全面超过 CUDA 和 FlashAttention
AWQ 同时实现显存与速度优化
已经实现生产级 CUDA FlashAttention
已经完整兼容 OpenAI API
```

## 28. 数据与扩展文档索引

```text
docs/qwen2_5_vl_offline_inference.md
docs/qwen2_5_vl_online_server.md
docs/qwen2_5_vl_awq.md
docs/attention_backends.md
docs/attention_backend_benchmark.md
docs/custom_kernels_and_triton.md
docs/qwen2_5_vl_vllm_comparison.md
docs/qwen2_5_vl_vllm_operator_comparison.md

profiles/attention_pytorch_cuda_triton_rtx5080.json
profiles/awq_online/summary.json
profiles/operator_comparison/summary.json
profiles/nsight/attention_backends_summary.json
profiles/nsight/attention_backends.nsys-rep
```
