# Qwen2.5-VL：vLLM 与 nano-vllm 性能对比

本文记录 2026-07-30 在同一台机器上，以同一张图片、同一模型和同一个 HTTP
benchmark 客户端对 vLLM 0.24.0 与 nano-vllm 进行的 C1/C2/C4 对比。

## 1. 结论先行

本轮是 `BF16 + eager + 单 GPU + 单图 + 32 token 输出` 的在线服务测试。

| 并发 | 框架 | 请求吞吐 req/s | 输出吞吐 tok/s | 平均 TTFT ms | 平均 TPOT ms | 平均 E2E ms |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| C1 | nano-vllm | 1.20 | 38.25 | **62.69** | 24.96 | 836.42 |
| C1 | vLLM | **1.56** | **49.79** | 354.42 | **9.30** | **642.60** |
| C2 | nano-vllm | 1.69 | 54.01 | **77.64** | 35.70 | 1184.33 |
| C2 | vLLM | **2.11** | **67.65** | 492.15 | **14.61** | **945.07** |
| C4 | nano-vllm | 2.07 | 66.38 | **152.77** | 57.20 | 1926.02 |
| C4 | vLLM | **2.69** | **86.16** | 889.13 | **19.12** | **1481.71** |

这组数据说明：

- nano-vllm 的平均 TTFT 是 vLLM 的约 `1/5.65` 到 `1/6.34`。
- vLLM 的平均 TPOT 是 nano-vllm 的约 `1/2.44` 到 `1/2.99`。
- vLLM 的请求吞吐比 nano-vllm 高约 `25.3%` 到 `30.2%`。
- vLLM 的平均端到端延迟低约 `20.2%` 到 `23.2%`。

因此不能用“谁全面更快”概括结果。当前 nano-vllm 的单图首 token 路径很轻，
但 decode 连续批处理和高并发吞吐仍明显落后于 vLLM。

## 2. 测试环境

| 项目 | 配置 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 5080，16303 MiB |
| Driver | 595.58.03 |
| GPU power limit | 360 W |
| Python | 3.10.20 |
| PyTorch | 2.11.0+cu130 |
| CUDA runtime | 13.0 |
| cuDNN | 9.19.0 |
| Transformers | 5.12.1 |
| FlashAttention | 2.8.3.post1 |
| Triton | 3.6.0 |
| vLLM | 0.24.0 |
| 模型 | `/home/agua/models/Qwen2.5-VL-3B-Instruct` |

## 3. 输入与正确性基线

测试数据位于：

```text
benchmarks/data/qwen2_5_vl_dog.jsonl
```

请求内容：

```text
image: assets/dog.png
prompt: 描述这张图片
```

Processor 结果：

```text
image size: 1254 x 1254
input tokens: 2049
image tokens: 2025
image_grid_thw: [[1, 90, 90]]
pixel_values: [8100, 1176], FP32
```

正式性能测试前，已经单独做过 greedy 正确性 smoke。vLLM 与 nano-vllm
前 4 个输出 token 完全一致：

```text
token_ids: [108893, 45930, 101987, 99593]
text: 这张图片展示了一
```

性能请求设置 `temperature=0`、`top_p=1` 和 `ignore_eos=true`，确保每个请求
输出 32 个 token。并发执行时，vLLM 的少量后续 token 会随 batch shape 出现
BF16 数值路径差异，但输出语义一致。性能测试不替代独立的逐 token 正确性回归。

## 4. 控制变量

两个服务共同保持：

```text
tensor_parallel_size = 1
dtype = BF16
enforce_eager = true
max_model_len = 4096
max_num_seqs = 4
max_num_batched_tokens = 4096
gpu_memory_utilization = 0.72
decoder attention = FlashAttention
vision attention = FlashAttention
prefix caching = disabled
同一张本地图片
同一 chat template 和 prompt
同一个 vllm bench serve 客户端
```

vLLM 的 chunked prefill 保持开启，但本例 2049-token prompt 小于 4096-token
单步预算，因此不会因预算不足被切成多个 prefill chunk。

vLLM 同时关闭 Torch Compile、CUDA Graph、异步调度和多模态 processor cache。
这是为了接近 nano-vllm 当前始终 eager、无 CUDA Graph、无图片缓存的执行条件，
不是 vLLM 的最佳生产配置。

测试协议：

```text
warmup requests: 3
measured requests: 12
arrival: request_rate=inf
concurrency: 1 / 2 / 4
output length: 32
streaming: enabled
percentiles: P50 / P90 / P99
```

## 5. 完整结果

| 并发 | 框架 | TTFT mean/P90 ms | TPOT mean/P90 ms | E2E mean/P90 ms | 成功/失败 |
| ---: | --- | ---: | ---: | ---: | ---: |
| C1 | nano-vllm | 62.69 / 64.08 | 24.96 / 25.38 | 836.42 / 848.61 | 12 / 0 |
| C1 | vLLM | 354.42 / 359.64 | 9.30 / 9.37 | 642.60 / 647.46 | 12 / 0 |
| C2 | nano-vllm | 77.64 / 90.46 | 35.70 / 36.42 | 1184.33 / 1198.62 | 12 / 0 |
| C2 | vLLM | 492.15 / 628.13 | 14.61 / 19.03 | 945.07 / 954.38 | 12 / 0 |
| C4 | nano-vllm | 152.77 / 239.14 | 57.20 / 58.14 | 1926.02 / 2005.22 | 12 / 0 |
| C4 | vLLM | 889.13 / 1166.18 | 19.12 / 36.12 | 1481.71 / 1494.04 | 12 / 0 |

原始 JSON：

```text
profiles/framework_comparison/nano-c1-o32.json
profiles/framework_comparison/nano-c2-o32.json
profiles/framework_comparison/nano-c4-o32.json
profiles/framework_comparison/vllm-c1-o32.json
profiles/framework_comparison/vllm-c2-o32.json
profiles/framework_comparison/vllm-c4-o32.json
```

这些 JSON 由 `vllm bench serve --save-detailed` 生成，包含每个请求的 input
length、output length、TTFT、ITL、生成文本和错误信息，而不只有汇总值。

## 6. 指标如何理解

### TTFT

Time To First Token，从客户端发出请求到收到第一个 token。它包含：

```text
HTTP 和请求解析
base64 图片解码
AutoProcessor 预处理
视觉 Encoder
文本 prefill
第一个 token 采样和 SSE 返回
```

本轮 nano-vllm TTFT 更低，只能说明这套固定 eager 配置下的整条首 token 路径
更短，不能单独归因于某一个 Attention kernel。

### TPOT

Time Per Output Token，不含第一个 token 的平均输出 token 时间。它主要反映：

```text
连续 decode forward
paged KV Cache 读取
batch 调度
采样
流式返回
```

vLLM 在 C1/C2/C4 的 TPOT 都明显更低。这是下一轮 nano-vllm profiling 应优先
定位的路径。

### 请求吞吐与输出吞吐

请求吞吐取决于完整请求时长；输出吞吐只统计生成 token。因为本轮所有请求都固定
生成 32 token，两者趋势一致。并发从 C1 提高到 C4 后：

```text
nano-vllm output throughput: 38.25 -> 66.38 tok/s，1.74x
vLLM output throughput:      49.79 -> 86.16 tok/s，1.73x
```

两者都通过 batching 获益，但 vLLM 在每个并发档位仍领先约 30%。

## 7. 复现步骤

### 7.1 启动 nano-vllm

```bash
/home/agua/anaconda3/envs/yolo26/bin/python \
  examples/qwen2_5_vl_server.py \
  --model /home/agua/models/Qwen2.5-VL-3B-Instruct \
  --served-model-name /home/agua/models/Qwen2.5-VL-3B-Instruct \
  --host 127.0.0.1 \
  --port 8000 \
  --max-model-len 4096 \
  --max-num-seqs 4 \
  --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.72 \
  --max-concurrent-requests 4 \
  --attention-backend flash_attn \
  --vision-attention-backend flash_attn
```

另一个终端执行：

```bash
/home/agua/anaconda3/envs/yolo26/bin/python \
  benchmarks/qwen2_5_vl_serving_benchmark.py \
  --framework-label nano-vllm
```

### 7.2 启动 vLLM

先停止 nano-vllm，确认 GPU 上没有残留模型进程，再执行：

```bash
VLLM_WORKER_MULTIPROC_METHOD=spawn \
/home/agua/anaconda3/envs/yolo26/bin/vllm serve \
  /home/agua/models/Qwen2.5-VL-3B-Instruct \
  --served-model-name /home/agua/models/Qwen2.5-VL-3B-Instruct \
  --host 127.0.0.1 \
  --port 8000 \
  --model-impl vllm \
  --dtype bfloat16 \
  --tensor-parallel-size 1 \
  --max-model-len 4096 \
  --max-num-seqs 4 \
  --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.72 \
  --enforce-eager \
  --no-async-scheduling \
  --no-enable-prefix-caching \
  --mm-processor-cache-gb 0 \
  --limit-mm-per-prompt '{"image":1,"video":0}' \
  --attention-backend FLASH_ATTN \
  --mm-encoder-attn-backend FLASH_ATTN \
  --generation-config vllm \
  --stream-interval 1 \
  --disable-log-stats
```

另一个终端执行：

```bash
/home/agua/anaconda3/envs/yolo26/bin/python \
  benchmarks/qwen2_5_vl_serving_benchmark.py \
  --framework-label vllm
```

`VLLM_WORKER_MULTIPROC_METHOD=spawn` 是必要的，避免已经初始化 CUDA 后再通过
`fork` 创建 worker 导致 CUDA re-initialization 错误。

## 8. 结果边界

这是一轮开发机上的小样本定点测试，不是完整 benchmark suite：

- 只有一张 1254x1254 图片和一个 prompt。
- 只有 32-token 短输出。
- 每个并发档位只有 12 个正式请求。
- 未固定 GPU application clock，也未独占整台机器。
- vLLM 为控制变量关闭了 CUDA Graph 和 Compile，不能代表其最佳生产性能。
- nano-vllm 与 vLLM 内部的调度、缓存和预处理实现不同，无法做到逐内部组件相同。

后续严谨对比应增加至少三条轴：

```text
图片尺寸: 小 / 中 / 大
输出长度: 8 / 32 / 128
并发: C1 / C2 / C4 / C8
```

每组至少重复 3 个独立 run，报告 run 间 median，并同时抓取 GPU 功耗、SM 利用率
和显存峰值。

## 9. 算子级差距

端到端结果之后，已在同一张 `assets/dog.png` 上补充 Kineto CUDA kernel profile。
测试分别采集 O1 与 O32，并用 `(O32 - O1) / 31` 估算 decode 单 token：

| 阶段 | nano CUDA 时间 | vLLM CUDA 时间 | nano 相对差距 | nano/vLLM launches |
| --- | ---: | ---: | ---: | ---: |
| O1：视觉编码 + prefill + 首 token | 354.42 ms | 284.24 ms | +24.7% | 2.59x |
| O32：完整 32-token 请求 | 623.65 ms | 537.97 ms | +15.9% | 1.90x |
| 估算 decode/token | 8.685 ms | 8.185 ms | +6.1% | 1.86x |

O32 中 GEMM 为 `447.89/449.30 ms`，Attention 为 `42.80/42.43 ms`，两边基本
持平。主要差距是 nano-vllm 的 Elementwise/Layout 为 `108.32 ms`，vLLM 只有
`12.26 ms`；decode 每 token 的 kernel launches 为 `1061/571`。

详细方法、分类结果、复现命令和优化优先级见：

```text
docs/qwen2_5_vl_vllm_operator_comparison.md
profiles/operator_comparison/summary.json
```

## 10. 面试回答

### 为什么使用 vLLM 自己的 benchmark 客户端测 nano-vllm？

因为 nano-vllm 已实现 OpenAI 兼容的 `/v1/chat/completions` 和 SSE。统一客户端
可以保证请求构造、token 统计、TTFT/TPOT 定义和计时位置一致，减少“两个框架
各报一套指标”的偏差。

### 为什么同时看 TTFT 和 TPOT？

VLM 的 TTFT 包含图片预处理、视觉编码和文本 prefill；TPOT 更接近自回归 decode。
只看总延迟会把两个阶段混在一起，无法判断优化方向。本轮结果正好显示 nano-vllm
首 token 快，但 decode 慢。

### 为什么要设置 ignore_eos？

若某个框架提前生成 EOS，每个请求的实际输出长度不同，吞吐和 E2E 就不可直接
比较。`ignore_eos=true` 强制每个请求生成相同的 32 token。

### 下一步优先优化什么？

先 profile C1 和 C4 的 decode step，分解 scheduler、KV cache write、
`flash_attn_with_kvcache`、LM head、采样和 SSE 开销。当前证据不支持直接重写
Decoder Attention；需要先确认 2.4x 到 3.0x 的 TPOT 差距具体落在哪些 kernel
和 CPU gap 上。

## 11. 官方参考

- [vLLM Supported Models](https://docs.vllm.ai/en/latest/models/supported_models/)
- [vLLM Multimodal Offline Inference](https://docs.vllm.ai/en/latest/examples/generate/multimodal/)
- [vLLM Benchmark CLI](https://docs.vllm.ai/en/latest/benchmarking/cli/)
- [vLLM Engine Arguments](https://docs.vllm.ai/en/latest/configuration/engine_args/)
- [vLLM OpenAI-Compatible Server](https://docs.vllm.ai/en/latest/serving/online_serving/openai_compatible_server/)
