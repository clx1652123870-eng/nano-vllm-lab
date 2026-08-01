# Qwen2.5-VL：nano-vllm 与 vLLM 算子级性能对比

本文记录 2026-08-01 在同一张 `assets/dog.png` 上，对 nano-vllm 与 vLLM
进行的 CUDA 算子级 profile。目标不是只给出一个端到端延迟，而是回答：

```text
nano-vllm 慢在哪里？
差距来自 Attention、GEMM，还是大量小算子和 CPU launch gap？
下一步应该先优化什么？
```

## 1. 结论

在本轮 `BF16 + eager + 单 GPU + 单请求 + 固定 32 token` 测试中：

| 阶段 | nano CUDA 时间 | vLLM CUDA 时间 | nano 相对差距 | nano/vLLM launches |
| --- | ---: | ---: | ---: | ---: |
| O1：视觉编码 + prefill + 首 token | 354.42 ms | 284.24 ms | +24.7% | 2534 / 978，2.59x |
| O32：完整 32-token 请求 | 623.65 ms | 537.97 ms | +15.9% | 35425 / 18679，1.90x |
| 估算 decode/token | 8.685 ms | 8.185 ms | +6.1% | 1061 / 571，1.86x |

关键判断：

- GEMM 和 Attention 不是当前主要差距。O32 中两边 GEMM 都约 `448 ms`，
  Attention 都约 `42.5 ms`。
- nano-vllm 的主要 GPU 差距来自逐元素运算、dtype/layout 转换、`cat/copy`
  等碎片化路径。O32 为 `108.32 ms`，vLLM 为 `12.26 ms`。
- nano-vllm 每个 decode token 多启动约 `490` 个 CUDA kernel。小 kernel 本身不一定
  很慢，但会产生明显的 CPU launch 和同步间隙。
- nano-vllm 的 greedy 请求仍计算随机采样分支，sampling CUDA 时间约为 vLLM 的
  `19.3x`。绝对值不大，但这是明确且容易修复的冗余。
- 按非 profile warmup 粗略估算，decode 每 token wall time 为 `14.82 ms` 对
`9.38 ms`。扣除 CUDA kernel 之和后，nano-vllm 约有 `6.14 ms/token` 的主机侧与
  launch 间隙，vLLM 约为 `1.20 ms/token`。

所以当前优化顺序不应该是重写 FlashAttention。更合理的是先减少 MRoPE、
RMSNorm/residual、布局转换和 sampler 产生的小 kernel，再优化 CPU 调度与 launch。

## 2. “算子加载”具体指什么

这里 profile 的是模型完成初始化之后，推理过程中实际执行的 CUDA kernel：

```text
Linear/GEMM
Attention
RoPE/MRoPE
RMSNorm
SwiGLU/SiLU-and-Mul
KV Cache 写入
Sampling
逐元素、复制和布局转换
```

模型权重读取、模块构造、首次 CUDA context 初始化和首次 JIT 编译不计入 CUDA
算子时间。严格说这应叫“算子执行 profile”，而不是“算子加载时间”。

## 3. 控制变量

| 项目 | 设置 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 5080 |
| PyTorch / CUDA | 2.11.0+cu130 / 13.0 |
| vLLM | 0.24.0 |
| FlashAttention / Triton | 2.8.3.post1 / 3.6.0 |
| 模型 | `/home/agua/models/Qwen2.5-VL-3B-Instruct` |
| 图片 | `assets/dog.png`，1254x1254 |
| prompt | `描述这张图片` |
| 输入 | 2049 tokens，其中 2025 image tokens |
| pixel values | `[8100, 1176]`，FP32 processor 输出 |
| dtype / TP | BF16 / TP=1 |
| 执行模式 | eager，关闭 CUDA Graph |
| prefix cache | 关闭 |
| 输出 | greedy，`ignore_eos=true`，固定 1 或 32 tokens |
| profile 请求 | warmup 2 次，Kineto 正式采集 2 次 |

nano-vllm 使用当前最佳配置：视觉侧 `hybrid`，文本 decoder 使用
`flash_attn`。vLLM 使用其 Qwen2.5-VL 原生实现并保持 eager。为让 PyTorch Kineto
能看到 vLLM worker 内部 kernel，脚本设置：

```text
VLLM_ENABLE_V1_MULTIPROCESSING=0
```

图片的 PIL/Processor 预处理在 profile 区间之外，但视觉编码器 forward 在区间内。
因此 O1 适合比较 GPU 视觉编码和 prefill，不等价于 HTTP 服务的完整 TTFT。

## 4. 测量方法

单次 O32 无法准确分开 prefill 和 decode，因此分别启动独立进程测两组：

```text
O1  = 多模态 prefill + 首 token sampling
O32 = 多模态 prefill + 首 token + 31 个 decode token
decode/token ~= (O32 - O1) / 31
```

脚本读取 Kineto 的原始 CUDA device events，按 kernel 名称聚合。这样可以避免把
PyTorch 父算子与它的子 kernel 重复相加。若多个 stream 存在重叠，kernel 时间之和
仍可能高于真实 wall time；本轮请求基本为单流执行。

O1 和 O32 来自不同进程，差分会包含少量 run-to-run 噪声，因此 decode 分类中极小
的负值应视为 0，而不能解释成负耗时。

## 5. 分阶段结果

### 5.1 O1：视觉编码、prefill 与首 token

| 类别 | nano ms | vLLM ms | nano launches | vLLM launches |
| --- | ---: | ---: | ---: | ---: |
| GEMM | 229.632 | 225.491 | 307 | 276 |
| Attention | 29.851 | 29.368 | 68 | 68 |
| Elementwise/Layout | 78.806 | 5.829 | 1691 | 260 |
| Activation | 8.274 | 16.568 | 69 | 69 |
| Normalization | 1.391 | 2.952 | 138 | 138 |
| Memory | 2.407 | 0.781 | 101 | 61 |
| Sampling | 0.090 | 0.005 | 2 | 1 |

prefill 的核心矩阵乘和 Attention 接近。`70.18 ms` 的 CUDA 总差距中，
Elementwise/Layout 单类差距约 `72.98 ms`，已经足以解释主要问题。

### 5.2 O32：完整 32-token 请求

| 类别 | nano ms | vLLM ms | nano/vLLM |
| --- | ---: | ---: | ---: |
| GEMM | 447.886 | 449.296 | 1.00x |
| Attention | 42.797 | 42.430 | 1.01x |
| Elementwise/Layout | 108.321 | 12.260 | 8.84x |
| Activation | 9.577 | 18.910 | 0.51x |
| Normalization | 4.458 | 7.091 | 0.63x |
| Sampling | 2.878 | 0.149 | 19.34x |
| KV Cache | 1.153 | 1.902 | 0.61x |

分类结果不能直接理解为 nano 的 RMSNorm 比 vLLM 快，因为 nano 的 MRoPE、
RMSNorm 和视觉运算中有部分通用 PyTorch/Triton kernel 会被归入
Elementwise/Layout。该分类最适合定位“融合算子”与“碎片化小算子”的总体差异。

### 5.3 估算 decode/token

| 类别 | nano ms/token | vLLM ms/token | nano/vLLM | launches nano/vLLM |
| --- | ---: | ---: | ---: | ---: |
| GEMM | 7.040 | 7.220 | 0.98x | 145 / 145 |
| Attention | 0.418 | 0.421 | 0.99x | 72 / 72 |
| Elementwise/Layout | 0.952 | 0.207 | 4.59x | 688 / 121 |
| Normalization | 0.099 | 0.134 | 0.74x | 73 / 73 |
| Activation | 0.042 | 0.076 | 0.56x | 36 / 36 |
| Sampling | 0.090 | 0.0046 | 19.35x | 2 / 1 |
| KV Cache | 0.033 | 0.054 | 0.62x | 36 / 36 |

两边 decode 中 GEMM 都启动 145 次、Attention 都启动 72 次，而且时间接近。
nano-vllm 的额外 launches 主要集中在通用逐元素和布局操作。

## 6. 实现差异

| 功能 | nano-vllm 当前路径 | vLLM profile 中的路径 |
| --- | --- | --- |
| Decoder Attention | 外部 `flash_attn_with_kvcache` | FlashAttention split-KV kernels |
| Vision Attention | Hybrid：window Triton、full FlashAttention | FlashAttention kernels |
| GEMM/GEMV | PyTorch 调用 cuBLAS/CUTLASS | cuBLAS/CUTLASS |
| SwiGLU | `torch.compile` 生成 Triton fused kernel | `vllm::act_and_mul_kernel` |
| RMSNorm + residual | `torch.compile` | `vllm::fused_add_rms_norm_kernel` |
| Text MRoPE | 多个 PyTorch float/cat/mul/add 操作 | `rotary_kernel`、Triton MRoPE |
| KV Cache write | nano-vllm Triton store kernel | `reshape_and_cache_flash_kernel` |
| Greedy sampling | greedy 与随机分支都计算，再 `where` | greedy fast path |

原始 kernel 名进一步显示，nano-vllm 中频繁出现：

```text
at::native::elementwise_kernel
at::native::vectorized_elementwise_kernel
CatArrayBatchedCopy
bfloat16_copy_kernel_cuda
```

vLLM 则能看到明确的领域融合 kernel：

```text
vllm::fused_add_rms_norm_kernel
vllm::act_and_mul_kernel
rotary_kernel
_triton_mrope_forward
reshape_and_cache_flash_kernel
```

## 7. 为什么 wall time 差距大于 CUDA 算子差距

用每个进程最快的初始化后 warmup 作为参考，得到：

| 指标 | nano-vllm | vLLM | nano/vLLM |
| --- | ---: | ---: | ---: |
| O1 wall | 367.08 ms | 307.63 ms | 1.19x |
| O32 wall | 826.52 ms | 598.52 ms | 1.38x |
| 估算 decode wall/token | 14.82 ms | 9.38 ms | 1.58x |
| wall 减 CUDA/token | 6.14 ms | 1.20 ms | 5.12x |

这里的 wall 数字只用于解释趋势，不是正式延迟结果。它说明大量小 kernel 除了增加
GPU 执行时间，还增加 Python、dispatcher、kernel launch 和同步之间的空隙。

在线服务此前测得的 C1 TPOT 为 nano-vllm `24.96 ms`、vLLM `9.30 ms`，差距比
本次离线 eager profile 更大，说明服务调度、请求状态更新和 SSE 路径也需要单独用
Nsight Systems 分析，不能全部归因于模型算子。

## 8. 优化优先级

1. 为 `temperature=0` 增加真正的 greedy sampler fast path，跳过 FP32 logits
   缩放、softmax、exponential 和随机采样。
2. 为 Qwen2.5-VL 文本 MRoPE 实现单 kernel CUDA/Triton 路径，融合位置读取、
   sin/cos 选择、rotate-half 和输出写回。
3. 核对 `torch.compile` 的 residual + RMSNorm 是否在所有 prefill/decode shape 上
   稳定融合；必要时接入已有自定义 CUDA/Triton RMSNorm。
4. 用 trace 定位 `cat/copy/BF16<->FP32` 的来源，优先去掉视觉 RoPE 和文本 MRoPE
   中的中间 tensor 与布局转换。
5. 优化 `prepare_decode`、scheduler 和 Python launch 路径。完成 eager 对齐后，再用
   CUDA Graph 降低 decode launch 开销。
6. Attention 和 GEMM 暂不重写。本轮数据表明它们已与 vLLM 基本处于同一水平。

## 9. 复现命令

以下四条命令应分别运行；每条都会单独加载模型，以保持 O1/O32 互不污染：

```bash
PYTHON=/home/agua/anaconda3/envs/yolo26/bin/python

$PYTHON benchmarks/qwen2_5_vl_operator_profile.py \
  --engine nano --max-new-tokens 1 \
  --output-json profiles/operator_comparison/nano-dog-o1.json

$PYTHON benchmarks/qwen2_5_vl_operator_profile.py \
  --engine nano --max-new-tokens 32 \
  --output-json profiles/operator_comparison/nano-dog-o32.json

$PYTHON benchmarks/qwen2_5_vl_operator_profile.py \
  --engine vllm --max-new-tokens 1 \
  --output-json profiles/operator_comparison/vllm-dog-o1.json

$PYTHON benchmarks/qwen2_5_vl_operator_profile.py \
  --engine vllm --max-new-tokens 32 \
  --output-json profiles/operator_comparison/vllm-dog-o32.json

$PYTHON benchmarks/compare_operator_profiles.py \
  --output-json profiles/operator_comparison/summary.json
```

需要在 Perfetto 或 Chrome trace viewer 中查看 CPU/CUDA 时间线时，可以在任一 profile
命令后增加：

```bash
--trace-output profiles/operator_comparison/nano-dog-o32-trace.json
```

## 10. 正确性与结果边界

两边 O32 的前 `21` 个 greedy token 完全一致，之后因 BF16 backend 和归约顺序差异
发生分叉，但都正确描述了图片中的金毛幼犬。本 profile 用于性能定位，不能替代
Transformers 对齐测试。

本轮仍有以下边界：

- 只有一张高分辨率图片、一个 prompt 和 batch size 1。
- O1/O32 各自只 profile 2 个请求，未固定 GPU application clock。
- vLLM 被固定为 eager、关闭 multiprocessing 和 prefix cache，不代表其生产最优性能。
- kernel 名称分类是启发式规则，不是框架内部算子耗时的严格调用树。
- decode/token 是两次独立 profile 的差分估算，应该用 Nsight Systems 单步标记继续验证。

## 11. 面试表述

可以这样说明这轮工作：

> 我没有只比较端到端延迟，而是固定图片、模型、精度、输出长度和 eager 模式，分别
> profile 首 token 与 32-token 请求，再通过差分估算 decode 单 token。结果显示
> nano-vllm 的 GEMM 和 FlashAttention 与 vLLM 基本持平，但每个 decode token 的
> CUDA kernel launch 是 1061 对 571，Elementwise/Layout 时间是 4.59 倍，最终
> decode GPU 时间高 6.1%，估算 wall time 高 57.9%。因此下一步优先做 greedy sampler
> fast path、融合 Qwen2.5-VL MRoPE、减少布局转换和 CPU launch gap，而不是盲目重写
> Attention。

原始数据：

```text
profiles/operator_comparison/nano-dog-o1.json
profiles/operator_comparison/nano-dog-o32.json
profiles/operator_comparison/vllm-dog-o1.json
profiles/operator_comparison/vllm-dog-o32.json
profiles/operator_comparison/summary.json
```
