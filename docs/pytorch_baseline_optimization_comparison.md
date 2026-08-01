# PyTorch 基线与优化后端性能对比

本文专门记录 nano-vllm 的 PyTorch 参考路径与优化后端对比，用于性能回归和简历
数据。所有结论都限定在 Qwen2.5-VL 的视觉 Encoder，不把尚未替换的文本 Decoder
算入自定义 Attention 收益。

原始结果：

```text
profiles/attention_pytorch_vs_hybrid_rtx5080.json
profiles/attention_pytorch_math_vs_hybrid_rtx5080.json
profiles/qwen2_5_vl_pytorch_vision_baseline.json
profiles/qwen2_5_vl_hybrid_vs_pytorch.json
profiles/qwen2_5_vl_dog_pytorch_math_oom.json
profiles/qwen2_5_vl_logo_pytorch_math.json
profiles/qwen2_5_vl_logo_hybrid.json
profiles/kernels_pytorch_vs_custom_rtx5080.json
```

## 1. 为什么有两种 PyTorch 基线

### `torch_sdpa`

调用：

```python
torch.nn.functional.scaled_dot_product_attention(...)
```

PyTorch 根据 shape 和硬件自动选择内部实现。nano-vllm 的参考 adapter 仍需要把
packed-varlen 输入拆成单条 sequence，因此 Window case 会调用 144 次 SDPA。

它代表：

```text
直接使用 PyTorch 高层 SDPA API 的工程基线
```

但不能把它称为纯 cuBLAS 或最基础 Attention，因为 PyTorch 可能自动选择 fused
SDPA kernel。

### `torch_math`

新增后端：

```python
with sdpa_kernel(SDPBackend.MATH):
    F.scaled_dot_product_attention(...)
```

它强制执行 PyTorch Math 路径，逻辑上对应：

```text
QK^T -> Softmax -> P @ V
```

矩阵乘由 PyTorch CUDA/cuBLAS 等基础算子完成，并物化 Attention 中间结果。它是
本文使用的严格“PyTorch/基础 CUDA 数学路径”基线。

## 2. 优化后端

优化端使用 `hybrid`：

```text
Window max sequence length <= 64:
    自定义 Triton packed-varlen online-softmax Attention

Full sequence length > 64:
    外部 FlashAttention CUDA
```

因此这是 shape-aware Triton/CUDA Hybrid，不是全 Triton 实现。文本 Decoder 在
所有端到端实验中都保持相同的 FlashAttention Paged-KV 路径。

## 3. 高分辨率真实 Shape 微基准

输入：

```text
GPU: RTX 5080 16 GB
image: assets/dog.png
raw image: 1254 x 1254
pixel_values: [8100, 1176]
vision patches: 8100
heads: 16
head_dim: 80
```

Window：

```text
144 个 packed sequences
121 x length 64
22 x length 16
1 x length 4
```

Full：

```text
1 x length 8100
```

结果使用 5 次 warmup、20 次 measure 的 P50：

| Case | PyTorch Math | Hybrid | 加速比 |
| --- | ---: | ---: | ---: |
| Window | 9.736 ms | **0.115 ms** | **84.7x** |
| Full | 66.489 ms | **4.433 ms** | **15.0x** |

按照 Qwen2.5-VL 的 28 个 Window layer 和 4 个 Full layer加权：

```text
PyTorch Math:
28 * 9.736 + 4 * 66.489 = 538.56 ms

Hybrid:
28 * 0.115 + 4 * 4.433 = 20.95 ms
```

即：

```text
视觉 Attention 子项 P50 加速约 25.7x
延迟降低约 96.1%
```

这不是完整模型 TTFT，而是 32 层 Attention kernel/adapter 调用的加权估算。

## 4. 高分辨率端到端结果

在同一个 8100-patch 输入上，严格 PyTorch Math 端到端运行失败：

```text
torch.OutOfMemoryError
Full Attention tried to allocate: 3.91 GiB
GPU capacity: 15.44 GiB
```

原因是 Math backend 需要物化长序列 Attention 中间矩阵。仅一个 FP32：

```text
[1, 16, 8100, 8100]
```

就需要约：

```text
16 * 8100 * 8100 * 4 bytes ~= 3.91 GiB
```

而 Hybrid 使用分块 online softmax，不需要保存完整矩阵，可以正常完成推理。

因此该输入不能给出严格 Math baseline 的端到端加速比，只能报告：

```text
PyTorch Math: OOM
Hybrid:       可运行
```

## 5. PyTorch 自动 SDPA 的高分辨率端到端对比

为了在同一张高分辨率图片上获得可完成的端到端基线，使用 PyTorch 自动 SDPA。
控制变量：

```text
same BF16 model
same assets/dog.png
2049 input tokens
32 output tokens
temperature=0
single GPU, eager
3 warmup + 20 measure
```

P50 结果：

| 指标 | PyTorch SDPA | Hybrid | 改善 |
| --- | ---: | ---: | ---: |
| TTFT | 457.94 ms | **364.77 ms** | **降低 20.3% / 1.26x** |
| Prefill throughput | 4474 tok/s | **5617 tok/s** | **提升 25.5%** |
| 32-token E2E | 885.94 ms | **797.87 ms** | **降低 9.9%** |
| E2E throughput | 36.12 tok/s | **40.11 tok/s** | **提升 11.0%** |
| Decode TPOT | **13.84 ms** | 14.03 ms | 基本不变 |
| Peak allocated | 8.07 GB | 8.07 GB | 相同 |

Decode 没有加速是预期结果，因为图片只在 Prefill 运行，两个实验的文本 Decoder
完全相同。

两边前 21 个 greedy token 一致，之后因 BF16 Attention 后端数值路径不同而分叉，
输出语义一致。每个后端的 20 次测量内部输出保持确定。

## 6. 严格 Math 基线的可运行端到端对比

为了测量严格 Math/cuBLAS 路径，补充较小图片：

```text
image: assets/logo.png
input tokens: 690
image tokens after merge: 666
pixel_values: [2664, 1176]
3 warmup + 20 measure
```

P50 结果：

| 指标 | PyTorch Math | Hybrid | 改善 |
| --- | ---: | ---: | ---: |
| TTFT | 237.03 ms | **105.16 ms** | **降低 55.6% / 2.25x** |
| Prefill throughput | 2911 tok/s | **6562 tok/s** | **提升 125.4%** |
| 32-token E2E | 658.18 ms | **532.54 ms** | **降低 19.1%** |
| E2E throughput | 48.62 tok/s | **60.09 tok/s** | **提升 23.6%** |
| Peak allocated | 8.88 GB | **8.05 GB** | **降低 0.83 GB** |

两个实验的 Decode TPOT 接近，差异来自测量波动。两边前 15 个 greedy token
一致，后续文本都能正确描述 Nano-vLLM 标志。

## 7. 基础算子 PyTorch 对比

基础算子使用 20 次 warmup、100 次 measure，以下为 P50。这里的“Best”允许保留
更快的 PyTorch 实现，不强制使用自定义 kernel。

| 算子和 Shape | PyTorch | Best | 加速比 |
| --- | ---: | ---: | ---: |
| Softmax `[1024,2048]` | 0.0197 ms | CUDA 0.0158 ms | 1.24x |
| SiLU-Mul `[1,22016]` | 0.0117 ms | CUDA 0.0089 ms | 1.31x |
| SiLU-Mul `[32,22016]` | 0.0132 ms | CUDA 0.0090 ms | 1.47x |
| SiLU-Mul `[2049,22016]` | 0.2815 ms | CUDA 0.1720 ms | 1.64x |

本轮重复测试中，RMSNorm 和 Matmul 的最快实现仍是 PyTorch：

```text
RMSNorm -> PyTorch CUDA
BF16 Matmul -> PyTorch/cuBLAS
```

这说明最佳 dispatcher 应该按算子和 shape 选择，而不是所有算子都强制换成
Triton/CUDA。基础算子目前只完成 standalone benchmark，不能把这些微基准收益
叠加后宣称为模型端到端收益。

## 8. 推荐简历表述

可以写：

> 为 nano-vLLM 的 Qwen2.5-VL 视觉编码器实现 shape-aware Triton/CUDA Hybrid
> Attention，针对短 Window 使用自定义 Triton online-softmax kernel、长序列使用
> 分块 FlashAttention；在 RTX 5080、8100 视觉 patch 下，相较 PyTorch Math
> 基线将 32 层视觉 Attention 加权 P50 延迟降低 96.1%（25.7x），并避免基础路径
> 3.91 GiB 中间张量导致的 OOM；在可运行的 2664-patch 端到端测试中将 TTFT 从
> 237.0 ms 降至 105.2 ms（2.25x），Prefill 吞吐提升 125%。

较保守、适合一行简历的版本：

> 实现 Qwen2.5-VL Vision packed-varlen Hybrid Attention 与 CUDA/Triton 算子
> benchmark；相较 PyTorch SDPA reference，在 8100-patch 单图离线推理中降低
> TTFT 20.3%、提升 Prefill 吞吐 25.5%，并通过 shape-aware dispatch 保留 cuBLAS
> GEMM 等更快库实现。

不能写：

```text
整个 nano-vllm 比 PyTorch 快 25.7x
文本 Decode 加速 25.7x
全部算子都由 Triton 实现
Hybrid 完全不使用 FlashAttention
```

## 9. 面试追问

### 为什么严格 Math 基线比自动 SDPA 慢很多？

Math backend 会物化 Attention score/probability，并通过多个基础 kernel 完成 QK、
Softmax 和 PV。自动 SDPA 可以选择 fused backend，避免部分中间结果和 launch。

### 25.7x 为什么没有全部转化为 TTFT？

25.7x 只属于 Vision Attention 子项。完整 TTFT 还包括图片预处理、PatchEmbed、
QKV/Output projection、视觉 MLP、PatchMerger、文本 Prefill、LM head 和调度。

### 为什么高分辨率只报告 OOM、不报告端到端加速比？

基线没有完成就不存在合法的 latency。可以报告优化路径解决了 OOM，也可以报告
独立 Attention 微基准，但不能虚构端到端时间。

### 为什么还保留 PyTorch/cuBLAS？

实测 BF16 Matmul 和多数 RMSNorm shape 仍是 PyTorch 更快。高性能框架的目标是
最短端到端延迟，不是自定义 kernel 数量最多。

## 10. 复现命令

严格 Math Attention 微基准：

```bash
python benchmarks/attention_backend_benchmark.py \
  --backends torch_math,hybrid \
  --cases window,full \
  --warmup-iters 5 \
  --measure-iters 20 \
  --output-json profiles/attention_pytorch_math_vs_hybrid_rtx5080.json
```

PyTorch 自动 SDPA 高分辨率基线：

```bash
python examples/qwen2_5_vl_profile.py \
  --image assets/dog.png \
  --vision-attention-backend torch_sdpa \
  --max-new-tokens 32 \
  --warmup-iters 3 \
  --measure-iters 20 \
  --max-model-len 4096 \
  --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.72 \
  --output-json profiles/qwen2_5_vl_pytorch_vision_baseline.json
```

严格 Math 的较小图片基线只需改为：

```text
--image assets/logo.png
--vision-attention-backend torch_math
--max-model-len 2048
--max-num-batched-tokens 2048
```

基础算子：

```bash
TORCH_CUDA_ARCH_LIST=12.0 \
python benchmarks/kernel_backend_benchmark.py \
  --warmup-iters 20 \
  --measure-iters 100 \
  --output-json profiles/kernels_pytorch_vs_custom_rtx5080.json
```
