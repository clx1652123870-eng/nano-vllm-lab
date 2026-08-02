# CUDA 与 Triton 算子实验

本文记录 nano-vllm 中自定义 CUDA/Triton 算子的实现、正确性验证和 RTX 5080
性能结果。这里的目标是用同一组 shape 回答三个问题：

1. 自定义 kernel 是否真的比 PyTorch/CUDA 默认实现快。
2. Triton 与手写 CUDA 各自适合什么算子和 shape。
3. 单算子更快后，是否值得替换模型生产路径。

原始结果：

```text
profiles/kernel_backends_rtx5080.json
profiles/attention_backends_qwen2_5_vl_dog.json
profiles/qwen2_5_vl_flash_attention.json
profiles/qwen2_5_vl_hybrid_attention.json
```

## 1. 三种实现不是同一个概念

本文中的三列含义如下：

| 名称 | 实际实现 |
| --- | --- |
| `torch_cuda` | PyTorch CUDA 算子；Matmul 最终通常使用 cuBLAS/cuBLASLt |
| `custom_cuda` | 本项目 `.cu` 源码，通过 PyTorch C++/CUDA extension 编译 |
| `triton` | 本项目 Triton JIT kernel |

视觉 Attention 还有第四类：

```text
flash_attn
```

它来自外部 `flash-attn` CUDA 库，不是本项目手写的 CUDA kernel。项目内 Triton
Attention 则是独立实现的 packed-varlen FlashAttention 思路：分块计算、online
softmax，不物化完整 `S x S` score matrix。

## 2. 代码位置

```text
nanovllm/kernels/csrc/bindings.cpp
    PyTorch extension 的 Python/C++ 绑定

nanovllm/kernels/csrc/kernels.cu
    BF16 Softmax、RMSNorm、SiLU-and-Mul、Matmul CUDA kernel

nanovllm/kernels/cuda_ops.py
    torch.utils.cpp_extension.load 懒加载

nanovllm/kernels/triton_ops.py
    对应的四个 Triton kernel

nanovllm/attention/triton_attn.py
    packed-varlen Vision FlashAttention Triton kernel

nanovllm/attention/hybrid.py
    短窗口走 Triton、长序列走外部 FlashAttention

benchmarks/kernel_backend_benchmark.py
benchmarks/attention_backend_benchmark.py
    可重复运行的算子基准
```

CUDA extension 第一次调用时编译，编译产物进入 PyTorch extension cache。首次编译
时间不计入稳态 kernel benchmark。

## 3. 基础算子测试条件

```text
GPU: NVIDIA GeForce RTX 5080
PyTorch: 2.11.0+cu130
CUDA runtime: 13.0
dtype: BF16
warmup: 10
measure: 30
timing: perf_counter + 每次调用前后 cuda synchronize
```

以下表格使用 P50，单位均为毫秒。粗体表示当前 shape 的最快实现。

## 4. Softmax

| shape | PyTorch CUDA | 自定义 CUDA | Triton |
| --- | ---: | ---: | ---: |
| `[1024, 64]` | **0.0097** | 0.0114 | 0.0135 |
| `[1024, 2048]` | 0.0196 | **0.0150** | 0.0166 |
| `[128, 8192]` | **0.0130** | 0.0148 | 0.0164 |

自定义 CUDA 只在 `[1024, 2048]` 上胜出，不能据此全局替换 PyTorch Softmax。
更重要的是，模型的 Attention 已经把 Softmax 融合进 FlashAttention，单独替换
`torch.softmax` 不会优化那条路径。当前独立 Softmax 主要用于算子学习和回归。

## 5. RMSNorm

| shape | PyTorch CUDA | 自定义 CUDA | Triton |
| --- | ---: | ---: | ---: |
| `[1, 2048]` | 0.0115 | **0.0115** | 0.0136 |
| `[32, 2048]` | 0.0115 | **0.0114** | 0.0137 |
| `[2049, 2048]` | **0.0161** | 0.0204 | 0.0200 |
| `[8100, 1280]` | 0.0269 | 0.0341 | **0.0263** |

Decode 小 batch 上 PyTorch 与自定义 CUDA 基本相同；视觉 `8100 x 1280` shape
上 Triton 略快约 2.4%。这个差距较小，仍需用完整视觉层验证，不能仅凭微基准
替换模型的 `torch.compile` RMSNorm。

## 6. SiLU-and-Mul

| shape | PyTorch CUDA | 自定义 CUDA | Triton |
| --- | ---: | ---: | ---: |
| `[1, 22016]` | 0.0117 | **0.0093** | 0.0156 |
| `[32, 22016]` | 0.0132 | **0.0104** | 0.0156 |
| `[2049, 22016]` | 0.2699 | 0.1702 | **0.1697** |

SiLU 与逐元素乘法融合后避免中间 Tensor 和额外 launch，因此是四个基础算子中
收益最稳定的一类。自定义 CUDA 在 M=1/32 上分别比 PyTorch eager 快约 20% 和
21%，大 prefill 上 CUDA/Triton 都快约 37%。

注意当前模型里的 `SiluAndMul` 使用 `torch.compile`，而本表 PyTorch 列是 eager
表达式。因此这个结果证明“融合 kernel 有价值”，但不能直接宣称完整模型已经
获得相同比例收益。生产替换前还要对 compiled baseline 做同条件测试。

## 7. Matmul

| `M x K x N` | PyTorch/cuBLAS | 自定义 CUDA | Triton |
| --- | ---: | ---: | ---: |
| `1 x 2048 x 2048` | **0.0209** | 0.0793 | 0.0260 |
| `32 x 2048 x 2048` | **0.0200** | 0.1294 | 0.0262 |
| `256 x 2048 x 2048` | **0.0352** | 0.6867 | 0.0443 |

手写 CUDA Matmul 是教学性质的 tiled kernel，没有使用 Tensor Core，明显慢于
cuBLAS。Triton 的 `tl.dot` 能使用 Tensor Core，因此比手写 CUDA 快，但当前固定
tile 和调度仍落后于 cuBLAS。

该结果说明 Matmul 不能只做到“结果正确”就替换框架实现。高性能 GEMM 还涉及：

```text
Tensor Core 指令
共享内存和寄存器流水
layout/swizzle
不同 M/N/K 的 dispatch
split-K 或 persistent scheduling
epilogue fusion
```

因此 BF16 Linear 继续走 PyTorch/cuBLAS。自定义 Matmul 保留作教学和后续融合
epilogue 的基线。

## 8. Triton Vision FlashAttention

真实 `assets/dog.png` 经过 Qwen2.5-VL Processor 后得到：

```text
pixel_values: [8100, 1176]
vision hidden: 1280
heads: 16
head_dim: 80
Window case: 8100 packed tokens，144 条长度 4/16/64 的序列
Full case: 8100 tokens，1 条长序列
```

Triton kernel 使用：

```text
grid = (query block, query head, packed sequence)
QK 分块计算
online softmax
按 GQA group 映射 KV head
分块累加 P @ V
```

调优后按最大 sequence length 选择两套配置：

```text
max_seqlen <= 64:
    BLOCK_M=32, BLOCK_N=32, num_warps=4

max_seqlen > 64:
    BLOCK_M=128, BLOCK_N=64, num_warps=8
```

正式结果为 5 次 warmup、20 次 measure：

| case | 外部 FlashAttention mean | Triton mean | 结论 |
| --- | ---: | ---: | --- |
| Window | 0.146 ms | **0.130 ms** | Triton 快 11.0% |
| Full | **4.354 ms** | 5.956 ms | Triton 慢 36.8% |

Triton 确实在大量短窗口的真实 shape 上超过了外部 CUDA FlashAttention，但没有
在 8100-token Full Attention 上超过它。

## 9. Hybrid Attention

Qwen2.5-VL 的 32 个视觉层中有 28 个 Window Attention、4 个 Full
Attention。因此新增：

```text
vision_attention_backend=hybrid
```

dispatch 规则：

```text
max(max_seqlen_q, max_seqlen_k) <= 64 -> Triton
otherwise                              -> external flash-attn
```

微基准中 Hybrid 的结果：

| case | Hybrid mean | 实际后端 |
| --- | ---: | --- |
| Window | **0.128 ms** | Triton |
| Full | 4.363 ms | 外部 FlashAttention |

按 `28 * Window + 4 * Full` 粗略相加：

```text
纯 FlashAttention: 28 * 0.146 + 4 * 4.354 = 21.504 ms
Hybrid:            28 * 0.128 + 4 * 4.363 = 21.036 ms
理论 Attention 子项改善约 2.2%
```

端到端离线 profile 使用同一模型、图片、32 个输出 token、2 次 warmup 和 10 次
measure：

| Vision backend | mean TTFT | mean E2E |
| --- | ---: | ---: |
| `flash_attn` | 358.19 ms | 787.71 ms |
| `hybrid` | **357.73 ms** | **783.58 ms** |

TTFT 只改善 `0.45 ms / 0.13%`，接近系统测量噪声；E2E 的 decode 波动也不能
归因于 Vision backend，因为 decode 不再运行视觉塔。正确结论是：

```text
Triton Window kernel 单算子胜出。
Hybrid dispatch 工作正常。
当前没有证据证明它带来稳定、显著的端到端收益。
默认后端因此仍保持 flash_attn。
```

## 10. 复现命令

第一次构建自定义 CUDA extension 并运行基础算子：

```bash
cd /home/agua/tensorrtlearning/nano-vllm

TORCH_CUDA_ARCH_LIST=12.0 \
/home/agua/anaconda3/envs/yolo26/bin/python \
  benchmarks/kernel_backend_benchmark.py \
  --warmup-iters 10 \
  --measure-iters 30 \
  --output-json profiles/kernel_backends_rtx5080.json
```

Attention 微基准：

```bash
/home/agua/anaconda3/envs/yolo26/bin/python \
  benchmarks/attention_backend_benchmark.py \
  --image assets/dog.png \
  --backends flash_attn,torch_sdpa,cudnn_sdpa,triton,hybrid \
  --cases window,full \
  --warmup-iters 5 \
  --measure-iters 20 \
  --output-json profiles/attention_backends_qwen2_5_vl_dog.json
```

Hybrid 端到端 profile：

```bash
/home/agua/anaconda3/envs/yolo26/bin/python \
  examples/qwen2_5_vl_profile.py \
  --model /home/agua/models/Qwen2.5-VL-3B-Instruct \
  --image assets/dog.png \
  --max-new-tokens 32 \
  --warmup-iters 2 \
  --measure-iters 10 \
  --max-model-len 4096 \
  --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.72 \
  --vision-attention-backend hybrid \
  --output-json profiles/qwen2_5_vl_hybrid_attention.json
```

## 11. 面试回答

### 为什么 Triton 能在 Window 上赢、Full 上输？

Window 序列只有 4/16/64 token，主要受 launch、固定开销和小块利用率影响。当前
Triton tile 对这组固定 shape 更合适。Full 是 8100-token 长序列，对流水线、
访存、warp 调度和长序列分块要求更高，成熟 FlashAttention CUDA 实现更充分。

### 自己写的 CUDA Matmul 为什么比库慢？

正确性 kernel 只做了基础 tiling，没有 Tensor Core、复杂流水、shape dispatch
和 epilogue fusion；cuBLAS/cuBLASLt 是多年调优的生产 GEMM 库。手写 CUDA 的
价值在于理解执行和验证融合点，不代表天然比库快。

### 单算子快 10%，为什么 TTFT 几乎没变化？

Vision Attention 只是视觉塔的一部分，而且 Window Attention 仅是其中 28 次很短
的调用。根据 Amdahl 定律，如果被优化部分只占 TTFT 很小比例，即使 kernel 本身
加速明显，端到端收益也会很小。

### 为什么不把所有算子都替换成 Triton？

后端选择按真实 shape、数值误差和端到端收益决定。当前 Matmul 是
PyTorch/cuBLAS 更快，Softmax/RMSNorm 也只在个别 shape 胜出。无条件替换会造成
性能回退，并增加维护成本。

## 12. 自定义 CUDA Attention 与最终三方对比

项目新增 `nanovllm/attention/cuda_attn.py` 和
`nanovllm/kernels/csrc/kernels.cu::packed_attention_bf16_kernel`。该 CUDA kernel：

```text
输入: packed-varlen BF16 Q/K/V + cu_seqlens
调度: 一个 block 处理一个 (sequence, query head)，8 warps
融合: QK + scale + causal mask + online softmax + PV
中间状态: K/V 放 shared memory，softmax 和 output accumulator 放寄存器
能力: GQA、causal/non-causal、head_dim <= 128、sequence <= 64
```

它不物化 `S x S` score matrix，但内部是 FP32 scalar FMA，没有 WMMA/Tensor Core，
所以是可解释的 CUDA fused baseline。`cuda_hybrid` 在长序列自动回退外部
FlashAttention，避免把短窗口 kernel 错用到 8100-token Full Attention。

RTX 5080、BF16、`assets/dog.png`、5 warmup、20 measure 的正式 P50：

| Window backend | P50 | 相对 Triton |
| --- | ---: | ---: |
| PyTorch Math | 10.917 ms | 91.39x |
| PyTorch SDPA | 2.548 ms | 21.33x |
| 自定义 CUDA fused | 1.080 ms | 9.04x |
| 自定义 Triton fused | **0.119 ms** | 1.00x |
| 外部 FlashAttention | 0.124 ms | 1.04x |

Full Attention 的 Triton、外部 FlashAttention P50 分别为 `5.937 ms` 和
`4.361 ms`，因此生产选择仍是 shape-aware Hybrid：短窗口 Triton，长序列外部
FlashAttention。注意 `91.39x/21.33x/9.04x` 都是 Attention 子项微基准，不是
整个 Qwen2.5-VL 的端到端加速比。

Nsight Systems 用 NVTX range 验证了差距来源：

| Backend | GPU projected time/iter | GPU operations/iter |
| --- | ---: | ---: |
| PyTorch Math | 16.553 ms | 2162 |
| PyTorch SDPA | 5.097 ms | 290 |
| CUDA fused | 1.073 ms | 1 |
| Triton fused | **0.092 ms** | **1** |

PyTorch 的主要问题是 packed adapter 和多算子 launch；CUDA/Triton 都只有一个
operation 后，`11.66x` 的差距来自 kernel 内部实现，下一步应分析 Tensor Core、
occupancy、寄存器、shared memory 和 warp stall，而不是继续减少 launch。当前 NCU
由于 `ERR_NVGPUCTRPERM` 无法读取硬件 counter，未虚构 NCU 数值。

结果与复现入口：

```text
profiles/attention_pytorch_cuda_triton_rtx5080.json
profiles/nsight/attention_backends_summary.json
profiles/nsight/attention_backends.nsys-rep
benchmarks/nsight_attention_workload.py
docs/nano_vllm_qwen2_5_vl_interview_guide.md
```
