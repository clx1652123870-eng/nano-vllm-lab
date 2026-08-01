# Qwen2.5-VL Attention Backend 性能与 Triton 实验

本文记录 nano-vllm 完成 Attention backend 抽象后，对五种高性能/自动视觉
Encoder Attention 实现进行的真实 shape 微基准与端到端正确性验证。严格
PyTorch Math 基线另见 `docs/pytorch_baseline_optimization_comparison.md`。

接口设计见 `docs/attention_backends.md`，原始结果见：

```text
profiles/attention_backends_qwen2_5_vl_dog.json
```

## 1. 本轮实现

本轮候选性能对比包含五个 backend：

| backend | 实现 | packed varlen 方式 | 定位 |
| --- | --- | --- | --- |
| `flash_attn` | `flash_attn_varlen_func` | 原生 | 默认高性能实现 |
| `torch_sdpa` | PyTorch SDPA | 按序列逐段调用 | 正确性和可移植性参考 |
| `cudnn_sdpa` | 强制 cuDNN SDPA | 按序列逐段调用 | cuDNN 对比 |
| `triton` | 自定义 fused forward kernel | 原生 | 实验性实现 |
| `hybrid` | 短序列 Triton、长序列 FlashAttention | 原生 | shape-aware dispatch |

文本 Decoder 仍只支持 `flash_attn`，因为它需要处理 causal prefill、Paged KV
Cache、`block_tables` 和 decode。视觉后端可独立通过
`vision_attention_backend` 切换。

## 2. 为什么使用真实视觉 shape

本轮没有任意选择一个 `[B,H,S,D]`，而是先用真实图片经过 Qwen2.5-VL
Processor 和窗口划分逻辑，得到模型实际传给视觉 Attention 的 layout。

输入：

```text
image: assets/dog.png
prompt: 描述这张图片
input tokens: 2049
image tokens: 2025
image_grid_thw: [[1, 90, 90]]
pixel_values: [8100, 1176]
vision depth: 32
num heads: 16
head dim: 80
```

32 个视觉层包括：

```text
28 个 Window Attention 层
4 个 Full Attention 层: [7, 15, 23, 31]
```

两个 case：

| case | packed token | 序列数 | 序列长度 |
| --- | ---: | ---: | --- |
| Window | 8100 | 144 | 4、16、64 |
| Full | 8100 | 1 | 8100 |

这两个 case 的算子特征完全不同。Window 是大量短序列，容易受 Python 循环、
CPU 同步和 kernel launch 开销影响；Full 是单条长序列，更考验 kernel 的矩阵
计算和显存带宽效率。

## 3. 测试方法

环境：

```text
GPU: NVIDIA GeForce RTX 5080
dtype: BF16
PyTorch: 2.11.0+cu130
CUDA: 13.0
cuDNN: 9.19.0
Triton: 3.6.0
FlashAttention: 2.8.3.post1
```

每个 case 使用同一组随机 Q/K/V：

```text
warmup: 5
measure: 20
seed: 0
timing: perf_counter + 每次前后 torch.cuda.synchronize()
reference: FlashAttention output
```

估算 Attention FLOPs：

```text
4 * num_heads * head_dim * sum(sequence_length^2)
```

该 FLOPs 只覆盖 `QK^T` 和 `PV`，不包含 QKV projection、RoPE、MLP、归一化
和数据重排，因此不能当作整个视觉 Encoder 的 TFLOPS。

## 4. 测试结果

### 4.1 Window Attention

| backend | mean ms | P50 ms | P90 ms | 估算 TFLOPS | max error vs Flash |
| --- | ---: | ---: | ---: | ---: | ---: |
| `flash_attn` | 0.146 | **0.124** | 0.167 | 17.55 | 0 |
| `torch_sdpa` | 2.726 | 2.694 | 3.156 | 0.94 | 0.003906 |
| `cudnn_sdpa` | 3.696 | 3.573 | 3.871 | 0.69 | 0.003906 |
| `triton` | 0.130 | 0.129 | 0.132 | 19.72 | 0.007812 |
| `hybrid` | **0.128** | 0.128 | **0.129** | **19.98** | 0.007812 |

Window case 中，调优后的 Triton mean 比外部 FlashAttention 快约 11%；
`hybrid` 在该 shape 自动选择 Triton。Torch/cuDNN 并不是单个 kernel 很慢，而是当前
packed adapter 会：

```text
cu_seqlens GPU -> CPU -> Python list
Python 循环 144 个序列
逐段发起 SDPA kernel
```

因此它们在大量短窗口上主要输给调度和 launch 开销。

### 4.2 Full Attention

| backend | mean ms | P50 ms | P90 ms | 估算 TFLOPS | max error vs Flash |
| --- | ---: | ---: | ---: | ---: | ---: |
| `flash_attn` | **4.354** | **4.349** | **4.659** | **77.15** | 0 |
| `torch_sdpa` | 4.494 | 4.454 | 4.824 | 74.75 | 0.000244 |
| `cudnn_sdpa` | 5.846 | 5.808 | 6.116 | 57.46 | 0.000488 |
| `triton` | 5.956 | 5.930 | 6.328 | 56.40 | 0.000488 |
| `hybrid` | 4.363 | **4.349** | 4.676 | 76.99 | 0 |

Full case 只有一个 8100-token 序列，Torch SDPA 不再承担 144 次循环开销，
因此距离 FlashAttention 只有约 3%。调优后的 Triton 已从最初约 7.09 ms 降到
5.96 ms，但在长序列上仍慢约 37%。`hybrid` 在该 shape 自动回到外部
FlashAttention。

### 4.3 32 层粗略加权

仅将每层 Attention 调用按 `28 * Window + 4 * Full` 相加：

| backend | 估算 32 层 Attention 时间 ms | 相对 Flash |
| --- | ---: | ---: |
| `hybrid` | **21.05** | **0.98x** |
| `flash_attn` | 21.51 | 1.00x |
| `triton` | 27.47 | 1.28x |
| `torch_sdpa` | 94.32 | 4.38x |
| `cudnn_sdpa` | 126.87 | 5.90x |

这不是端到端视觉 Encoder 时间。它说明 shape-aware dispatch 在 Attention 子项
上理论改善约 2.1%，但是否采用仍要看完整模型 TTFT。

### 4.4 端到端 TTFT

同一 BF16 模型、图片、32 个输出 token，2 次 warmup 和 10 次 measure：

| backend | mean TTFT | mean E2E |
| --- | ---: | ---: |
| `flash_attn` | 358.19 ms | 787.71 ms |
| `hybrid` | **357.73 ms** | **783.58 ms** |

TTFT 只改善 0.45 ms，即 0.13%，接近测量噪声。因此当前默认值仍保持
`flash_attn`；Hybrid 是已验证可用的实验选项，而不是默认性能结论。

## 5. Triton kernel 是怎么实现的

代码位于：

```text
nanovllm/attention/triton_attn.py
```

输入保持 backend 的统一 packed layout：

```text
Q: [total_q_tokens, num_q_heads, head_dim]
K: [total_k_tokens, num_kv_heads, head_dim]
V: [total_k_tokens, num_kv_heads, head_dim]
cu_seqlens_q / cu_seqlens_k: int32 CUDA Tensor
```

kernel grid：

```text
(query block, query head, packed sequence)
```

每个 program：

1. 从 `cu_seqlens` 读取当前 packed sequence 的 Q/KV 起止位置。
2. 根据 `query_head // group_size` 将 GQA query head 映射到 KV head。
3. 分块加载 Q 和 K，计算 `QK^T * scale`。
4. 使用 online softmax 维护每行 `row_max` 和 `row_sum`。
5. 分块加载 V，累加 softmax probability 与 V 的乘积。
6. 写回 `[total_q_tokens, num_q_heads, head_dim]`。

它不会物化完整的 `S x S` Attention matrix，额外存储随 block 大小增长，而不是
随序列长度平方增长。`head_dim=80` 会 pad 到 128 以满足 Triton dot 的块形状，
超出 80 的部分通过 mask 屏蔽。

当前约束：

```text
CUDA only
BF16 only
head_dim <= 256
forward only
无 dropout
causal 模式要求 Q/KV packed segment 完全一致
未实现 Paged KV Cache decode
```

这是可执行的实验后端，不是为了替代成熟 FlashAttention 库而写的等价产品。

## 6. 数值与端到端验证

微基准中的最大绝对误差都在 BF16 可接受量级：

```text
Window:
    torch_sdpa  0.00390625
    cudnn_sdpa  0.00390625
    triton      0.00781250

Full:
    torch_sdpa  0.00024414
    cudnn_sdpa  0.00048828
    triton      0.00048828
```

五种视觉后端都运行了真实 Qwen2.5-VL greedy smoke，前 4 个 token 一致：

```text
flash_attn: [108893, 45930, 101987, 99593]
torch_sdpa: [108893, 45930, 101987, 99593]
cudnn_sdpa: [108893, 45930, 101987, 99593]
triton:     [108893, 45930, 101987, 99593]
hybrid:     [108893, 45930, 101987, 99593]
```

输出文本均为：

```text
这张图片展示了一
```

token 对齐证明四条端到端路径在当前输入上可运行。它不能替代多图片尺寸、多
sequence length、不同 causal/GQA shape 的数值测试。

## 7. 复现命令

运行五后端真实 shape 微基准：

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

验证 Triton 端到端输出：

```bash
/home/agua/anaconda3/envs/yolo26/bin/python \
  examples/qwen2_5_vl_offline.py \
  --engine nano \
  --image assets/dog.png \
  --max-new-tokens 4 \
  --no-tqdm \
  --gpu-memory-utilization 0.72 \
  --max-model-len 4096 \
  --max-num-batched-tokens 4096 \
  --attention-backend flash_attn \
  --vision-attention-backend triton
```

将最后一个参数依次改为 `flash_attn`、`torch_sdpa`、`cudnn_sdpa` 和
`hybrid` 即可完成五后端回归。

## 8. 下一步优化判断

第一优先级不是继续扩写一个通用 Triton Vision Attention，而是：

1. 默认继续使用 FlashAttention，Hybrid 保留为实验后端。
2. 用 Nsight Systems/Compute 分解在线 C1/C4 的 decode TPOT 差距。
3. 检查 CPU scheduler gap、KV cache write、paged decode Attention、LM head
   和 sampler 各自占比。
4. 只有确认 decode Attention 是主瓶颈后，再实现 Triton Paged Attention。

视觉侧仍有两个清晰实验：

- 将长度相同的 Window 分组为 dense batch，使 Torch/cuDNN 从 144 次调用降到
  3 次，验证 adapter 开销与 kernel 性能的边界。
- 继续调优 8100-token Full Triton kernel，包括 persistent scheduling 和更好的
  K/V tile 流水；短窗口已经超过外部 FlashAttention。

## 9. 面试回答

### 为什么 Torch SDPA 在 Full 很快，在 Window 很慢？

Full 只有一个长序列，只调用一次 fused SDPA；Window 被 packed 成 144 个短
序列，当前参考 adapter 通过 Python 逐段调用 144 次。后者主要受 CPU 同步和
kernel launch 开销限制，而不是 FLOPs 限制。

### 强制 cuDNN 后端后更慢，能否说明 cuDNN Attention 不行？

不能。这个结论只适用于“cuDNN SDPA + 当前逐段 packed adapter + RTX 5080 +
这些 shape”。后端性能依赖 dtype、head dim、长度、mask、layout 和硬件，
不能从一个 shape 推广到所有场景。

### 为什么还要自己写 Triton，FlashAttention 已经更快？

目的有两个：验证 backend 抽象确实能接入新 kernel；掌握 packed varlen、
GQA、online softmax 和数值稳定性。本轮 Triton 在 Window shape 胜出，但 Full
仍落后，所以用 Hybrid 表达 shape-specific 选择，默认生产路径仍不改变。

### Triton Attention 最关键的数值稳定设计是什么？

不能直接对整行 score 做 `exp` 后求和。kernel 按 K block 流式处理，用运行中的
最大值修正历史累加：

```text
new_max = max(old_max, block_max)
old_acc *= exp(old_max - new_max)
block_prob = exp(block_score - new_max)
```

这样避免长序列 score 导致指数上溢，并且无需保存完整 Attention matrix。

## 10. 官方参考

- [PyTorch scaled_dot_product_attention](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention)
- [PyTorch sdpa_kernel](https://docs.pytorch.org/docs/main/generated/torch.nn.attention.sdpa_kernel.html)
- [NVIDIA cuDNN Attention](https://docs.nvidia.com/deeplearning/cudnn/latest/operations/Attention.html)
- [Triton Fused Attention Tutorial](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)
