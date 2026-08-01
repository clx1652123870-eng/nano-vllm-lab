# Qwen2.5-VL AWQ W4A16 离线推理

本文记录 nano-vllm 对 Hugging Face
`Qwen/Qwen2.5-VL-3B-Instruct-AWQ` checkpoint 的第一版支持，包括权重格式、
Linear 接入、Triton kernel、正确性和 BF16/AWQ 性能对比。

## 1. 当前支持范围

```text
模型: Qwen2.5-VL-3B-Instruct-AWQ
量化: AWQ W4A16
权重: INT4，group_size=128，asymmetric zero point
激活: BF16
GPU: 单 GPU
tensor_parallel_size: 1
视觉塔: BF16，不量化
文本 Decoder Linear: AWQ INT4 checkpoint
离线单图: 已验证
```

本地模型入口：

```text
/home/agua/models/Qwen2.5-VL-3B-Instruct-AWQ
```

它当前是指向 Hugging Face snapshot 的符号链接。代码不依赖这个固定绝对路径，
运行时可通过 `--model` 指定其他兼容 checkpoint。

## 2. 为什么叫 W4A16

```text
W4: 权重以 4 bit 整数存储
A16: 计算输入激活保持 FP16/BF16
```

第一版不是 INT4 Tensor Core 的“纯整数 GEMM”。运行时有两条研究路径：

```text
dequantize:
    INT4 -> BF16 weight
    BF16 activation @ BF16 weight

triton_fused:
    Triton GEMM 内部读取 packed INT4
    按 group 读取 zero/scale
    反量化后执行 tl.dot
```

默认选择 `dequantize`，因为在 RTX 5080 当前实现上它明显更快。

## 3. Checkpoint 数据格式

每个 AWQ Linear 保存：

```text
qweight: int32 [K, N / 8]
qzeros:  int32 [K / group_size, N / 8]
scales:  fp16  [K / group_size, N]
bias:    optional
```

一个 int32 装 8 个 INT4 值。该 checkpoint 的 nibble 逻辑顺序不是简单的
`0,1,2,3,4,5,6,7`，而是：

```text
[0, 4, 1, 5, 2, 6, 3, 7]
```

因此第 `n` 个逻辑输出列对应的位移为：

```text
shift = ((n % 2) * 4 + n // 2) * 4
```

反量化公式：

```text
group = k // group_size
W[k, n] = (qweight[k, n] - qzero[group, n]) * scale[group, n]
```

这里的 `qweight[k, n]` 和 `qzero[group, n]` 指解包后的 4-bit 值。

## 4. 模型接入

代码位置：

```text
nanovllm/layers/quantization/awq.py
    AWQConfig
    AWQColumnParallelLinear
    AWQMergedColumnParallelLinear
    AWQQKVParallelLinear
    AWQRowParallelLinear

nanovllm/layers/quantization/awq_kernels.py
    AWQ INT4 解包/反量化 Triton kernel
    实验性 fused W4A16 GEMM

nanovllm/models/qwen2_5_vl.py
    从 hf_config.quantization_config 识别 AWQ
    文本 Attention/MLP/LM head 创建 AWQ Linear

nanovllm/config.py
    第一版限制 tensor_parallel_size=1
```

Qwen2.5-VL 的合并权重需要特殊处理。例如 `gate_proj` 和 `up_proj` 在模型内合并
为一个 Merged Linear；`q_proj/k_proj/v_proj` 合并为 QKV Linear。加载 packed
权重时：

```text
qweight/qzeros 的 output offset 和 shard size 要除以 8
scales 使用正常 output channel offset
bias 使用一维 offset
```

如果把所有 component 都按普通 FP16 weight 切片，会造成 shard 放置错误。

视觉塔由 checkpoint 的：

```json
"modules_to_not_convert": ["visual"]
```

明确排除量化，因此视觉 patch embed、Attention 和 MLP 仍加载 BF16 权重。

## 5. Triton AWQ kernel

### 5.1 Dequantize kernel

每个 Triton program 处理一段逻辑 `[K, N]`：

1. 根据逻辑输出列计算 packed int32 offset 和 nibble shift。
2. 从 `qweight` 解出 4-bit 权重。
3. 根据 `k // group_size` 读取对应 packed zero point。
4. 读取 FP16 scale。
5. 计算 `(q - z) * scale` 并写入 BF16 weight。

之后矩阵乘交给 PyTorch/cuBLAS。

### 5.2 Fused W4A16 GEMM

实验 kernel 在 K loop 内进行：

```text
load BF16 activation tile
load packed INT4 weight
unpack qweight/qzero
load per-group scale
dequantize tile
tl.dot activation and weight
```

它避免完整 BF16 weight Tensor 的全局写回，但当前每个 M/N tile 会重复读取和解包
packed 权重，且没有 Marlin 类 kernel 的专用 layout 和调度，因此实际更慢。

## 6. AWQ kernel 性能

测试从真实 checkpoint 第一层提取三种 Linear shape，使用 BF16 activation、
5 次 warmup、20 次 measure。表格是 P50，单位毫秒。

### Q projection: `K=2048, N=2048`

| M | Triton fused W4A16 | 反量化 + cuBLAS | 缓存 BF16 + cuBLAS |
| ---: | ---: | ---: | ---: |
| 1 | 0.2290 | 0.0374 | **0.0214** |
| 32 | 0.2308 | 0.0367 | **0.0191** |
| 256 | 0.4548 | 0.0506 | **0.0331** |

### Gate projection: `K=2048, N=11008`

| M | Triton fused W4A16 | 反量化 + cuBLAS | 缓存 BF16 + cuBLAS |
| ---: | ---: | ---: | ---: |
| 1 | 0.6295 | 0.1035 | **0.0688** |
| 32 | 0.6224 | 0.1068 | **0.0381** |
| 256 | 2.1152 | 0.1864 | **0.1223** |

### Down projection: `K=11008, N=2048`

| M | Triton fused W4A16 | 反量化 + cuBLAS | 缓存 BF16 + cuBLAS |
| ---: | ---: | ---: | ---: |
| 1 | 1.2324 | 0.0965 | **0.0620** |
| 32 | 1.2439 | 0.1060 | **0.0367** |
| 256 | 2.5887 | 0.1939 | **0.1219** |

当前 fused Triton 比“反量化 + cuBLAS”慢约 6 到 13 倍，所以默认不启用。这个
负结果很重要：融合减少一次中间写回，并不保证更快；如果解包工作被大量 tile
重复、GEMM Tensor Core 利用率下降，总成本反而更高。

运行时选择：

```bash
# 默认、当前更快
export NANOVLLM_AWQ_KERNEL=dequantize

# 仅用于研究和复现
export NANOVLLM_AWQ_KERNEL=triton_fused
```

## 7. 正确性验证

### 7.1 Kernel 级

单元测试会生成可控的 packed INT4 weight，对比显式 PyTorch 反量化：

```text
Triton dequantize == explicit reference
Triton fused GEMM == dequantized BF16 GEMM，误差在 BF16 范围内
Merged/QKV packed shard loader offset 正确
```

### 7.2 模型级

同一 AWQ checkpoint、`assets/dog.png`、greedy 解码时，nano-vllm 与外部 vLLM
前 4 个 token 完全一致：

```text
[108893, 45930, 101987, 99593]
```

文本：

```text
这张图片展示了一
```

BF16 与 AWQ 生成 32 token 时，前 16 token 一致，之后出现正常量化差异：

```text
BF16:
这张图片展示了一只可爱的小金毛幼犬。幼犬的毛色是浅金色，眼睛大而明亮，
显得非常活泼和好奇。

AWQ:
这张图片展示了一只可爱的小金毛幼犬。幼犬的毛发是浅金色的，眼睛大而明亮，
显得非常活泼和好奇
```

语义一致，但不应要求量化模型和 BF16 长序列逐 token 完全相同。

当前 Transformers 环境缺少其 AWQ 加载所需的 `gptqmodel` 依赖，因此 AWQ
checkpoint 的外部参考使用了支持该模型的 vLLM。BF16 正确性基线仍使用
Transformers 对齐脚本。

## 8. BF16 与 AWQ 端到端性能

控制变量：

```text
同一 RTX 5080
同一 assets/dog.png
同一 prompt
同一 2049 input tokens
同一 32 output tokens
temperature=0
单 GPU、eager
gpu_memory_utilization=0.72
1 warmup + 5 measure
```

| 指标 | BF16 | AWQ | AWQ 相对变化 |
| --- | ---: | ---: | ---: |
| checkpoint/model unique storage | 6.99 GB | **3.17 GB** | 减少 54.7% |
| engine init | 3775.60 ms | **1953.83 ms** | 1.93x |
| 可分配 KV blocks | 118 | **563** | 4.77x |
| mean TTFT | **357.09 ms** | 366.93 ms | 慢 2.76% |
| mean decode TPOT | **13.65 ms** | 21.54 ms | 慢 1.58x |
| mean decode throughput | **73.40 tok/s** | 46.43 tok/s | 下降 36.7% |
| mean E2E | **780.45 ms** | 1034.93 ms | 慢 32.6% |

AWQ 显著降低模型权重存储，使同一显存预算下 KV Cache 从 118 blocks 增加到
563 blocks。但当前 kernel 没有 Marlin 等高度优化的 fused INT4 GEMM，运行时每
次 Linear 都需要反量化，所以 decode 比 BF16 慢。

两个 profile 的 peak allocated 都约 8.5 GB，这不表示 AWQ 没有省显存。引擎会
按 `gpu_memory_utilization` 把权重之外的可用预算尽量分给 KV Cache：

```text
BF16: 大权重 + 小 KV Cache
AWQ:  小权重 + 大 KV Cache
```

比较量化收益时应同时看 `model_unique_storage_gb` 和 `num_kvcache_blocks`，不能
只看引擎初始化后的总 allocated。

## 9. 复现命令

AWQ 离线推理：

```bash
cd /home/agua/tensorrtlearning/nano-vllm

/home/agua/anaconda3/envs/yolo26/bin/python \
  examples/qwen2_5_vl_offline.py \
  --engine nano \
  --model /home/agua/models/Qwen2.5-VL-3B-Instruct-AWQ \
  --image assets/dog.png \
  --max-new-tokens 32 \
  --no-tqdm \
  --max-model-len 4096 \
  --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.72
```

AWQ profile：

```bash
/home/agua/anaconda3/envs/yolo26/bin/python \
  examples/qwen2_5_vl_profile.py \
  --model /home/agua/models/Qwen2.5-VL-3B-Instruct-AWQ \
  --image assets/dog.png \
  --max-new-tokens 32 \
  --warmup-iters 1 \
  --measure-iters 5 \
  --max-model-len 4096 \
  --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.72 \
  --output-json profiles/qwen2_5_vl_awq.json
```

AWQ kernel benchmark：

```bash
/home/agua/anaconda3/envs/yolo26/bin/python \
  benchmarks/awq_kernel_benchmark.py \
  --model /home/agua/models/Qwen2.5-VL-3B-Instruct-AWQ \
  --token-counts 1,32,256 \
  --warmup-iters 5 \
  --measure-iters 20 \
  --output-json profiles/awq_kernels_qwen2_5_vl_3b.json
```

外部 vLLM AWQ 输出参考：

```bash
/home/agua/anaconda3/envs/yolo26/bin/python \
  examples/qwen2_5_vl_vllm_offline.py \
  --model /home/agua/models/Qwen2.5-VL-3B-Instruct-AWQ \
  --image assets/dog.png \
  --max-new-tokens 4
```

## 10. 当前限制与下一步

当前限制：

```text
只支持 bits=4、group_size=128、zero_point=true、version=gemm
只支持 tensor_parallel_size=1
视觉塔保持 BF16
未实现 Marlin/GPTQ-Marlin 风格重排和专用 INT4 Tensor Core kernel
尚未做多 checkpoint、多图片尺寸和长文本质量评测
```

下一步优化优先级：

1. 以 vLLM 的 Marlin 路径作为性能参考，确认 weight layout 和 dispatch。
2. 为 decode 的 M=1 和小 batch 单独设计 W4A16 kernel。
3. 避免不同 output tile 重复解包同一 packed weight。
4. 将 quantized GEMM 与 bias/activation 等 epilogue 融合。
5. 再跑 C1/C2/C4 在线吞吐，判断显存容量收益能否转化为并发收益。

## 11. 面试回答

### AWQ 为什么省显存却可能更慢？

权重以 INT4 存储能降低容量，但计算前必须解包、减 zero point、乘 scale。没有
高效 fused W4A16 kernel 时，这些额外工作可能超过 BF16 GEMM 的成本。容量优化
与计算加速是两个不同目标。

### 为什么视觉塔没有量化？

checkpoint 的 `modules_to_not_convert` 明确包含 `visual`。模型兼容首先要遵循
checkpoint 实际格式；擅自把视觉塔当作 INT4 加载既不匹配权重，也会改变精度。

### 为什么 AWQ 与 BF16 不要求所有 token 一致？

INT4 权重会引入量化误差。早期 logits 排名接近时可能保持相同 token，后续误差
通过 autoregressive decode 累积后会产生分支。应该比较模型语义质量、任务指标
和外部 AWQ 实现，而不是把 BF16 全 token 一致当作必要条件。

### 这版 AWQ 最核心的工程点是什么？

不是读取 `quantization_config`，而是正确处理 packed `qweight/qzeros/scales`、
非线性 nibble 顺序，以及 merged QKV/gate-up 权重的 packed shard offset。格式
正确后，kernel 才有优化意义。
