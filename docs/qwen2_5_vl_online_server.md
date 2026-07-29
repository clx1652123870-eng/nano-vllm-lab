# Qwen2.5-VL 在线推理服务

本文说明 nano-vllm 第四版在线服务。当前版本已经具备异步请求队列、
continuous batching、SSE token streaming、请求取消、超时和并发保护，以及
OpenAI Chat Completions 兼容接口。

本文既是使用教程，也是面试时解释系统设计和实现取舍的参考。

## 1. 当前能力

已支持：

- 单进程、单 GPU、`tensor_parallel_size=1`。
- 服务启动时只加载一次 `AutoProcessor` 和模型。
- 单请求单图片。
- 原生非流式接口 `POST /generate`。
- 原生流式接口 `POST /generate_stream`。
- OpenAI 兼容接口 `POST /v1/chat/completions`。
- OpenAI 非流式和流式响应。
- FastAPI async handler。
- 图片预处理线程池。
- 专用 `nanovllm-engine` 线程。
- 跨线程 request queue、Future 和 per-request stream queue。
- 单图 multimodal prefill。
- 多请求 continuous batched decode。
- 客户端断开后的请求取消。
- waiting/running sequence 删除和 KV Cache block 回收。
- 全请求超时。
- 服务级最大并发和 HTTP 429 backpressure。
- C1/C2/C4 正确性与吞吐回归。
- JSON profiling 报告。

当前限制：

- 一个请求只能包含一张图片。
- multimodal prefill 每个 step 只能处理一个图片请求。
- multimodal chunked prefill 尚未实现。
- OpenAI `image_url.url` 目前只接受 Base64 data URL。
- OpenAI 接口只支持 `n=1`、`top_p=1`，不支持自定义 `stop`。
- 没有 `/v1/models`、工具调用、JSON Schema 和 logprobs。
- 没有 API/EngineCore 多进程拆分和 ZMQ。

## 2. 代码位置

- `examples/qwen2_5_vl_server.py`
  FastAPI、SSE、OpenAI 协议、并发限制、超时和图片预处理。
- `nanovllm/engine/async_llm_engine.py`
  engine thread、请求队列、Future、stream event、取消和 profiling。
- `nanovllm/engine/llm_engine.py`
  step metadata、token 路由和 abort 入口。
- `nanovllm/engine/scheduler.py`
  waiting/running 调度、continuous batching 和 sequence abort。
- `nanovllm/engine/block_manager.py`
  Paged KV Cache block 分配与释放。
- `examples/qwen2_5_vl_online_profile.py`
  单档并发 profiling。
- `examples/qwen2_5_vl_online_regression.py`
  C1/C2/C4 自动回归和总报告。
- `tests/test_async_llm_engine.py`
  engine streaming、continuous batching、取消和 KV Cache 回收测试。
- `tests/test_online_server.py`
  HTTP/SSE/OpenAI/429/504 协议测试。
- `tests/test_online_regression.py`
  回归报告判定测试。

## 3. 启动服务

先停止旧服务，再启动当前代码：

```bash
/home/agua/anaconda3/envs/yolo26/bin/python \
  examples/qwen2_5_vl_server.py \
  --host 127.0.0.1 \
  --port 8000 \
  --model /home/agua/models/Qwen2.5-VL-3B-Instruct \
  --served-model-name Qwen2.5-VL-3B-Instruct \
  --max-model-len 4096 \
  --max-num-seqs 4 \
  --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.72 \
  --max-concurrent-requests 16 \
  --request-timeout-seconds 300
```

关键参数：

| 参数 | 含义 |
|---|---|
| `max_num_seqs` | 最多同时进入 Scheduler 的活跃 sequence 数量 |
| `max_num_batched_tokens` | 一个 prefill step 最多调度的输入 token 数 |
| `max_concurrent_requests` | 从预处理到响应结束的 HTTP in-flight 请求上限 |
| `request_timeout_seconds` | 预处理、排队和生成共用的总超时时间 |
| `served_model_name` | OpenAI 请求中必须使用的模型名称 |

`max_num_seqs=4` 和 `max_concurrent_requests=16` 表示：

```text
最多 16 个 HTTP 请求占用服务槽位
最多 4 个请求进入 Scheduler 成为活跃 sequence
其余已预处理请求留在 AsyncLLMEngine request queue
第 17 个并发 HTTP 请求立即返回 429
```

模型只在 `nanovllm-engine` 线程中加载一次。HTTP 请求不会创建新的模型实例。

## 4. 健康检查

```bash
curl -s http://127.0.0.1:8000/health | python -m json.tool
```

关键字段：

```json
{
  "status": "ok",
  "served_model_name": "Qwen2.5-VL-3B-Instruct",
  "mode": "single-process-single-gpu-continuous-batching",
  "capabilities": {
    "native_generate": true,
    "native_sse": true,
    "openai_chat_completions": true,
    "openai_sse": true,
    "request_cancellation": true
  },
  "concurrency": {
    "limit": 16,
    "in_flight": 0,
    "available": 16,
    "rejected": 0
  },
  "engine": {
    "active_request_count": 0,
    "queue_depth": 0,
    "submitted_requests": 0,
    "completed_requests": 0,
    "failed_requests": 0,
    "cancelled_requests": 0,
    "outstanding_requests": 0,
    "max_decode_batch_size": 0
  }
}
```

`in_flight` 和 `outstanding_requests` 不完全相同：

- `in_flight` 从 HTTP 请求开始预处理时计数。
- `outstanding_requests` 从请求提交到 AsyncLLMEngine 时计数。
- 因此正在图片预处理的请求只出现在 `in_flight` 中。

## 5. 原生非流式接口

### 5.1 文件路径

文件路径只适合客户端和服务端共享文件系统的情况：

```bash
curl -s http://127.0.0.1:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "描述这张图片",
    "image_path": "assets/dog.png",
    "max_new_tokens": 32,
    "temperature": 0.0,
    "profile": false
  }' | python -m json.tool
```

### 5.2 Base64 图片

真实远程 HTTP 请求应使用 Base64：

```bash
IMAGE_BASE64=$(base64 -w 0 assets/dog.png)

curl -s http://127.0.0.1:8000/generate \
  -H 'Content-Type: application/json' \
  -d "{
    \"prompt\": \"描述这张图片\",
    \"image_base64\": \"$IMAGE_BASE64\",
    \"max_new_tokens\": 32,
    \"temperature\": 0.0
  }" | python -m json.tool
```

返回结构：

```json
{
  "request_id": "1f...",
  "model": "/home/agua/models/Qwen2.5-VL-3B-Instruct",
  "text": "这张图片中有一只狗。",
  "token_ids": [108893, 45930],
  "finish_reason": "length",
  "usage": {
    "input_tokens": 2049,
    "image_tokens": 2025,
    "image_grid_thw": [[1, 90, 90]],
    "completion_tokens": 32,
    "total_tokens": 2081
  },
  "latency_ms": {
    "preprocess": 70.0,
    "queue_wait": 0.0,
    "generation": 800.0,
    "total": 870.0
  },
  "engine": {
    "queue_depth_at_submit": 0,
    "requests_ahead_at_submit": 0
  }
}
```

`finish_reason`：

- `stop`：生成到 tokenizer 的 EOS。
- `length`：达到 `max_new_tokens`。

## 6. 原生 SSE 流式接口

使用 `curl -N` 禁止客户端缓冲：

```bash
IMAGE_BASE64=$(base64 -w 0 assets/dog.png)

curl -N http://127.0.0.1:8000/generate_stream \
  -H 'Content-Type: application/json' \
  -d "{
    \"prompt\": \"描述这张图片\",
    \"image_base64\": \"$IMAGE_BASE64\",
    \"max_new_tokens\": 32,
    \"temperature\": 0.0,
    \"profile\": true
  }"
```

响应事件顺序：

```text
event: metadata
data: {"request_id":"...","model":"...","usage":{...}}

event: token
data: {"index":0,"token_id":108893,"delta":"这","text":"这",...}

event: token
data: {"index":1,"token_id":45930,"delta":"张图片","text":"这张图片",...}

event: done
data: {"text":"这张图片...","token_ids":[...],"finish_reason":"length",...}
```

字段说明：

- `delta`：相对上一个事件新增的文本，适合直接追加到终端。
- `text`：截至当前 token 的累计解码文本，用于校正或展示。
- `token_id`：当前生成的 token。
- `index`：从 0 开始的 completion token 下标。
- `client_visible_ttft_ms`：服务端第一次把 token 交给 SSE 响应生成器的时间。
- `done`：包含与 `/generate` 相同的最终结果和 profiling。

最终 `done.text` 和 `done.token_ids` 是权威结果。流式增量解码遇到特殊的
byte-level token 边界时，客户端可以用累计 `text` 校正展示。

## 7. OpenAI Chat Completions

### 7.1 非流式请求

```bash
IMAGE_BASE64=$(base64 -w 0 assets/dog.png)

curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d "{
    \"model\": \"Qwen2.5-VL-3B-Instruct\",
    \"messages\": [
      {
        \"role\": \"system\",
        \"content\": \"You are a helpful assistant.\"
      },
      {
        \"role\": \"user\",
        \"content\": [
          {
            \"type\": \"image_url\",
            \"image_url\": {
              \"url\": \"data:image/png;base64,$IMAGE_BASE64\"
            }
          },
          {
            \"type\": \"text\",
            \"text\": \"描述这张图片\"
          }
        ]
      }
    ],
    \"max_tokens\": 32,
    \"temperature\": 0,
    \"stream\": false
  }" | python -m json.tool
```

标准字段：

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "created": 178...",
  "model": "Qwen2.5-VL-3B-Instruct",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "..."
      },
      "finish_reason": "length"
    }
  ],
  "usage": {
    "prompt_tokens": 2049,
    "completion_tokens": 32,
    "total_tokens": 2081
  }
}
```

额外的 `nano_vllm` 字段包含 token IDs、多模态信息和阶段延迟。标准 OpenAI
客户端会忽略不认识的扩展字段。

### 7.2 OpenAI SSE

将请求中的 `stream` 改成 `true`：

```json
{
  "stream": true,
  "stream_options": {
    "include_usage": true
  }
}
```

返回顺序：

```text
data: {"object":"chat.completion.chunk","choices":[{"delta":{"role":"assistant"}}]}

data: {"object":"chat.completion.chunk","choices":[{"delta":{"content":"这"}}]}

data: {"object":"chat.completion.chunk","choices":[{"delta":{},"finish_reason":"length"}]}

data: {"object":"chat.completion.chunk","choices":[],"usage":{...}}

data: [DONE]
```

可选的 OpenAI Python SDK 用法：

```python
import base64
from openai import OpenAI

client = OpenAI(
    api_key="EMPTY",
    base_url="http://127.0.0.1:8000/v1",
)
image = base64.b64encode(open("assets/dog.png", "rb").read()).decode()

stream = client.chat.completions.create(
    model="Qwen2.5-VL-3B-Instruct",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image}",
                    },
                },
                {
                    "type": "text",
                    "text": "描述这张图片",
                },
            ],
        }
    ],
    max_tokens=32,
    temperature=0,
    stream=True,
)

for chunk in stream:
    delta = chunk.choices[0].delta.content if chunk.choices else None
    if delta:
        print(delta, end="", flush=True)
```

`openai` SDK 只是可选客户端，不是 nano-vllm 服务端依赖。

## 8. 总体架构

```text
HTTP client
    |
    v
FastAPI event loop
    |
    +-- ConcurrencyLimiter.try_acquire()
    |
    +-- asyncio.to_thread(image decode + AutoProcessor)
    |
    +-- AsyncLLMEngine request queue
            |
            v
       nanovllm-engine thread
            |
            +-- admit up to max_num_seqs
            +-- LLMEngine.add_request()
            +-- Scheduler.schedule()
            +-- ModelRunner.run()
            +-- step metadata(seq_id, token_id, finish_reason)
            |
            +-- non-stream: Future.set_result()
            |
            +-- stream:
                loop.call_soon_threadsafe(
                    asyncio_queue.put_nowait(event)
                )
                    |
                    v
              FastAPI StreamingResponse
```

### 8.1 为什么使用专用 engine thread

`LLMEngine`、Scheduler、BlockManager 和 CUDA 执行状态都由一个线程独占。这样可以：

- 避免多个 HTTP handler 同时修改 waiting/running 队列。
- 避免 KV Cache block 被并发分配或释放。
- 避免多个线程无序调用同一个 CUDA execution context。
- 让 continuous batching 由一个稳定的 step loop 驱动。

FastAPI 的 async 并不意味着 GPU forward 也使用 Python asyncio。这里的异步是：

```text
HTTP 协程不阻塞 event loop
GPU engine 仍由一个同步、可预测的专用线程顺序驱动
```

### 8.2 Future 和 stream queue 的职责

非流式请求只需要最终结果：

```text
engine thread -> concurrent.futures.Future
              -> asyncio.wrap_future
              -> HTTP coroutine
```

流式请求需要多个中间事件：

```text
engine thread -> per-request asyncio.Queue
              -> token/token/.../done
              -> StreamingResponse
```

`asyncio.Queue` 不能从另一个线程直接写，因此 engine thread 使用：

```python
loop.call_soon_threadsafe(queue.put_nowait, event)
```

这保证真正的 queue mutation 在 FastAPI event loop 所在线程执行。

## 9. Multimodal prefill 与 batched decode

对一个图文请求，prefill 并不是“只处理图片”：

```text
pixel_values
    -> Vision Transformer
    -> vision embeddings
    -> 替换 <image_pad> 的 token embeddings
    -> image + text mixed embeddings
    -> Text Transformer prefill
    -> KV Cache
    -> 第一个 completion token
```

两个图片请求 A、B 当前的执行顺序：

```text
step 1: multimodal prefill A
step 2: multimodal prefill B
step 3: decode batch [A, B]
step 4: decode batch [A, B]
...
```

原因是 `ModelRunner.prepare_prefill()` 当前只允许一个 `pixel_values`：

```text
only one image request per prefill batch is supported
```

而 decode 不再运行 Vision Transformer。每个 sequence 只提供最后一个 token、
position、context length 和 block table，因此多个请求可以共享一次 decode
forward。

## 10. Streaming 的实现

原来的 step metadata 只有：

```text
seq_id
num_tokens
produced_token
```

流式版本增加：

```text
token_id
finish_reason
```

每次 `LLMEngine.step_with_metadata()` 完成后，AsyncLLMEngine：

1. 根据 `seq_id` 找到 `_ActiveRequest`。
2. 把新 token 加入该请求自己的 token 列表。
3. 增量 decode，形成 `delta_text` 和累计 `text`。
4. 向对应请求的 stream queue 发送 `token` event。
5. sequence 完成后发送 `done` event。
6. 非流式路径仍然完成原来的 Future。

多个请求共享 decode forward，但事件不会混淆，因为每个 metadata 都携带
`seq_id`，并且每个 request 有独立的 token 状态和 event sink。

## 11. 请求取消与 KV Cache 回收

### 11.1 客户端断开

客户端关闭 SSE 连接后：

```text
ASGI 取消 StreamingResponse body iterator
    -> async generator finally
    -> AsyncLLMEngine.cancel(request_id)
    -> cancel_requested_ids
    -> engine thread 在最近 step 边界检查
    -> LLMEngine.abort_request(seq_id)
    -> Scheduler.abort(seq_id)
    -> 从 waiting 或 running 删除 Sequence
    -> BlockManager.deallocate(seq)
    -> KV Cache block ref_count--
    -> block 返回 free_block_ids
```

取消不能中断已经提交到 GPU 的单个 CUDA kernel。它在当前 model step 返回后执行，
这是有意的安全边界。

### 11.2 尚未进入 Scheduler 的请求

如果请求仍在外部 request queue，它还没有分配 KV Cache。engine thread 取到它时
发现 cancellation flag，直接完成取消，不调用 `LLMEngine.add_request()`。

### 11.3 竞态处理

请求取消可能和正常完成同时发生。terminal accounting 使用 request ID 集合保证：

- `completed/failed/cancelled` 只增加一次。
- `outstanding_requests` 只减少一次。
- 完成后再次调用 `cancel()` 返回 `False`。
- 已取消请求的最终输出不会路由给其他客户端。

## 12. 超时

`request_timeout_seconds` 是完整请求 deadline：

```text
图片解码
+ AutoProcessor
+ engine queue wait
+ multimodal prefill
+ decode
```

非流式请求超时：

```text
HTTP 504
+ engine.cancel(request_id)
```

流式请求可能已经发送 HTTP 200 header，无法再改成 504，因此使用协议内错误：

原生 SSE：

```text
event: error
data: {"error":{"message":"request timed out","code":"request_timeout"}}
```

OpenAI SSE：

```text
data: {"error":{"message":"request timed out",...}}
data: [DONE]
```

## 13. 并发上限与 backpressure

两个上限解决不同问题：

| 上限 | 保护对象 | 达到上限后的行为 |
|---|---|---|
| `max_num_seqs` | GPU Scheduler、KV Cache | 请求留在 engine 外部 queue |
| `max_concurrent_requests` | HTTP、CPU 预处理、内存和总排队长度 | 立即返回 HTTP 429 |

只有 `max_num_seqs` 而没有 HTTP 上限，会导致高流量下 request queue 和 Base64
图片对象持续增长。只有 HTTP 上限而没有 Scheduler admission capacity，则可能让
过多 sequence 抢占 KV Cache。

当前是立即拒绝策略，而不是在 FastAPI semaphore 上无限等待。客户端收到 429 和
`Retry-After: 1` 后可以重试。

## 14. Profiling

单请求中设置：

```json
{
  "profile": true
}
```

会开启逐 engine step 的 CUDA synchronize。关键指标：

| 指标 | 含义 |
|---|---|
| `request_ttft_ms` | HTTP 请求开始到首 token 在 engine 中生成 |
| `engine_ttft_ms` | 首个 prefill step 的 GPU 同步耗时 |
| `prefill_latency_ms` | 该请求所有 prefill step 总时间 |
| `decode_tpot_ms` | decode step 平均每 token 时间 |
| `mean_decode_batch_size` | 该请求参与的 decode step 平均 batch |
| `max_decode_batch_size` | 该请求观察到的最大 decode batch |
| `queue_wait` | 提交 engine 到第一次被调度 |
| `generation` | 第一次调度到 sequence 完成 |
| `client_e2e_latency_ms` | profiling 客户端测得的完整 HTTP 时间 |
| `requests_per_s` | 整轮完成请求数除以整轮时间 |
| `output_tokens_per_s` | 整轮 completion token 数除以整轮时间 |

`profile=true` 会增加同步开销，只用于 benchmark，不应作为生产默认值。

## 15. 单档并发 Profiling

C1：

```bash
/home/agua/anaconda3/envs/yolo26/bin/python \
  examples/qwen2_5_vl_online_profile.py \
  --url http://127.0.0.1:8000/generate \
  --image assets/dog.png \
  --image-input base64 \
  --max-new-tokens 8 \
  --warmup-iters 2 \
  --measure-iters 10 \
  --concurrency 1 \
  --output-json profiles/qwen2_5_vl_online_c1.json
```

C2/C4 只修改：

```text
--concurrency 2
--output-json profiles/qwen2_5_vl_online_c2.json
```

或：

```text
--concurrency 4
--output-json profiles/qwen2_5_vl_online_c4.json
```

## 16. C1/C2/C4 自动回归

推荐使用一个命令同时完成正确性、batching 和吞吐检查：

```bash
/home/agua/anaconda3/envs/yolo26/bin/python \
  examples/qwen2_5_vl_online_regression.py \
  --url http://127.0.0.1:8000/generate \
  --image assets/dog.png \
  --image-input base64 \
  --prompt "描述这张图片" \
  --max-new-tokens 8 \
  --warmup-iters 2 \
  --measure-iters 12 \
  --concurrencies 1,2,4 \
  --output-json profiles/qwen2_5_vl_online_regression.json
```

生成：

```text
profiles/qwen2_5_vl_online_regression_c1.json
profiles/qwen2_5_vl_online_regression_c2.json
profiles/qwen2_5_vl_online_regression_c4.json
profiles/qwen2_5_vl_online_regression.json
```

回归规则：

1. 必须使用 `temperature=0`。
2. 矩阵脚本中的 `warmup_iters` 表示每档并发的 warmup 波次数。例如
   C4 + `warmup_iters=2` 会发送 8 个 warmup 请求，确保 batch=4 路径预热。
3. C1 第一个正式请求作为 greedy token baseline。
4. C1/C2/C4 每个正式请求的 completion token IDs 必须与 baseline 完全一致。
5. C2/C4 的最大 decode batch 应达到
   `min(concurrency, server.max_num_seqs)`。
6. 记录每档 `requests/s`、`output_tokens/s` 和平均 E2E。
7. 计算相对 C1 的吞吐 speedup 和延迟 ratio。

这项在线正确性回归检查“并发不会串结果或改变 greedy 输出”。它不能代替已有的
Transformers vs nano-vllm 离线对齐；两者覆盖不同问题：

```text
离线对齐：
    检查 nano-vllm 模型实现是否正确

在线 C1/C2/C4：
    检查异步调度、batching 和结果路由是否正确
```

如果生成 token 太少，例如 `max_new_tokens=1`，请求可能在 prefill 产生首 token 后
立即结束，无法观察 decode batch。并发回归建议至少生成 8 个 token。

## 17. 本机 C1/C2/C4 实测

2026-07-28 使用以下条件：

```text
model: Qwen2.5-VL-3B-Instruct
image: assets/dog.png
input tokens: 2049
image tokens: 2025
max_new_tokens: 8
temperature: 0
measure requests: 每档 12
profile: true
```

最终结果：

| 并发 | Max decode batch | Requests/s | Output tok/s | 相对 C1 吞吐 | Mean E2E |
|---:|---:|---:|---:|---:|---:|
| C1 | 1 | 1.922 | 15.380 | 1.000x | 510.02 ms |
| C2 | 2 | 2.195 | 17.561 | 1.142x | 891.38 ms |
| C4 | 4 | 2.441 | 19.526 | 1.270x | 1615.06 ms |

正确性：

```text
passed: true
mismatches: 0
reference token IDs:
[108893, 45930, 101987, 99593, 91680, 102783, 104006, 34230]
text:
这张图片展示了一只可爱的小金
```

decode TPOT：

```text
C1: 13.81 ms
C2: 14.20 ms
C4: 14.31 ms
```

完整结果：

```text
profiles/qwen2_5_vl_online_regression.json
profiles/qwen2_5_vl_online_regression_c1.json
profiles/qwen2_5_vl_online_regression_c2.json
profiles/qwen2_5_vl_online_regression_c4.json
```

为什么 C4 不是 4 倍吞吐：

- 每个请求包含 2025 个 image tokens。
- 每个图片请求的 multimodal prefill 仍然单独执行。
- 单请求 prefill 约 350 ms，是当前请求成本的主要部分。
- 只有首 token 之后的 7 个 decode step 可以合批。
- 因此 decode batch 从 1 增加到 4 已经生效，但端到端吞吐受串行视觉/prefill 限制。

这组结果说明 continuous batching 实现正确，但也暴露了下一项核心优化方向：
multimodal prefill batching/chunking 和视觉侧吞吐，而不是继续增加 HTTP 并发。

## 18. 如何解读 C1/C2/C4

continuous batching 的目标是系统吞吐，不保证单请求延迟更低。

可能出现：

```text
C2/C4 mean E2E 高于 C1
但 requests/s 和 output_tokens/s 高于 C1
```

原因：

- 每张图片仍然需要串行 multimodal prefill。
- 先到的请求可能等待后续图片 prefill。
- batch decode 的单 step 可能比 batch=1 稍慢。
- 但一次 forward 同时推进多个 sequence，所以系统总吞吐提高。

判断优先级：

1. token correctness 必须通过。
2. `max_decode_batch_size` 必须证明请求确实合批。
3. 比较 `output_tokens_per_s` 和 `requests_per_s`。
4. 再分析 TTFT、E2E 和 TPOT 的取舍。

## 19. 测试

运行所有无 GPU 单元测试：

```bash
/home/agua/anaconda3/envs/yolo26/bin/python \
  -m unittest discover -s tests -p 'test_*.py' -v
```

测试覆盖：

- 多请求 decode batch 和 `seq_id` 结果隔离。
- token/token/done 事件顺序。
- 关闭 stream 后请求取消。
- Scheduler abort 后 KV Cache block 归还。
- shutdown 后拒绝新请求。
- 原生非流式和 SSE schema。
- OpenAI 非流式、SSE 和 `[DONE]`。
- HTTP 429。
- 超时返回 504 并触发 engine cancel。
- C1/C2/C4 token mismatch 和 batch 判定。

单元测试使用 FakeLLM，不要求 GPU。真实模型行为仍需启动服务后运行在线回归。

## 20. 常见面试问题

### 20.1 FastAPI 是异步的，为什么还需要 engine thread？

FastAPI async 解决的是网络连接和协程等待问题，不会自动让同步 CUDA engine
线程安全。专用 engine thread 负责独占 Scheduler、KV Cache 和模型执行状态，
同时持续驱动 continuous batching。

### 20.2 为什么非流式用 Future，流式用 Queue？

Future 只能表示一次完成，适合最终结果。流式生成是一系列 token 事件，需要
多生产、多消费时序，因此每个请求使用独立 asyncio Queue。

### 20.3 两个客户端是否真的同时推理？

两个 HTTP 请求可以同时存在，也可以同时成为 Scheduler active sequence。图片
multimodal prefill 仍逐个执行；完成 prefill 后，两个 sequence 在同一个 decode
forward 中组成 batch。

### 20.4 waiting 和 running 分别是什么？

- waiting：尚未完成完整 prefill 的 sequence。
- running：已经具备 KV Cache、可以 decode 的 sequence。

它们是 Scheduler 状态队列，不是两个线程。

### 20.5 客户端断开为什么必须释放 KV Cache？

如果只关闭 HTTP 连接而不通知 engine，模型仍会继续生成，sequence 的 KV blocks
也一直占用显存。取消路径必须到达 Scheduler 和 BlockManager，不能只取消 FastAPI
协程。

### 20.6 能否立即中断正在运行的 CUDA kernel？

当前不能。取消在 engine step 边界执行。强行从另一个线程修改 Scheduler 或释放
正在被 kernel 使用的 block 会产生竞态和显存安全问题。

### 20.7 `max_num_seqs` 为什么不能替代 API 并发限制？

它只限制进入 Scheduler 的 sequence 数，不限制外部 queue、Base64 图片、Processor
任务和 HTTP 连接数量。服务层仍需要 backpressure。

### 20.8 SSE 之后 TTFT 有什么变化？

计算 TTFT 不一定变化，但客户端不再等待整段 decode 完成。首 token 生成后立即通过
SSE 到达客户端，因此“首 token 计算完成”变成了用户真正可感知的时间。

### 20.9 如何证明 continuous batching 生效？

同时看：

```text
max_decode_batch_size > 1
output_tokens_per_s 相比 C1 提升
所有并发请求 greedy token IDs 保持一致
```

只看到 HTTP 请求并发不等于 GPU decode 已合批。

### 20.10 为什么 C2 单请求延迟可能比 C1 更高？

请求 A 完成 prefill 后，Scheduler 可能优先处理请求 B 的 prefill。A 暂停一个
step 后再与 B 合批 decode。系统吞吐提高不代表每个请求的 E2E 都降低。

### 20.11 OpenAI compatible 是否等于完整实现 OpenAI API？

不是。这里兼容的是 Chat Completions 的主要请求和响应 schema，以及标准 SSE
chunk 和 `[DONE]`。工具调用、`n>1`、logprobs、远程 image URL 等能力仍未实现。

### 20.12 为什么现在不拆 ZMQ/EngineCore 子进程？

当前单 GPU 架构中，线程队列已经解决 HTTP event loop 和同步 engine 的隔离。
过早增加进程、序列化和 IPC 会扩大调试面。先稳定 request lifecycle、streaming、
cancellation 和 batching，再根据隔离或多实例需求拆进程。

### 20.13 profiling 为什么要 CUDA synchronize？

CUDA kernel 默认异步提交。只测 Python 调用时间会低估真实 GPU step 时间。
benchmark 模式在 step 前后同步以得到可解释的 prefill/decode latency，代价是会
扰动正常流水。

### 20.14 在线正确性和 Transformers 对齐有什么区别？

Transformers 对齐验证模型数学实现和 token 结果；在线回归验证并发调度、状态隔离、
stream routing 和 continuous batching 没有改变结果。

### 20.15 为什么 C4 decode batch=4，吞吐却只提高约 27%？

因为被 batch 的只是 decode。当前请求有 2049 个输入 token，其中 2025 个是 image
tokens，每个 multimodal prefill 仍然串行执行；生成又只有 8 个 token，所以
decode 在总耗时中的占比较小。Amdahl 定律决定了只优化较短阶段不可能得到 4 倍
端到端加速。

## 21. 当前在线阶段结论

这一版已经形成完整的单机在线控制面：

```text
协议接入
-> CPU 多模态预处理
-> 有界并发
-> 异步请求队列
-> continuous batching
-> token streaming
-> cancellation / timeout
-> OpenAI schema
-> profiling / regression
```

后续工作应回到 nano-vllm 核心能力：Attention backend、prefix cache、
multimodal prefill 改进和 AWQ W4A16，而不是继续扩展外围应用。
