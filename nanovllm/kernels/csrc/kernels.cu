#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cmath>
#include <limits>

namespace {

constexpr int kThreads = 256;
constexpr int kTile = 16;
constexpr int kWarpSize = 32;
constexpr int kAttentionWarps = 8;
constexpr int kMaxAttentionHeadDim = 128;

__inline__ __device__ float warp_reduce_sum(float value) {
  for (int offset = 16; offset > 0; offset /= 2) {
    value += __shfl_down_sync(0xffffffff, value, offset);
  }
  return value;
}

__inline__ __device__ float warp_reduce_max(float value) {
  for (int offset = 16; offset > 0; offset /= 2) {
    value = fmaxf(value, __shfl_down_sync(0xffffffff, value, offset));
  }
  return value;
}

__inline__ __device__ float block_reduce_sum(float value) {
  __shared__ float shared[32];
  const int lane = threadIdx.x % 32;
  const int warp = threadIdx.x / 32;
  value = warp_reduce_sum(value);
  if (lane == 0) {
    shared[warp] = value;
  }
  __syncthreads();
  value = threadIdx.x < blockDim.x / 32 ? shared[lane] : 0.0f;
  if (warp == 0) {
    value = warp_reduce_sum(value);
  }
  if (threadIdx.x == 0) {
    shared[0] = value;
  }
  __syncthreads();
  return shared[0];
}

__inline__ __device__ float block_reduce_max(float value) {
  __shared__ float shared[32];
  const int lane = threadIdx.x % 32;
  const int warp = threadIdx.x / 32;
  value = warp_reduce_max(value);
  if (lane == 0) {
    shared[warp] = value;
  }
  __syncthreads();
  value = threadIdx.x < blockDim.x / 32
      ? shared[lane]
      : -std::numeric_limits<float>::infinity();
  if (warp == 0) {
    value = warp_reduce_max(value);
  }
  if (threadIdx.x == 0) {
    shared[0] = value;
  }
  __syncthreads();
  return shared[0];
}

__global__ void softmax_bf16_kernel(
    const __nv_bfloat16* input,
    __nv_bfloat16* output,
    int rows,
    int cols) {
  const int row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  const __nv_bfloat16* row_input = input + static_cast<int64_t>(row) * cols;
  __nv_bfloat16* row_output = output + static_cast<int64_t>(row) * cols;
  float local_max = -std::numeric_limits<float>::infinity();
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    local_max = fmaxf(local_max, __bfloat162float(row_input[col]));
  }
  const float row_max = block_reduce_max(local_max);
  float local_sum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    local_sum += expf(__bfloat162float(row_input[col]) - row_max);
  }
  const float inverse_sum = 1.0f / block_reduce_sum(local_sum);
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const float value = expf(__bfloat162float(row_input[col]) - row_max);
    row_output[col] = __float2bfloat16(value * inverse_sum);
  }
}

__global__ void rms_norm_bf16_kernel(
    const __nv_bfloat16* input,
    const __nv_bfloat16* weight,
    __nv_bfloat16* output,
    int rows,
    int cols,
    float eps) {
  const int row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  const __nv_bfloat16* row_input = input + static_cast<int64_t>(row) * cols;
  __nv_bfloat16* row_output = output + static_cast<int64_t>(row) * cols;
  float local_sum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const float value = __bfloat162float(row_input[col]);
    local_sum += value * value;
  }
  const float inverse_rms = rsqrtf(block_reduce_sum(local_sum) / cols + eps);
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const float value = __bfloat162float(row_input[col]);
    const float scale = __bfloat162float(weight[col]);
    row_output[col] = __float2bfloat16(value * inverse_rms * scale);
  }
}

__global__ void silu_and_mul_bf16_kernel(
    const __nv_bfloat16* input,
    __nv_bfloat16* output,
    int64_t elements,
    int hidden_size) {
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= elements) {
    return;
  }
  const int64_t row = index / hidden_size;
  const int col = index % hidden_size;
  const int64_t base = row * hidden_size * 2;
  const float gate = __bfloat162float(input[base + col]);
  const float up = __bfloat162float(input[base + hidden_size + col]);
  output[index] = __float2bfloat16((gate / (1.0f + expf(-gate))) * up);
}

__global__ void matmul_bf16_kernel(
    const __nv_bfloat16* a,
    const __nv_bfloat16* b,
    __nv_bfloat16* output,
    int M,
    int N,
    int K) {
  __shared__ __nv_bfloat16 tile_a[kTile][kTile];
  __shared__ __nv_bfloat16 tile_b[kTile][kTile];
  const int row = blockIdx.y * kTile + threadIdx.y;
  const int col = blockIdx.x * kTile + threadIdx.x;
  float accumulator = 0.0f;

  for (int tile = 0; tile < (K + kTile - 1) / kTile; ++tile) {
    const int a_col = tile * kTile + threadIdx.x;
    const int b_row = tile * kTile + threadIdx.y;
    tile_a[threadIdx.y][threadIdx.x] =
        row < M && a_col < K ? a[static_cast<int64_t>(row) * K + a_col]
                            : __float2bfloat16(0.0f);
    tile_b[threadIdx.y][threadIdx.x] =
        b_row < K && col < N ? b[static_cast<int64_t>(b_row) * N + col]
                            : __float2bfloat16(0.0f);
    __syncthreads();
    #pragma unroll
    for (int k = 0; k < kTile; ++k) {
      accumulator += __bfloat162float(tile_a[threadIdx.y][k]) *
          __bfloat162float(tile_b[k][threadIdx.x]);
    }
    __syncthreads();
  }
  if (row < M && col < N) {
    output[static_cast<int64_t>(row) * N + col] =
        __float2bfloat16(accumulator);
  }
}

// One block handles one packed sequence and one query head. Each warp owns a
// query row and keeps the online-softmax state and output fragment in registers.
// K/V are staged in shared memory because Qwen2.5-VL window attention has at
// most 64 tokens per packed sequence.
__global__ void packed_attention_bf16_kernel(
    const __nv_bfloat16* query,
    const __nv_bfloat16* key,
    const __nv_bfloat16* value,
    __nv_bfloat16* output,
    const int32_t* cu_seqlens_q,
    const int32_t* cu_seqlens_k,
    int64_t stride_qt,
    int64_t stride_qh,
    int64_t stride_kt,
    int64_t stride_kh,
    int64_t stride_vt,
    int64_t stride_vh,
    int64_t stride_ot,
    int64_t stride_oh,
    int num_query_heads,
    int num_kv_heads,
    int head_dim,
    int max_seqlen_k,
    float softmax_scale,
    bool causal) {
  const int sequence = blockIdx.x / num_query_heads;
  const int query_head = blockIdx.x % num_query_heads;
  const int kv_group_size = num_query_heads / num_kv_heads;
  const int kv_head = query_head / kv_group_size;
  const int q_start = cu_seqlens_q[sequence];
  const int q_length = cu_seqlens_q[sequence + 1] - q_start;
  const int k_start = cu_seqlens_k[sequence];
  const int k_length = cu_seqlens_k[sequence + 1] - k_start;
  const int lane = threadIdx.x % kWarpSize;
  const int warp = threadIdx.x / kWarpSize;

  extern __shared__ __nv_bfloat16 shared_bf16[];
  __nv_bfloat16* shared_key = shared_bf16;
  __nv_bfloat16* shared_value =
      shared_key + static_cast<int64_t>(max_seqlen_k) * head_dim;

  const int packed_kv_elements = k_length * head_dim;
  for (int index = threadIdx.x; index < packed_kv_elements;
       index += blockDim.x) {
    const int token = index / head_dim;
    const int dim = index % head_dim;
    shared_key[index] =
        key[static_cast<int64_t>(k_start + token) * stride_kt +
            static_cast<int64_t>(kv_head) * stride_kh + dim];
    shared_value[index] =
        value[static_cast<int64_t>(k_start + token) * stride_vt +
              static_cast<int64_t>(kv_head) * stride_vh + dim];
  }
  __syncthreads();

  constexpr int kValuesPerLane = kMaxAttentionHeadDim / kWarpSize;
  for (int local_q = warp; local_q < q_length;
       local_q += kAttentionWarps) {
    float query_fragment[kValuesPerLane];
    float output_accumulator[kValuesPerLane];
    #pragma unroll
    for (int index = 0; index < kValuesPerLane; ++index) {
      const int dim = lane + index * kWarpSize;
      query_fragment[index] = dim < head_dim
          ? __bfloat162float(
                query[static_cast<int64_t>(q_start + local_q) * stride_qt +
                      static_cast<int64_t>(query_head) * stride_qh + dim])
          : 0.0f;
      output_accumulator[index] = 0.0f;
    }

    float row_max = -std::numeric_limits<float>::infinity();
    float row_sum = 0.0f;
    int key_limit = k_length;
    if (causal && local_q + 1 < key_limit) {
      key_limit = local_q + 1;
    }
    for (int local_k = 0; local_k < key_limit; ++local_k) {
      float partial_dot = 0.0f;
      #pragma unroll
      for (int index = 0; index < kValuesPerLane; ++index) {
        const int dim = lane + index * kWarpSize;
        if (dim < head_dim) {
          partial_dot += query_fragment[index] * __bfloat162float(
              shared_key[local_k * head_dim + dim]);
        }
      }
      float score = warp_reduce_sum(partial_dot);
      score = __shfl_sync(0xffffffff, score, 0) * softmax_scale;
      const float new_max = fmaxf(row_max, score);
      const float previous_correction = expf(row_max - new_max);
      const float probability_numerator = expf(score - new_max);
      #pragma unroll
      for (int index = 0; index < kValuesPerLane; ++index) {
        const int dim = lane + index * kWarpSize;
        if (dim < head_dim) {
          const float value_element = __bfloat162float(
              shared_value[local_k * head_dim + dim]);
          output_accumulator[index] =
              output_accumulator[index] * previous_correction +
              probability_numerator * value_element;
        }
      }
      row_sum = row_sum * previous_correction + probability_numerator;
      row_max = new_max;
    }

    const float inverse_sum = 1.0f / row_sum;
    #pragma unroll
    for (int index = 0; index < kValuesPerLane; ++index) {
      const int dim = lane + index * kWarpSize;
      if (dim < head_dim) {
        output[static_cast<int64_t>(q_start + local_q) * stride_ot +
               static_cast<int64_t>(query_head) * stride_oh + dim] =
            __float2bfloat16(output_accumulator[index] * inverse_sum);
      }
    }
  }
}

void check_bf16_cuda_contiguous(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(tensor.scalar_type() == torch::kBFloat16, name, " must be BF16");
}

void check_bf16_cuda_last_dim_contiguous(
    const torch::Tensor& tensor,
    const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.scalar_type() == torch::kBFloat16, name, " must be BF16");
  TORCH_CHECK(tensor.stride(-1) == 1, name, " last dimension must be contiguous");
}

}  // namespace

torch::Tensor nanovllm_softmax_cuda(torch::Tensor input) {
  check_bf16_cuda_contiguous(input, "input");
  TORCH_CHECK(input.dim() >= 1, "input must have at least one dimension");
  const int64_t cols64 = input.size(-1);
  const int64_t rows64 = input.numel() / cols64;
  TORCH_CHECK(cols64 <= std::numeric_limits<int>::max(), "too many columns");
  TORCH_CHECK(rows64 <= std::numeric_limits<int>::max(), "too many rows");
  auto output = torch::empty_like(input);
  softmax_bf16_kernel<<<rows64, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
      static_cast<int>(rows64),
      static_cast<int>(cols64));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor nanovllm_rms_norm_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    double eps) {
  check_bf16_cuda_contiguous(input, "input");
  check_bf16_cuda_contiguous(weight, "weight");
  TORCH_CHECK(weight.dim() == 1, "weight must be one-dimensional");
  TORCH_CHECK(weight.numel() == input.size(-1), "weight shape mismatch");
  const int64_t cols64 = input.size(-1);
  const int64_t rows64 = input.numel() / cols64;
  auto output = torch::empty_like(input);
  rms_norm_bf16_kernel<<<rows64, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
      static_cast<int>(rows64),
      static_cast<int>(cols64),
      static_cast<float>(eps));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor nanovllm_silu_and_mul_cuda(torch::Tensor input) {
  check_bf16_cuda_contiguous(input, "input");
  TORCH_CHECK(input.size(-1) % 2 == 0, "last dimension must be even");
  const int hidden_size = input.size(-1) / 2;
  auto output_sizes = input.sizes().vec();
  output_sizes.back() = hidden_size;
  auto output = torch::empty(output_sizes, input.options());
  const int64_t elements = output.numel();
  const int blocks = (elements + kThreads - 1) / kThreads;
  silu_and_mul_bf16_kernel<<<
      blocks,
      kThreads,
      0,
      at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
      elements,
      hidden_size);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor nanovllm_matmul_cuda(torch::Tensor a, torch::Tensor b) {
  check_bf16_cuda_contiguous(a, "a");
  check_bf16_cuda_contiguous(b, "b");
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "matmul inputs must be matrices");
  TORCH_CHECK(a.size(1) == b.size(0), "matmul shape mismatch");
  const int M = a.size(0);
  const int K = a.size(1);
  const int N = b.size(1);
  auto output = torch::empty({M, N}, a.options());
  const dim3 threads(kTile, kTile);
  const dim3 blocks((N + kTile - 1) / kTile, (M + kTile - 1) / kTile);
  matmul_bf16_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(a.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(b.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
      M,
      N,
      K);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor nanovllm_packed_attention_cuda(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor cu_seqlens_q,
    torch::Tensor cu_seqlens_k,
    int64_t max_seqlen_q,
    int64_t max_seqlen_k,
    double softmax_scale,
    bool causal) {
  check_bf16_cuda_last_dim_contiguous(query, "query");
  check_bf16_cuda_last_dim_contiguous(key, "key");
  check_bf16_cuda_last_dim_contiguous(value, "value");
  TORCH_CHECK(
      query.dim() == 3 && key.dim() == 3 && value.dim() == 3,
      "query, key and value must have shape [tokens, heads, dim]");
  TORCH_CHECK(key.sizes() == value.sizes(), "key/value shape mismatch");
  TORCH_CHECK(query.size(2) == key.size(2), "Q/KV head_dim mismatch");
  TORCH_CHECK(
      query.size(1) % key.size(1) == 0,
      "query heads must be divisible by KV heads");
  TORCH_CHECK(
      query.size(2) > 0 && query.size(2) <= kMaxAttentionHeadDim,
      "CUDA fused attention requires 1 <= head_dim <= ",
      kMaxAttentionHeadDim);
  TORCH_CHECK(
      max_seqlen_q > 0 && max_seqlen_q <= 64 &&
          max_seqlen_k > 0 && max_seqlen_k <= 64,
      "CUDA fused attention requires max sequence length <= 64");
  TORCH_CHECK(
      cu_seqlens_q.is_cuda() && cu_seqlens_k.is_cuda(),
      "cu_seqlens must be CUDA tensors");
  TORCH_CHECK(
      cu_seqlens_q.is_contiguous() && cu_seqlens_k.is_contiguous(),
      "cu_seqlens must be contiguous");
  TORCH_CHECK(
      cu_seqlens_q.scalar_type() == torch::kInt32 &&
          cu_seqlens_k.scalar_type() == torch::kInt32,
      "cu_seqlens must use int32");
  TORCH_CHECK(
      cu_seqlens_q.dim() == 1 &&
          cu_seqlens_q.numel() == cu_seqlens_k.numel() &&
          cu_seqlens_q.numel() >= 2,
      "Q/KV cu_seqlens shape mismatch");

  const int num_sequences = cu_seqlens_q.numel() - 1;
  const int num_query_heads = query.size(1);
  const int num_kv_heads = key.size(1);
  const int head_dim = query.size(2);
  const int blocks = num_sequences * num_query_heads;
  const size_t shared_bytes =
      2 * max_seqlen_k * head_dim * sizeof(__nv_bfloat16);
  auto output = torch::empty_like(query);
  packed_attention_bf16_kernel<<<
      blocks,
      kThreads,
      shared_bytes,
      at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(
          query.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(key.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(value.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
      cu_seqlens_q.data_ptr<int32_t>(),
      cu_seqlens_k.data_ptr<int32_t>(),
      query.stride(0),
      query.stride(1),
      key.stride(0),
      key.stride(1),
      value.stride(0),
      value.stride(1),
      output.stride(0),
      output.stride(1),
      num_query_heads,
      num_kv_heads,
      head_dim,
      max_seqlen_k,
      static_cast<float>(softmax_scale),
      causal);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}
