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

void check_bf16_cuda_contiguous(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(tensor.scalar_type() == torch::kBFloat16, name, " must be BF16");
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
