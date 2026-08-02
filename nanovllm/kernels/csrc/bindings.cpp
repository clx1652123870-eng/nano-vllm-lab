#include <torch/extension.h>

torch::Tensor nanovllm_softmax_cuda(torch::Tensor input);
torch::Tensor nanovllm_rms_norm_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    double eps);
torch::Tensor nanovllm_silu_and_mul_cuda(torch::Tensor input);
torch::Tensor nanovllm_matmul_cuda(torch::Tensor a, torch::Tensor b);
torch::Tensor nanovllm_packed_attention_cuda(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor cu_seqlens_q,
    torch::Tensor cu_seqlens_k,
    int64_t max_seqlen_q,
    int64_t max_seqlen_k,
    double softmax_scale,
    bool causal);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("softmax", &nanovllm_softmax_cuda);
  module.def("rms_norm", &nanovllm_rms_norm_cuda);
  module.def("silu_and_mul", &nanovllm_silu_and_mul_cuda);
  module.def("matmul", &nanovllm_matmul_cuda);
  module.def("packed_attention", &nanovllm_packed_attention_cuda);
}
