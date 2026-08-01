import unittest

import torch
import torch.nn.functional as F

from nanovllm.kernels import (
    cuda_matmul,
    cuda_rms_norm,
    cuda_silu_and_mul,
    cuda_softmax,
    triton_matmul,
    triton_rms_norm,
    triton_silu_and_mul,
    triton_softmax,
)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class CustomKernelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(5)

    def test_softmax_cuda_and_triton(self):
        x = torch.randn(13, 80, dtype=torch.bfloat16, device="cuda")
        expected = torch.softmax(x.float(), dim=-1).bfloat16()
        torch.testing.assert_close(cuda_softmax(x), expected, rtol=0, atol=2e-5)
        torch.testing.assert_close(triton_softmax(x), expected, rtol=0, atol=2e-5)

    def test_rms_norm_cuda_and_triton(self):
        x = torch.randn(13, 80, dtype=torch.bfloat16, device="cuda")
        weight = torch.randn(80, dtype=torch.bfloat16, device="cuda")
        expected = (
            x.float()
            * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6)
            * weight.float()
        ).bfloat16()
        torch.testing.assert_close(
            cuda_rms_norm(x, weight, 1e-6),
            expected,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            triton_rms_norm(x, weight, 1e-6),
            expected,
            rtol=0,
            atol=0,
        )

    def test_silu_and_mul_cuda_and_triton(self):
        x = torch.randn(13, 160, dtype=torch.bfloat16, device="cuda")
        expected = (
            F.silu(x[:, :80].float()) * x[:, 80:].float()
        ).bfloat16()
        torch.testing.assert_close(cuda_silu_and_mul(x), expected, rtol=0, atol=0)
        torch.testing.assert_close(
            triton_silu_and_mul(x),
            expected,
            rtol=0,
            atol=0,
        )

    def test_matmul_cuda_and_triton(self):
        a = torch.randn(7, 64, dtype=torch.bfloat16, device="cuda")
        b = torch.randn(64, 96, dtype=torch.bfloat16, device="cuda")
        expected = (a.float() @ b.float()).bfloat16()
        torch.testing.assert_close(cuda_matmul(a, b), expected, rtol=0.02, atol=0.02)
        torch.testing.assert_close(
            triton_matmul(a, b),
            expected,
            rtol=0.02,
            atol=0.02,
        )
