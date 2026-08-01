import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from nanovllm.layers.quantization.awq import (
    AWQConfig,
    AWQMergedColumnParallelLinear,
)
from nanovllm.layers.quantization.awq_kernels import (
    awq_dequantize,
    awq_gemm,
)


def pack_awq(values: torch.Tensor) -> torch.Tensor:
    order = [0, 4, 1, 5, 2, 6, 3, 7]
    packed = torch.zeros(
        values.shape[0],
        values.shape[1] // 8,
        dtype=torch.int32,
        device=values.device,
    )
    for index, shift_index in enumerate(order):
        packed |= values[:, index::8] << (shift_index * 4)
    return packed


class AWQConfigTest(unittest.TestCase):
    def test_parses_supported_huggingface_config(self):
        hf_config = SimpleNamespace(
            quantization_config={
                "bits": 4,
                "group_size": 128,
                "modules_to_not_convert": ["visual"],
                "quant_method": "awq",
                "version": "gemm",
                "zero_point": True,
            }
        )
        config = AWQConfig.from_hf_config(hf_config)
        self.assertEqual(config.bits, 4)
        self.assertEqual(config.group_size, 128)
        self.assertEqual(config.modules_to_not_convert, ("visual",))

    @patch(
        "nanovllm.layers.quantization.awq.dist.get_world_size",
        return_value=1,
    )
    def test_merged_loader_places_packed_and_scale_shards(self, _):
        config = AWQConfig(4, 128, True, "gemm", ("visual",))
        layer = AWQMergedColumnParallelLinear(
            128,
            [64, 32],
            config,
        )
        first_qweight = torch.full((128, 8), 11, dtype=torch.int32)
        second_qweight = torch.full((128, 4), 22, dtype=torch.int32)
        layer.weight_loader(layer.qweight, first_qweight, 0)
        layer.weight_loader(layer.qweight, second_qweight, 1)
        self.assertTrue(torch.equal(layer.qweight[:, :8], first_qweight))
        self.assertTrue(torch.equal(layer.qweight[:, 8:], second_qweight))

        first_scales = torch.full((1, 64), 1.5, dtype=torch.float16)
        second_scales = torch.full((1, 32), 2.5, dtype=torch.float16)
        layer.weight_loader(layer.scales, first_scales, 0)
        layer.weight_loader(layer.scales, second_scales, 1)
        self.assertTrue(torch.equal(layer.scales[:, :64], first_scales))
        self.assertTrue(torch.equal(layer.scales[:, 64:], second_scales))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class AWQKernelTest(unittest.TestCase):
    def test_dequantize_and_gemm_match_explicit_reference(self):
        torch.manual_seed(3)
        K, N, group_size = 256, 96, 128
        quantized = torch.randint(
            0,
            16,
            (K, N),
            dtype=torch.int32,
            device="cuda",
        )
        zeros = torch.randint(
            0,
            16,
            (K // group_size, N),
            dtype=torch.int32,
            device="cuda",
        )
        scales = torch.rand(
            K // group_size,
            N,
            dtype=torch.float16,
            device="cuda",
        )
        qweight = pack_awq(quantized)
        qzeros = pack_awq(zeros)
        expected_weight = (
            (quantized - zeros.repeat_interleave(group_size, dim=0))
            * scales.repeat_interleave(group_size, dim=0)
        ).bfloat16()
        actual_weight = awq_dequantize(
            qweight,
            qzeros,
            scales,
            group_size,
            torch.bfloat16,
        )
        torch.testing.assert_close(actual_weight, expected_weight, rtol=0, atol=0)

        inputs = torch.randn(7, K, dtype=torch.bfloat16, device="cuda")
        expected = inputs @ expected_weight
        actual = awq_gemm(inputs, qweight, qzeros, scales, group_size)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
