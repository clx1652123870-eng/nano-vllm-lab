import math
import unittest
from types import SimpleNamespace

import torch

from nanovllm.attention import (
    create_decoder_attention_backend,
    create_encoder_attention_backend,
    normalize_attention_backend_name,
    supported_decoder_attention_backends,
    supported_encoder_attention_backends,
)
from nanovllm.models.qwen2_5_vl import Qwen2_5_VisionAttention


def reference_packed_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    scale: float,
    causal: bool,
):
    output = torch.empty_like(query)
    offsets = cu_seqlens.tolist()
    repeats = query.size(1) // key.size(1)
    key = key.repeat_interleave(repeats, dim=1)
    value = value.repeat_interleave(repeats, dim=1)

    for start, end in zip(offsets[:-1], offsets[1:]):
        q_seq = query[start:end]
        k_seq = key[start:end]
        v_seq = value[start:end]
        scores = torch.einsum("qhd,khd->hqk", q_seq, k_seq) * scale
        if causal:
            mask = torch.triu(
                torch.ones(
                    end - start,
                    end - start,
                    dtype=torch.bool,
                    device=query.device,
                ),
                diagonal=1,
            )
            scores = scores.masked_fill(mask.unsqueeze(0), float("-inf"))
        probabilities = torch.softmax(scores, dim=-1)
        output[start:end] = torch.einsum(
            "hqk,khd->qhd",
            probabilities,
            v_seq,
        )
    return output


class AttentionBackendFactoryTest(unittest.TestCase):
    def test_names_and_aliases(self):
        self.assertEqual(normalize_attention_backend_name("Flash-Attention"), "flash_attn")
        self.assertEqual(normalize_attention_backend_name("SDPA"), "torch_sdpa")
        self.assertEqual(normalize_attention_backend_name("cuDNN"), "cudnn_sdpa")
        self.assertEqual(supported_decoder_attention_backends(), ("flash_attn",))
        self.assertEqual(
            supported_encoder_attention_backends(),
            (
                "cuda_fused",
                "cuda_hybrid",
                "cudnn_sdpa",
                "flash_attn",
                "hybrid",
                "torch_math",
                "torch_sdpa",
                "triton",
            ),
        )
        self.assertEqual(
            create_decoder_attention_backend("flash").name,
            "flash_attn",
        )
        self.assertEqual(
            create_encoder_attention_backend("custom-cuda").name,
            "cuda_fused",
        )
        self.assertEqual(
            create_encoder_attention_backend("torch").name,
            "torch_sdpa",
        )
        self.assertEqual(
            create_encoder_attention_backend("cudnn").name,
            "cudnn_sdpa",
        )
        self.assertEqual(
            create_encoder_attention_backend("math").name,
            "torch_math",
        )
        self.assertEqual(
            create_encoder_attention_backend("triton").name,
            "triton",
        )
        self.assertEqual(
            create_encoder_attention_backend("flash-triton").name,
            "hybrid",
        )

    def test_decoder_rejects_unimplemented_paged_sdpa(self):
        with self.assertRaisesRegex(
            ValueError,
            "unsupported decoder attention backend",
        ):
            create_decoder_attention_backend("torch_sdpa")

    def test_backend_choice_does_not_change_vision_parameter_names(self):
        config = SimpleNamespace(hidden_size=16, num_heads=4)
        modules = [
            Qwen2_5_VisionAttention(config, backend)
            for backend in (
                "cuda_fused",
                "cuda_hybrid",
                "flash_attn",
                "torch_sdpa",
                "torch_math",
                "cudnn_sdpa",
                "triton",
                "hybrid",
            )
        ]
        expected_keys = ["qkv.weight", "qkv.bias", "proj.weight", "proj.bias"]
        for module in modules:
            self.assertEqual(list(module.state_dict()), expected_keys)
        self.assertEqual(
            list(modules[0].state_dict()),
            ["qkv.weight", "qkv.bias", "proj.weight", "proj.bias"],
        )


class TorchSDPAEncoderBackendTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.backend = create_encoder_attention_backend("torch_sdpa")
        self.cu_seqlens = torch.tensor([0, 3, 5], dtype=torch.int32)
        self.query = torch.randn(5, 4, 8)
        self.key = torch.randn(5, 2, 8)
        self.value = torch.randn(5, 2, 8)
        self.scale = 1.0 / math.sqrt(self.query.size(-1))

    def run_backend(self, causal: bool):
        return self.backend.forward(
            self.query,
            self.key,
            self.value,
            cu_seqlens_q=self.cu_seqlens,
            cu_seqlens_k=self.cu_seqlens,
            max_seqlen_q=3,
            max_seqlen_k=3,
            softmax_scale=self.scale,
            causal=causal,
        )

    def test_packed_noncausal_gqa_matches_reference(self):
        actual = self.run_backend(causal=False)
        expected = reference_packed_attention(
            self.query,
            self.key,
            self.value,
            self.cu_seqlens,
            scale=self.scale,
            causal=False,
        )
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_packed_causal_gqa_matches_reference(self):
        actual = self.run_backend(causal=True)
        expected = reference_packed_attention(
            self.query,
            self.key,
            self.value,
            self.cu_seqlens,
            scale=self.scale,
            causal=True,
        )
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_rejects_incompatible_gqa_head_counts(self):
        with self.assertRaisesRegex(
            ValueError,
            "query heads must be divisible by KV heads",
        ):
            self.backend.forward(
                torch.randn(5, 3, 8),
                self.key,
                self.value,
                cu_seqlens_q=self.cu_seqlens,
                cu_seqlens_k=self.cu_seqlens,
                max_seqlen_q=3,
                max_seqlen_k=3,
                softmax_scale=self.scale,
                causal=False,
            )

    def test_rejects_invalid_packed_offsets(self):
        with self.assertRaisesRegex(
            ValueError,
            "query cu_seqlens must start at 0 and end at 5",
        ):
            self.backend.forward(
                self.query,
                self.key,
                self.value,
                cu_seqlens_q=torch.tensor([0, 3, 4], dtype=torch.int32),
                cu_seqlens_k=self.cu_seqlens,
                max_seqlen_q=3,
                max_seqlen_k=3,
                softmax_scale=self.scale,
                causal=False,
            )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class CUDAFusedAttentionEncoderBackendTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.backend = create_encoder_attention_backend("cuda_fused")
        self.cu_seqlens = torch.tensor(
            [0, 4, 7],
            dtype=torch.int32,
            device="cuda",
        )
        qkv = torch.randn(
            7,
            64,
            dtype=torch.bfloat16,
            device="cuda",
        )
        query, key, value = qkv.split((32, 16, 16), dim=-1)
        self.query = query.view(7, 4, 8)
        self.key = key.view(7, 2, 8)
        self.value = value.view(7, 2, 8)
        self.assertFalse(self.query.is_contiguous())
        self.assertFalse(self.key.is_contiguous())
        self.assertFalse(self.value.is_contiguous())
        self.scale = 1.0 / math.sqrt(self.query.size(-1))

    def run_backend(self, causal):
        return self.backend.forward(
            self.query,
            self.key,
            self.value,
            cu_seqlens_q=self.cu_seqlens,
            cu_seqlens_k=self.cu_seqlens,
            max_seqlen_q=4,
            max_seqlen_k=4,
            softmax_scale=self.scale,
            causal=causal,
        )

    def assert_matches_reference(self, causal):
        actual = self.run_backend(causal)
        expected = reference_packed_attention(
            self.query,
            self.key,
            self.value,
            self.cu_seqlens,
            scale=self.scale,
            causal=causal,
        )
        torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)

    def test_packed_noncausal_gqa_matches_reference(self):
        self.assert_matches_reference(causal=False)

    def test_packed_causal_gqa_matches_reference(self):
        self.assert_matches_reference(causal=True)

    def test_rejects_long_sequence(self):
        with self.assertRaisesRegex(ValueError, "sequence lengths <= 64"):
            self.backend.forward(
                self.query,
                self.key,
                self.value,
                cu_seqlens_q=self.cu_seqlens,
                cu_seqlens_k=self.cu_seqlens,
                max_seqlen_q=65,
                max_seqlen_k=65,
                softmax_scale=self.scale,
                causal=False,
            )

if __name__ == "__main__":
    unittest.main()
