import argparse
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from attention_backend_benchmark import build_vision_metadata
from nanovllm.attention import create_encoder_attention_backend


def parse_args():
    parser = argparse.ArgumentParser(
        description="NVTX-annotated packed Attention workload for Nsight Systems."
    )
    parser.add_argument(
        "--model",
        default="/home/agua/models/Qwen2.5-VL-3B-Instruct",
    )
    parser.add_argument("--image", default="assets/dog.png")
    parser.add_argument("--prompt", default="描述这张图片")
    parser.add_argument(
        "--backends",
        default="torch_math,torch_sdpa,cuda_fused,triton",
    )
    parser.add_argument("--warmup-iters", type=int, default=3)
    parser.add_argument("--measure-iters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    config, _, offsets = build_vision_metadata(args)
    cu_seqlens = offsets["window"].to(dtype=torch.int32, device="cuda")
    lengths = torch.diff(cu_seqlens).cpu().tolist()
    total_tokens = int(cu_seqlens[-1].item())
    max_seqlen = max(lengths)
    num_heads = config.num_heads
    head_dim = config.hidden_size // num_heads
    softmax_scale = 1.0 / math.sqrt(head_dim)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    query = torch.randn(
        total_tokens,
        num_heads,
        head_dim,
        dtype=torch.bfloat16,
        device="cuda",
    )
    key = torch.randn_like(query)
    value = torch.randn_like(query)

    backend_names = [
        value.strip() for value in args.backends.split(",") if value.strip()
    ]
    for backend_name in backend_names:
        backend = create_encoder_attention_backend(backend_name)

        def run_once():
            return backend.forward(
                query,
                key,
                value,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                softmax_scale=softmax_scale,
                causal=False,
            )

        for _ in range(args.warmup_iters):
            output = run_once()
        torch.cuda.synchronize()

        torch.cuda.nvtx.range_push(f"attention::{backend_name}::measured")
        for _ in range(args.measure_iters):
            torch.cuda.nvtx.range_push(f"attention::{backend_name}::iteration")
            output = run_once()
            torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()
        print(
            f"{backend_name}: output={tuple(output.shape)} "
            f"checksum={output.float().sum().item():.6f}"
        )


if __name__ == "__main__":
    main()
