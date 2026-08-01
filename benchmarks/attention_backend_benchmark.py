import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from time import perf_counter

import torch
from PIL import Image
from transformers import AutoConfig, AutoProcessor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nanovllm.attention import create_encoder_attention_backend
from nanovllm.models.qwen2_5_vl import (
    get_vision_cu_seqlens,
    get_vision_window_index,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark Qwen2.5-VL packed Vision Attention backends."
    )
    parser.add_argument(
        "--model",
        default="/home/agua/models/Qwen2.5-VL-3B-Instruct",
    )
    parser.add_argument("--image", default="assets/dog.png")
    parser.add_argument("--prompt", default="描述这张图片")
    parser.add_argument(
        "--backends",
        default="flash_attn,torch_sdpa,cudnn_sdpa,triton,hybrid",
    )
    parser.add_argument("--cases", default="window,full")
    parser.add_argument("--warmup-iters", type=int, default=3)
    parser.add_argument("--measure-iters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-json")
    return parser.parse_args()


def build_vision_metadata(args):
    processor = AutoProcessor.from_pretrained(args.model)
    image = Image.open(args.image).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": args.prompt},
            ],
        }
    ]
    chat_text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = processor(
        text=[chat_text],
        images=[image],
        return_tensors="pt",
    )
    config = AutoConfig.from_pretrained(args.model).vision_config
    grid_thw = inputs["image_grid_thw"].cpu()
    _, window_cu_seqlens = get_vision_window_index(
        grid_thw,
        config.spatial_merge_size,
        config.window_size,
        config.patch_size,
    )
    full_cu_seqlens = get_vision_cu_seqlens(grid_thw)
    return config, inputs, {
        "window": window_cu_seqlens.cpu(),
        "full": full_cu_seqlens.cpu(),
    }


def benchmark_backend(
    backend_name,
    query,
    key,
    value,
    cu_seqlens,
    max_seqlen,
    softmax_scale,
    reference,
    warmup_iters,
    measure_iters,
):
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

    for _ in range(warmup_iters):
        output = run_once()
    torch.cuda.synchronize()
    del output

    baseline_allocated = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    latencies_ms = []
    output = None
    for _ in range(measure_iters):
        torch.cuda.synchronize()
        started_at = perf_counter()
        output = run_once()
        torch.cuda.synchronize()
        latencies_ms.append((perf_counter() - started_at) * 1000)

    difference = output.float() - reference.float()
    max_abs_error = difference.abs().max().item()
    mean_abs_error = difference.abs().mean().item()
    peak_extra_bytes = max(
        0,
        torch.cuda.max_memory_allocated() - baseline_allocated,
    )
    del output
    torch.cuda.empty_cache()
    return {
        "status": "ok",
        "latency_ms": summarize(latencies_ms),
        "peak_extra_memory_mb": peak_extra_bytes / 1024**2,
        "max_abs_error_vs_flash": max_abs_error,
        "mean_abs_error_vs_flash": mean_abs_error,
    }


def summarize(values):
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "mean": statistics.fmean(ordered),
        "min": ordered[0],
        "p50": percentile(ordered, 50),
        "p90": percentile(ordered, 90),
        "max": ordered[-1],
    }


def percentile(values, percent):
    if len(values) == 1:
        return values[0]
    rank = (len(values) - 1) * percent / 100
    lower = int(rank)
    upper = min(lower + 1, len(values) - 1)
    weight = rank - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def attention_flops(lengths, num_heads, head_dim):
    return 4 * num_heads * head_dim * sum(length * length for length in lengths)


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.warmup_iters < 1 or args.measure_iters < 1:
        raise ValueError("warmup and measure iterations must be positive")

    backend_names = [item.strip() for item in args.backends.split(",") if item.strip()]
    case_names = [item.strip() for item in args.cases.split(",") if item.strip()]
    config, inputs, case_offsets = build_vision_metadata(args)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    num_heads = config.num_heads
    head_dim = config.hidden_size // num_heads
    softmax_scale = 1.0 / math.sqrt(head_dim)
    report = {
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(),
            "dtype": "torch.bfloat16",
        },
        "config": {
            "model": args.model,
            "image": args.image,
            "prompt": args.prompt,
            "backends": backend_names,
            "cases": case_names,
            "warmup_iters": args.warmup_iters,
            "measure_iters": args.measure_iters,
            "seed": args.seed,
        },
        "input": {
            "input_tokens": int(inputs["input_ids"].numel()),
            "image_tokens": int((inputs["mm_token_type_ids"] == 1).sum().item()),
            "image_grid_thw": inputs["image_grid_thw"].tolist(),
            "pixel_values_shape": list(inputs["pixel_values"].shape),
            "vision_depth": config.depth,
            "full_attention_blocks": list(config.fullatt_block_indexes),
            "num_heads": num_heads,
            "head_dim": head_dim,
        },
        "cases": {},
    }

    for case_name in case_names:
        if case_name not in case_offsets:
            raise ValueError(f"unsupported case: {case_name}")
        cu_seqlens_cpu = case_offsets[case_name].to(torch.int32)
        lengths = torch.diff(cu_seqlens_cpu).tolist()
        total_tokens = int(cu_seqlens_cpu[-1])
        max_seqlen = max(lengths)
        cu_seqlens = cu_seqlens_cpu.cuda()
        query = torch.randn(
            total_tokens,
            num_heads,
            head_dim,
            device="cuda",
            dtype=torch.bfloat16,
        )
        key = torch.randn_like(query)
        value = torch.randn_like(query)
        flash_backend = create_encoder_attention_backend("flash_attn")
        reference = flash_backend.forward(
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
        torch.cuda.synchronize()

        case_report = {
            "total_tokens": total_tokens,
            "num_sequences": len(lengths),
            "sequence_lengths": {
                "min": min(lengths),
                "max": max(lengths),
                "unique": sorted(set(lengths)),
            },
            "attention_flops": attention_flops(lengths, num_heads, head_dim),
            "backends": {},
        }
        print(
            f"\n[{case_name}] tokens={total_tokens} sequences={len(lengths)} "
            f"lengths={sorted(set(lengths))}"
        )
        for backend_name in backend_names:
            try:
                result = benchmark_backend(
                    backend_name,
                    query,
                    key,
                    value,
                    cu_seqlens,
                    max_seqlen,
                    softmax_scale,
                    reference,
                    args.warmup_iters,
                    args.measure_iters,
                )
                mean_ms = result["latency_ms"]["mean"]
                result["query_tokens_per_s"] = total_tokens / (mean_ms / 1000)
                result["estimated_tflops"] = (
                    case_report["attention_flops"] / (mean_ms / 1000) / 1e12
                )
                print(
                    f"{backend_name:12s} mean={mean_ms:9.3f}ms "
                    f"p50={result['latency_ms']['p50']:9.3f}ms "
                    f"tflops={result['estimated_tflops']:7.3f} "
                    f"max_err={result['max_abs_error_vs_flash']:.6f}"
                )
            except Exception as exc:
                result = {
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                print(f"{backend_name:12s} ERROR {type(exc).__name__}: {exc}")
            case_report["backends"][backend_name] = result
        report["cases"][case_name] = case_report
        del reference, query, key, value, cu_seqlens
        torch.cuda.empty_cache()

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n"
        )
        print(f"\nwrote: {output_path}")


if __name__ == "__main__":
    main()
