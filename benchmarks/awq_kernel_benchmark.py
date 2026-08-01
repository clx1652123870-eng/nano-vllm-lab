import argparse
import json
import statistics
import sys
from pathlib import Path
from time import perf_counter

import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nanovllm.layers.quantization.awq_kernels import (
    awq_dequantize,
    awq_gemm,
)


LAYERS = {
    "q_proj": "model.layers.0.self_attn.q_proj",
    "gate_proj": "model.layers.0.mlp.gate_proj",
    "down_proj": "model.layers.0.mlp.down_proj",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark nano-vllm AWQ kernels on checkpoint weights."
    )
    parser.add_argument(
        "--model",
        default="/home/agua/models/Qwen2.5-VL-3B-Instruct-AWQ",
    )
    parser.add_argument("--token-counts", default="1,32,256")
    parser.add_argument("--warmup-iters", type=int, default=5)
    parser.add_argument("--measure-iters", type=int, default=20)
    parser.add_argument("--output-json")
    return parser.parse_args()


def summarize(values):
    ordered = sorted(values)
    return {
        "mean": statistics.fmean(ordered),
        "p50": percentile(ordered, 50),
        "p90": percentile(ordered, 90),
        "min": ordered[0],
        "max": ordered[-1],
        "count": len(ordered),
    }


def percentile(values, percent):
    if len(values) == 1:
        return values[0]
    rank = (len(values) - 1) * percent / 100
    lower = int(rank)
    upper = min(lower + 1, len(values) - 1)
    weight = rank - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def benchmark(function, warmup_iters, measure_iters):
    for _ in range(warmup_iters):
        output = function()
    torch.cuda.synchronize()
    latencies = []
    for _ in range(measure_iters):
        torch.cuda.synchronize()
        started_at = perf_counter()
        output = function()
        torch.cuda.synchronize()
        latencies.append((perf_counter() - started_at) * 1000)
    return output, summarize(latencies)


def load_layer(file, prefix):
    return (
        file.get_tensor(prefix + ".qweight").cuda(),
        file.get_tensor(prefix + ".qzeros").cuda(),
        file.get_tensor(prefix + ".scales").cuda(),
    )


def main():
    args = parse_args()
    token_counts = [
        int(item.strip())
        for item in args.token_counts.split(",")
        if item.strip()
    ]
    if not token_counts or any(item < 1 for item in token_counts):
        raise ValueError("--token-counts must contain positive integers")
    checkpoint = Path(args.model) / "model.safetensors"
    report = {
        "environment": {
            "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "activation_dtype": "torch.bfloat16",
        },
        "config": {
            "model": args.model,
            "group_size": 128,
            "token_counts": token_counts,
            "warmup_iters": args.warmup_iters,
            "measure_iters": args.measure_iters,
        },
        "layers": {},
    }

    with safe_open(checkpoint, framework="pt", device="cpu") as file:
        for label, prefix in LAYERS.items():
            qweight, qzeros, scales = load_layer(file, prefix)
            K, packed_n = qweight.shape
            N = packed_n * 8
            dequantized = awq_dequantize(
                qweight,
                qzeros,
                scales,
                128,
                torch.bfloat16,
            )
            torch.cuda.synchronize()
            layer_report = {
                "shape": {"K": K, "N": N},
                "cases": {},
            }
            print(f"\n[{label}] K={K} N={N}")
            for M in token_counts:
                inputs = torch.randn(
                    M,
                    K,
                    dtype=torch.bfloat16,
                    device="cuda",
                )
                reference = inputs @ dequantized
                implementations = {
                    "triton_fused_w4a16": lambda: awq_gemm(
                        inputs,
                        qweight,
                        qzeros,
                        scales,
                        128,
                    ),
                    "dequantize_then_torch": lambda: inputs
                    @ awq_dequantize(
                        qweight,
                        qzeros,
                        scales,
                        128,
                        inputs.dtype,
                    ),
                    "cached_bf16_matmul": lambda: inputs @ dequantized,
                }
                case = {}
                print(f"  M={M}")
                for name, function in implementations.items():
                    output, latency = benchmark(
                        function,
                        args.warmup_iters,
                        args.measure_iters,
                    )
                    difference = output.float() - reference.float()
                    case[name] = {
                        "latency_ms": latency,
                        "max_abs_error": difference.abs().max().item(),
                        "mean_abs_error": difference.abs().mean().item(),
                    }
                    print(
                        f"    {name:24s} p50={latency['p50']:9.4f}ms "
                        f"max_err={case[name]['max_abs_error']:.6f}"
                    )
                layer_report["cases"][str(M)] = case
            report["layers"][label] = layer_report
            del qweight, qzeros, scales, dequantized
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
