import argparse
import json
import statistics
import sys
from pathlib import Path
from time import perf_counter

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare PyTorch, custom CUDA and Triton kernels."
    )
    parser.add_argument("--warmup-iters", type=int, default=10)
    parser.add_argument("--measure-iters", type=int, default=30)
    parser.add_argument("--output-json")
    return parser.parse_args()


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


def compare_case(name, shape, functions, reference, args):
    reference_output = reference()
    torch.cuda.synchronize()
    case = {"shape": list(shape), "implementations": {}}
    print(f"\n[{name}] shape={shape}")
    for implementation, function in functions.items():
        try:
            output, latency = benchmark(
                function,
                args.warmup_iters,
                args.measure_iters,
            )
            difference = output.float() - reference_output.float()
            result = {
                "status": "ok",
                "latency_ms": latency,
                "max_abs_error": difference.abs().max().item(),
                "mean_abs_error": difference.abs().mean().item(),
            }
            print(
                f"{implementation:12s} mean={latency['mean']:9.4f}ms "
                f"p50={latency['p50']:9.4f}ms "
                f"max_err={result['max_abs_error']:.6f}"
            )
        except Exception as exc:
            result = {
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            print(f"{implementation:12s} ERROR {type(exc).__name__}: {exc}")
        case["implementations"][implementation] = result
    return case


def run_softmax(report, args):
    for rows, cols in ((1024, 64), (1024, 2048), (128, 8192)):
        x = torch.randn(rows, cols, device="cuda", dtype=torch.bfloat16)
        key = f"{rows}x{cols}"
        report["softmax"][key] = compare_case(
            "softmax",
            x.shape,
            {
                "torch_cuda": lambda: torch.softmax(x, dim=-1),
                "custom_cuda": lambda: cuda_softmax(x),
                "triton": lambda: triton_softmax(x),
            },
            lambda: torch.softmax(x.float(), dim=-1).bfloat16(),
            args,
        )


def run_rms_norm(report, args):
    for rows, cols in ((1, 2048), (32, 2048), (2049, 2048), (8100, 1280)):
        x = torch.randn(rows, cols, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(cols, device="cuda", dtype=torch.bfloat16)
        key = f"{rows}x{cols}"
        report["rms_norm"][key] = compare_case(
            "rms_norm",
            x.shape,
            {
                "torch_cuda": lambda: F.rms_norm(x, (cols,), weight, 1e-6),
                "custom_cuda": lambda: cuda_rms_norm(x, weight, 1e-6),
                "triton": lambda: triton_rms_norm(x, weight, 1e-6),
            },
            lambda: (
                x.float()
                * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6)
                * weight.float()
            ).bfloat16(),
            args,
        )


def run_silu_and_mul(report, args):
    for rows, hidden in ((1, 11008), (32, 11008), (2049, 11008)):
        x = torch.randn(
            rows,
            hidden * 2,
            device="cuda",
            dtype=torch.bfloat16,
        )
        key = f"{rows}x{hidden * 2}"
        reference = lambda: (
            F.silu(x[:, :hidden].float()) * x[:, hidden:].float()
        ).bfloat16()
        report["silu_and_mul"][key] = compare_case(
            "silu_and_mul",
            x.shape,
            {
                "torch_cuda": lambda: F.silu(x[:, :hidden]) * x[:, hidden:],
                "custom_cuda": lambda: cuda_silu_and_mul(x),
                "triton": lambda: triton_silu_and_mul(x),
            },
            reference,
            args,
        )


def run_matmul(report, args):
    for M, K, N in ((1, 2048, 2048), (32, 2048, 2048), (256, 2048, 2048)):
        a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
        key = f"{M}x{K}x{N}"
        report["matmul"][key] = compare_case(
            "matmul",
            (M, K, N),
            {
                "torch_cuda": lambda: a @ b,
                "custom_cuda": lambda: cuda_matmul(a, b),
                "triton": lambda: triton_matmul(a, b),
            },
            lambda: (a.float() @ b.float()).bfloat16(),
            args,
        )


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.warmup_iters < 1 or args.measure_iters < 1:
        raise ValueError("warmup and measure iterations must be positive")
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    report = {
        "environment": {
            "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "dtype": "torch.bfloat16",
        },
        "config": {
            "warmup_iters": args.warmup_iters,
            "measure_iters": args.measure_iters,
        },
        "softmax": {},
        "rms_norm": {},
        "silu_and_mul": {},
        "matmul": {},
    }
    run_softmax(report, args)
    run_rms_norm(report, args)
    run_silu_and_mul(report, args)
    run_matmul(report, args)

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n"
        )
        print(f"\nwrote: {output_path}")


if __name__ == "__main__":
    main()
