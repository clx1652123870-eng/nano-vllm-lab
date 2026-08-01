import argparse
import json
import os
import statistics
import sys
from collections import defaultdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import perf_counter

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import torch
from PIL import Image
from transformers import AutoProcessor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


CATEGORY_PATTERNS = (
    ("kv_cache", ("cache", "reshape_and_cache", "concat_and_cache")),
    (
        "attention",
        ("attention", "flash", "fmha", "fused_attn", "paged_attention"),
    ),
    ("normalization", ("rms", "layer_norm", "layernorm", "rsqrt")),
    ("rotary_embedding", ("rotary", "rope")),
    ("activation", ("silu", "gelu", "sigmoid")),
    (
        "sampling",
        ("sampling", "sample", "argmax", "topk", "multinomial", "exponential"),
    ),
    ("convolution", ("conv", "cudnn")),
    ("gemm", ("gemm", "cublas", "cutlass", "matmul")),
    ("memory", ("memcpy", "memset")),
    ("reduction", ("softmax", "reduce", "reduction")),
    (
        "elementwise_layout",
        (
            "elementwise",
            "pointwise",
            "triton_poi",
            "vectorized",
            "index",
            "gather",
            "scatter",
            "copy",
        ),
    ),
)


def package_version(package):
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Profile CUDA operators for nano-vllm or vLLM Qwen2.5-VL."
    )
    parser.add_argument("--engine", choices=["nano", "vllm"], required=True)
    parser.add_argument(
        "--model",
        default="/home/agua/models/Qwen2.5-VL-3B-Instruct",
    )
    parser.add_argument("--image", default="assets/dog.png")
    parser.add_argument("--prompt", default="描述这张图片")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--warmup-iters", type=int, default=2)
    parser.add_argument("--profile-iters", type=int, default=2)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-batched-tokens", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.72)
    parser.add_argument("--vision-attention-backend", default="hybrid")
    parser.add_argument("--top-kernels", type=int, default=40)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--trace-output")
    return parser.parse_args()


def build_processor_input(args):
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
    prompt_text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = processor(
        text=[prompt_text],
        images=[image],
        return_tensors="pt",
    )
    prompt_summary = {
        "input_tokens": int(inputs["input_ids"].numel()),
        "image_tokens": int((inputs["mm_token_type_ids"] == 1).sum().item()),
        "image_grid_thw": inputs["image_grid_thw"].tolist(),
        "pixel_values_shape": list(inputs["pixel_values"].shape),
    }
    return processor, image, prompt_text, inputs, prompt_summary


def build_nano_runner(args, inputs):
    from nanovllm import LLM, MultiModalPrompt, SamplingParams

    llm = LLM(
        args.model,
        enforce_eager=True,
        tensor_parallel_size=1,
        max_model_len=args.max_model_len,
        max_num_seqs=1,
        max_num_batched_tokens=args.max_num_batched_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        attention_backend="flash_attn",
        vision_attention_backend=args.vision_attention_backend,
    )
    prompt = MultiModalPrompt.from_processor(inputs)
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_new_tokens,
        ignore_eos=True,
    )

    def run_once():
        output = llm.generate([prompt], sampling_params, use_tqdm=False)[0]
        return {
            "token_ids": list(output["token_ids"]),
            "text": output["text"],
        }

    return llm, run_once


def build_vllm_runner(args, image, prompt_text):
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        tensor_parallel_size=1,
        max_model_len=args.max_model_len,
        max_num_seqs=1,
        max_num_batched_tokens=args.max_num_batched_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        enable_prefix_caching=False,
        mm_processor_cache_gb=0,
        limit_mm_per_prompt={"image": 1, "video": 0},
        generation_config="vllm",
    )
    prompts = [
        {
            "prompt": prompt_text,
            "multi_modal_data": {"image": image},
        }
    ]
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_new_tokens,
        ignore_eos=True,
    )

    def run_once():
        output = llm.generate(
            prompts,
            sampling_params=sampling_params,
            use_tqdm=False,
        )[0].outputs[0]
        return {
            "token_ids": list(output.token_ids),
            "text": output.text,
        }

    return llm, run_once


def classify_kernel(name):
    normalized = name.lower()
    for category, patterns in CATEGORY_PATTERNS:
        if any(pattern in normalized for pattern in patterns):
            return category
    return "other"


def aggregate_cuda_events(profiler, top_kernels, profile_iters):
    kernels = defaultdict(lambda: {"count": 0, "cuda_time_us": 0.0})
    categories = defaultdict(lambda: {"count": 0, "cuda_time_us": 0.0})
    for event in profiler.events():
        if str(event.device_type) != "DeviceType.CUDA":
            continue
        duration_us = float(event.time_range.elapsed_us())
        kernel = kernels[event.name]
        kernel["count"] += 1
        kernel["cuda_time_us"] += duration_us
        category = categories[classify_kernel(event.name)]
        category["count"] += 1
        category["cuda_time_us"] += duration_us

    total_us = sum(item["cuda_time_us"] for item in kernels.values())

    def finalize(name, values):
        time_us = values["cuda_time_us"]
        return {
            "name": name,
            "count": values["count"],
            "count_per_request": values["count"] / profile_iters,
            "cuda_time_ms": time_us / 1000,
            "cuda_time_ms_per_request": time_us / 1000 / profile_iters,
            "share_percent": 0.0 if total_us == 0 else time_us / total_us * 100,
        }

    kernel_rows = sorted(
        (finalize(name, values) for name, values in kernels.items()),
        key=lambda item: item["cuda_time_ms"],
        reverse=True,
    )
    category_rows = sorted(
        (finalize(name, values) for name, values in categories.items()),
        key=lambda item: item["cuda_time_ms"],
        reverse=True,
    )
    return {
        "total_cuda_kernel_time_ms": total_us / 1000,
        "cuda_kernel_time_ms_per_request": total_us / 1000 / profile_iters,
        "kernel_launches": sum(item["count"] for item in kernels.values()),
        "kernel_launches_per_request": (
            sum(item["count"] for item in kernels.values()) / profile_iters
        ),
        "unique_kernels": len(kernels),
        "categories": category_rows,
        "kernels": kernel_rows,
        "top_kernels": kernel_rows[:top_kernels],
    }


def summarize(values):
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "mean": statistics.fmean(ordered),
        "min": ordered[0],
        "p50": ordered[len(ordered) // 2],
        "max": ordered[-1],
    }


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.max_new_tokens < 1 or args.warmup_iters < 1 or args.profile_iters < 1:
        raise ValueError(
            "max-new-tokens, warmup-iters and profile-iters must be positive"
        )

    _, image, prompt_text, inputs, prompt_summary = build_processor_input(args)
    if args.engine == "nano":
        llm, run_once = build_nano_runner(args, inputs)
    else:
        llm, run_once = build_vllm_runner(args, image, prompt_text)

    warmup_latencies_ms = []
    warmup_output = None
    for index in range(args.warmup_iters):
        torch.cuda.synchronize()
        started_at = perf_counter()
        warmup_output = run_once()
        torch.cuda.synchronize()
        latency_ms = (perf_counter() - started_at) * 1000
        warmup_latencies_ms.append(latency_ms)
        print(f"warmup[{index}]={latency_ms:.2f}ms")

    activities = [
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ]
    torch.cuda.synchronize()
    started_at = perf_counter()
    profiled_outputs = []
    with torch.profiler.profile(
        activities=activities,
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as profiler:
        for _ in range(args.profile_iters):
            profiled_outputs.append(run_once())
        torch.cuda.synchronize()
    profiled_latency_ms = (perf_counter() - started_at) * 1000

    if args.trace_output:
        trace_path = Path(args.trace_output)
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        profiler.export_chrome_trace(str(trace_path))

    cuda_profile = aggregate_cuda_events(
        profiler,
        args.top_kernels,
        args.profile_iters,
    )
    output = profiled_outputs[-1]
    report = {
        "environment": {
            "gpu": torch.cuda.get_device_name(),
            "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "transformers": package_version("transformers"),
            "flash_attn": package_version("flash_attn"),
            "triton": package_version("triton"),
            "vllm": package_version("vllm"),
        },
        "config": {
            "engine": args.engine,
            "model": args.model,
            "image": args.image,
            "prompt": args.prompt,
            "max_new_tokens": args.max_new_tokens,
            "warmup_iters": args.warmup_iters,
            "profile_iters": args.profile_iters,
            "max_model_len": args.max_model_len,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "vision_attention_backend": (
                args.vision_attention_backend if args.engine == "nano" else None
            ),
            "enforce_eager": True,
            "tensor_parallel_size": 1,
            "prefix_cache": False,
            "vllm_v1_multiprocessing": False,
        },
        "prompt": prompt_summary,
        "timing": {
            "warmup_latency_ms": summarize(warmup_latencies_ms),
            "profiled_wall_latency_ms": profiled_latency_ms,
            "profiled_wall_latency_ms_per_request": (
                profiled_latency_ms / args.profile_iters
            ),
            "profiler_overhead_warning": (
                "Profiled wall latency includes Kineto tracing overhead. "
                "Use CUDA kernel time and non-profiled benchmarks for latency."
            ),
        },
        "correctness": {
            "warmup_matches_profiled": (
                warmup_output is not None
                and warmup_output["token_ids"] == output["token_ids"]
            ),
            "profiled_requests_are_deterministic": all(
                item["token_ids"] == output["token_ids"]
                for item in profiled_outputs
            ),
            "token_ids": output["token_ids"],
            "text": output["text"],
        },
        "cuda_profile": cuda_profile,
    }

    del llm
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"profiled_wall={profiled_latency_ms:.2f}ms")
    print(
        "cuda_kernel_time="
        f"{cuda_profile['cuda_kernel_time_ms_per_request']:.2f}ms/request, "
        f"launches={cuda_profile['kernel_launches_per_request']:.0f}/request"
    )
    for item in cuda_profile["categories"]:
        print(
            f"{item['name']:20s} "
            f"{item['cuda_time_ms_per_request']:9.3f}ms/request "
            f"{item['share_percent']:6.2f}% "
            f"launches={item['count_per_request']:.0f}/request"
        )
    print(f"wrote: {output_path}")


if __name__ == "__main__":
    main()
