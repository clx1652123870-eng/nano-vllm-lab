import argparse
import json
import statistics
import sys
from pathlib import Path
from time import perf_counter

import torch
from PIL import Image
from transformers import AutoProcessor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nanovllm import LLM, MultiModalPrompt, SamplingParams


def parse_args():
    parser = argparse.ArgumentParser(
        description="Profile Qwen2.5-VL single-image offline inference in nano-vllm."
    )
    parser.add_argument("--model", default="/home/agua/models/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--image", default="assets/dog.png")
    parser.add_argument("--prompt", default="描述这张图片")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--warmup-iters", type=int, default=1)
    parser.add_argument("--measure-iters", type=int, default=3)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-batched-tokens", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.72)
    parser.add_argument("--output-json")
    parser.add_argument("--print-text", action="store_true")
    return parser.parse_args()


def build_prompt(model_path: str, image_path: str, question: str):
    t0 = perf_counter()
    processor = AutoProcessor.from_pretrained(model_path)
    processor_load_ms = elapsed_ms(t0)

    t0 = perf_counter()
    image = Image.open(image_path).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        }
    ]
    chat_text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = processor(text=[chat_text], images=[image], return_tensors="pt")
    prompt = MultiModalPrompt.from_processor(inputs)
    request_preprocess_ms = elapsed_ms(t0)

    return processor, inputs, prompt, chat_text, {
        "processor_load_ms": processor_load_ms,
        "request_preprocess_ms": request_preprocess_ms,
    }


def create_llm(args):
    t0 = perf_counter()
    llm = LLM(
        args.model,
        enforce_eager=True,
        tensor_parallel_size=1,
        max_model_len=args.max_model_len,
        max_num_seqs=1,
        max_num_batched_tokens=args.max_num_batched_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    sync_cuda()
    return llm, elapsed_ms(t0)


def run_once(llm: LLM, prompt: MultiModalPrompt, sampling_params: SamplingParams):
    llm.add_request(prompt, sampling_params)
    outputs = {}
    prefill_latencies_ms = []
    decode_latencies_ms = []
    prefill_tokens = 0
    decode_tokens = 0

    sync_cuda()
    t_total = perf_counter()
    while not llm.is_finished():
        sync_cuda()
        t_step = perf_counter()
        output, num_tokens = llm.step()
        sync_cuda()
        step_ms = elapsed_ms(t_step)

        if num_tokens > 0:
            prefill_latencies_ms.append(step_ms)
            prefill_tokens += num_tokens
        else:
            decode_latencies_ms.append(step_ms)
            decode_tokens += -num_tokens

        for seq_id, token_ids in output:
            outputs[seq_id] = token_ids

    end_to_end_latency_ms = elapsed_ms(t_total)
    token_ids = outputs[min(outputs.keys())]
    text = llm.tokenizer.decode(token_ids)
    prefill_latency_ms = sum(prefill_latencies_ms)
    decode_latency_ms = sum(decode_latencies_ms)
    generated_tokens = len(token_ids)

    return {
        "input_tokens": len(prompt.input_ids),
        "generated_tokens": generated_tokens,
        "prefill_steps": len(prefill_latencies_ms),
        "decode_steps": len(decode_latencies_ms),
        "prefill_tokens": prefill_tokens,
        "decode_tokens": decode_tokens,
        "ttft_ms": prefill_latencies_ms[0] if prefill_latencies_ms else None,
        "prefill_latency_ms": prefill_latency_ms,
        "prefill_tokens_per_s": safe_div(prefill_tokens, prefill_latency_ms / 1000),
        "decode_latency_ms": decode_latency_ms,
        "decode_tpot_ms": safe_div(decode_latency_ms, decode_tokens),
        "decode_tokens_per_s": safe_div(decode_tokens, decode_latency_ms / 1000),
        "end_to_end_latency_ms": end_to_end_latency_ms,
        "end_to_end_tokens_per_s": safe_div(generated_tokens, end_to_end_latency_ms / 1000),
        "step_latencies_ms": {
            "prefill": prefill_latencies_ms,
            "decode": decode_latencies_ms,
        },
        "token_ids": token_ids,
        "text": text,
    }


def add_memory_stats(result: dict):
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    result.update(
        {
            "peak_allocated_gb": bytes_to_gb(torch.cuda.max_memory_allocated()),
            "peak_reserved_gb": bytes_to_gb(torch.cuda.max_memory_reserved()),
            "current_allocated_gb": bytes_to_gb(torch.cuda.memory_allocated()),
            "current_reserved_gb": bytes_to_gb(torch.cuda.memory_reserved()),
            "cuda_free_gb": bytes_to_gb(free_bytes),
            "cuda_total_gb": bytes_to_gb(total_bytes),
        }
    )


def build_summary(measurements: list[dict]):
    metric_names = [
        "ttft_ms",
        "prefill_latency_ms",
        "prefill_tokens_per_s",
        "decode_latency_ms",
        "decode_tpot_ms",
        "decode_tokens_per_s",
        "end_to_end_latency_ms",
        "end_to_end_tokens_per_s",
        "peak_allocated_gb",
        "peak_reserved_gb",
    ]
    return {
        name: summarize([item[name] for item in measurements if item[name] is not None])
        for name in metric_names
    }


def prompt_summary(inputs, prompt: MultiModalPrompt):
    return {
        "input_tokens": len(prompt.input_ids),
        "image_tokens": int((inputs["mm_token_type_ids"] == 1).sum().item()),
        "image_grid_thw": inputs["image_grid_thw"].tolist(),
        "pixel_values_shape": list(inputs["pixel_values"].shape),
        "pixel_values_dtype": str(inputs["pixel_values"].dtype),
    }


def validate_args(args, prompt: MultiModalPrompt):
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be at least 1")
    if args.warmup_iters < 0 or args.measure_iters < 1:
        raise ValueError("--warmup-iters must be >= 0 and --measure-iters must be >= 1")
    if len(prompt.input_ids) > args.max_num_batched_tokens:
        raise ValueError(
            f"prompt has {len(prompt.input_ids)} tokens, but max_num_batched_tokens="
            f"{args.max_num_batched_tokens}; multimodal prefill is not chunked yet"
        )
    if len(prompt.input_ids) + args.max_new_tokens > args.max_model_len:
        raise ValueError(
            f"prompt + generation needs {len(prompt.input_ids) + args.max_new_tokens} "
            f"tokens, but max_model_len={args.max_model_len}"
        )


def print_iteration(prefix: str, result: dict):
    print(
        f"{prefix}: "
        f"ttft={result['ttft_ms']:.2f}ms, "
        f"prefill={result['prefill_latency_ms']:.2f}ms "
        f"({result['prefill_tokens_per_s']:.2f}tok/s), "
        f"decode_tpot={format_optional(result['decode_tpot_ms'])}ms, "
        f"decode={format_optional(result['decode_tokens_per_s'])}tok/s, "
        f"e2e={result['end_to_end_latency_ms']:.2f}ms, "
        f"peak_alloc={result['peak_allocated_gb']:.2f}GB"
    )


def summarize(values: list[float]):
    if not values:
        return None
    values = sorted(values)
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "min": values[0],
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
        "max": values[-1],
    }


def percentile(values: list[float], percent: float):
    if len(values) == 1:
        return values[0]
    rank = (len(values) - 1) * percent / 100
    lower = int(rank)
    upper = min(lower + 1, len(values) - 1)
    weight = rank - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def safe_div(numerator: float, denominator: float):
    if denominator == 0:
        return None
    return numerator / denominator


def format_optional(value):
    return "n/a" if value is None else f"{value:.2f}"


def elapsed_ms(start: float):
    return (perf_counter() - start) * 1000


def bytes_to_gb(value: int):
    return value / 1024**3


def sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this profiling script")

    _, inputs, prompt, chat_text, preprocess_times = build_prompt(
        args.model,
        args.image,
        args.prompt,
    )
    validate_args(args, prompt)
    prompt_info = prompt_summary(inputs, prompt)

    print("[prompt]")
    print(f"chat_template: {chat_text!r}")
    print(json.dumps(prompt_info, ensure_ascii=False, indent=2))

    llm, engine_init_ms = create_llm(args)
    print("\n[setup]")
    print(f"processor_load_ms: {preprocess_times['processor_load_ms']:.2f}")
    print(f"request_preprocess_ms: {preprocess_times['request_preprocess_ms']:.2f}")
    print(f"engine_init_ms: {engine_init_ms:.2f}")
    print(f"current_allocated_gb: {bytes_to_gb(torch.cuda.memory_allocated()):.2f}")
    print(f"current_reserved_gb: {bytes_to_gb(torch.cuda.memory_reserved()):.2f}")

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_new_tokens,
    )

    warmups = []
    for idx in range(args.warmup_iters):
        torch.cuda.reset_peak_memory_stats()
        result = run_once(llm, prompt, sampling_params)
        add_memory_stats(result)
        warmups.append(result)
        print_iteration(f"warmup[{idx}]", result)

    measurements = []
    for idx in range(args.measure_iters):
        torch.cuda.reset_peak_memory_stats()
        result = run_once(llm, prompt, sampling_params)
        add_memory_stats(result)
        measurements.append(result)
        print_iteration(f"measure[{idx}]", result)

    report = {
        "config": {
            "model": args.model,
            "image": args.image,
            "prompt": args.prompt,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "warmup_iters": args.warmup_iters,
            "measure_iters": args.measure_iters,
            "max_model_len": args.max_model_len,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "gpu_memory_utilization": args.gpu_memory_utilization,
        },
        "prompt": prompt_info,
        "setup": {
            **preprocess_times,
            "engine_init_ms": engine_init_ms,
        },
        "warmups": warmups,
        "measurements": measurements,
        "summary": build_summary(measurements),
    }

    print("\n[summary]")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    if args.print_text:
        print("\n[text]")
        print(measurements[-1]["text"])

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(f"\nwrote: {output_path}")


if __name__ == "__main__":
    main()
