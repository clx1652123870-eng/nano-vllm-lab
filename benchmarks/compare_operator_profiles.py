import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare nano-vllm and vLLM CUDA operator profiles."
    )
    parser.add_argument(
        "--nano-prefill",
        default="profiles/operator_comparison/nano-dog-o1.json",
    )
    parser.add_argument(
        "--nano-total",
        default="profiles/operator_comparison/nano-dog-o32.json",
    )
    parser.add_argument(
        "--vllm-prefill",
        default="profiles/operator_comparison/vllm-dog-o1.json",
    )
    parser.add_argument(
        "--vllm-total",
        default="profiles/operator_comparison/vllm-dog-o32.json",
    )
    parser.add_argument(
        "--output-json",
        default="profiles/operator_comparison/summary.json",
    )
    return parser.parse_args()


def load(path):
    return json.loads(Path(path).read_text())


def category_map(report):
    return {
        item["name"]: {
            "cuda_time_ms": item["cuda_time_ms_per_request"],
            "launches": item["count_per_request"],
        }
        for item in report["cuda_profile"]["categories"]
    }


def raw_phase(report):
    profile = report["cuda_profile"]
    return {
        "cuda_kernel_time_ms": profile["cuda_kernel_time_ms_per_request"],
        "kernel_launches": profile["kernel_launches_per_request"],
        "steady_wall_latency_ms": report["timing"]["warmup_latency_ms"]["min"],
        "categories": category_map(report),
    }


def subtract_phase(total, prefill, decode_tokens):
    total_categories = total["categories"]
    prefill_categories = prefill["categories"]
    categories = {}
    for name in sorted(set(total_categories) | set(prefill_categories)):
        total_item = total_categories.get(name, {})
        prefill_item = prefill_categories.get(name, {})
        categories[name] = {
            "cuda_time_ms": (
                total_item.get("cuda_time_ms", 0.0)
                - prefill_item.get("cuda_time_ms", 0.0)
            )
            / decode_tokens,
            "launches": (
                total_item.get("launches", 0.0)
                - prefill_item.get("launches", 0.0)
            )
            / decode_tokens,
        }
    kernel_time_ms = (
        total["cuda_kernel_time_ms"] - prefill["cuda_kernel_time_ms"]
    ) / decode_tokens
    wall_latency_ms = (
        total["steady_wall_latency_ms"] - prefill["steady_wall_latency_ms"]
    ) / decode_tokens
    return {
        "cuda_kernel_time_ms": kernel_time_ms,
        "kernel_launches": (
            total["kernel_launches"] - prefill["kernel_launches"]
        )
        / decode_tokens,
        "steady_wall_latency_ms": wall_latency_ms,
        "estimated_cpu_and_launch_gap_ms": wall_latency_ms - kernel_time_ms,
        "categories": categories,
        "derivation": "(O32 profile - O1 profile) / 31 decode tokens",
    }


def comparison(nano_value, vllm_value):
    return {
        "nano": nano_value,
        "vllm": vllm_value,
        "nano_over_vllm": None if vllm_value == 0 else nano_value / vllm_value,
        "nano_delta_percent": (
            None if vllm_value == 0 else (nano_value / vllm_value - 1) * 100
        ),
    }


def compare_phase(nano, vllm):
    categories = {}
    for name in sorted(set(nano["categories"]) | set(vllm["categories"])):
        nano_item = nano["categories"].get(name, {})
        vllm_item = vllm["categories"].get(name, {})
        categories[name] = {
            "cuda_time_ms": comparison(
                nano_item.get("cuda_time_ms", 0.0),
                vllm_item.get("cuda_time_ms", 0.0),
            ),
            "launches": comparison(
                nano_item.get("launches", 0.0),
                vllm_item.get("launches", 0.0),
            ),
        }
    result = {
        "cuda_kernel_time_ms": comparison(
            nano["cuda_kernel_time_ms"],
            vllm["cuda_kernel_time_ms"],
        ),
        "kernel_launches": comparison(
            nano["kernel_launches"],
            vllm["kernel_launches"],
        ),
        "steady_wall_latency_ms": comparison(
            nano["steady_wall_latency_ms"],
            vllm["steady_wall_latency_ms"],
        ),
        "categories": categories,
    }
    if "estimated_cpu_and_launch_gap_ms" in nano:
        result["estimated_cpu_and_launch_gap_ms"] = comparison(
            nano["estimated_cpu_and_launch_gap_ms"],
            vllm["estimated_cpu_and_launch_gap_ms"],
        )
        result["derivation"] = nano["derivation"]
    return result


def matching_prefix(left, right):
    count = 0
    for left_token, right_token in zip(left, right):
        if left_token != right_token:
            break
        count += 1
    return count


def validate_reports(reports):
    reference = reports["nano"]["total"]
    expected_prompt = reference["prompt"]
    expected_model = reference["config"]["model"]
    expected_image = reference["config"]["image"]
    for framework, phases in reports.items():
        if phases["prefill"]["config"]["max_new_tokens"] != 1:
            raise ValueError(f"{framework} prefill profile must use one output token")
        for phase_name, report in phases.items():
            if report["prompt"] != expected_prompt:
                raise ValueError(f"{framework} {phase_name} prompt shape differs")
            if report["config"]["model"] != expected_model:
                raise ValueError(f"{framework} {phase_name} model differs")
            if report["config"]["image"] != expected_image:
                raise ValueError(f"{framework} {phase_name} image differs")


def main():
    args = parse_args()
    reports = {
        "nano": {
            "prefill": load(args.nano_prefill),
            "total": load(args.nano_total),
        },
        "vllm": {
            "prefill": load(args.vllm_prefill),
            "total": load(args.vllm_total),
        },
    }
    validate_reports(reports)
    output_tokens = reports["nano"]["total"]["config"]["max_new_tokens"]
    if output_tokens != reports["vllm"]["total"]["config"]["max_new_tokens"]:
        raise ValueError("total profiles must use the same output length")
    decode_tokens = output_tokens - 1
    if decode_tokens < 1:
        raise ValueError("total profile must include at least one decode token")

    phases = {}
    for framework in ("nano", "vllm"):
        prefill = raw_phase(reports[framework]["prefill"])
        total = raw_phase(reports[framework]["total"])
        phases[framework] = {
            "prefill_o1": prefill,
            "total_o32": total,
            "decode_per_token": subtract_phase(total, prefill, decode_tokens),
        }

    nano_tokens = reports["nano"]["total"]["correctness"]["token_ids"]
    vllm_tokens = reports["vllm"]["total"]["correctness"]["token_ids"]
    summary = {
        "environment": reports["nano"]["total"]["environment"],
        "workload": {
            **reports["nano"]["total"]["prompt"],
            "image": reports["nano"]["total"]["config"]["image"],
            "output_tokens": output_tokens,
            "decode_tokens": decode_tokens,
            "dtype": "torch.bfloat16",
            "enforce_eager": True,
            "tensor_parallel_size": 1,
        },
        "correctness": {
            "matching_greedy_prefix_tokens": matching_prefix(
                nano_tokens,
                vllm_tokens,
            ),
            "nano_token_ids": nano_tokens,
            "vllm_token_ids": vllm_tokens,
            "nano_text": reports["nano"]["total"]["correctness"]["text"],
            "vllm_text": reports["vllm"]["total"]["correctness"]["text"],
        },
        "phases": phases,
        "comparison": {
            "prefill_o1": compare_phase(
                phases["nano"]["prefill_o1"],
                phases["vllm"]["prefill_o1"],
            ),
            "total_o32": compare_phase(
                phases["nano"]["total_o32"],
                phases["vllm"]["total_o32"],
            ),
            "decode_per_token": compare_phase(
                phases["nano"]["decode_per_token"],
                phases["vllm"]["decode_per_token"],
            ),
        },
        "methodology_notes": [
            "O1 contains multimodal prefill, first-token sampling, and no decode step.",
            "Decode per-token values are derived from separate O32 and O1 profiles.",
            "PIL and processor work is outside the profile; vision encoder forward is inside.",
            "CUDA times are sums of raw Kineto CUDA kernel events; overlapping kernels would be double-counted.",
            "Operator categories are heuristic classifications based on CUDA kernel names.",
            "Profiled wall time contains Kineto overhead and is not used as production latency.",
            "steady_wall_latency_ms uses the fastest post-initialization warmup from each process and is indicative only.",
            "vLLM V1 multiprocessing is disabled so both frameworks are captured in-process.",
        ],
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary["comparison"], ensure_ascii=False, indent=2))
    print(f"wrote: {output_path}")


if __name__ == "__main__":
    main()
