import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run C1/C2/C4 correctness and throughput regression against the "
            "nano-vllm Qwen2.5-VL online server."
        )
    )
    parser.add_argument("--url", default="http://127.0.0.1:8000/generate")
    parser.add_argument("--image", default="assets/dog.png")
    parser.add_argument(
        "--image-input",
        choices=("base64", "path"),
        default="base64",
    )
    parser.add_argument("--prompt", default="描述这张图片")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--warmup-iters", type=int, default=2)
    parser.add_argument("--measure-iters", type=int, default=12)
    parser.add_argument("--concurrencies", default="1,2,4")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument(
        "--output-json",
        default="profiles/qwen2_5_vl_online_regression.json",
    )
    parser.add_argument(
        "--skip-batch-assertion",
        action="store_true",
        help="Record observed decode batches without failing the regression.",
    )
    return parser.parse_args()


def parse_concurrencies(value: str):
    try:
        values = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError("--concurrencies must be comma-separated integers") from exc
    if not values or any(value < 1 for value in values):
        raise ValueError("--concurrencies must contain positive integers")
    if len(values) != len(set(values)):
        raise ValueError("--concurrencies must not contain duplicates")
    if 1 not in values:
        raise ValueError("--concurrencies must include 1 as the correctness baseline")
    return values


def scenario_output_path(output_path: Path, concurrency: int):
    return output_path.with_name(
        f"{output_path.stem}_c{concurrency}{output_path.suffix}"
    )


def run_scenario(args, concurrency: int, output_path: Path):
    profile_script = Path(__file__).with_name("qwen2_5_vl_online_profile.py")
    command = [
        sys.executable,
        str(profile_script),
        "--url",
        args.url,
        "--image",
        args.image,
        "--image-input",
        args.image_input,
        "--prompt",
        args.prompt,
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--temperature",
        str(args.temperature),
        "--warmup-iters",
        str(args.warmup_iters * concurrency),
        "--measure-iters",
        str(args.measure_iters),
        "--concurrency",
        str(concurrency),
        "--timeout",
        str(args.timeout),
        "--output-json",
        str(output_path),
    ]
    if args.ignore_eos:
        command.append("--ignore-eos")

    print(f"\n[regression C{concurrency}]")
    subprocess.run(command, check=True)
    return json.loads(output_path.read_text(encoding="utf-8"))


def build_correctness(scenarios: dict[str, dict]):
    reference_measurement = scenarios["c1"]["measurements"][0]
    reference_token_ids = reference_measurement["token_ids"]
    reference_text = reference_measurement["text"]
    mismatches = []

    for scenario_name, report in scenarios.items():
        for index, measurement in enumerate(report["measurements"]):
            if measurement["token_ids"] == reference_token_ids:
                continue
            mismatches.append(
                {
                    "scenario": scenario_name,
                    "measurement_index": index,
                    "expected_token_ids": reference_token_ids,
                    "actual_token_ids": measurement["token_ids"],
                    "actual_text": measurement["text"],
                }
            )

    return {
        "passed": not mismatches,
        "comparison": "exact greedy completion token IDs",
        "reference_scenario": "c1",
        "reference_token_ids": reference_token_ids,
        "reference_text": reference_text,
        "mismatches": mismatches,
    }


def build_batching_checks(
    scenarios: dict[str, dict],
    skip_assertion: bool,
):
    checks = {}
    passed = True
    for scenario_name, report in scenarios.items():
        concurrency = report["config"]["concurrency"]
        max_num_seqs = report["server"]["limits"]["max_num_seqs"]
        target = min(concurrency, max_num_seqs)
        observed_values = [
            (measurement.get("profile") or {}).get("max_decode_batch_size")
            for measurement in report["measurements"]
        ]
        observed_values = [
            value for value in observed_values if value is not None
        ]
        observed = max(observed_values) if observed_values else None
        scenario_passed = target == 1 or (
            observed is not None and observed >= target
        )
        checks[scenario_name] = {
            "target_decode_batch_size": target,
            "observed_max_decode_batch_size": observed,
            "passed": scenario_passed,
        }
        if not scenario_passed:
            passed = False

    return {
        "passed": passed or skip_assertion,
        "assertion_skipped": skip_assertion,
        "scenarios": checks,
    }


def build_throughput_comparison(scenarios: dict[str, dict]):
    baseline = scenarios["c1"]
    baseline_requests_per_s = baseline["benchmark"]["requests_per_s"]
    baseline_tokens_per_s = baseline["benchmark"]["output_tokens_per_s"]
    baseline_e2e = baseline["summary"]["client_e2e_latency_ms"]["mean"]
    comparisons = {}
    for scenario_name, report in scenarios.items():
        requests_per_s = report["benchmark"]["requests_per_s"]
        tokens_per_s = report["benchmark"]["output_tokens_per_s"]
        mean_e2e = report["summary"]["client_e2e_latency_ms"]["mean"]
        comparisons[scenario_name] = {
            "concurrency": report["config"]["concurrency"],
            "requests_per_s": requests_per_s,
            "requests_per_s_speedup_vs_c1": safe_div(
                requests_per_s,
                baseline_requests_per_s,
            ),
            "output_tokens_per_s": tokens_per_s,
            "output_tokens_per_s_speedup_vs_c1": safe_div(
                tokens_per_s,
                baseline_tokens_per_s,
            ),
            "mean_client_e2e_latency_ms": mean_e2e,
            "mean_client_e2e_ratio_vs_c1": safe_div(
                mean_e2e,
                baseline_e2e,
            ),
        }
    return comparisons


def safe_div(numerator: float, denominator: float):
    if denominator == 0:
        return None
    return numerator / denominator


def main():
    args = parse_args()
    concurrencies = parse_concurrencies(args.concurrencies)
    if args.temperature != 0:
        raise ValueError(
            "online correctness regression requires --temperature 0 "
            "for deterministic greedy decoding"
        )
    if args.warmup_iters < 0 or args.measure_iters < 1:
        raise ValueError("--warmup-iters must be >= 0 and --measure-iters must be >= 1")

    output_path = Path(args.output_json).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scenarios = {}
    for concurrency in concurrencies:
        scenario_path = scenario_output_path(output_path, concurrency)
        scenarios[f"c{concurrency}"] = run_scenario(
            args,
            concurrency,
            scenario_path,
        )

    correctness = build_correctness(scenarios)
    batching = build_batching_checks(
        scenarios,
        args.skip_batch_assertion,
    )
    throughput = build_throughput_comparison(scenarios)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "url": args.url,
            "image": str(Path(args.image).expanduser().resolve()),
            "image_input": args.image_input,
            "prompt": args.prompt,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "ignore_eos": args.ignore_eos,
            "warmup_iters": args.warmup_iters,
            "warmup_interpretation": "concurrent rounds per scenario",
            "measure_iters": args.measure_iters,
            "concurrencies": concurrencies,
        },
        "correctness": correctness,
        "batching": batching,
        "throughput_comparison": throughput,
        "scenarios": scenarios,
    }
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\n[correctness]")
    print(json.dumps(correctness, ensure_ascii=False, indent=2))
    print("\n[batching]")
    print(json.dumps(batching, ensure_ascii=False, indent=2))
    print("\n[throughput comparison]")
    print(json.dumps(throughput, ensure_ascii=False, indent=2))
    print(f"\nwrote: {output_path}")

    if not correctness["passed"] or not batching["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
