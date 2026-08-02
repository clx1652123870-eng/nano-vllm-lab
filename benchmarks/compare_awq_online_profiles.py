import argparse
import json
from pathlib import Path


METRICS = (
    "request_throughput",
    "output_throughput",
    "mean_ttft_ms",
    "mean_tpot_ms",
    "mean_e2el_ms",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare nano-vllm BF16 and AWQ online benchmark JSONs."
    )
    parser.add_argument(
        "--bf16-dir",
        default="profiles/framework_comparison",
    )
    parser.add_argument("--awq-dir", default="profiles/awq_online")
    parser.add_argument("--concurrencies", default="1,2,4")
    parser.add_argument("--output-len", type=int, default=32)
    parser.add_argument(
        "--output-json",
        default="profiles/awq_online/summary.json",
    )
    return parser.parse_args()


def compare_value(bf16, awq):
    return {
        "bf16": bf16,
        "awq": awq,
        "awq_over_bf16": awq / bf16,
        "awq_delta_percent": (awq / bf16 - 1) * 100,
    }


def load(path):
    return json.loads(path.read_text())


def main():
    args = parse_args()
    concurrencies = [
        int(value.strip())
        for value in args.concurrencies.split(",")
        if value.strip()
    ]
    if not concurrencies or any(value < 1 for value in concurrencies):
        raise ValueError("--concurrencies must contain positive integers")

    bf16_dir = Path(args.bf16_dir)
    awq_dir = Path(args.awq_dir)
    comparisons = {}
    for concurrency in concurrencies:
        bf16_path = bf16_dir / f"nano-c{concurrency}-o{args.output_len}.json"
        awq_path = awq_dir / f"nano-awq-c{concurrency}-o{args.output_len}.json"
        bf16 = load(bf16_path)
        awq = load(awq_path)
        if bf16["completed"] != awq["completed"] or bf16["failed"] or awq["failed"]:
            raise ValueError(
                f"C{concurrency} profiles must have equal successful request counts"
            )
        comparisons[f"c{concurrency}"] = {
            "completed_requests": bf16["completed"],
            "metrics": {
                metric: compare_value(bf16[metric], awq[metric])
                for metric in METRICS
            },
            "source": {
                "bf16": str(bf16_path),
                "awq": str(awq_path),
            },
        }

    output = {
        "workload": {
            "image": "assets/dog.png",
            "output_tokens": args.output_len,
            "concurrencies": concurrencies,
            "request_count_per_case": next(iter(comparisons.values()))[
                "completed_requests"
            ],
            "awq_kernel": "dequantize",
        },
        "comparisons": comparisons,
        "notes": [
            "BF16 and AWQ runs use the same OpenAI-compatible SSE benchmark.",
            "AWQ reduces stored weight bytes but dequantizes before every Linear.",
            "Lower latency and higher throughput are better; interpret delta signs accordingly.",
        ],
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(comparisons, ensure_ascii=False, indent=2))
    print(f"wrote: {output_path}")


if __name__ == "__main__":
    main()
