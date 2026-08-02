import argparse
import csv
import json
from pathlib import Path


PREFIX = ":attention::"
SUFFIX = "::iteration"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize Nsight Systems NVTX GPU projection CSV."
    )
    parser.add_argument(
        "--input-csv",
        default=(
            "profiles/nsight/"
            "attention_backends_stats_nvtx_gpu_proj_sum.csv"
        ),
    )
    parser.add_argument(
        "--output-json",
        default="profiles/nsight/attention_backends_summary.json",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input_csv)
    backends = {}
    with input_path.open(newline="") as file:
        for row in csv.DictReader(file):
            name = row["Range"]
            if not name.startswith(PREFIX) or not name.endswith(SUFFIX):
                continue
            backend = name[len(PREFIX) : -len(SUFFIX)]
            instances = int(row["Range Instances"])
            gpu_ops = int(row["Total GPU Ops"])
            backends[backend] = {
                "range_instances": instances,
                "gpu_projected_time_ms_per_iteration": (
                    float(row["Total Proj Time (ns)"]) / instances / 1e6
                ),
                "cpu_range_time_us_per_iteration": (
                    float(row["Total Range Time (ns)"]) / instances / 1e3
                ),
                "gpu_operations_per_iteration": gpu_ops / instances,
            }
    if "triton" not in backends:
        raise ValueError("input CSV does not contain a Triton iteration range")
    triton_gpu_ms = backends["triton"]["gpu_projected_time_ms_per_iteration"]
    for values in backends.values():
        values["gpu_time_over_triton"] = (
            values["gpu_projected_time_ms_per_iteration"] / triton_gpu_ms
        )

    output = {
        "workload": {
            "image": "assets/dog.png",
            "case": "window",
            "tokens": 8100,
            "packed_sequences": 144,
            "sequence_lengths": [4, 16, 64],
            "iterations": next(iter(backends.values()))["range_instances"],
        },
        "backends": backends,
        "source": {
            "nsys_report": "profiles/nsight/attention_backends.nsys-rep",
            "nvtx_gpu_projection_csv": str(input_path),
        },
        "notes": [
            "GPU projected time is measured inside each NVTX iteration range.",
            "GPU operations count includes kernels and memory operations.",
            "Nsight instrumentation changes absolute latency; use ratios and launch structure for diagnosis.",
        ],
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(output, ensure_ascii=False, indent=2))
    print(f"wrote: {output_path}")


if __name__ == "__main__":
    main()
