import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the same Qwen2.5-VL HTTP benchmark against one server."
    )
    parser.add_argument(
        "--framework-label",
        required=True,
        choices=["nano-vllm", "vllm"],
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--model",
        default="/home/agua/models/Qwen2.5-VL-3B-Instruct",
    )
    parser.add_argument(
        "--dataset",
        default=str(ROOT / "benchmarks/data/qwen2_5_vl_dog.jsonl"),
    )
    parser.add_argument("--concurrencies", default="1,2,4")
    parser.add_argument("--output-len", type=int, default=32)
    parser.add_argument("--num-warmups", type=int, default=3)
    parser.add_argument("--num-prompts", type=int, default=12)
    parser.add_argument(
        "--result-dir",
        default=str(ROOT / "profiles/framework_comparison"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    concurrencies = [
        int(value.strip())
        for value in args.concurrencies.split(",")
        if value.strip()
    ]
    if not concurrencies or any(value < 1 for value in concurrencies):
        raise ValueError("--concurrencies must contain positive integers")
    if args.output_len < 1 or args.num_prompts < 1 or args.num_warmups < 0:
        raise ValueError("output length and prompt count must be positive")

    vllm_cli = Path(sys.executable).with_name("vllm")
    if not vllm_cli.exists():
        raise FileNotFoundError(
            f"vllm CLI was not found next to the active Python: {vllm_cli}"
        )

    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    short_label = "nano" if args.framework_label == "nano-vllm" else "vllm"
    for concurrency in concurrencies:
        filename = f"{short_label}-c{concurrency}-o{args.output_len}.json"
        command = [
            str(vllm_cli),
            "bench",
            "serve",
            "--base-url",
            args.base_url,
            "--backend",
            "openai-chat",
            "--endpoint",
            "/v1/chat/completions",
            "--model",
            args.model,
            "--dataset-name",
            "custom_image",
            "--dataset-path",
            args.dataset,
            "--custom-ensure-client-side-data",
            "--custom-output-len",
            str(args.output_len),
            "--num-warmups",
            str(args.num_warmups),
            "--num-prompts",
            str(args.num_prompts),
            "--max-concurrency",
            str(concurrency),
            "--request-rate",
            "inf",
            "--temperature",
            "0",
            "--top-p",
            "1",
            "--ignore-eos",
            "--seed",
            "0",
            "--disable-tqdm",
            "--percentile-metrics",
            "ttft,tpot,itl,e2el",
            "--metric-percentiles",
            "50,90,99",
            "--save-result",
            "--save-detailed",
            "--result-dir",
            str(result_dir),
            "--result-filename",
            filename,
            "--metadata",
            f"framework={args.framework_label}",
            f"concurrency={concurrency}",
            "image=assets/dog.png",
        ]
        print(
            f"\n[{args.framework_label}] concurrency={concurrency} "
            f"output={args.output_len}"
        )
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
