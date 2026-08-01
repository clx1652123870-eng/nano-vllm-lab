import argparse
import os

from PIL import Image
from transformers import AutoProcessor

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

from vllm import LLM, SamplingParams


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Qwen2.5-VL with the vLLM offline API."
    )
    parser.add_argument(
        "--model",
        default="/home/agua/models/Qwen2.5-VL-3B-Instruct-AWQ",
    )
    parser.add_argument("--image", default="assets/dog.png")
    parser.add_argument("--prompt", default="描述这张图片")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-batched-tokens", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.72)
    return parser.parse_args()


def main():
    args = parse_args()
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
    output = llm.generate(
        [
            {
                "prompt": prompt_text,
                "multi_modal_data": {"image": image},
            }
        ],
        SamplingParams(
            temperature=0,
            max_tokens=args.max_new_tokens,
        ),
        use_tqdm=False,
    )[0].outputs[0]
    print(f"token_ids: {output.token_ids}")
    print(f"text: {output.text!r}")


if __name__ == "__main__":
    main()
