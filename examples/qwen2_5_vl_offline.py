import argparse
import sys
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nanovllm import LLM, MultiModalPrompt, SamplingParams
from nanovllm.attention import supported_encoder_attention_backends


def parse_args():
    parser = argparse.ArgumentParser(description="Run Qwen2.5-VL single-image offline inference.")
    parser.add_argument("--model", default="/home/agua/models/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--image", default="assets/dog.png")
    parser.add_argument("--prompt", default="描述这张图片")
    parser.add_argument("--engine", choices=["nano", "transformers", "both"], default="nano")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--max-num-batched-tokens", type=int, default=1024)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--attention-backend", default="flash_attn")
    parser.add_argument(
        "--vision-attention-backend",
        choices=supported_encoder_attention_backends(),
        default="flash_attn",
    )
    parser.add_argument("--no-tqdm", action="store_true")
    return parser.parse_args()


def build_prompt(model_path: str, image_path: str, question: str):
    processor = AutoProcessor.from_pretrained(model_path)
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
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = processor(text=[text], images=[image], return_tensors="pt")
    prompt = MultiModalPrompt.from_processor(inputs)
    return processor, inputs, prompt, text


def run_nano(args, prompt: MultiModalPrompt):
    llm = LLM(
        args.model,
        enforce_eager=True,
        tensor_parallel_size=1,
        max_model_len=args.max_model_len,
        max_num_seqs=1,
        max_num_batched_tokens=args.max_num_batched_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        attention_backend=args.attention_backend,
        vision_attention_backend=args.vision_attention_backend,
    )
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_new_tokens,
    )
    return llm.generate([prompt], sampling_params, use_tqdm=not args.no_tqdm)[0]


@torch.inference_mode()
def run_transformers(args, processor, inputs):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
    ).cuda()
    model.eval()
    cuda_inputs = {
        key: value.cuda() if isinstance(value, torch.Tensor) else value
        for key, value in inputs.items()
    }
    generated = model.generate(
        **cuda_inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.temperature > 0,
        temperature=args.temperature if args.temperature > 0 else None,
    )
    completion = generated[:, inputs["input_ids"].shape[1]:]
    return {
        "text": processor.batch_decode(
            completion,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0],
        "token_ids": completion[0].detach().cpu().tolist(),
    }


def print_prompt_summary(inputs, prompt: MultiModalPrompt):
    image_tokens = int((inputs["mm_token_type_ids"] == 1).sum().item())
    print(f"input_tokens: {len(prompt.input_ids)}")
    print(f"image_tokens: {image_tokens}")
    print(f"image_grid_thw: {inputs['image_grid_thw'].tolist()}")
    print(f"pixel_values: {tuple(inputs['pixel_values'].shape)} {inputs['pixel_values'].dtype}")


def main():
    args = parse_args()
    processor, inputs, prompt, text = build_prompt(args.model, args.image, args.prompt)
    print_prompt_summary(inputs, prompt)
    print(f"chat_template: {text!r}")

    if args.engine in {"transformers", "both"}:
        output = run_transformers(args, processor, inputs)
        print("\n[transformers]")
        print(f"token_ids: {output['token_ids']}")
        print(f"text: {output['text']!r}")
        del output
        torch.cuda.empty_cache()

    if args.engine in {"nano", "both"}:
        output = run_nano(args, prompt)
        print("\n[nano-vllm]")
        print(f"token_ids: {output['token_ids']}")
        print(f"text: {output['text']!r}")


if __name__ == "__main__":
    main()
