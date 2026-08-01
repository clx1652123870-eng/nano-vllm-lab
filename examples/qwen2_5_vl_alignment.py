import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoConfig, AutoProcessor, Qwen2_5_VLForConditionalGeneration
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nanovllm import LLM, MultiModalPrompt, SamplingParams
from nanovllm.multimodal import compute_qwen2_5_vl_mrope_positions


ALIGN_MARKER = "NANO_VLLM_ALIGN_JSON="


def parse_args():
    parser = argparse.ArgumentParser(
        description="Check Qwen2.5-VL single-image alignment between Transformers and nano-vllm."
    )
    parser.add_argument("--mode", choices=["check", "generate"], default="check")
    parser.add_argument("--engine", choices=["nano", "transformers"], default="nano")
    parser.add_argument("--model", default="/home/agua/models/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--image", default="assets/dog.png")
    parser.add_argument("--prompt", default="描述这张图片")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-batched-tokens", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.72)
    parser.add_argument("--attention-backend", default="flash_attn")
    parser.add_argument(
        "--vision-attention-backend",
        choices=[
            "flash_attn",
            "torch_sdpa",
            "torch_math",
            "cudnn_sdpa",
            "triton",
            "hybrid",
        ],
        default="flash_attn",
    )
    parser.add_argument("--python", default=sys.executable)
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


def compare_prompt_inputs(inputs, prompt: MultiModalPrompt) -> bool:
    return prompt.input_ids == inputs["input_ids"][0].tolist()


def compare_mrope_positions(model_path: str, inputs, prompt: MultiModalPrompt):
    config = AutoConfig.from_pretrained(model_path)
    ours, ours_delta = compute_qwen2_5_vl_mrope_positions(
        prompt.input_ids,
        prompt.mm_token_type_ids,
        prompt.image_grid_thw,
        config.vision_config.spatial_merge_size,
        prompt.attention_mask,
    )

    # Call the Transformers helper without allocating model weights.
    dummy_model = object.__new__(Qwen2_5_VLModel)
    dummy_model.config = config
    hf_positions, hf_delta = Qwen2_5_VLModel.get_rope_index(
        dummy_model,
        inputs["input_ids"],
        inputs["mm_token_type_ids"],
        image_grid_thw=inputs["image_grid_thw"],
        attention_mask=inputs.get("attention_mask"),
    )
    hf_positions = hf_positions[:, 0].cpu()
    hf_delta = int(hf_delta[0, 0].item())
    return {
        "positions_equal": torch.equal(ours, hf_positions),
        "delta_equal": ours_delta == hf_delta,
        "ours_shape": tuple(ours.shape),
        "hf_shape": tuple(hf_positions.shape),
        "ours_delta": ours_delta,
        "hf_delta": hf_delta,
    }


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
    return llm.generate([prompt], sampling_params, use_tqdm=False)[0]


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
        "token_ids": completion[0].detach().cpu().tolist(),
        "text": processor.batch_decode(
            completion,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0],
    }


def run_generate_mode(args):
    processor, inputs, prompt, _ = build_prompt(args.model, args.image, args.prompt)
    if args.engine == "nano":
        output = run_nano(args, prompt)
    else:
        output = run_transformers(args, processor, inputs)
    print(ALIGN_MARKER + json.dumps(output, ensure_ascii=False))


def run_child(args, engine: str):
    command = [
        args.python,
        str(Path(__file__).resolve()),
        "--mode",
        "generate",
        "--engine",
        engine,
        "--model",
        args.model,
        "--image",
        args.image,
        "--prompt",
        args.prompt,
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--temperature",
        str(args.temperature),
        "--max-model-len",
        str(args.max_model_len),
        "--max-num-batched-tokens",
        str(args.max_num_batched_tokens),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--attention-backend",
        args.attention_backend,
        "--vision-attention-backend",
        args.vision_attention_backend,
    ]
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        print(result.stdout)
        raise RuntimeError(f"{engine} generation failed with code {result.returncode}")
    for line in reversed(result.stdout.splitlines()):
        if line.startswith(ALIGN_MARKER):
            return json.loads(line[len(ALIGN_MARKER):])
    print(result.stdout)
    raise RuntimeError(f"{engine} generation did not emit {ALIGN_MARKER}")


def print_check(name: str, passed: bool):
    print(f"{name}: {'PASS' if passed else 'FAIL'}")


def run_check_mode(args):
    processor, inputs, prompt, chat_text = build_prompt(args.model, args.image, args.prompt)
    image_tokens = int((inputs["mm_token_type_ids"] == 1).sum().item())
    print("[prompt]")
    print(f"chat_template: {chat_text!r}")
    print(f"input_tokens: {len(prompt.input_ids)}")
    print(f"image_tokens: {image_tokens}")
    print(f"image_grid_thw: {inputs['image_grid_thw'].tolist()}")
    print(f"pixel_values: {tuple(inputs['pixel_values'].shape)} {inputs['pixel_values'].dtype}")

    input_ids_equal = compare_prompt_inputs(inputs, prompt)
    mrope = compare_mrope_positions(args.model, inputs, prompt)
    print("\n[metadata]")
    print_check("input_ids", input_ids_equal)
    print_check("mrope_positions", mrope["positions_equal"])
    print_check("mrope_delta", mrope["delta_equal"])
    print(f"mrope_shape: ours={mrope['ours_shape']} hf={mrope['hf_shape']}")
    print(f"mrope_delta: ours={mrope['ours_delta']} hf={mrope['hf_delta']}")

    print("\n[generation]")
    print("running transformers reference...")
    hf_output = run_child(args, "transformers")
    print("running nano-vllm...")
    nano_output = run_child(args, "nano")
    token_ids_equal = nano_output["token_ids"] == hf_output["token_ids"]
    print_check("greedy_token_ids", token_ids_equal)
    print(f"transformers_token_ids: {hf_output['token_ids']}")
    print(f"nano_token_ids:         {nano_output['token_ids']}")
    print(f"transformers_text: {hf_output['text']!r}")
    print(f"nano_text:         {nano_output['text']!r}")

    if not (
        input_ids_equal
        and mrope["positions_equal"]
        and mrope["delta_equal"]
        and token_ids_equal
    ):
        raise SystemExit(1)


def main():
    args = parse_args()
    if args.mode == "generate":
        run_generate_mode(args)
    else:
        run_check_mode(args)


if __name__ == "__main__":
    main()
