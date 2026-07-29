import atexit
import socket
from dataclasses import dataclass, fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.multimodal import MultiModalPrompt, compute_qwen2_5_vl_mrope_positions
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


@dataclass(slots=True)
class ScheduledSequenceMetadata:
    seq_id: int
    num_tokens: int
    produced_token: bool
    token_id: int | None = None
    finish_reason: str | None = None


@dataclass(slots=True)
class EngineStepOutput:
    outputs: list[tuple[int, list[int]]]
    scheduled: list[ScheduledSequenceMetadata]
    is_prefill: bool
    num_tokens: int


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        if config.distributed_init_port == 0:
            config.distributed_init_port = find_free_port()
        self.config = config
        Sequence.block_size = config.kvcache_block_size
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        self._exited = False
        atexit.register(self.exit)

    def exit(self):
        if self._exited:
            return
        self._exited = True
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(
        self,
        prompt: str | list[int] | MultiModalPrompt,
        sampling_params: SamplingParams,
    ):
        multimodal = None
        mrope_positions = None
        mrope_position_delta = 0
        if isinstance(prompt, MultiModalPrompt):
            if self.config.tensor_parallel_size != 1:
                raise ValueError("multimodal inference currently requires tensor_parallel_size=1")
            if not hasattr(self.config.hf_config, "vision_config"):
                raise ValueError("multimodal prompts require a vision-language model")
            multimodal = prompt
            prompt = prompt.input_ids
            mrope_positions, mrope_position_delta = compute_qwen2_5_vl_mrope_positions(
                prompt,
                multimodal.mm_token_type_ids,
                multimodal.image_grid_thw,
                self.config.hf_config.vision_config.spatial_merge_size,
                multimodal.attention_mask,
            )
            if len(prompt) > self.config.max_num_batched_tokens:
                raise ValueError(
                    "multimodal prompt length exceeds max_num_batched_tokens; "
                    "increase max_num_batched_tokens because chunked multimodal "
                    "prefill is not supported yet"
                )
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(
            prompt,
            sampling_params,
            multimodal=multimodal,
            mrope_positions=mrope_positions,
            mrope_position_delta=mrope_position_delta,
        )
        self.scheduler.add(seq)
        return seq.seq_id

    def step(self):
        result = self.step_with_metadata()
        return result.outputs, result.num_tokens

    def step_with_metadata(self):
        seqs, is_prefill = self.scheduler.schedule()
        scheduled_token_counts = [seq.num_scheduled_tokens for seq in seqs]
        completion_counts = [seq.num_completion_tokens for seq in seqs]
        num_tokens = (
            sum(scheduled_token_counts)
            if is_prefill
            else -len(seqs)
        )
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids, is_prefill)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        scheduled = [
            ScheduledSequenceMetadata(
                seq_id=seq.seq_id,
                num_tokens=num_scheduled_tokens,
                produced_token=seq.num_completion_tokens > completion_count,
                token_id=(
                    token_id
                    if seq.num_completion_tokens > completion_count
                    else None
                ),
                finish_reason=seq.finish_reason,
            )
            for seq, token_id, num_scheduled_tokens, completion_count in zip(
                seqs,
                token_ids,
                scheduled_token_counts,
                completion_counts,
            )
        ]
        return EngineStepOutput(
            outputs=outputs,
            scheduled=scheduled,
            is_prefill=is_prefill,
            num_tokens=num_tokens,
        )

    def abort_request(self, seq_id: int):
        return self.scheduler.abort(seq_id)

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]] | list[MultiModalPrompt],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            if num_tokens > 0:
                prefill_throughput = num_tokens / (perf_counter() - t)
            else:
                decode_throughput = -num_tokens / (perf_counter() - t)
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                pbar.update(1)
        pbar.close()
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        return outputs


def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]
