import argparse
import asyncio
import base64
import binascii
import io
import json
import sys
import threading
import uuid
from pathlib import Path
from time import perf_counter, time
from typing import Literal

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, model_validator
from PIL import Image, UnidentifiedImageError
from transformers import AutoProcessor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nanovllm import (
    AsyncEngineClosedError,
    AsyncEngineOutput,
    AsyncEngineStreamEvent,
    AsyncLLMEngine,
    MultiModalPrompt,
    SamplingParams,
)
from nanovllm.attention import normalize_attention_backend_name


class GenerateRequest(BaseModel):
    prompt: str = Field(default="描述这张图片")
    image_path: str | None = None
    image_base64: str | None = None
    max_new_tokens: int = Field(default=32, ge=1)
    temperature: float = Field(default=0.0, ge=0.0)
    ignore_eos: bool = False
    profile: bool = False

    @model_validator(mode="after")
    def validate_image_source(self):
        if bool(self.image_path) == bool(self.image_base64):
            raise ValueError("provide exactly one of image_path or image_base64")
        return self


class TextContentPart(BaseModel):
    type: Literal["text"]
    text: str


class ChatImageURL(BaseModel):
    url: str
    detail: Literal["auto", "low", "high"] | None = None


class ImageContentPart(BaseModel):
    type: Literal["image_url"]
    image_url: ChatImageURL


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str | list[TextContentPart | ImageContentPart]


class StreamOptions(BaseModel):
    include_usage: bool = False


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage] = Field(min_length=1)
    max_tokens: int | None = Field(default=None, ge=1)
    max_completion_tokens: int | None = Field(default=None, ge=1)
    temperature: float = Field(default=0.0, ge=0.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    n: int = Field(default=1, ge=1, le=1)
    stream: bool = False
    stream_options: StreamOptions | None = None
    stop: str | list[str] | None = None
    ignore_eos: bool = False
    profile: bool = False

    @model_validator(mode="after")
    def validate_supported_options(self):
        if self.max_tokens is not None and self.max_completion_tokens is not None:
            raise ValueError("provide only one of max_tokens or max_completion_tokens")
        if self.top_p != 1.0:
            raise ValueError("top_p sampling is not supported yet; use top_p=1")
        if self.stop is not None:
            raise ValueError("custom stop strings are not supported yet")
        return self

    @property
    def resolved_max_tokens(self):
        return self.max_completion_tokens or self.max_tokens or 32


class ServerConfig(BaseModel):
    model: str
    served_model_name: str
    max_model_len: int
    max_num_seqs: int
    max_num_batched_tokens: int
    gpu_memory_utilization: float
    max_concurrent_requests: int
    request_timeout_seconds: float
    attention_backend: str = "flash_attn"
    vision_attention_backend: str = "flash_attn"


class ConcurrencyLimiter:
    def __init__(self, limit: int):
        self.limit = limit
        self._in_flight = 0
        self._rejected = 0
        self._lock = threading.Lock()

    def try_acquire(self):
        with self._lock:
            if self._in_flight >= self.limit:
                self._rejected += 1
                return False
            self._in_flight += 1
            return True

    def release(self):
        with self._lock:
            if self._in_flight <= 0:
                raise RuntimeError("concurrency limiter released without acquire")
            self._in_flight -= 1

    def stats(self):
        with self._lock:
            return {
                "limit": self.limit,
                "in_flight": self._in_flight,
                "available": self.limit - self._in_flight,
                "rejected": self._rejected,
            }


app = FastAPI(title="nano-vllm Qwen2.5-VL server")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Serve Qwen2.5-VL single-image inference with nano-vllm."
    )
    parser.add_argument("--model", default="/home/agua/models/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--served-model-name")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=4)
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
    parser.add_argument("--max-concurrent-requests", type=int, default=16)
    parser.add_argument("--request-timeout-seconds", type=float, default=300.0)
    return parser.parse_args()


def setup_engine(args):
    if args.max_concurrent_requests < 1:
        raise ValueError("--max-concurrent-requests must be at least 1")
    if args.request_timeout_seconds <= 0:
        raise ValueError("--request-timeout-seconds must be greater than 0")

    t0 = perf_counter()
    processor = AutoProcessor.from_pretrained(args.model)
    processor_load_ms = elapsed_ms(t0)

    attention_backend = normalize_attention_backend_name(args.attention_backend)
    vision_attention_backend = normalize_attention_backend_name(
        args.vision_attention_backend
    )
    engine = AsyncLLMEngine(
        args.model,
        enforce_eager=True,
        tensor_parallel_size=1,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        attention_backend=attention_backend,
        vision_attention_backend=vision_attention_backend,
    )

    served_model_name = args.served_model_name or Path(args.model).name
    app.state.processor = processor
    app.state.processor_lock = threading.Lock()
    app.state.engine = engine
    app.state.limiter = ConcurrencyLimiter(args.max_concurrent_requests)
    app.state.config = ServerConfig(
        model=args.model,
        served_model_name=served_model_name,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_concurrent_requests=args.max_concurrent_requests,
        request_timeout_seconds=args.request_timeout_seconds,
        attention_backend=attention_backend,
        vision_attention_backend=vision_attention_backend,
    )
    app.state.startup = {
        "processor_load_ms": processor_load_ms,
        "engine_load_ms": engine.startup_ms,
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": app.state.config.model,
        "served_model_name": app.state.config.served_model_name,
        "mode": "single-process-single-gpu-continuous-batching",
        "capabilities": {
            "native_generate": True,
            "native_sse": True,
            "openai_chat_completions": True,
            "openai_sse": True,
            "request_cancellation": True,
        },
        "limits": app.state.config.model_dump(),
        "concurrency": app.state.limiter.stats(),
        "startup": app.state.startup,
        "engine": app.state.engine.stats(),
    }


@app.post("/generate")
async def generate(request: GenerateRequest):
    request_id = str(uuid.uuid4())
    if not app.state.limiter.try_acquire():
        raise HTTPException(
            status_code=429,
            detail="server request concurrency limit reached",
            headers={"Retry-After": "1"},
        )

    try:
        return await asyncio.wait_for(
            run_native_generation(request_id, request),
            timeout=app.state.config.request_timeout_seconds,
        )
    except asyncio.TimeoutError as exc:
        app.state.engine.cancel(request_id)
        raise HTTPException(status_code=504, detail="request timed out") from exc
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="invalid image") from exc
    except AsyncEngineClosedError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    finally:
        app.state.limiter.release()


@app.post("/generate_stream")
async def generate_stream(http_request: Request, request: GenerateRequest):
    request_id = str(uuid.uuid4())
    if not app.state.limiter.try_acquire():
        raise HTTPException(
            status_code=429,
            detail="server request concurrency limit reached",
            headers={"Retry-After": "1"},
        )

    request_started_at = perf_counter()
    try:
        prompt, input_summary, preprocess_ms = await preprocess_native_with_timeout(
            request,
            request_started_at,
        )
        validate_prompt_budget(prompt, request.max_new_tokens)
    except asyncio.TimeoutError as exc:
        app.state.limiter.release()
        raise HTTPException(status_code=504, detail="request timed out") from exc
    except (ValueError, FileNotFoundError) as exc:
        app.state.limiter.release()
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except UnidentifiedImageError as exc:
        app.state.limiter.release()
        raise HTTPException(status_code=400, detail="invalid image") from exc

    sampling_params = SamplingParams(
        temperature=request.temperature,
        max_tokens=request.max_new_tokens,
        ignore_eos=request.ignore_eos,
    )
    body = native_event_stream(
        http_request,
        request_id,
        prompt,
        sampling_params,
        input_summary,
        preprocess_ms,
        request.profile,
        request_started_at,
    )
    return sse_response(body)


@app.post("/v1/chat/completions")
async def chat_completions(
    http_request: Request,
    request: ChatCompletionRequest,
):
    model_error = validate_requested_model(request.model)
    if model_error is not None:
        return model_error
    if not app.state.limiter.try_acquire():
        return openai_error_response(
            429,
            "server request concurrency limit reached",
            "rate_limit_error",
            "concurrency_limit",
            headers={"Retry-After": "1"},
        )

    request_id = str(uuid.uuid4())
    request_started_at = perf_counter()
    try:
        prompt, input_summary, preprocess_ms = await preprocess_chat_with_timeout(
            request,
            request_started_at,
        )
        validate_prompt_budget(prompt, request.resolved_max_tokens)
    except asyncio.TimeoutError:
        app.state.limiter.release()
        return openai_error_response(
            504,
            "request timed out during preprocessing",
            "server_error",
            "request_timeout",
        )
    except (ValueError, FileNotFoundError) as exc:
        app.state.limiter.release()
        return openai_error_response(
            400,
            str(exc),
            "invalid_request_error",
            "invalid_request",
        )
    except UnidentifiedImageError:
        app.state.limiter.release()
        return openai_error_response(
            400,
            "invalid image",
            "invalid_request_error",
            "invalid_image",
        )

    sampling_params = SamplingParams(
        temperature=request.temperature,
        max_tokens=request.resolved_max_tokens,
        ignore_eos=request.ignore_eos,
    )
    completion_id = f"chatcmpl-{request_id.replace('-', '')}"
    created = int(time())

    if request.stream:
        body = openai_event_stream(
            http_request,
            request_id,
            completion_id,
            created,
            request.model,
            prompt,
            sampling_params,
            input_summary,
            preprocess_ms,
            request.profile,
            request.stream_options.include_usage
            if request.stream_options is not None
            else False,
            request_started_at,
        )
        return sse_response(body)

    try:
        remaining = remaining_timeout(request_started_at)
        engine_result = await asyncio.wait_for(
            app.state.engine.generate(
                request_id,
                prompt,
                sampling_params,
                profile=request.profile,
                request_started_at=request_started_at,
            ),
            timeout=remaining,
        )
        total_ms = elapsed_ms(request_started_at)
        return build_openai_response(
            completion_id,
            created,
            request.model,
            engine_result,
            input_summary,
            preprocess_ms,
            total_ms,
        )
    except asyncio.TimeoutError:
        app.state.engine.cancel(request_id)
        return openai_error_response(
            504,
            "request timed out",
            "server_error",
            "request_timeout",
        )
    except AsyncEngineClosedError as exc:
        return openai_error_response(
            503,
            str(exc),
            "server_error",
            "engine_closed",
        )
    except ValueError as exc:
        return openai_error_response(
            400,
            str(exc),
            "invalid_request_error",
            "invalid_request",
        )
    except Exception as exc:
        return openai_error_response(
            500,
            str(exc),
            "server_error",
            "generation_error",
        )
    finally:
        app.state.limiter.release()


async def run_native_generation(request_id: str, request: GenerateRequest):
    request_started_at = perf_counter()
    t_preprocess = perf_counter()
    prompt, input_summary = await asyncio.to_thread(
        preprocess_native_request,
        request,
    )
    preprocess_ms = elapsed_ms(t_preprocess)
    validate_prompt_budget(prompt, request.max_new_tokens)

    sampling_params = SamplingParams(
        temperature=request.temperature,
        max_tokens=request.max_new_tokens,
        ignore_eos=request.ignore_eos,
    )
    engine_result = await app.state.engine.generate(
        request_id,
        prompt,
        sampling_params,
        profile=request.profile,
        request_started_at=request_started_at,
    )
    return build_native_response(
        request_id,
        engine_result,
        input_summary,
        preprocess_ms,
        elapsed_ms(request_started_at),
    )


async def preprocess_native_with_timeout(
    request: GenerateRequest,
    request_started_at: float,
):
    t_preprocess = perf_counter()
    prompt, input_summary = await asyncio.wait_for(
        asyncio.to_thread(preprocess_native_request, request),
        timeout=remaining_timeout(request_started_at),
    )
    return prompt, input_summary, elapsed_ms(t_preprocess)


async def preprocess_chat_with_timeout(
    request: ChatCompletionRequest,
    request_started_at: float,
):
    t_preprocess = perf_counter()
    prompt, input_summary = await asyncio.wait_for(
        asyncio.to_thread(preprocess_chat_request, request),
        timeout=remaining_timeout(request_started_at),
    )
    return prompt, input_summary, elapsed_ms(t_preprocess)


async def native_event_stream(
    http_request: Request,
    request_id: str,
    prompt: MultiModalPrompt,
    sampling_params: SamplingParams,
    input_summary: dict,
    preprocess_ms: float,
    profile: bool,
    request_started_at: float,
):
    engine_stream = None
    first_token_sent = False
    try:
        yield encode_sse(
            {
                "request_id": request_id,
                "model": app.state.config.model,
                "usage": input_summary,
            },
            event="metadata",
        )
        engine_stream = app.state.engine.stream_generate(
            request_id,
            prompt,
            sampling_params,
            profile=profile,
            request_started_at=request_started_at,
        )
        while True:
            if await http_request.is_disconnected():
                app.state.engine.cancel(request_id)
                return
            event = await next_engine_event(engine_stream, request_started_at)
            if event.event_type == "token":
                payload = {
                    "request_id": request_id,
                    "index": event.token_index,
                    "token_id": event.token_id,
                    "delta": event.delta_text,
                    "text": event.text,
                }
                if not first_token_sent:
                    payload["client_visible_ttft_ms"] = elapsed_ms(request_started_at)
                    first_token_sent = True
                yield encode_sse(payload, event="token")
            elif event.event_type == "done":
                total_ms = elapsed_ms(request_started_at)
                response = build_native_response(
                    request_id,
                    event.output,
                    input_summary,
                    preprocess_ms,
                    total_ms,
                )
                yield encode_sse(response, event="done")
                return
            elif event.event_type == "error":
                yield encode_sse(
                    stream_error_payload(event.error),
                    event="error",
                )
                return
    except asyncio.TimeoutError:
        app.state.engine.cancel(request_id)
        yield encode_sse(
            stream_error_payload(
                RuntimeError("request timed out"),
                code="request_timeout",
            ),
            event="error",
        )
    except asyncio.CancelledError:
        app.state.engine.cancel(request_id)
        raise
    except Exception as exc:
        yield encode_sse(stream_error_payload(exc), event="error")
    finally:
        if engine_stream is not None:
            await engine_stream.aclose()
        app.state.engine.cancel(request_id)
        app.state.limiter.release()


async def openai_event_stream(
    http_request: Request,
    request_id: str,
    completion_id: str,
    created: int,
    model: str,
    prompt: MultiModalPrompt,
    sampling_params: SamplingParams,
    input_summary: dict,
    preprocess_ms: float,
    profile: bool,
    include_usage: bool,
    request_started_at: float,
):
    engine_stream = None
    try:
        yield encode_sse(
            openai_chunk(
                completion_id,
                created,
                model,
                delta={"role": "assistant"},
            )
        )
        engine_stream = app.state.engine.stream_generate(
            request_id,
            prompt,
            sampling_params,
            profile=profile,
            request_started_at=request_started_at,
        )
        while True:
            if await http_request.is_disconnected():
                app.state.engine.cancel(request_id)
                return
            event = await next_engine_event(engine_stream, request_started_at)
            if event.event_type == "token":
                yield encode_sse(
                    openai_chunk(
                        completion_id,
                        created,
                        model,
                        delta={"content": event.delta_text},
                    )
                )
            elif event.event_type == "done":
                yield encode_sse(
                    openai_chunk(
                        completion_id,
                        created,
                        model,
                        delta={},
                        finish_reason=event.output.finish_reason,
                    )
                )
                if include_usage:
                    yield encode_sse(
                        {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [],
                            "usage": build_usage(input_summary, event.output),
                        }
                    )
                yield "data: [DONE]\n\n"
                return
            elif event.event_type == "error":
                yield encode_sse(
                    {
                        "error": openai_error_body(
                            str(event.error),
                            "server_error",
                            "generation_error",
                        )
                    }
                )
                yield "data: [DONE]\n\n"
                return
    except asyncio.TimeoutError:
        app.state.engine.cancel(request_id)
        yield encode_sse(
            {
                "error": openai_error_body(
                    "request timed out",
                    "server_error",
                    "request_timeout",
                )
            }
        )
        yield "data: [DONE]\n\n"
    except asyncio.CancelledError:
        app.state.engine.cancel(request_id)
        raise
    except Exception as exc:
        yield encode_sse(
            {
                "error": openai_error_body(
                    str(exc),
                    "server_error",
                    "generation_error",
                )
            }
        )
        yield "data: [DONE]\n\n"
    finally:
        if engine_stream is not None:
            await engine_stream.aclose()
        app.state.engine.cancel(request_id)
        app.state.limiter.release()


async def next_engine_event(engine_stream, request_started_at: float):
    return await asyncio.wait_for(
        anext(engine_stream),
        timeout=remaining_timeout(request_started_at),
    )


def preprocess_native_request(request: GenerateRequest):
    image = load_native_image(request)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": request.prompt},
            ],
        }
    ]
    with app.state.processor_lock:
        return build_multimodal_prompt(
            app.state.processor,
            messages,
            [image],
        )


def preprocess_chat_request(request: ChatCompletionRequest):
    messages, images = convert_openai_messages(request.messages)
    with app.state.processor_lock:
        return build_multimodal_prompt(
            app.state.processor,
            messages,
            images,
        )


def convert_openai_messages(messages: list[ChatMessage]):
    processor_messages = []
    images = []
    for message in messages:
        if isinstance(message.content, str):
            content = [{"type": "text", "text": message.content}]
        else:
            content = []
            for part in message.content:
                if isinstance(part, TextContentPart):
                    content.append({"type": "text", "text": part.text})
                else:
                    image = load_openai_image(part.image_url.url)
                    images.append(image)
                    content.append({"type": "image", "image": image})
        processor_messages.append(
            {
                "role": message.role,
                "content": content,
            }
        )
    if len(images) != 1:
        raise ValueError(
            f"this server requires exactly one image, received {len(images)}"
        )
    return processor_messages, images


def build_multimodal_prompt(processor, messages: list[dict], images: list[Image.Image]):
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = processor(text=[text], images=images, return_tensors="pt")
    prompt = MultiModalPrompt.from_processor(inputs)
    summary = {
        "input_tokens": len(prompt.input_ids),
        "image_tokens": int((inputs["mm_token_type_ids"] == 1).sum().item()),
        "image_grid_thw": inputs["image_grid_thw"].tolist(),
    }
    return prompt, summary


def build_native_response(
    request_id: str,
    engine_result: AsyncEngineOutput,
    input_summary: dict,
    preprocess_ms: float,
    total_ms: float,
):
    output = engine_result.output
    completion_tokens = len(output["token_ids"])
    response = {
        "request_id": request_id,
        "model": app.state.config.model,
        "text": output["text"],
        "token_ids": output["token_ids"],
        "finish_reason": engine_result.finish_reason,
        "usage": {
            **input_summary,
            "completion_tokens": completion_tokens,
            "total_tokens": input_summary["input_tokens"] + completion_tokens,
        },
        "latency_ms": {
            "preprocess": preprocess_ms,
            "queue_wait": engine_result.queue_wait_ms,
            "generation": engine_result.generation_ms,
            "total": total_ms,
        },
        "engine": {
            "queue_depth_at_submit": engine_result.queue_depth_at_submit,
            "requests_ahead_at_submit": engine_result.requests_ahead_at_submit,
        },
    }
    if engine_result.profile is not None:
        engine_result.profile["server_tokens_per_s"] = safe_div(
            completion_tokens,
            total_ms / 1000,
        )
        response["profile"] = engine_result.profile
    return response


def build_openai_response(
    completion_id: str,
    created: int,
    model: str,
    engine_result: AsyncEngineOutput,
    input_summary: dict,
    preprocess_ms: float,
    total_ms: float,
):
    response = {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": engine_result.output["text"],
                },
                "finish_reason": engine_result.finish_reason,
            }
        ],
        "usage": build_usage(input_summary, engine_result),
        "nano_vllm": {
            "token_ids": engine_result.output["token_ids"],
            "image_tokens": input_summary["image_tokens"],
            "image_grid_thw": input_summary["image_grid_thw"],
            "latency_ms": {
                "preprocess": preprocess_ms,
                "queue_wait": engine_result.queue_wait_ms,
                "generation": engine_result.generation_ms,
                "total": total_ms,
            },
        },
    }
    if engine_result.profile is not None:
        response["nano_vllm"]["profile"] = engine_result.profile
    return response


def build_usage(input_summary: dict, engine_result: AsyncEngineOutput):
    completion_tokens = len(engine_result.output["token_ids"])
    return {
        "prompt_tokens": input_summary["input_tokens"],
        "completion_tokens": completion_tokens,
        "total_tokens": input_summary["input_tokens"] + completion_tokens,
    }


def openai_chunk(
    completion_id: str,
    created: int,
    model: str,
    *,
    delta: dict,
    finish_reason: str | None = None,
):
    return {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }


def validate_prompt_budget(prompt: MultiModalPrompt, max_new_tokens: int):
    config = app.state.config
    if len(prompt.input_ids) > config.max_num_batched_tokens:
        raise ValueError(
            f"prompt has {len(prompt.input_ids)} tokens, but max_num_batched_tokens="
            f"{config.max_num_batched_tokens}; multimodal prefill is not chunked yet"
        )
    if len(prompt.input_ids) + max_new_tokens > config.max_model_len:
        raise ValueError(
            f"prompt + generation needs {len(prompt.input_ids) + max_new_tokens} "
            f"tokens, but max_model_len={config.max_model_len}"
        )


def validate_requested_model(model: str):
    served_model_name = app.state.config.served_model_name
    if model == served_model_name:
        return None
    return openai_error_response(
        404,
        f"model {model!r} is not served; use {served_model_name!r}",
        "invalid_request_error",
        "model_not_found",
    )


def load_native_image(request: GenerateRequest):
    if request.image_path:
        path = Path(request.image_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"image_path does not exist: {path}")
        return Image.open(path).convert("RGB")
    return image_from_bytes(decode_base64_image(request.image_base64 or ""))


def load_openai_image(value: str):
    if not value.startswith("data:"):
        raise ValueError(
            "image_url.url currently supports data URLs only "
            "(data:image/...;base64,...)"
        )
    return image_from_bytes(decode_base64_image(value))


def image_from_bytes(value: bytes):
    return Image.open(io.BytesIO(value)).convert("RGB")


def decode_base64_image(value: str):
    if "," in value and value.split(",", 1)[0].startswith("data:"):
        value = value.split(",", 1)[1]
    try:
        return base64.b64decode(value, validate=True)
    except binascii.Error as exc:
        raise ValueError("image data is not valid base64") from exc


def remaining_timeout(request_started_at: float):
    remaining = (
        app.state.config.request_timeout_seconds
        - (perf_counter() - request_started_at)
    )
    if remaining <= 0:
        raise asyncio.TimeoutError
    return remaining


def sse_response(body):
    return StreamingResponse(
        body,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


def encode_sse(payload: dict, *, event: str | None = None):
    prefix = f"event: {event}\n" if event is not None else ""
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return f"{prefix}data: {data}\n\n"


def stream_error_payload(exc: BaseException | None, code: str = "generation_error"):
    return {
        "error": {
            "message": str(exc or "unknown generation error"),
            "code": code,
        }
    }


def openai_error_body(message: str, error_type: str, code: str):
    return {
        "message": message,
        "type": error_type,
        "param": None,
        "code": code,
    }


def openai_error_response(
    status_code: int,
    message: str,
    error_type: str,
    code: str,
    headers: dict | None = None,
):
    return JSONResponse(
        status_code=status_code,
        content={
            "error": openai_error_body(message, error_type, code),
        },
        headers=headers,
    )


def elapsed_ms(start: float):
    return (perf_counter() - start) * 1000


def safe_div(numerator: float, denominator: float):
    if denominator == 0:
        return None
    return numerator / denominator


def main():
    args = parse_args()
    setup_engine(args)
    try:
        uvicorn.run(app, host=args.host, port=args.port, workers=1)
    finally:
        app.state.engine.shutdown()


if __name__ == "__main__":
    main()
