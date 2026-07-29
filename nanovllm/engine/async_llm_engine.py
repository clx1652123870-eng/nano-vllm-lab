import asyncio
import threading
from collections.abc import AsyncIterator, Callable
from concurrent.futures import Future
from dataclasses import dataclass, field
from queue import Empty, Queue
from time import perf_counter
from typing import Any

import torch

from nanovllm.llm import LLM
from nanovllm.multimodal import MultiModalPrompt
from nanovllm.sampling_params import SamplingParams


class AsyncEngineClosedError(RuntimeError):
    pass


class AsyncEngineRequestCancelledError(RuntimeError):
    pass


@dataclass(slots=True)
class AsyncEngineOutput:
    output: dict[str, Any]
    finish_reason: str
    queue_wait_ms: float
    generation_ms: float
    queue_depth_at_submit: int
    requests_ahead_at_submit: int
    profile: dict[str, Any] | None


@dataclass(slots=True)
class AsyncEngineStreamEvent:
    event_type: str
    request_id: str
    token_id: int | None = None
    token_index: int | None = None
    delta_text: str = ""
    text: str = ""
    output: AsyncEngineOutput | None = None
    error: BaseException | None = None


@dataclass(slots=True)
class _EngineRequest:
    request_id: str
    prompt: str | list[int] | MultiModalPrompt
    sampling_params: SamplingParams
    profile: bool
    request_started_at: float
    submitted_at: float
    queue_depth_at_submit: int
    requests_ahead_at_submit: int
    future: Future[AsyncEngineOutput]
    event_sink: Callable[[AsyncEngineStreamEvent], None] | None = None


@dataclass(slots=True)
class _ActiveRequest:
    request: _EngineRequest
    seq_id: int
    queue_wait_ms: float | None = None
    generation_started_at: float | None = None
    request_ttft_ms: float | None = None
    prefill_latencies_ms: list[float] = field(default_factory=list)
    decode_latencies_ms: list[float] = field(default_factory=list)
    prefill_batch_sizes: list[int] = field(default_factory=list)
    decode_batch_sizes: list[int] = field(default_factory=list)
    prefill_tokens: int = 0
    decode_tokens: int = 0
    generated_token_ids: list[int] = field(default_factory=list)
    decoded_text: str = ""
    finish_reason: str | None = None


class AsyncLLMEngine:
    """Own an LLM on one engine thread and expose an asyncio request interface."""

    def __init__(self, model: str, **engine_kwargs):
        self.model = model
        self.engine_kwargs = engine_kwargs
        self._request_queue = Queue()
        self._stop = object()
        self._ready = threading.Event()
        self._state_lock = threading.Lock()
        self._startup_error: BaseException | None = None
        self._startup_ms: float | None = None
        self._accepting_requests = True
        self._active_request_ids: set[str] = set()
        self._outstanding_request_ids: set[str] = set()
        self._cancel_requested_ids: set[str] = set()
        self._submitted_requests = 0
        self._completed_requests = 0
        self._failed_requests = 0
        self._cancelled_requests = 0
        self._outstanding_requests = 0
        self._total_engine_steps = 0
        self._prefill_engine_steps = 0
        self._decode_engine_steps = 0
        self._last_batch_size = 0
        self._max_batch_size = 0
        self._max_decode_batch_size = 0
        self._thread = threading.Thread(
            target=self._engine_loop,
            name="nanovllm-engine",
            daemon=True,
        )
        self._thread.start()
        self._ready.wait()
        if self._startup_error is not None:
            with self._state_lock:
                self._accepting_requests = False
            raise RuntimeError("failed to initialize AsyncLLMEngine") from self._startup_error

    @property
    def startup_ms(self):
        return self._startup_ms

    async def generate(
        self,
        request_id: str,
        prompt: str | list[int] | MultiModalPrompt,
        sampling_params: SamplingParams,
        *,
        profile: bool = False,
        request_started_at: float | None = None,
    ) -> AsyncEngineOutput:
        future = self._submit(
            request_id,
            prompt,
            sampling_params,
            profile=profile,
            request_started_at=request_started_at,
        )
        try:
            return await asyncio.wrap_future(future)
        except asyncio.CancelledError:
            self.cancel(request_id)
            raise

    async def stream_generate(
        self,
        request_id: str,
        prompt: str | list[int] | MultiModalPrompt,
        sampling_params: SamplingParams,
        *,
        profile: bool = False,
        request_started_at: float | None = None,
    ) -> AsyncIterator[AsyncEngineStreamEvent]:
        loop = asyncio.get_running_loop()
        event_queue: asyncio.Queue[AsyncEngineStreamEvent] = asyncio.Queue()

        def event_sink(event: AsyncEngineStreamEvent):
            if loop.is_closed():
                return
            try:
                loop.call_soon_threadsafe(event_queue.put_nowait, event)
            except RuntimeError:
                pass

        future = self._submit(
            request_id,
            prompt,
            sampling_params,
            profile=profile,
            request_started_at=request_started_at,
            event_sink=event_sink,
        )
        future.add_done_callback(consume_future_exception)
        try:
            while True:
                event = await event_queue.get()
                yield event
                if event.event_type in ("done", "error"):
                    break
        finally:
            if not future.done():
                self.cancel(request_id)

    def cancel(self, request_id: str) -> bool:
        with self._state_lock:
            if request_id not in self._outstanding_request_ids:
                return False
            self._cancel_requested_ids.add(request_id)
            return True

    def _submit(
        self,
        request_id: str,
        prompt: str | list[int] | MultiModalPrompt,
        sampling_params: SamplingParams,
        *,
        profile: bool,
        request_started_at: float | None,
        event_sink: Callable[[AsyncEngineStreamEvent], None] | None = None,
    ) -> Future[AsyncEngineOutput]:
        future: Future[AsyncEngineOutput] = Future()
        submitted_at = perf_counter()
        if request_started_at is None:
            request_started_at = submitted_at

        with self._state_lock:
            if not self._accepting_requests:
                raise AsyncEngineClosedError("AsyncLLMEngine is shutting down")
            if request_id in self._outstanding_request_ids:
                raise ValueError(f"duplicate request_id: {request_id}")
            queue_depth = self._request_queue.qsize()
            requests_ahead = self._outstanding_requests
            request = _EngineRequest(
                request_id=request_id,
                prompt=prompt,
                sampling_params=sampling_params,
                profile=profile,
                request_started_at=request_started_at,
                submitted_at=submitted_at,
                queue_depth_at_submit=queue_depth,
                requests_ahead_at_submit=requests_ahead,
                future=future,
                event_sink=event_sink,
            )
            self._submitted_requests += 1
            self._outstanding_requests += 1
            self._outstanding_request_ids.add(request_id)
            self._request_queue.put(request)

        return future

    def stats(self):
        with self._state_lock:
            return {
                "accepting_requests": self._accepting_requests,
                "engine_thread_alive": self._thread.is_alive(),
                "active_request_ids": sorted(self._active_request_ids),
                "active_request_count": len(self._active_request_ids),
                "cancel_requested_count": len(self._cancel_requested_ids),
                "queue_depth": self._request_queue.qsize(),
                "submitted_requests": self._submitted_requests,
                "completed_requests": self._completed_requests,
                "failed_requests": self._failed_requests,
                "cancelled_requests": self._cancelled_requests,
                "outstanding_requests": self._outstanding_requests,
                "total_engine_steps": self._total_engine_steps,
                "prefill_engine_steps": self._prefill_engine_steps,
                "decode_engine_steps": self._decode_engine_steps,
                "last_batch_size": self._last_batch_size,
                "max_batch_size": self._max_batch_size,
                "max_decode_batch_size": self._max_decode_batch_size,
            }

    def shutdown(self, timeout: float | None = None):
        with self._state_lock:
            if self._accepting_requests:
                self._accepting_requests = False
                self._request_queue.put(self._stop)
        self._thread.join(timeout)
        if self._thread.is_alive():
            raise TimeoutError("AsyncLLMEngine did not stop before the timeout")

    def _engine_loop(self):
        llm = None
        try:
            t0 = perf_counter()
            llm = LLM(self.model, **self.engine_kwargs)
            sync_cuda()
            self._startup_ms = elapsed_ms(t0)
        except BaseException as exc:
            self._startup_error = exc
            self._ready.set()
            return

        self._ready.set()
        active: dict[int, _ActiveRequest] = {}
        stopping = False
        profile_window_initialized = False
        capacity = llm.config.max_num_seqs
        try:
            while active or not stopping:
                self._cancel_active_requests(llm, active)

                if not active and not stopping:
                    request = self._request_queue.get()
                    if request is self._stop:
                        stopping = True
                    else:
                        self._admit_request(llm, request, active)

                while not stopping and len(active) < capacity:
                    try:
                        request = self._request_queue.get_nowait()
                    except Empty:
                        break
                    if request is self._stop:
                        stopping = True
                        break
                    self._admit_request(llm, request, active)

                self._cancel_active_requests(llm, active)
                if not active:
                    profile_window_initialized = False
                    continue

                has_profile_request = any(
                    state.request.profile for state in active.values()
                )
                if has_profile_request and not profile_window_initialized:
                    torch.cuda.reset_peak_memory_stats()
                    profile_window_initialized = True

                if has_profile_request:
                    sync_cuda()
                step_started_at = perf_counter()
                try:
                    step_output = llm.step_with_metadata()
                    if has_profile_request:
                        sync_cuda()
                    step_finished_at = perf_counter()
                    step_ms = (step_finished_at - step_started_at) * 1000
                    self._record_step_stats(
                        step_output.is_prefill,
                        len(step_output.scheduled),
                    )
                except Exception as exc:
                    self._fail_active_requests(active, exc)
                    with self._state_lock:
                        self._accepting_requests = False
                    self._fail_queued_requests(exc)
                    stopping = True
                    continue

                self._cancel_active_requests(llm, active)
                batch_size = len(step_output.scheduled)
                for scheduled in step_output.scheduled:
                    state = active.get(scheduled.seq_id)
                    if state is None:
                        continue
                    if state.generation_started_at is None:
                        state.generation_started_at = step_started_at
                        state.queue_wait_ms = (
                            step_started_at - state.request.submitted_at
                        ) * 1000
                    if scheduled.produced_token and state.request_ttft_ms is None:
                        state.request_ttft_ms = (
                            step_finished_at - state.request.request_started_at
                        ) * 1000
                    if scheduled.produced_token:
                        self._emit_token(
                            llm,
                            state,
                            scheduled.token_id,
                        )
                    if scheduled.finish_reason is not None:
                        state.finish_reason = scheduled.finish_reason
                    if state.request.profile:
                        if step_output.is_prefill:
                            state.prefill_latencies_ms.append(step_ms)
                            state.prefill_batch_sizes.append(batch_size)
                            state.prefill_tokens += scheduled.num_tokens
                        else:
                            state.decode_latencies_ms.append(step_ms)
                            state.decode_batch_sizes.append(batch_size)
                            state.decode_tokens += scheduled.num_tokens

                for seq_id, token_ids in step_output.outputs:
                    state = active.pop(seq_id, None)
                    if state is None:
                        continue
                    self._complete_request(
                        llm,
                        state,
                        token_ids,
                        step_finished_at,
                    )

                if not active:
                    profile_window_initialized = False
        finally:
            llm.exit()

    def _admit_request(
        self,
        llm: LLM,
        request: _EngineRequest,
        active: dict[int, _ActiveRequest],
    ):
        if self._is_cancel_requested(request.request_id):
            self._finish_cancelled_request(request)
            return
        if not request.future.set_running_or_notify_cancel():
            self._finish_cancelled_request(request)
            return
        try:
            seq_id = llm.add_request(request.prompt, request.sampling_params)
        except Exception as exc:
            self._finish_failed_request(request, exc)
            return

        active[seq_id] = _ActiveRequest(request=request, seq_id=seq_id)
        with self._state_lock:
            self._active_request_ids.add(request.request_id)

    def _complete_request(
        self,
        llm: LLM,
        state: _ActiveRequest,
        token_ids: list[int],
        finished_at: float,
    ):
        generation_started_at = state.generation_started_at or finished_at
        generation_ms = (finished_at - generation_started_at) * 1000
        output = {
            "text": llm.tokenizer.decode(token_ids),
            "token_ids": token_ids,
            "finish_reason": state.finish_reason or "length",
        }
        profile = self._build_request_profile(state, len(token_ids), generation_ms)
        result = AsyncEngineOutput(
            output=output,
            finish_reason=state.finish_reason or "length",
            queue_wait_ms=state.queue_wait_ms or 0.0,
            generation_ms=generation_ms,
            queue_depth_at_submit=state.request.queue_depth_at_submit,
            requests_ahead_at_submit=state.request.requests_ahead_at_submit,
            profile=profile,
        )
        self._record_terminal(state.request.request_id, "completed")
        if not state.request.future.done():
            state.request.future.set_result(result)
        self._emit_event(
            state.request,
            AsyncEngineStreamEvent(
                event_type="done",
                request_id=state.request.request_id,
                text=output["text"],
                output=result,
            ),
        )

    def _emit_token(
        self,
        llm: LLM,
        state: _ActiveRequest,
        token_id: int | None,
    ):
        if token_id is None:
            raise RuntimeError("produced_token metadata is missing token_id")
        state.generated_token_ids.append(token_id)
        text = llm.tokenizer.decode(state.generated_token_ids)
        if text.startswith(state.decoded_text):
            delta_text = text[len(state.decoded_text):]
        else:
            delta_text = llm.tokenizer.decode([token_id])
        state.decoded_text = text
        self._emit_event(
            state.request,
            AsyncEngineStreamEvent(
                event_type="token",
                request_id=state.request.request_id,
                token_id=token_id,
                token_index=len(state.generated_token_ids) - 1,
                delta_text=delta_text,
                text=text,
            ),
        )

    def _build_request_profile(
        self,
        state: _ActiveRequest,
        completion_tokens: int,
        generation_ms: float,
    ):
        if not state.request.profile:
            return None
        prefill_latency_ms = sum(state.prefill_latencies_ms)
        decode_latency_ms = sum(state.decode_latencies_ms)
        return {
            "measurement_mode": "cuda-synchronized-per-step-continuous-batching",
            "engine_ttft_ms": (
                state.prefill_latencies_ms[0]
                if state.prefill_latencies_ms
                else None
            ),
            "request_ttft_ms": state.request_ttft_ms,
            "prefill_latency_ms": prefill_latency_ms,
            "prefill_tokens": state.prefill_tokens,
            "prefill_tokens_per_s": safe_div(
                state.prefill_tokens,
                prefill_latency_ms / 1000,
            ),
            "decode_latency_ms": decode_latency_ms,
            "decode_tokens": state.decode_tokens,
            "decode_tpot_ms": safe_div(decode_latency_ms, state.decode_tokens),
            "decode_tokens_per_s": safe_div(
                state.decode_tokens,
                decode_latency_ms / 1000,
            ),
            "generation_tokens_per_s": safe_div(
                completion_tokens,
                generation_ms / 1000,
            ),
            "mean_prefill_batch_size": mean(state.prefill_batch_sizes),
            "mean_decode_batch_size": mean(state.decode_batch_sizes),
            "max_decode_batch_size": (
                max(state.decode_batch_sizes)
                if state.decode_batch_sizes
                else None
            ),
            "step_latencies_ms": {
                "prefill": state.prefill_latencies_ms,
                "decode": state.decode_latencies_ms,
            },
            "batch_sizes": {
                "prefill": state.prefill_batch_sizes,
                "decode": state.decode_batch_sizes,
            },
            **cuda_memory_stats(),
        }

    def _record_step_stats(self, is_prefill: bool, batch_size: int):
        with self._state_lock:
            self._total_engine_steps += 1
            self._last_batch_size = batch_size
            self._max_batch_size = max(self._max_batch_size, batch_size)
            if is_prefill:
                self._prefill_engine_steps += 1
            else:
                self._decode_engine_steps += 1
                self._max_decode_batch_size = max(
                    self._max_decode_batch_size,
                    batch_size,
                )

    def _fail_active_requests(
        self,
        active: dict[int, _ActiveRequest],
        exc: Exception,
    ):
        states = list(active.values())
        active.clear()
        for state in states:
            self._finish_failed_request(state.request, exc)

    def _fail_queued_requests(self, exc: Exception):
        while True:
            try:
                request = self._request_queue.get_nowait()
            except Empty:
                return
            if request is self._stop:
                continue
            if self._is_cancel_requested(request.request_id):
                self._finish_cancelled_request(request)
                continue
            if not request.future.set_running_or_notify_cancel():
                self._finish_cancelled_request(request)
                continue
            self._finish_failed_request(request, exc)

    def _cancel_active_requests(
        self,
        llm: LLM,
        active: dict[int, _ActiveRequest],
    ):
        cancelled_seq_ids = [
            seq_id
            for seq_id, state in active.items()
            if state.request.future.cancelled()
            or self._is_cancel_requested(state.request.request_id)
        ]
        for seq_id in cancelled_seq_ids:
            state = active.pop(seq_id)
            llm.abort_request(seq_id)
            self._finish_cancelled_request(state.request)

    def _finish_cancelled_request(self, request: _EngineRequest):
        error = AsyncEngineRequestCancelledError(
            f"request {request.request_id} was cancelled"
        )
        if not self._record_terminal(request.request_id, "cancelled"):
            return
        if not request.future.done():
            request.future.cancel()
        self._emit_event(
            request,
            AsyncEngineStreamEvent(
                event_type="error",
                request_id=request.request_id,
                error=error,
            ),
        )

    def _finish_failed_request(self, request: _EngineRequest, exc: Exception):
        if request.future.cancelled():
            self._finish_cancelled_request(request)
            return
        if not self._record_terminal(request.request_id, "failed"):
            return
        if not request.future.done():
            request.future.set_exception(exc)
        self._emit_event(
            request,
            AsyncEngineStreamEvent(
                event_type="error",
                request_id=request.request_id,
                error=exc,
            ),
        )

    def _record_terminal(self, request_id: str, status: str):
        with self._state_lock:
            if request_id not in self._outstanding_request_ids:
                return False
            self._outstanding_request_ids.remove(request_id)
            self._active_request_ids.discard(request_id)
            self._cancel_requested_ids.discard(request_id)
            self._outstanding_requests -= 1
            if status == "completed":
                self._completed_requests += 1
            elif status == "failed":
                self._failed_requests += 1
            elif status == "cancelled":
                self._cancelled_requests += 1
            else:
                raise ValueError(f"unknown terminal request status: {status}")
            return True

    def _is_cancel_requested(self, request_id: str):
        with self._state_lock:
            return request_id in self._cancel_requested_ids

    @staticmethod
    def _emit_event(
        request: _EngineRequest,
        event: AsyncEngineStreamEvent,
    ):
        if request.event_sink is None:
            return
        try:
            request.event_sink(event)
        except Exception:
            pass


def elapsed_ms(start: float):
    return (perf_counter() - start) * 1000


def safe_div(numerator: float, denominator: float):
    if denominator == 0:
        return None
    return numerator / denominator


def mean(values: list[int]):
    if not values:
        return None
    return sum(values) / len(values)


def bytes_to_gb(value: int):
    return value / 1024**3


def cuda_memory_stats():
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    return {
        "peak_allocated_gb": bytes_to_gb(torch.cuda.max_memory_allocated()),
        "peak_reserved_gb": bytes_to_gb(torch.cuda.max_memory_reserved()),
        "current_allocated_gb": bytes_to_gb(torch.cuda.memory_allocated()),
        "current_reserved_gb": bytes_to_gb(torch.cuda.memory_reserved()),
        "cuda_free_gb": bytes_to_gb(free_bytes),
        "cuda_total_gb": bytes_to_gb(total_bytes),
    }


def sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def consume_future_exception(future: Future):
    if not future.cancelled():
        future.exception()
