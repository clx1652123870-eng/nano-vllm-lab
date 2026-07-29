import asyncio
import time
import unittest
from types import SimpleNamespace

import nanovllm.engine.async_llm_engine as async_engine_module
from nanovllm import AsyncEngineClosedError, AsyncLLMEngine, SamplingParams
from nanovllm.engine.llm_engine import EngineStepOutput, ScheduledSequenceMetadata
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence, SequenceStatus


class _FakeTokenizer:
    def decode(self, token_ids):
        return ",".join(map(str, token_ids))


class _FakeLLM:
    def __init__(self, model, **kwargs):
        self.tokenizer = _FakeTokenizer()
        self.config = SimpleNamespace(max_num_seqs=2)
        self._requests = {}
        self._prefilled = set()
        self._seq_id = 0
        self.aborted_seq_ids = []

    def add_request(self, prompt, sampling_params):
        self._seq_id += 1
        self._requests[self._seq_id] = {
            "seed": prompt[0],
            "tokens": [],
        }
        return self._seq_id

    def step_with_metadata(self):
        time.sleep(0.03)
        unprefilled = [
            seq_id
            for seq_id in self._requests
            if seq_id not in self._prefilled
        ]
        if unprefilled:
            seq_id = unprefilled[0]
            self._prefilled.add(seq_id)
            token_id = self._requests[seq_id]["seed"]
            self._requests[seq_id]["tokens"].append(token_id)
            return EngineStepOutput(
                outputs=[],
                scheduled=[
                    ScheduledSequenceMetadata(
                        seq_id=seq_id,
                        num_tokens=1,
                        produced_token=True,
                        token_id=token_id,
                    )
                ],
                is_prefill=True,
                num_tokens=1,
            )

        seq_ids = list(self._requests)
        for seq_id in seq_ids:
            self._requests[seq_id]["tokens"].append(99)
        outputs = [
            (seq_id, self._requests[seq_id]["tokens"])
            for seq_id in seq_ids
        ]
        self._requests.clear()
        return EngineStepOutput(
            outputs=outputs,
            scheduled=[
                ScheduledSequenceMetadata(
                    seq_id=seq_id,
                    num_tokens=1,
                    produced_token=True,
                    token_id=99,
                    finish_reason="length",
                )
                for seq_id in seq_ids
            ],
            is_prefill=False,
            num_tokens=-len(seq_ids),
        )

    def abort_request(self, seq_id):
        self.aborted_seq_ids.append(seq_id)
        self._prefilled.discard(seq_id)
        return self._requests.pop(seq_id, None) is not None

    def exit(self):
        pass


class AsyncLLMEngineTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.original_llm = async_engine_module.LLM
        self.original_sync_cuda = async_engine_module.sync_cuda
        self.original_memory_stats = async_engine_module.cuda_memory_stats
        self.original_reset_peak = (
            async_engine_module.torch.cuda.reset_peak_memory_stats
        )
        async_engine_module.LLM = _FakeLLM
        async_engine_module.sync_cuda = lambda: None
        async_engine_module.cuda_memory_stats = lambda: {}
        async_engine_module.torch.cuda.reset_peak_memory_stats = lambda: None

    def tearDown(self):
        async_engine_module.LLM = self.original_llm
        async_engine_module.sync_cuda = self.original_sync_cuda
        async_engine_module.cuda_memory_stats = self.original_memory_stats
        async_engine_module.torch.cuda.reset_peak_memory_stats = (
            self.original_reset_peak
        )

    async def test_concurrent_requests_share_decode_batch_without_mixing_results(self):
        engine = AsyncLLMEngine("fake")
        try:
            first, second, third = await asyncio.gather(
                engine.generate("first", [11], SamplingParams(), profile=True),
                engine.generate("second", [22], SamplingParams(), profile=True),
                engine.generate("third", [33], SamplingParams(), profile=True),
            )

            self.assertEqual(first.output["token_ids"], [11, 99])
            self.assertEqual(second.output["token_ids"], [22, 99])
            self.assertEqual(third.output["token_ids"], [33, 99])
            self.assertLess(first.queue_wait_ms, second.queue_wait_ms)
            self.assertGreater(second.queue_wait_ms, 20)
            self.assertGreater(third.queue_wait_ms, second.queue_wait_ms)
            self.assertEqual(first.requests_ahead_at_submit, 0)
            self.assertEqual(second.requests_ahead_at_submit, 1)
            self.assertEqual(third.requests_ahead_at_submit, 2)
            self.assertEqual(first.profile["mean_prefill_batch_size"], 1)
            self.assertEqual(second.profile["mean_prefill_batch_size"], 1)
            self.assertEqual(third.profile["mean_prefill_batch_size"], 1)
            self.assertEqual(first.profile["mean_decode_batch_size"], 2)
            self.assertEqual(second.profile["mean_decode_batch_size"], 2)
            self.assertEqual(third.profile["mean_decode_batch_size"], 1)
            self.assertEqual(engine.stats()["completed_requests"], 3)
            self.assertEqual(engine.stats()["failed_requests"], 0)
            self.assertEqual(engine.stats()["max_decode_batch_size"], 2)
        finally:
            engine.shutdown()

        self.assertFalse(engine.stats()["engine_thread_alive"])

    async def test_shutdown_rejects_new_requests(self):
        engine = AsyncLLMEngine("fake")
        engine.shutdown()

        with self.assertRaises(AsyncEngineClosedError):
            await engine.generate("late", [33], SamplingParams())

    async def test_streaming_emits_each_token_then_done(self):
        engine = AsyncLLMEngine("fake")
        try:
            events = [
                event
                async for event in engine.stream_generate(
                    "stream",
                    [44],
                    SamplingParams(),
                    profile=True,
                )
            ]

            self.assertEqual(
                [event.event_type for event in events],
                ["token", "token", "done"],
            )
            self.assertEqual(
                [event.token_id for event in events if event.event_type == "token"],
                [44, 99],
            )
            self.assertEqual(events[0].delta_text, "44")
            self.assertEqual(events[1].text, "44,99")
            self.assertEqual(events[-1].output.output["token_ids"], [44, 99])
            self.assertEqual(events[-1].output.finish_reason, "length")
        finally:
            engine.shutdown()

    async def test_closing_stream_cancels_active_request(self):
        engine = AsyncLLMEngine("fake")
        stream = engine.stream_generate(
            "cancel-me",
            [55],
            SamplingParams(),
        )
        first_event = await anext(stream)
        self.assertEqual(first_event.event_type, "token")
        await stream.aclose()

        for _ in range(20):
            if engine.stats()["outstanding_requests"] == 0:
                break
            await asyncio.sleep(0.01)

        try:
            self.assertEqual(engine.stats()["cancelled_requests"], 1)
            self.assertEqual(engine.stats()["outstanding_requests"], 0)
            self.assertTrue(engine._thread.is_alive())
        finally:
            engine.shutdown()


class SchedulerCancellationTest(unittest.TestCase):
    def test_abort_releases_allocated_kv_cache_blocks(self):
        original_block_size = Sequence.block_size
        Sequence.block_size = 4
        config = SimpleNamespace(
            max_num_seqs=2,
            max_num_batched_tokens=16,
            eos=2,
            kvcache_block_size=4,
            num_kvcache_blocks=8,
        )
        scheduler = Scheduler(config)
        seq = Sequence([1, 2, 3, 4], SamplingParams(max_tokens=4))
        try:
            scheduler.add(seq)
            scheduled, is_prefill = scheduler.schedule()
            self.assertTrue(is_prefill)
            self.assertEqual(scheduled, [seq])
            self.assertGreater(len(seq.block_table), 0)
            self.assertGreater(len(scheduler.block_manager.used_block_ids), 0)

            self.assertTrue(scheduler.abort(seq.seq_id))
            self.assertEqual(seq.status, SequenceStatus.CANCELLED)
            self.assertEqual(seq.block_table, [])
            self.assertEqual(len(scheduler.block_manager.used_block_ids), 0)
            self.assertTrue(scheduler.is_finished())
            self.assertFalse(scheduler.abort(seq.seq_id))
        finally:
            Sequence.block_size = original_block_size


if __name__ == "__main__":
    unittest.main()
