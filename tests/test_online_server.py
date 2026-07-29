import asyncio
import unittest

from fastapi.testclient import TestClient

import examples.qwen2_5_vl_server as server
from nanovllm import (
    AsyncEngineOutput,
    AsyncEngineStreamEvent,
    MultiModalPrompt,
)


class _FakeEngine:
    def __init__(self, delay: float = 0.0):
        self.delay = delay
        self.cancelled_request_ids = []

    async def generate(self, request_id, prompt, sampling_params, **kwargs):
        if self.delay:
            await asyncio.sleep(self.delay)
        return fake_engine_output()

    async def stream_generate(self, request_id, prompt, sampling_params, **kwargs):
        if self.delay:
            await asyncio.sleep(self.delay)
        yield AsyncEngineStreamEvent(
            event_type="token",
            request_id=request_id,
            token_id=101,
            token_index=0,
            delta_text="hello",
            text="hello",
        )
        yield AsyncEngineStreamEvent(
            event_type="done",
            request_id=request_id,
            text="hello",
            output=fake_engine_output(),
        )

    def cancel(self, request_id):
        self.cancelled_request_ids.append(request_id)
        return True

    def stats(self):
        return {
            "active_request_count": 0,
            "queue_depth": 0,
        }


def fake_engine_output():
    return AsyncEngineOutput(
        output={
            "text": "hello",
            "token_ids": [101],
            "finish_reason": "length",
        },
        finish_reason="length",
        queue_wait_ms=2.0,
        generation_ms=5.0,
        queue_depth_at_submit=0,
        requests_ahead_at_submit=0,
        profile=None,
    )


def fake_preprocess(_request):
    return (
        MultiModalPrompt(
            input_ids=[1, 2, 3],
            mm_token_type_ids=[0, 1, 0],
        ),
        {
            "input_tokens": 3,
            "image_tokens": 1,
            "image_grid_thw": [[1, 2, 2]],
        },
    )


class OnlineServerTest(unittest.TestCase):
    def setUp(self):
        self.original_native_preprocess = server.preprocess_native_request
        self.original_chat_preprocess = server.preprocess_chat_request
        server.preprocess_native_request = fake_preprocess
        server.preprocess_chat_request = fake_preprocess
        server.app.state.engine = _FakeEngine()
        server.app.state.limiter = server.ConcurrencyLimiter(2)
        server.app.state.config = server.ServerConfig(
            model="/models/test",
            served_model_name="test-vlm",
            max_model_len=64,
            max_num_seqs=2,
            max_num_batched_tokens=32,
            gpu_memory_utilization=0.5,
            max_concurrent_requests=2,
            request_timeout_seconds=1.0,
        )
        server.app.state.startup = {
            "processor_load_ms": 1.0,
            "engine_load_ms": 2.0,
        }
        self.client = TestClient(server.app)

    def tearDown(self):
        self.client.close()
        server.preprocess_native_request = self.original_native_preprocess
        server.preprocess_chat_request = self.original_chat_preprocess

    def test_native_non_streaming_response(self):
        response = self.client.post(
            "/generate",
            json={
                "prompt": "describe",
                "image_base64": "AA==",
                "max_new_tokens": 1,
            },
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["text"], "hello")
        self.assertEqual(body["token_ids"], [101])
        self.assertEqual(body["finish_reason"], "length")
        self.assertEqual(body["usage"]["total_tokens"], 4)
        self.assertEqual(server.app.state.limiter.stats()["in_flight"], 0)

    def test_native_sse_emits_metadata_token_and_done(self):
        response = self.client.post(
            "/generate_stream",
            json={
                "prompt": "describe",
                "image_base64": "AA==",
                "max_new_tokens": 1,
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn("event: metadata", response.text)
        self.assertIn("event: token", response.text)
        self.assertIn('"delta":"hello"', response.text)
        self.assertIn("event: done", response.text)
        self.assertEqual(server.app.state.limiter.stats()["in_flight"], 0)

    def test_openai_non_streaming_schema(self):
        response = self.client.post(
            "/v1/chat/completions",
            json=openai_request(stream=False),
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["object"], "chat.completion")
        self.assertEqual(body["model"], "test-vlm")
        self.assertEqual(
            body["choices"][0]["message"],
            {"role": "assistant", "content": "hello"},
        )
        self.assertEqual(body["choices"][0]["finish_reason"], "length")
        self.assertEqual(body["usage"]["completion_tokens"], 1)

    def test_openai_streaming_schema_and_done_marker(self):
        payload = openai_request(stream=True)
        payload["stream_options"] = {"include_usage": True}
        response = self.client.post("/v1/chat/completions", json=payload)

        self.assertEqual(response.status_code, 200)
        self.assertIn('"object":"chat.completion.chunk"', response.text)
        self.assertIn('"role":"assistant"', response.text)
        self.assertIn('"content":"hello"', response.text)
        self.assertIn('"choices":[]', response.text)
        self.assertTrue(response.text.rstrip().endswith("data: [DONE]"))

    def test_concurrency_limit_returns_429_without_preprocessing(self):
        server.app.state.limiter = server.ConcurrencyLimiter(1)
        self.assertTrue(server.app.state.limiter.try_acquire())
        try:
            response = self.client.post(
                "/generate",
                json={
                    "prompt": "describe",
                    "image_base64": "AA==",
                    "max_new_tokens": 1,
                },
            )
        finally:
            server.app.state.limiter.release()

        self.assertEqual(response.status_code, 429)
        self.assertEqual(response.headers["retry-after"], "1")

    def test_timeout_cancels_engine_request_and_returns_504(self):
        engine = _FakeEngine(delay=0.2)
        server.app.state.engine = engine
        server.app.state.config.request_timeout_seconds = 0.01

        response = self.client.post(
            "/generate",
            json={
                "prompt": "describe",
                "image_base64": "AA==",
                "max_new_tokens": 1,
            },
        )

        self.assertEqual(response.status_code, 504)
        self.assertEqual(len(engine.cancelled_request_ids), 1)
        self.assertEqual(server.app.state.limiter.stats()["in_flight"], 0)

    def test_stream_timeout_emits_error_event_and_releases_slot(self):
        engine = _FakeEngine(delay=0.2)
        server.app.state.engine = engine
        server.app.state.config.request_timeout_seconds = 0.01

        response = self.client.post(
            "/generate_stream",
            json={
                "prompt": "describe",
                "image_base64": "AA==",
                "max_new_tokens": 1,
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn("event: error", response.text)
        self.assertIn('"code":"request_timeout"', response.text)
        self.assertGreaterEqual(len(engine.cancelled_request_ids), 1)
        self.assertEqual(server.app.state.limiter.stats()["in_flight"], 0)

    def test_unknown_openai_model_returns_standard_error(self):
        payload = openai_request(stream=False)
        payload["model"] = "missing"
        response = self.client.post("/v1/chat/completions", json=payload)

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["error"]["code"], "model_not_found")


def openai_request(stream: bool):
    return {
        "model": "test-vlm",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,AA==",
                        },
                    },
                    {
                        "type": "text",
                        "text": "describe",
                    },
                ],
            }
        ],
        "max_tokens": 1,
        "temperature": 0,
        "stream": stream,
    }


if __name__ == "__main__":
    unittest.main()
