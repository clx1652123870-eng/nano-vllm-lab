import argparse
import base64
import http.client
import json
import statistics
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from urllib.parse import urlsplit


def parse_args():
    parser = argparse.ArgumentParser(
        description="Profile the nano-vllm Qwen2.5-VL HTTP inference service."
    )
    parser.add_argument("--url", default="http://127.0.0.1:8000/generate")
    parser.add_argument("--image", default="assets/dog.png")
    parser.add_argument(
        "--image-input",
        choices=("base64", "path"),
        default="base64",
        help="Use base64 for a real HTTP upload, or path when client and server share files.",
    )
    parser.add_argument("--prompt", default="描述这张图片")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--warmup-iters", type=int, default=1)
    parser.add_argument("--measure-iters", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--output-json")
    parser.add_argument("--print-text", action="store_true")
    return parser.parse_args()


class JsonHttpClient:
    def __init__(self, url: str, timeout: float):
        parsed = urlsplit(url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            raise ValueError("--url must be an http:// or https:// URL")
        connection_cls = (
            http.client.HTTPSConnection
            if parsed.scheme == "https"
            else http.client.HTTPConnection
        )
        self.connection = connection_cls(
            parsed.hostname,
            parsed.port,
            timeout=timeout,
        )
        self.path = parsed.path or "/"
        if parsed.query:
            self.path = f"{self.path}?{parsed.query}"

    def request_json(self, method: str, path: str, payload: dict | None = None):
        body = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"

        t0 = perf_counter()
        self.connection.request(method, path, body=body, headers=headers)
        response = self.connection.getresponse()
        raw = response.read()
        latency_ms = elapsed_ms(t0)
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"server returned non-JSON response: HTTP {response.status}"
            ) from exc
        if response.status >= 400:
            raise RuntimeError(f"server returned HTTP {response.status}: {data}")
        return data, latency_ms, len(body or b"")

    def close(self):
        self.connection.close()


def build_payload(args):
    image_path = Path(args.image).expanduser().resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"image does not exist: {image_path}")

    payload = {
        "prompt": args.prompt,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "ignore_eos": args.ignore_eos,
        "profile": True,
    }
    if args.image_input == "base64":
        payload["image_base64"] = base64.b64encode(image_path.read_bytes()).decode("ascii")
    else:
        payload["image_path"] = str(image_path)
    return payload


def run_iteration(client: JsonHttpClient, payload: dict):
    response, client_latency_ms, request_body_bytes = client.request_json(
        "POST",
        client.path,
        payload,
    )
    if "profile" not in response:
        raise RuntimeError(
            "server response has no profile field; restart the updated server "
            "and ensure the request contains profile=true"
        )
    response["client_e2e_latency_ms"] = client_latency_ms
    response["request_body_bytes"] = request_body_bytes
    return response


def build_summary(measurements: list[dict]):
    metric_paths = {
        "client_e2e_latency_ms": ("client_e2e_latency_ms",),
        "server_total_latency_ms": ("latency_ms", "total"),
        "preprocess_latency_ms": ("latency_ms", "preprocess"),
        "queue_wait_latency_ms": ("latency_ms", "queue_wait"),
        "generation_latency_ms": ("latency_ms", "generation"),
        "queue_depth_at_submit": ("engine", "queue_depth_at_submit"),
        "requests_ahead_at_submit": ("engine", "requests_ahead_at_submit"),
        "request_ttft_ms": ("profile", "request_ttft_ms"),
        "engine_ttft_ms": ("profile", "engine_ttft_ms"),
        "prefill_latency_ms": ("profile", "prefill_latency_ms"),
        "prefill_tokens_per_s": ("profile", "prefill_tokens_per_s"),
        "decode_latency_ms": ("profile", "decode_latency_ms"),
        "decode_tpot_ms": ("profile", "decode_tpot_ms"),
        "decode_tokens_per_s": ("profile", "decode_tokens_per_s"),
        "generation_tokens_per_s": ("profile", "generation_tokens_per_s"),
        "server_tokens_per_s": ("profile", "server_tokens_per_s"),
        "mean_prefill_batch_size": ("profile", "mean_prefill_batch_size"),
        "mean_decode_batch_size": ("profile", "mean_decode_batch_size"),
        "max_decode_batch_size": ("profile", "max_decode_batch_size"),
        "peak_allocated_gb": ("profile", "peak_allocated_gb"),
        "peak_reserved_gb": ("profile", "peak_reserved_gb"),
    }
    summary = {}
    for name, path in metric_paths.items():
        values = [nested_get(item, path) for item in measurements]
        summary[name] = summarize([value for value in values if value is not None])
    return summary


def run_measurements(
    args,
    payload: dict,
    serial_client: JsonHttpClient,
    *,
    iterations: int | None = None,
):
    iterations = args.measure_iters if iterations is None else iterations
    if args.concurrency == 1:
        return [
            run_iteration(serial_client, payload)
            for _ in range(iterations)
        ]

    def run_one(_):
        client = JsonHttpClient(args.url, args.timeout)
        try:
            return run_iteration(client, payload)
        finally:
            client.close()

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        return list(executor.map(run_one, range(iterations)))


def print_iteration(prefix: str, result: dict):
    profile = result["profile"]
    print(
        f"{prefix}: "
        f"client_e2e={result['client_e2e_latency_ms']:.2f}ms, "
        f"server={result['latency_ms']['total']:.2f}ms, "
        f"request_ttft={profile['request_ttft_ms']:.2f}ms, "
        f"engine_ttft={profile['engine_ttft_ms']:.2f}ms, "
        f"decode_tpot={format_optional(profile['decode_tpot_ms'])}ms, "
        f"decode={format_optional(profile['decode_tokens_per_s'])}tok/s, "
        f"decode_batch={format_optional(profile.get('mean_decode_batch_size'))}"
    )


def nested_get(value: dict, path: tuple[str, ...]):
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def build_benchmark(measurements: list[dict], elapsed_ms_value: float):
    completion_tokens = sum(
        item["usage"]["completion_tokens"]
        for item in measurements
    )
    return {
        "requests": len(measurements),
        "completion_tokens": completion_tokens,
        "elapsed_ms": elapsed_ms_value,
        "requests_per_s": safe_div(
            len(measurements),
            elapsed_ms_value / 1000,
        ),
        "output_tokens_per_s": safe_div(
            completion_tokens,
            elapsed_ms_value / 1000,
        ),
    }


def summarize(values: list[float]):
    if not values:
        return None
    values = sorted(values)
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "min": values[0],
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
        "max": values[-1],
    }


def percentile(values: list[float], percent: float):
    if len(values) == 1:
        return values[0]
    rank = (len(values) - 1) * percent / 100
    lower = int(rank)
    upper = min(lower + 1, len(values) - 1)
    weight = rank - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def format_optional(value):
    return "n/a" if value is None else f"{value:.2f}"


def elapsed_ms(start: float):
    return (perf_counter() - start) * 1000


def safe_div(numerator: float, denominator: float):
    if denominator == 0:
        return None
    return numerator / denominator


def validate_server(health: dict):
    expected_mode = "single-process-single-gpu-continuous-batching"
    if health.get("mode") != expected_mode:
        raise RuntimeError(
            f"server mode is {health.get('mode')!r}, expected {expected_mode!r}; "
            "restart examples/qwen2_5_vl_server.py to load the continuous "
            "batching implementation"
        )


def main():
    args = parse_args()
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be at least 1")
    if args.warmup_iters < 0 or args.measure_iters < 1:
        raise ValueError("--warmup-iters must be >= 0 and --measure-iters must be >= 1")
    if args.concurrency < 1:
        raise ValueError("--concurrency must be at least 1")

    payload = build_payload(args)
    client = JsonHttpClient(args.url, args.timeout)
    try:
        health_path = "/health"
        health, _, _ = client.request_json("GET", health_path)
        validate_server(health)

        warmups = run_measurements(
            args,
            payload,
            client,
            iterations=args.warmup_iters,
        )
        for idx, result in enumerate(warmups):
            print_iteration(f"warmup[{idx}]", result)

        measurement_started_at = perf_counter()
        measurements = run_measurements(args, payload, client)
        measurement_elapsed_ms = elapsed_ms(measurement_started_at)
        for idx, result in enumerate(measurements):
            print_iteration(f"measure[{idx}]", result)
    finally:
        client.close()

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "url": args.url,
            "image": str(Path(args.image).expanduser().resolve()),
            "image_input": args.image_input,
            "prompt": args.prompt,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "ignore_eos": args.ignore_eos,
            "warmup_iters": args.warmup_iters,
            "measure_iters": args.measure_iters,
            "concurrency": args.concurrency,
        },
        "server": health,
        "warmups": warmups,
        "measurements": measurements,
        "summary": build_summary(measurements),
        "benchmark": build_benchmark(measurements, measurement_elapsed_ms),
    }

    print("\n[summary]")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print("\n[benchmark]")
    print(json.dumps(report["benchmark"], ensure_ascii=False, indent=2))
    if args.print_text:
        print("\n[text]")
        print(measurements[-1]["text"])

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"\nwrote: {output_path}")


if __name__ == "__main__":
    main()
