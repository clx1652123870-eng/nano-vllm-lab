import unittest

from examples.qwen2_5_vl_online_regression import (
    build_batching_checks,
    build_correctness,
    build_throughput_comparison,
    parse_concurrencies,
)


class OnlineRegressionTest(unittest.TestCase):
    def test_matrix_checks_correctness_batching_and_throughput(self):
        scenarios = {
            "c1": make_report(1, 1, 2.0, 16.0, 100.0),
            "c2": make_report(2, 2, 3.5, 28.0, 150.0),
            "c4": make_report(4, 4, 5.0, 40.0, 260.0),
        }

        correctness = build_correctness(scenarios)
        batching = build_batching_checks(scenarios, skip_assertion=False)
        throughput = build_throughput_comparison(scenarios)

        self.assertTrue(correctness["passed"])
        self.assertTrue(batching["passed"])
        self.assertEqual(
            batching["scenarios"]["c4"]["observed_max_decode_batch_size"],
            4,
        )
        self.assertEqual(
            throughput["c2"]["requests_per_s_speedup_vs_c1"],
            1.75,
        )

    def test_token_mismatch_fails_correctness(self):
        scenarios = {
            "c1": make_report(1, 1, 2.0, 16.0, 100.0),
            "c2": make_report(
                2,
                2,
                3.0,
                24.0,
                150.0,
                token_ids=[7, 9],
            ),
        }

        correctness = build_correctness(scenarios)

        self.assertFalse(correctness["passed"])
        self.assertEqual(correctness["mismatches"][0]["scenario"], "c2")

    def test_concurrency_parser_requires_unique_c1_baseline(self):
        self.assertEqual(parse_concurrencies("1,2,4"), [1, 2, 4])
        with self.assertRaises(ValueError):
            parse_concurrencies("2,4")
        with self.assertRaises(ValueError):
            parse_concurrencies("1,2,2")


def make_report(
    concurrency,
    decode_batch,
    requests_per_s,
    output_tokens_per_s,
    mean_e2e,
    token_ids=None,
):
    token_ids = token_ids or [7, 8]
    measurement = {
        "token_ids": token_ids,
        "text": "answer",
        "profile": {
            "max_decode_batch_size": decode_batch,
        },
    }
    return {
        "config": {
            "concurrency": concurrency,
        },
        "server": {
            "limits": {
                "max_num_seqs": 4,
            }
        },
        "measurements": [measurement, measurement],
        "benchmark": {
            "requests_per_s": requests_per_s,
            "output_tokens_per_s": output_tokens_per_s,
        },
        "summary": {
            "client_e2e_latency_ms": {
                "mean": mean_e2e,
            }
        },
    }


if __name__ == "__main__":
    unittest.main()
