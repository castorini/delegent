from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from delegent.costs import CostRates, summarize_frontier_cost
from delegent.io import (
    DataValidationError,
    record_set_sha256,
    write_json,
    write_jsonl,
)


def _record(qid: str, generations: list[object]) -> dict:
    return {
        "schema_version": "delegent-routed-record-v1",
        "status": "complete",
        "qid": qid,
        "arm": "frontier2_delegate0",
        "generations": generations,
    }


def _write_completed_set(root: Path, records: list[dict]) -> Path:
    records_path = root / "records.jsonl"
    write_jsonl(records_path, records)
    write_json(
        root / "records.completion.json",
        {
            "schema_version": "delegent-record-set-completion-v1",
            "status": "complete",
            "arm": "frontier2_delegate0",
            "question_count": len(records),
            "record_set_sha256": record_set_sha256(records),
        },
    )
    return records_path


class CostTests(unittest.TestCase):
    def test_aggregate_only_summary_supports_common_usage_shapes(self) -> None:
        records = [
            _record(
                "synthetic-1",
                [
                    {
                        "actor": "frontier",
                        "attempts": 1,
                        "usage": {
                            "input_tokens": 100,
                            "cached_input_tokens": 40,
                            "cache_write_input_tokens": 10,
                            "output_tokens": 20,
                            "reasoning_output_tokens": 5,
                        },
                    },
                    {"actor": "delegate", "attempts": 1},
                ],
            ),
            _record(
                "synthetic-2",
                [
                    {
                        "actor": "frontier",
                        "attempts": 2,
                        "usage": {
                            "prompt_tokens": 50,
                            "prompt_tokens_details": {
                                "cached_tokens": 10,
                                "cache_write_tokens": 4,
                            },
                            "completion_tokens": 10,
                            "completion_tokens_details": {"reasoning_tokens": 2},
                        },
                    }
                ],
            ),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            records_path = _write_completed_set(Path(temporary), records)
            summary = summarize_frontier_cost(
                records_path=records_path,
                frontier_actor="frontier",
            )

        self.assertEqual(
            summary["token_totals"],
            {
                "input_tokens": 150,
                "cached_input_tokens": 50,
                "cache_write_input_tokens": 14,
                "uncached_input_tokens": 100,
                "output_tokens": 30,
                "reasoning_output_tokens": 7,
            },
        )
        self.assertEqual(summary["frontier_generations"], 2)
        self.assertEqual(summary["frontier_calls"], 3)
        self.assertEqual(summary["cost_usd"]["total"], 0.00102)
        self.assertEqual(summary["cost_usd"]["mean_per_question"], 0.00051)
        self.assertFalse(summary["reasoning_tokens_billed_separately"])
        self.assertTrue(summary["completion_validated_before_accounting"])
        self.assertNotIn("qid", json.dumps(summary).casefold())

    def test_custom_rates_are_reported_and_used(self) -> None:
        records = [
            _record(
                "synthetic-1",
                [
                    {
                        "actor": "frontier",
                        "usage": {
                            "input_tokens": 10,
                            "cached_input_tokens": 4,
                            "output_tokens": 2,
                        },
                    }
                ],
            )
        ]
        with tempfile.TemporaryDirectory() as temporary:
            records_path = _write_completed_set(Path(temporary), records)
            summary = summarize_frontier_cost(
                records_path=records_path,
                frontier_actor="frontier",
                rates=CostRates(uncached_input=1, cached_input=2, output=3),
            )
        self.assertEqual(
            summary["rates_usd_per_million_tokens"],
            {"uncached_input": 1.0, "cached_input": 2.0, "output": 3.0},
        )
        self.assertEqual(summary["cost_usd"]["total"], 0.00002)

    def test_zero_frontier_arm_has_zero_cost(self) -> None:
        records = [_record("synthetic-1", [{"actor": "delegate"}])]
        with tempfile.TemporaryDirectory() as temporary:
            records_path = _write_completed_set(Path(temporary), records)
            summary = summarize_frontier_cost(
                records_path=records_path,
                frontier_actor="frontier",
            )
        self.assertEqual(summary["frontier_generations"], 0)
        self.assertEqual(summary["cost_usd"]["total"], 0.0)

    def test_missing_completion_fails_before_usage_is_inspected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            records_path = Path(temporary) / "records.jsonl"
            write_jsonl(
                records_path,
                [_record("synthetic-1", [{"actor": "frontier", "usage": -1}])],
            )
            with self.assertRaisesRegex(DataValidationError, "completion is required"):
                summarize_frontier_cost(
                    records_path=records_path,
                    frontier_actor="frontier",
                )

    def test_completion_identity_drift_is_rejected(self) -> None:
        records = [_record("synthetic-1", [])]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records_path = _write_completed_set(root, records)
            completion = json.loads(
                (root / "records.completion.json").read_text(encoding="utf-8")
            )
            completion["record_set_sha256"] = "0" * 64
            write_json(
                root / "records.completion.json",
                completion,
                overwrite=True,
            )
            with self.assertRaisesRegex(DataValidationError, "field drifted"):
                summarize_frontier_cost(
                    records_path=records_path,
                    frontier_actor="frontier",
                )

    def test_inconsistent_and_negative_telemetry_is_rejected(self) -> None:
        cases = {
            "negative": {
                "input_tokens": -1,
                "output_tokens": 1,
            },
            "cached exceeds input": {
                "input_tokens": 5,
                "cached_input_tokens": 6,
                "output_tokens": 1,
            },
            "reasoning exceeds output": {
                "input_tokens": 5,
                "output_tokens": 1,
                "reasoning_output_tokens": 2,
            },
            "input aliases conflict": {
                "input_tokens": 5,
                "prompt_tokens": 6,
                "output_tokens": 1,
            },
            "boolean is not a count": {
                "input_tokens": True,
                "output_tokens": 1,
            },
            "null optional count": {
                "input_tokens": 5,
                "cached_input_tokens": None,
                "output_tokens": 1,
            },
        }
        for name, usage in cases.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                records = [
                    _record(
                        "synthetic-1",
                        [{"actor": "frontier", "usage": usage}],
                    )
                ]
                records_path = _write_completed_set(Path(temporary), records)
                with self.assertRaises(DataValidationError):
                    summarize_frontier_cost(
                        records_path=records_path,
                        frontier_actor="frontier",
                    )

    def test_invalid_rates_are_rejected(self) -> None:
        for value in (-1.0, float("inf"), float("nan")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                CostRates(uncached_input=value)
        with self.assertRaises(TypeError):
            CostRates(uncached_input=True)


if __name__ == "__main__":
    unittest.main()
