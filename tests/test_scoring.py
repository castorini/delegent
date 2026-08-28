from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from delegent.io import DataValidationError, write_jsonl
from delegent.scoring import score_records


def record(qid: str, arm: str, docids: list[str]) -> dict:
    return {
        "qid": qid,
        "arm": arm,
        "searches": [
            {
                "query": "synthetic",
                "hits": [
                    {"docid": docid, "rank": index, "score": 1.0, "text": "synthetic"}
                    for index, docid in enumerate(docids, start=1)
                ],
            }
        ],
    }


class ScoringTests(unittest.TestCase):
    def test_hotpot_score_is_aggregate_only_with_paired_comparison(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate = root / "candidate.jsonl"
            reference = root / "reference.jsonl"
            qrels = root / "qrels.jsonl"
            write_jsonl(
                candidate,
                [
                    record("demo-1", "candidate", ["a"]),
                    record("demo-2", "candidate", ["b"]),
                ],
            )
            write_jsonl(
                reference,
                [
                    record("demo-1", "reference", ["x"]),
                    record("demo-2", "reference", ["b"]),
                ],
            )
            write_jsonl(
                qrels,
                [
                    {
                        "qid": "demo-1",
                        "positive_docids": ["a"],
                        "doc_fact_ids": {"a": ["f1"]},
                    },
                    {
                        "qid": "demo-2",
                        "positive_docids": ["b"],
                        "doc_fact_ids": {"b": ["f2"]},
                    },
                ],
            )
            score = score_records(
                task="hotpotqa",
                records_path=candidate,
                reference_records_path=reference,
                qrels_path=qrels,
                bootstrap_repetitions=200,
                seed=11,
                require_completion=False,
            )
            self.assertEqual(score["question_count"], 2)
            self.assertEqual(
                score["metrics"]["union_required_fact_coverage"]["percent"], 100.0
            )
            self.assertFalse(score["per_question_outputs_included"])
            self.assertNotIn("qids", json.dumps(score))
            self.assertAlmostEqual(
                score["comparisons"]["union_required_fact_coverage"]["difference_pp"],
                50.0,
            )

    def test_squad_score_does_not_require_fact_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = root / "records.jsonl"
            qrels = root / "qrels.jsonl"
            write_jsonl(records, [record("demo-1", "arm", ["a"])])
            write_jsonl(qrels, [{"qid": "demo-1", "positive_docids": ["a"]}])
            score = score_records(
                task="squad",
                records_path=records,
                qrels_path=qrels,
                require_completion=False,
            )
            self.assertEqual(score["metrics"]["union_any_positive"]["percent"], 100.0)

    def test_missing_completion_fails_before_qrels_path_is_opened(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = root / "records.jsonl"
            write_jsonl(records, [record("demo-1", "arm", ["a"])])
            with self.assertRaisesRegex(DataValidationError, "completion is required"):
                score_records(
                    task="squad",
                    records_path=records,
                    qrels_path=root / "this-file-must-not-be-opened.jsonl",
                )


if __name__ == "__main__":
    unittest.main()
