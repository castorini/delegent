from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from delegent.derive import derive_prefix_family, derive_prefix_record
from delegent.io import read_json, read_jsonl, write_jsonl
from delegent.split import freeze_split, salted_order
from tests.helpers import demo_sources


class DeriveTests(unittest.TestCase):
    def test_prefix_slices_are_exact_and_zero_call(self) -> None:
        source = demo_sources()[0]
        derived = derive_prefix_record(source, cap=1, frontier_actor="frontier")
        self.assertEqual(derived["searches"], source["searches"][:2])
        self.assertEqual(derived["generations"], source["generations"][:1])
        self.assertTrue(derived["zero_model_calls_for_derivation"])

    def test_family_writes_each_cap(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            summary = derive_prefix_family(
                demo_sources(),
                caps=[0, 1],
                output_dir=temporary,
                frontier_actor="frontier",
            )
            self.assertEqual(summary["completed_records"], 4)
            self.assertEqual(
                read_json(
                    Path(temporary) / "frontier0" / "records" / "demo-route-001.json"
                )["searches"],
                demo_sources()[0]["searches"][:1],
            )


class SplitTests(unittest.TestCase):
    def test_salted_order_matches_sha256_salt_nul_qid(self) -> None:
        rows = [
            {"qid": "demo-c", "question": "three"},
            {"qid": "demo-a", "question": "one"},
            {"qid": "demo-b", "question": "two"},
        ]
        expected = sorted(
            (row["qid"] for row in rows),
            key=lambda qid: (hashlib.sha256(("salt\0" + qid).encode()).digest(), qid),
        )
        self.assertEqual(
            [row["qid"] for row in salted_order(rows, salt="salt")], expected
        )

    def test_freeze_writes_question_only_splits_and_aggregate_manifest(self) -> None:
        rows = [
            {"qid": "demo-a", "question": "one"},
            {"qid": "demo-b", "question": "two"},
            {"qid": "demo-c", "question": "three"},
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            questions = root / "questions.jsonl"
            write_jsonl(questions, rows)
            manifest = freeze_split(
                questions_path=questions,
                salt="demo-salt",
                selection_count=2,
                selected_path=root / "selected.jsonl",
                remainder_path=root / "remainder.jsonl",
                manifest_path=root / "manifest.json",
            )
            self.assertEqual(len(read_jsonl(root / "selected.jsonl")), 2)
            self.assertEqual(manifest["qrels_or_answers_opened"], False)
            self.assertNotIn("qids", manifest)

    def test_rejects_answer_bearing_rows(self) -> None:
        with self.assertRaises(ValueError):
            salted_order([{"qid": "demo", "question": "q", "answer": "a"}], salt="salt")


if __name__ == "__main__":
    unittest.main()
