from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from delegent.io import write_jsonl

ROOT = Path(__file__).resolve().parents[1]


def run_script(*arguments: str) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT / "src")
    return subprocess.run(
        [sys.executable, *arguments],
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )


class CliTests(unittest.TestCase):
    def test_standard_library_demo_dry_run_makes_no_calls_or_writes(self) -> None:
        demo_output = ROOT / "examples" / "demo-output"
        self.assertFalse(demo_output.exists())
        completed = run_script(
            "scripts/run_routed_family.py",
            "--config",
            "examples/demo_config.json",
            "--dry-run",
        )
        result = json.loads(completed.stdout)
        self.assertEqual(result["question_count"], 2)
        self.assertEqual(result["external_calls"], 0)
        self.assertEqual(result["files_written"], 0)
        self.assertFalse(demo_output.exists())

    def test_derive_and_freeze_split_clis(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            derived = run_script(
                "scripts/derive_prefixes.py",
                "--source-records",
                "examples/demo_source.jsonl",
                "--output-dir",
                str(root / "derived"),
                "--caps",
                "0",
                "1",
            )
            self.assertEqual(json.loads(derived.stdout)["model_calls"], 0)
            frozen = run_script(
                "scripts/freeze_split.py",
                "--questions",
                "examples/demo_questions.jsonl",
                "--salt",
                "demo-salt",
                "--selection-count",
                "1",
                "--selected",
                str(root / "selected.jsonl"),
                "--remainder",
                str(root / "remainder.jsonl"),
                "--manifest",
                str(root / "manifest.json"),
            )
            self.assertEqual(json.loads(frozen.stdout)["selection_count"], 1)

            qrels = root / "synthetic-qrels.jsonl"
            write_jsonl(
                qrels,
                [
                    {"qid": "demo-route-001", "positive_docids": ["demo-doc-b"]},
                    {"qid": "demo-route-002", "positive_docids": ["demo-doc-e"]},
                ],
            )
            scored = run_script(
                "scripts/score_squad.py",
                "--records",
                str(root / "derived" / "frontier1" / "records"),
                "--qrels",
                str(qrels),
                "--output",
                str(root / "score.json"),
            )
            score = json.loads(scored.stdout)
            self.assertTrue(score["completion_validated_before_qrels_opened"])
            self.assertEqual(score["question_count"], 2)

            costed = run_script(
                "scripts/summarize_cost.py",
                "--records",
                str(root / "derived" / "frontier1" / "records"),
                "--frontier-actor",
                "frontier",
                "--output",
                str(root / "cost.json"),
            )
            cost = json.loads(costed.stdout)
            self.assertTrue(cost["completion_validated_before_accounting"])
            self.assertEqual(cost["token_totals"]["input_tokens"], 23)
            self.assertEqual(cost["token_totals"]["output_tokens"], 15)
            self.assertEqual(cost["cost_usd"]["total"], 0.000392)
            self.assertNotIn("qid", json.dumps(cost).casefold())

    def test_every_public_script_has_help(self) -> None:
        for script in (
            "run_routed_family.py",
            "derive_prefixes.py",
            "freeze_split.py",
            "score_hotpotqa.py",
            "score_squad.py",
            "summarize_cost.py",
        ):
            with self.subTest(script=script):
                completed = run_script(f"scripts/{script}", "--help")
                self.assertIn("usage:", completed.stdout)


if __name__ == "__main__":
    unittest.main()
