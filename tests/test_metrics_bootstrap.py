from __future__ import annotations

import unittest

from delegent.bootstrap import paired_bootstrap
from delegent.metrics import (
    QuestionQrels,
    hotpotqa_metrics,
    rrf_docids,
    squad_metrics,
    union_docids,
)


class MetricTests(unittest.TestCase):
    def setUp(self) -> None:
        self.searches = [
            {"hits": [{"docid": "a"}, {"docid": "b"}]},
            {"hits": [{"docid": "b"}, {"docid": "a"}, {"docid": "c"}]},
        ]

    def test_union_and_rrf_are_deterministic(self) -> None:
        self.assertEqual(union_docids(self.searches), ["a", "b", "c"])
        self.assertEqual(rrf_docids(self.searches), ["a", "b", "c"])

    def test_hotpotqa_union_and_fact_metrics(self) -> None:
        qrels = QuestionQrels(
            positive_docids=frozenset({"a", "c", "missing"}),
            doc_fact_ids={
                "a": frozenset({"f1"}),
                "c": frozenset({"f2"}),
                "missing": frozenset({"f3"}),
            },
        )
        metrics = hotpotqa_metrics(self.searches, qrels)
        self.assertAlmostEqual(metrics["union_required_fact_coverage"], 2 / 3)
        self.assertAlmostEqual(metrics["union_positive_doc_recall"], 2 / 3)

    def test_squad_union_any_and_ndcg(self) -> None:
        qrels = QuestionQrels(frozenset({"c", "missing"}), {})
        metrics = squad_metrics(self.searches, qrels)
        self.assertEqual(metrics["union_any_positive"], 1.0)
        self.assertEqual(metrics["rrf20_any_positive"], 1.0)
        self.assertAlmostEqual(metrics["union_positive_doc_recall"], 0.5)


class BootstrapTests(unittest.TestCase):
    def test_paired_bootstrap_is_seeded_and_reports_direction_counts(self) -> None:
        first = paired_bootstrap([0, 1, 0], [1, 1, 0], repetitions=200, seed=7)
        second = paired_bootstrap([0, 1, 0], [1, 1, 0], repetitions=200, seed=7)
        self.assertEqual(first, second)
        self.assertAlmostEqual(first["mean_delta"], 1 / 3)
        self.assertEqual(
            first["worse_equal_better"], {"worse": 0, "equal": 2, "better": 1}
        )


if __name__ == "__main__":
    unittest.main()
