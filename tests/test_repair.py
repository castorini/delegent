from __future__ import annotations

import unittest

from delegent.repair import deterministic_repair


class RepairTests(unittest.TestCase):
    def test_uses_frequency_then_first_occurrence_for_novel_tokens(self) -> None:
        searches = [
            {
                "query": "Project Lumen",
                "hits": [
                    {
                        "text": "Northwind Exampleton Northwind launch Exampleton",
                        "docid": "d1",
                    }
                ],
            }
        ]
        action = deterministic_repair(
            question="Project Lumen", searches=searches, action_index=0
        )
        self.assertEqual(action.plan, "deterministic qrel-blind evidence repair")
        self.assertEqual(action.query, "Project Lumen northwind exampleton launch")

    def test_fallback_is_deterministic_and_novel(self) -> None:
        searches = [{"query": "Question history", "hits": []}]
        first = deterministic_repair(
            question="Question", searches=searches, action_index=2
        )
        second = deterministic_repair(
            question="Question", searches=searches, action_index=2
        )
        self.assertEqual(first, second)
        self.assertNotEqual(first.query.casefold(), "question history")


if __name__ == "__main__":
    unittest.main()
