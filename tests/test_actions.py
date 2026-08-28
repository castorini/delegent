from __future__ import annotations

import json
import unittest

from delegent.actions import ActionValidationError, parse_action


class ActionTests(unittest.TestCase):
    def test_accepts_only_plan_and_query_and_normalizes_whitespace(self) -> None:
        action = parse_action(
            '{"plan":" find   another source ","query":" Lumen   launch announcement "}',
            question="Where was Lumen launched?",
            prior_queries=["Lumen project"],
        )
        self.assertEqual(action.plan, "find another source")
        self.assertEqual(action.query, "Lumen launch announcement")

    def test_rejects_markdown_extra_fields_and_non_strings(self) -> None:
        bad = [
            '```json\n{"plan":"p","query":"q"}\n```',
            '{"plan":"p","query":"q","answer":"x"}',
            '{"plan":1,"query":"q"}',
        ]
        for value in bad:
            with self.subTest(value=value), self.assertRaises(ActionValidationError):
                parse_action(value, question="question")

    def test_rejects_duplicate_json_keys(self) -> None:
        with self.assertRaisesRegex(ActionValidationError, "duplicate JSON key"):
            parse_action(
                '{"plan":"first","plan":"second","query":"novel query"}',
                question="question",
            )

    def test_rejects_question_and_prior_query_duplicates_case_insensitively(
        self,
    ) -> None:
        for query in ("  THE QUESTION ", "Prior Query"):
            with self.subTest(query=query), self.assertRaises(ActionValidationError):
                parse_action(
                    json.dumps({"plan": "p", "query": query}),
                    question="the question",
                    prior_queries=["prior query"],
                )


if __name__ == "__main__":
    unittest.main()
