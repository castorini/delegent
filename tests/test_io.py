from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from delegent.io import DataValidationError, read_json, read_jsonl


class JsonInputTests(unittest.TestCase):
    def test_duplicate_keys_are_rejected_in_json_and_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            json_path = root / "duplicate.json"
            jsonl_path = root / "duplicate.jsonl"
            payload = '{"qid":"first","qid":"second"}\n'
            json_path.write_text(payload, encoding="utf-8")
            jsonl_path.write_text(payload, encoding="utf-8")

            with self.assertRaisesRegex(DataValidationError, "duplicate JSON key"):
                read_json(json_path)
            with self.assertRaisesRegex(DataValidationError, "duplicate JSON key"):
                read_jsonl(jsonl_path)


if __name__ == "__main__":
    unittest.main()
