#!/usr/bin/env python3
"""Freeze a question-only split with ascending salted SHA-256 ordering."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from delegent.split import freeze_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--questions", type=Path, required=True, help="Question-only JSONL"
    )
    parser.add_argument(
        "--salt", required=True, help="Public deterministic protocol salt"
    )
    parser.add_argument("--selection-count", type=int, required=True)
    parser.add_argument("--selected", type=Path, required=True)
    parser.add_argument("--remainder", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = freeze_split(
        questions_path=args.questions,
        salt=args.salt,
        selection_count=args.selection_count,
        selected_path=args.selected,
        remainder_path=args.remainder,
        manifest_path=args.manifest,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
