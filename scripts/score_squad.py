#!/usr/bin/env python3
"""Score SQuAD union-any/RRF retrieval metrics with optional paired bootstrap."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from delegent.io import write_json
from delegent.scoring import score_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--qrels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-records", type=Path)
    parser.add_argument(
        "--completion",
        type=Path,
        help="Record-set completion (otherwise inferred beside records/)",
    )
    parser.add_argument("--reference-completion", type=Path)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10_000)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = score_records(
        task="squad",
        records_path=args.records,
        qrels_path=args.qrels,
        reference_records_path=args.reference_records,
        completion_path=args.completion,
        reference_completion_path=args.reference_completion,
        bootstrap_repetitions=args.bootstrap_repetitions,
        seed=args.seed,
    )
    write_json(args.output, result, overwrite=args.overwrite)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
