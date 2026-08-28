#!/usr/bin/env python3
"""Summarize completed frontier token usage and USD cost without QIDs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from delegent.costs import CostRates, summarize_frontier_cost
from delegent.io import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--frontier-actor", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--completion",
        type=Path,
        help="Record-set completion (otherwise inferred beside records/)",
    )
    parser.add_argument(
        "--uncached-input-rate",
        type=float,
        default=4.00,
        help="USD per million uncached input tokens (default: 4.00)",
    )
    parser.add_argument(
        "--cached-input-rate",
        type=float,
        default=0.40,
        help="USD per million cached input tokens (default: 0.40)",
    )
    parser.add_argument(
        "--output-rate",
        type=float,
        default=20.00,
        help="USD per million output tokens (default: 20.00)",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = summarize_frontier_cost(
        records_path=args.records,
        frontier_actor=args.frontier_actor,
        completion_path=args.completion,
        rates=CostRates(
            uncached_input=args.uncached_input_rate,
            cached_input=args.cached_input_rate,
            output=args.output_rate,
        ),
    )
    write_json(args.output, result, overwrite=args.overwrite)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
