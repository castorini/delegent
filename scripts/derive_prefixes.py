#!/usr/bin/env python3
"""Zero-call derive exact all-frontier prefixes from a longer trajectory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from delegent.derive import load_and_derive


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-records", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--caps", type=int, nargs="+", required=True)
    parser.add_argument("--frontier-actor", default="frontier")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = load_and_derive(
        args.source_records,
        caps=args.caps,
        output_dir=args.output_dir,
        frontier_actor=args.frontier_actor,
        resume=args.resume,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
