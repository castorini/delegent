#!/usr/bin/env python3
"""Generate a fixed-schedule routed family from completed all-frontier records."""

from __future__ import annotations

import argparse
import json

from delegent.routing import dry_run_family, load_family_config, run_family


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", required=True, help="Path to delegent-routed-family-config-v1 JSON"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate records/schedules and hash first delegate prompts; make no calls or writes",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse only existing records whose source and schedule identities match",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.dry_run and args.resume:
        raise SystemExit("--resume has no meaning with --dry-run")
    config = load_family_config(args.config, dry_run=args.dry_run)
    result = (
        dry_run_family(config)
        if args.dry_run
        else run_family(config, resume=args.resume)
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
