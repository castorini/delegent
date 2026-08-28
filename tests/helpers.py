from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def demo_sources() -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in (ROOT / "examples" / "demo_source.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
