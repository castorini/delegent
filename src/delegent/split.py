"""Question-only, salted SHA-256 split freezing."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .actions import normalize_text
from .io import canonical_sha256, read_jsonl, sha256_file, write_json, write_jsonl

FORBIDDEN_QUESTION_FIELDS = frozenset(
    {
        "answer",
        "answers",
        "gold",
        "qrel",
        "qrels",
        "positive_docids",
        "doc_fact_ids",
        "fact_ids",
    }
)


def validate_question_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("question input is empty")
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if set(row) != {"qid", "question"}:
            forbidden = FORBIDDEN_QUESTION_FIELDS.intersection(
                str(key).casefold() for key in row
            )
            if forbidden:
                raise ValueError(
                    f"question row {index} contains evaluation-only fields"
                )
            raise ValueError(
                f"question row {index} must contain exactly qid and question"
            )
        qid = str(row["qid"])
        if not qid or qid in seen or not normalize_text(row["question"]):
            raise ValueError(
                f"question row {index} has empty/duplicate qid or empty question"
            )
        seen.add(qid)


def salted_order(
    rows: Sequence[Mapping[str, Any]], *, salt: str
) -> list[dict[str, str]]:
    """Sort by ``sha256(salt + NUL + qid)`` with qid as a collision tie-break."""

    if not salt:
        raise ValueError("salt must be non-empty")
    validate_question_rows(rows)
    normalized = [
        {"qid": str(row["qid"]), "question": normalize_text(row["question"])}
        for row in rows
    ]
    return sorted(
        normalized,
        key=lambda row: (
            hashlib.sha256((salt + "\0" + row["qid"]).encode("utf-8")).digest(),
            row["qid"],
        ),
    )


def freeze_split(
    *,
    questions_path: str | Path,
    salt: str,
    selection_count: int,
    selected_path: str | Path,
    remainder_path: str | Path,
    manifest_path: str | Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    rows = read_jsonl(questions_path)
    ordered = salted_order(rows, salt=salt)
    if not 0 < selection_count < len(ordered):
        raise ValueError("selection_count must be between one and question_count - 1")
    selected = ordered[:selection_count]
    remainder = ordered[selection_count:]
    write_jsonl(selected_path, selected, overwrite=overwrite)
    write_jsonl(remainder_path, remainder, overwrite=overwrite)
    manifest = {
        "schema_version": "delegent-salted-split-manifest-v1",
        "status": "frozen",
        "algorithm": "ascending sha256(salt + NUL + qid), qid collision tie-break",
        "salt": salt,
        "question_count": len(ordered),
        "selection_count": len(selected),
        "remainder_count": len(remainder),
        "input_sha256": sha256_file(questions_path),
        "selected_sha256": sha256_file(selected_path),
        "remainder_sha256": sha256_file(remainder_path),
        "selected_order_sha256": canonical_sha256([row["qid"] for row in selected]),
        "remainder_order_sha256": canonical_sha256([row["qid"] for row in remainder]),
        "row_keys_exact": ["qid", "question"],
        "qrels_or_answers_opened": False,
    }
    write_json(manifest_path, manifest, overwrite=overwrite)
    return manifest
