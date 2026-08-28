"""Small, deterministic JSON helpers shared by Delegent commands."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any
from urllib.parse import quote


class DataValidationError(ValueError):
    """Raised when an input artifact does not satisfy its public contract."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise DataValidationError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def canonical_json_bytes(value: object) -> bytes:
    """Serialize JSON deterministically, rejecting non-finite numbers."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: str | Path) -> Any:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle, object_pairs_hook=_reject_duplicate_keys)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line, object_pairs_hook=_reject_duplicate_keys)
            except json.JSONDecodeError as exc:
                raise DataValidationError(
                    f"invalid JSON on line {line_number}: {exc.msg}"
                ) from exc
            if not isinstance(row, dict):
                raise DataValidationError(f"JSONL line {line_number} must be an object")
            rows.append(row)
    return rows


def load_records(path: str | Path) -> list[dict[str, Any]]:
    """Load records from JSONL, one JSON object, or a directory of JSON files."""

    source = Path(path)
    if source.is_dir():
        rows = []
        for record_path in sorted(source.glob("*.json")):
            row = read_json(record_path)
            if not isinstance(row, dict):
                raise DataValidationError(
                    f"record is not an object: {record_path.name}"
                )
            rows.append(row)
    elif source.suffix == ".jsonl":
        rows = read_jsonl(source)
    else:
        value = read_json(source)
        if isinstance(value, list):
            rows = value
        elif isinstance(value, dict) and isinstance(value.get("records"), list):
            rows = value["records"]
        elif isinstance(value, dict):
            rows = [value]
        else:
            raise DataValidationError(
                "record input must be an object, list, or {records: [...]} object"
            )
        if any(not isinstance(row, dict) for row in rows):
            raise DataValidationError("every record must be an object")
    seen: set[str] = set()
    for row in rows:
        qid = str(row.get("qid") or "")
        if not qid:
            raise DataValidationError("every record requires a non-empty qid")
        if qid in seen:
            raise DataValidationError(f"duplicate qid: {qid}")
        seen.add(qid)
    return rows


def record_set_sha256(records: Iterable[Mapping[str, Any]]) -> str:
    """Hash a set of records by qid and canonical record content."""

    identities: dict[str, str] = {}
    for record in records:
        qid = str(record.get("qid") or "")
        if not qid or qid in identities:
            raise DataValidationError(
                "record-set identity requires unique non-empty qids"
            )
        identities[qid] = canonical_sha256(dict(record))
    if not identities:
        raise DataValidationError("cannot identify an empty record set")
    return canonical_sha256(identities)


def _json_with_newline(value: object, *, pretty: bool) -> bytes:
    if pretty:
        text = json.dumps(
            value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
        )
    else:
        text = canonical_json_bytes(value).decode("utf-8")
    return (text + "\n").encode("utf-8")


def write_json(path: str | Path, value: object, *, overwrite: bool = False) -> None:
    """Atomically write JSON, refusing to replace artifacts unless requested."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        raise FileExistsError(destination)
    payload = _json_with_newline(value, pretty=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if destination.exists() and not overwrite:
            raise FileExistsError(destination)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_jsonl(
    path: str | Path,
    rows: Iterable[Mapping[str, Any]],
    *,
    overwrite: bool = False,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        raise FileExistsError(destination)
    payload = b"".join(canonical_json_bytes(dict(row)) + b"\n" for row in rows)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if destination.exists() and not overwrite:
            raise FileExistsError(destination)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def qid_filename(qid: str) -> str:
    """Map an arbitrary identifier to a reversible, traversal-safe filename."""

    if not qid or qid in {".", ".."}:
        raise DataValidationError("qid must be non-empty and cannot be '.' or '..'")
    return quote(qid, safe="") + ".json"


def resolve_config_path(config_path: str | Path, value: object, *, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise DataValidationError(f"{field} must be a non-empty path string")
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = Path(config_path).resolve().parent / candidate
    return candidate.resolve()
