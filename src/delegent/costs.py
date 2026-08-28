"""Completion-gated, aggregate-only frontier token cost accounting."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .io import (
    DataValidationError,
    load_records,
    read_json,
    record_set_sha256,
    sha256_file,
)


@dataclass(frozen=True)
class CostRates:
    """USD prices per million tokens.

    ``input_tokens`` is interpreted as the total input count, including cached
    input. Reasoning tokens are a diagnostic subset of output tokens and are
    never billed a second time.
    """

    uncached_input: float = 4.00
    cached_input: float = 0.40
    output: float = 20.00

    def __post_init__(self) -> None:
        for name, value in (
            ("uncached_input", self.uncached_input),
            ("cached_input", self.cached_input),
            ("output", self.output),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} rate must be a finite non-negative number")
            if not math.isfinite(float(value)) or value < 0:
                raise ValueError(f"{name} rate must be a finite non-negative number")

    def as_dict(self) -> dict[str, float]:
        return {
            "uncached_input": float(self.uncached_input),
            "cached_input": float(self.cached_input),
            "output": float(self.output),
        }


DEFAULT_COST_RATES = CostRates()
_MISSING = object()


def _record_input_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_dir() and (candidate / "records").is_dir():
        return candidate / "records"
    return candidate


def _default_completion_path(records_path: Path) -> Path:
    if records_path.is_dir():
        return (
            records_path.parent / "completion.json"
            if records_path.name == "records"
            else records_path / "completion.json"
        )
    return records_path.with_name(records_path.stem + ".completion.json")


def _validate_records(records: Sequence[Mapping[str, Any]]) -> str:
    if not records:
        raise DataValidationError("record input is empty")
    arms: set[str] = set()
    for record_index, record in enumerate(records):
        arm = record.get("arm")
        generations = record.get("generations")
        if not isinstance(arm, str) or not arm:
            raise DataValidationError(f"record {record_index} requires a non-empty arm")
        if not isinstance(generations, list):
            raise DataValidationError(
                f"record {record_index} generations must be an array"
            )
        arms.add(arm)
    if len(arms) != 1:
        raise DataValidationError("one cost input must contain exactly one arm")
    return next(iter(arms))


def _completion_gate(
    *,
    records: Sequence[Mapping[str, Any]],
    arm: str,
    records_path: Path,
    completion_path: str | Path | None,
) -> dict[str, Any]:
    path = (
        Path(completion_path)
        if completion_path is not None
        else _default_completion_path(records_path)
    )
    if not path.is_file():
        raise DataValidationError(
            "record-set completion is required before cost accounting; pass "
            "--completion or place completion.json beside records/"
        )
    completion = read_json(path)
    if not isinstance(completion, Mapping):
        raise DataValidationError("record-set completion must be an object")
    expected = {
        "schema_version": "delegent-record-set-completion-v1",
        "status": "complete",
        "arm": arm,
        "question_count": len(records),
        "record_set_sha256": record_set_sha256(records),
    }
    for key, expected_value in expected.items():
        if completion.get(key) != expected_value:
            raise DataValidationError(f"record-set completion field drifted: {key}")
    return {"sha256": sha256_file(path), "content": dict(completion)}


def _details(usage: Mapping[str, Any], key: str) -> Mapping[str, Any] | None:
    if key not in usage:
        return None
    value = usage[key]
    if not isinstance(value, Mapping):
        raise DataValidationError(f"usage.{key} must be an object")
    return value


def _counter(
    candidates: Sequence[tuple[str, Any]],
    *,
    name: str,
    required: bool,
    default: int = 0,
) -> int:
    present: list[tuple[str, int]] = []
    for location, value in candidates:
        if value is _MISSING:
            continue
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise DataValidationError(f"{location} must be a non-negative integer")
        present.append((location, value))
    if not present:
        if required:
            raise DataValidationError(f"frontier usage requires {name}")
        return default
    values = {value for _, value in present}
    if len(values) != 1:
        locations = ", ".join(location for location, _ in present)
        raise DataValidationError(f"conflicting {name} telemetry across {locations}")
    return present[0][1]


def _usage_counts(generation: Mapping[str, Any]) -> dict[str, int]:
    usage = generation.get("usage")
    if not isinstance(usage, Mapping):
        raise DataValidationError("frontier generation requires an object usage field")

    input_details = _details(usage, "input_tokens_details")
    prompt_details = _details(usage, "prompt_tokens_details")
    output_details = _details(usage, "output_tokens_details")
    completion_details = _details(usage, "completion_tokens_details")

    input_tokens = _counter(
        (
            ("usage.input_tokens", usage.get("input_tokens", _MISSING)),
            ("usage.prompt_tokens", usage.get("prompt_tokens", _MISSING)),
        ),
        name="input_tokens",
        required=True,
    )
    cached_input_tokens = _counter(
        (
            (
                "usage.cached_input_tokens",
                usage.get("cached_input_tokens", _MISSING),
            ),
            (
                "usage.cache_read_input_tokens",
                usage.get("cache_read_input_tokens", _MISSING),
            ),
            (
                "usage.input_tokens_details.cached_tokens",
                input_details.get("cached_tokens", _MISSING)
                if input_details is not None
                else _MISSING,
            ),
            (
                "usage.prompt_tokens_details.cached_tokens",
                prompt_details.get("cached_tokens", _MISSING)
                if prompt_details is not None
                else _MISSING,
            ),
        ),
        name="cached_input_tokens",
        required=False,
    )
    cache_write_input_tokens = _counter(
        (
            (
                "usage.cache_write_input_tokens",
                usage.get("cache_write_input_tokens", _MISSING),
            ),
            (
                "usage.cache_creation_input_tokens",
                usage.get("cache_creation_input_tokens", _MISSING),
            ),
            (
                "usage.input_tokens_details.cache_write_tokens",
                input_details.get("cache_write_tokens", _MISSING)
                if input_details is not None
                else _MISSING,
            ),
            (
                "usage.prompt_tokens_details.cache_write_tokens",
                prompt_details.get("cache_write_tokens", _MISSING)
                if prompt_details is not None
                else _MISSING,
            ),
        ),
        name="cache_write_input_tokens",
        required=False,
    )
    output_tokens = _counter(
        (
            ("usage.output_tokens", usage.get("output_tokens", _MISSING)),
            (
                "usage.completion_tokens",
                usage.get("completion_tokens", _MISSING),
            ),
        ),
        name="output_tokens",
        required=True,
    )
    reasoning_output_tokens = _counter(
        (
            (
                "usage.reasoning_output_tokens",
                usage.get("reasoning_output_tokens", _MISSING),
            ),
            (
                "usage.output_tokens_details.reasoning_tokens",
                output_details.get("reasoning_tokens", _MISSING)
                if output_details is not None
                else _MISSING,
            ),
            (
                "usage.completion_tokens_details.reasoning_tokens",
                completion_details.get("reasoning_tokens", _MISSING)
                if completion_details is not None
                else _MISSING,
            ),
        ),
        name="reasoning_output_tokens",
        required=False,
    )
    if cached_input_tokens > input_tokens:
        raise DataValidationError("cached input tokens exceed total input tokens")
    if reasoning_output_tokens > output_tokens:
        raise DataValidationError("reasoning output tokens exceed total output tokens")
    return {
        "input_tokens": input_tokens,
        "cached_input_tokens": cached_input_tokens,
        "cache_write_input_tokens": cache_write_input_tokens,
        "uncached_input_tokens": input_tokens - cached_input_tokens,
        "output_tokens": output_tokens,
        "reasoning_output_tokens": reasoning_output_tokens,
    }


def _attempts(generation: Mapping[str, Any]) -> int:
    value = generation.get("attempts", 1)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise DataValidationError(
            "frontier generation attempts must be a positive integer"
        )
    return value


def _usd(tokens: int, rate: float) -> float:
    return round(tokens * float(rate) / 1_000_000, 12)


def summarize_frontier_cost(
    *,
    records_path: str | Path,
    frontier_actor: str,
    completion_path: str | Path | None = None,
    rates: CostRates = DEFAULT_COST_RATES,
) -> dict[str, Any]:
    """Summarize frontier token telemetry after validating record-set completion."""

    if not isinstance(frontier_actor, str) or not frontier_actor:
        raise ValueError("frontier_actor must be a non-empty string")
    if not isinstance(rates, CostRates):
        raise TypeError("rates must be a CostRates instance")

    resolved_records_path = _record_input_path(records_path)
    records = load_records(resolved_records_path)
    arm = _validate_records(records)
    completion = _completion_gate(
        records=records,
        arm=arm,
        records_path=resolved_records_path,
        completion_path=completion_path,
    )

    totals = {
        "input_tokens": 0,
        "cached_input_tokens": 0,
        "cache_write_input_tokens": 0,
        "uncached_input_tokens": 0,
        "output_tokens": 0,
        "reasoning_output_tokens": 0,
    }
    frontier_generations = 0
    frontier_calls = 0
    for record_index, record in enumerate(records):
        for generation_index, generation in enumerate(record["generations"]):
            if not isinstance(generation, Mapping):
                raise DataValidationError(
                    f"record {record_index} generation {generation_index} must be an object"
                )
            actor = generation.get("actor")
            if not isinstance(actor, str) or not actor:
                raise DataValidationError(
                    f"record {record_index} generation {generation_index} requires a non-empty actor"
                )
            if actor != frontier_actor:
                continue
            counts = _usage_counts(generation)
            for name, value in counts.items():
                totals[name] += value
            frontier_calls += _attempts(generation)
            frontier_generations += 1

    rate_values = rates.as_dict()
    cost_components = {
        "uncached_input": _usd(
            totals["uncached_input_tokens"], rate_values["uncached_input"]
        ),
        "cached_input": _usd(
            totals["cached_input_tokens"], rate_values["cached_input"]
        ),
        "output": _usd(totals["output_tokens"], rate_values["output"]),
    }
    total_cost = round(sum(cost_components.values()), 12)
    return {
        "schema_version": "delegent-frontier-cost-summary-v1",
        "status": "complete",
        "arm": arm,
        "question_count": len(records),
        "frontier_actor": frontier_actor,
        "frontier_generations": frontier_generations,
        "frontier_calls": frontier_calls,
        "token_totals": totals,
        "rates_usd_per_million_tokens": rate_values,
        "cost_usd": {
            **cost_components,
            "total": total_cost,
            "mean_per_question": round(total_cost / len(records), 12),
        },
        "input_tokens_include_cached_tokens": True,
        "reasoning_tokens_included_in_output_tokens": True,
        "reasoning_tokens_billed_separately": False,
        "record_set_sha256": record_set_sha256(records),
        "completion_sha256": completion["sha256"],
        "completion_validated_before_accounting": True,
        "per_question_outputs_included": False,
    }
