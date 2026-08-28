"""Aggregate-only scoring for completed HotpotQA and SQuAD trajectories."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .bootstrap import deterministic_seed, paired_bootstrap
from .io import (
    DataValidationError,
    load_records,
    read_json,
    read_jsonl,
    record_set_sha256,
    sha256_file,
)
from .metrics import QuestionQrels, hotpotqa_metrics, macro_average, squad_metrics

MetricFunction = Callable[
    [Sequence[Mapping[str, Any]], QuestionQrels], dict[str, float]
]


def load_qrels(path: str | Path, *, require_facts: bool) -> dict[str, QuestionQrels]:
    """Load one synthetic/public qrels object per question.

    Each JSONL row has ``qid``, ``positive_docids``, and, for HotpotQA,
    ``doc_fact_ids`` mapping document IDs to lists of required-fact IDs.
    """

    rows = read_jsonl(path)
    result: dict[str, QuestionQrels] = {}
    allowed = {"qid", "positive_docids", "doc_fact_ids"}
    for line_index, row in enumerate(rows, start=1):
        if set(row) - allowed:
            raise DataValidationError(f"qrels row {line_index} contains unknown fields")
        qid = str(row.get("qid") or "")
        positives_value = row.get("positive_docids")
        if not qid or not isinstance(positives_value, list):
            raise DataValidationError(
                f"qrels row {line_index} requires qid and positive_docids"
            )
        positives = frozenset(str(item) for item in positives_value if str(item))
        if len(positives) != len(positives_value) or not positives:
            raise DataValidationError(
                f"qrels row {line_index} has empty/duplicate positive_docids"
            )
        mapping_value = row.get("doc_fact_ids", {})
        if not isinstance(mapping_value, Mapping):
            raise DataValidationError(
                f"qrels row {line_index} doc_fact_ids must be an object"
            )
        doc_fact_ids: dict[str, frozenset[str]] = {}
        for docid, facts_value in mapping_value.items():
            if not isinstance(facts_value, list):
                raise DataValidationError(
                    f"qrels row {line_index} fact lists must be arrays"
                )
            facts = frozenset(str(fact) for fact in facts_value if str(fact))
            if len(facts) != len(facts_value) or not facts:
                raise DataValidationError(
                    f"qrels row {line_index} has empty/duplicate fact IDs"
                )
            if str(docid) not in positives:
                raise DataValidationError(
                    f"qrels row {line_index} maps facts for a non-positive document"
                )
            doc_fact_ids[str(docid)] = facts
        qrels = QuestionQrels(positive_docids=positives, doc_fact_ids=doc_fact_ids)
        if require_facts and not qrels.fact_ids:
            raise DataValidationError(f"qrels row {line_index} has no required facts")
        if qid in result:
            raise DataValidationError(f"duplicate qrels qid: {qid}")
        result[qid] = qrels
    if not result:
        raise DataValidationError("qrels input is empty")
    return result


def _validate_records(
    records: Sequence[Mapping[str, Any]],
) -> tuple[str, dict[str, Mapping[str, Any]]]:
    if not records:
        raise DataValidationError("record input is empty")
    by_qid: dict[str, Mapping[str, Any]] = {}
    arms: set[str] = set()
    for record in records:
        qid = str(record.get("qid") or "")
        arm = str(record.get("arm") or "")
        searches = record.get("searches")
        if not qid or not arm or not isinstance(searches, list) or not searches:
            raise DataValidationError(
                "each scored record requires qid, arm, and non-empty searches"
            )
        if qid in by_qid:
            raise DataValidationError(f"duplicate record qid: {qid}")
        by_qid[qid] = record
        arms.add(arm)
    if len(arms) != 1:
        raise DataValidationError("one scoring input must contain exactly one arm")
    return next(iter(arms)), by_qid


def _metric_rows(
    records: Sequence[Mapping[str, Any]],
    qrels: Mapping[str, QuestionQrels],
    *,
    metric_function: MetricFunction,
) -> tuple[str, dict[str, dict[str, float]]]:
    arm, by_qid = _validate_records(records)
    if set(by_qid) != set(qrels):
        missing = len(set(qrels) - set(by_qid))
        extra = len(set(by_qid) - set(qrels))
        raise DataValidationError(
            f"record/qrels qid sets differ: missing={missing}, extra={extra}"
        )
    rows = {
        qid: metric_function(by_qid[qid]["searches"], qrels[qid])
        for qid in sorted(by_qid)
    }
    return arm, rows


def _record_input_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_dir() and (candidate / "records").is_dir():
        return candidate / "records"
    return candidate


def _default_completion_path(records_path: Path) -> Path | None:
    if records_path.is_dir():
        candidate = (
            records_path.parent / "completion.json"
            if records_path.name == "records"
            else records_path / "completion.json"
        )
        return candidate
    candidate = records_path.with_name(records_path.stem + ".completion.json")
    return candidate


def _completion_gate(
    *,
    records: Sequence[Mapping[str, Any]],
    arm: str,
    records_path: Path,
    completion_path: str | Path | None,
    required: bool,
) -> dict[str, Any] | None:
    path = (
        Path(completion_path)
        if completion_path is not None
        else _default_completion_path(records_path)
    )
    if path is None or not path.is_file():
        if required:
            raise DataValidationError(
                "record-set completion is required before qrels may be opened; pass --completion or place completion.json beside records/"
            )
        return None
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
    for key, value in expected.items():
        if completion.get(key) != value:
            raise DataValidationError(f"record-set completion field drifted: {key}")
    return {"sha256": sha256_file(path), "content": dict(completion)}


def _comparison(
    reference_arm: str,
    candidate_arm: str,
    reference: Mapping[str, Mapping[str, float]],
    candidate: Mapping[str, Mapping[str, float]],
    *,
    repetitions: int,
    seed: int | None,
) -> dict[str, Any]:
    if set(reference) != set(candidate):
        raise DataValidationError("reference/candidate qid sets differ")
    qids = sorted(reference)
    metric_names = sorted(next(iter(reference.values())))
    result: dict[str, Any] = {}
    for metric_index, metric in enumerate(metric_names):
        metric_seed = (
            deterministic_seed(reference_arm, candidate_arm, metric)
            if seed is None
            else deterministic_seed(str(seed), str(metric_index), metric)
        )
        summary = paired_bootstrap(
            [reference[qid][metric] for qid in qids],
            [candidate[qid][metric] for qid in qids],
            repetitions=repetitions,
            seed=metric_seed,
        )
        result[metric] = {
            **summary,
            "reference_percent": summary["reference_mean"] * 100,
            "candidate_percent": summary["candidate_mean"] * 100,
            "difference_pp": summary["mean_delta"] * 100,
            "paired_bootstrap_95ci_pp": [
                value * 100 for value in summary["paired_bootstrap_95ci"]
            ],
        }
    return result


def score_records(
    *,
    task: str,
    records_path: str | Path,
    qrels_path: str | Path,
    reference_records_path: str | Path | None = None,
    completion_path: str | Path | None = None,
    reference_completion_path: str | Path | None = None,
    require_completion: bool = True,
    bootstrap_repetitions: int = 10_000,
    seed: int | None = None,
) -> dict[str, Any]:
    if task == "hotpotqa":
        metric_function: MetricFunction = hotpotqa_metrics
        schema_version = "delegent-hotpotqa-score-v1"
        require_facts = True
    elif task == "squad":
        metric_function = squad_metrics
        schema_version = "delegent-squad-score-v1"
        require_facts = False
    else:
        raise ValueError("task must be hotpotqa or squad")
    resolved_records_path = _record_input_path(records_path)
    records = load_records(resolved_records_path)
    arm, _ = _validate_records(records)
    completion = _completion_gate(
        records=records,
        arm=arm,
        records_path=resolved_records_path,
        completion_path=completion_path,
        required=require_completion,
    )
    reference_records: list[dict[str, Any]] | None = None
    reference_arm: str | None = None
    reference_completion: dict[str, Any] | None = None
    if reference_records_path is not None:
        resolved_reference_path = _record_input_path(reference_records_path)
        reference_records = load_records(resolved_reference_path)
        reference_arm, _ = _validate_records(reference_records)
        reference_completion = _completion_gate(
            records=reference_records,
            arm=reference_arm,
            records_path=resolved_reference_path,
            completion_path=reference_completion_path,
            required=require_completion,
        )
    # The label file is deliberately opened only after every requested record set
    # passes its completion and content-identity gate.
    qrels = load_qrels(qrels_path, require_facts=require_facts)
    arm, rows = _metric_rows(records, qrels, metric_function=metric_function)
    aggregate = macro_average(list(rows.values()))
    result: dict[str, Any] = {
        "schema_version": schema_version,
        "status": "complete",
        "task": task,
        "arm": arm,
        "question_count": len(rows),
        "record_set_sha256": record_set_sha256(records),
        "qrels_sha256": sha256_file(qrels_path),
        "metrics": {
            name: {"mean": value, "percent": value * 100}
            for name, value in aggregate.items()
        },
        "per_question_outputs_included": False,
        "completion_sha256": completion["sha256"] if completion else None,
        "completion_validated_before_qrels_opened": completion is not None,
    }
    if reference_records is not None and reference_arm is not None:
        _, reference_rows = _metric_rows(
            reference_records,
            qrels,
            metric_function=metric_function,
        )
        result["reference_arm"] = reference_arm
        result["reference_record_set_sha256"] = record_set_sha256(reference_records)
        result["reference_completion_sha256"] = (
            reference_completion["sha256"] if reference_completion else None
        )
        result["comparisons"] = _comparison(
            reference_arm,
            arm,
            reference_rows,
            rows,
            repetitions=bootstrap_repetitions,
            seed=seed,
        )
    return result
