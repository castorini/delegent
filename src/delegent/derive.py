"""Zero-call derivation of exact all-frontier trajectory prefixes."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .io import (
    canonical_sha256,
    load_records,
    qid_filename,
    read_json,
    record_set_sha256,
    write_json,
)
from .routing import validate_source_record


def derive_prefix_record(
    source: Mapping[str, Any],
    *,
    cap: int,
    source_cap: int | None = None,
    frontier_actor: str = "frontier",
) -> dict[str, Any]:
    """Derive one shorter view using exactly ``[:cap+1]`` and ``[:cap]``."""

    searches = source.get("searches")
    generations = source.get("generations")
    inferred_cap = len(generations) if isinstance(generations, list) else -1
    full_cap = inferred_cap if source_cap is None else source_cap
    validate_source_record(source, cap=full_cap, frontier_actor=frontier_actor)
    if not 0 <= cap <= full_cap:
        raise ValueError("derived cap must be between zero and the source cap")
    derived_searches = copy.deepcopy(searches[: cap + 1])
    derived_generations = copy.deepcopy(generations[:cap])
    if (
        derived_searches != searches[: cap + 1]
        or derived_generations != generations[:cap]
    ):
        raise AssertionError("derived prefix differs from source slices")
    return {
        "schema_version": "delegent-derived-prefix-record-v1",
        "status": "complete",
        "qid": str(source["qid"]),
        "question": str(source["question"]),
        "arm": f"frontier{cap}",
        "selected_cap": cap,
        "source_cap": full_cap,
        "frontier_actor": frontier_actor,
        "fixed_direct_searches": 1,
        "source_record_sha256": canonical_sha256(source),
        "source_prefix_reused_exactly": True,
        "zero_model_calls_for_derivation": True,
        "generation_reads_qrels": False,
        "generation_reads_answers": False,
        "searches": derived_searches,
        "generations": derived_generations,
    }


def derive_prefix_family(
    sources: Sequence[Mapping[str, Any]],
    *,
    caps: Sequence[int],
    output_dir: str | Path,
    frontier_actor: str = "frontier",
    resume: bool = False,
) -> dict[str, Any]:
    if not sources:
        raise ValueError("at least one source record is required")
    if not caps or len(set(caps)) != len(caps):
        raise ValueError("caps must be a non-empty unique sequence")
    source_cap = len(sources[0].get("generations") or [])
    if any(len(source.get("generations") or []) != source_cap for source in sources):
        raise ValueError("all source records must have the same cap")
    destination_root = Path(output_dir)
    completed = 0
    reused = 0
    source_set_sha = record_set_sha256(sources)
    completion_identities: dict[str, str] = {}
    for cap in caps:
        arm_records: list[dict[str, Any]] = []
        for source in sources:
            destination = (
                destination_root
                / f"frontier{cap}"
                / "records"
                / qid_filename(str(source["qid"]))
            )
            derived = derive_prefix_record(
                source,
                cap=cap,
                source_cap=source_cap,
                frontier_actor=frontier_actor,
            )
            if destination.exists():
                if not resume:
                    raise FileExistsError(destination)
                existing = read_json(destination)
                if existing != derived:
                    raise ValueError(f"existing derived record differs: {destination}")
                arm_records.append(derived)
                reused += 1
                continue
            write_json(destination, derived)
            arm_records.append(derived)
            completed += 1
        completion = {
            "schema_version": "delegent-record-set-completion-v1",
            "status": "complete",
            "producer": "derived-prefix",
            "arm": f"frontier{cap}",
            "question_count": len(sources),
            "cap": cap,
            "record_set_sha256": record_set_sha256(arm_records),
            "source_record_set_sha256": source_set_sha,
            "model_calls": 0,
            "qrels_or_answers_opened": False,
        }
        completion_path = destination_root / f"frontier{cap}" / "completion.json"
        if completion_path.exists():
            if not resume or read_json(completion_path) != completion:
                raise ValueError(
                    f"derived completion identity differs: {completion_path}"
                )
        else:
            write_json(completion_path, completion)
        completion_identities[f"frontier{cap}"] = canonical_sha256(completion)
    summary = {
        "schema_version": "delegent-derived-prefix-family-summary-v1",
        "status": "complete",
        "question_count": len(sources),
        "source_cap": source_cap,
        "derived_caps": list(caps),
        "completed_records": completed,
        "reused_records": reused,
        "model_calls": 0,
        "qrels_or_answers_opened": False,
        "source_record_set_sha256": source_set_sha,
        "arm_completion_sha256": completion_identities,
    }
    write_json(destination_root / "summary.json", summary, overwrite=resume)
    return summary


def load_and_derive(
    source_path: str | Path,
    *,
    caps: Sequence[int],
    output_dir: str | Path,
    frontier_actor: str,
    resume: bool,
) -> dict[str, Any]:
    return derive_prefix_family(
        load_records(source_path),
        caps=caps,
        output_dir=output_dir,
        frontier_actor=frontier_actor,
        resume=resume,
    )
