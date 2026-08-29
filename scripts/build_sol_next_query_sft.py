#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "sol-next-query-complement-sft-v1"
SOURCE_NAME = "gpt_query_bundle"
TARGET_TRANSITIONS = ((4, 5), (5, 6))
TRAIN_FILENAME = "sol_next_query_train.jsonl"
VAL_FILENAME = "sol_next_query_val.jsonl"
MANIFEST_FILENAME = "manifest.json"


@dataclass(frozen=True)
class BuildConfig:
    min_fact_coverage: float = 0.5
    min_ndcg_at_20: float = 0.15
    min_doc_recall_at_20: float = 0.0
    max_duplicate_doc_rate: float | None = None
    require_parse_success: bool = True
    require_no_tool_errors: bool = True
    hits: int = 5
    max_docs_per_call: int = 2
    max_chars_per_passage: int = 240
    val_fraction: float = 0.15
    seed: int = 13


@dataclass(frozen=True)
class SFTExample:
    qid: str
    prefix_length: int
    target_query_number: int
    messages: list[dict[str, str]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build grouped Gemma SFT examples that predict Sol's fifth or sixth "
            "retrieval query from the question, prior queries, and prior passages."
        )
    )
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-fact-coverage", type=float, default=0.5)
    parser.add_argument("--min-ndcg-at-20", type=float, default=0.15)
    parser.add_argument("--min-doc-recall-at-20", type=float, default=0.0)
    parser.add_argument("--max-duplicate-doc-rate", type=float, default=None)
    parser.add_argument("--allow-parse-failures", action="store_true")
    parser.add_argument("--allow-tool-errors", action="store_true")
    parser.add_argument("--hits", type=int, default=5)
    parser.add_argument("--max-docs-per-call", type=int, default=2)
    parser.add_argument("--max-chars-per-passage", type=int, default=240)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = BuildConfig(
        min_fact_coverage=args.min_fact_coverage,
        min_ndcg_at_20=args.min_ndcg_at_20,
        min_doc_recall_at_20=args.min_doc_recall_at_20,
        max_duplicate_doc_rate=args.max_duplicate_doc_rate,
        require_parse_success=not args.allow_parse_failures,
        require_no_tool_errors=not args.allow_tool_errors,
        hits=args.hits,
        max_docs_per_call=args.max_docs_per_call,
        max_chars_per_passage=args.max_chars_per_passage,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    validate_config(config)
    rows = read_jsonl(args.results)
    train, val, manifest = build_dataset(
        rows,
        config=config,
        source_path=args.results,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / TRAIN_FILENAME
    val_path = args.output_dir / VAL_FILENAME
    manifest_path = args.output_dir / MANIFEST_FILENAME
    write_conversations(train_path, train)
    write_conversations(val_path, val)
    manifest["outputs"] = {
        "train": file_metadata(train_path),
        "validation": file_metadata(val_path),
    }
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


def build_dataset(
    rows: list[dict[str, Any]],
    *,
    config: BuildConfig,
    source_path: Path | None = None,
) -> tuple[list[SFTExample], list[SFTExample], dict[str, Any]]:
    validate_config(config)
    seen_qids: set[str] = set()
    examples: list[SFTExample] = []
    filter_reasons: Counter[str] = Counter()
    source_rows_kept = 0

    for row_number, row in enumerate(rows, start=1):
        qid = normalized_required_text(row.get("qid"), f"row {row_number} qid")
        if qid in seen_qids:
            raise ValueError(f"duplicate source qid: {qid}")
        seen_qids.add(qid)
        keep, reason = source_row_filter(row, config)
        filter_reasons[reason] += 1
        if not keep:
            continue
        source_rows_kept += 1
        examples.extend(build_examples_for_row(row, config=config))

    qids_with_examples = sorted({example.qid for example in examples})
    train_qids, val_qids = grouped_split(
        qids_with_examples,
        val_fraction=config.val_fraction,
        seed=config.seed,
    )
    train_qid_set = set(train_qids)
    val_qid_set = set(val_qids)
    train = [example for example in examples if example.qid in train_qid_set]
    val = [example for example in examples if example.qid in val_qid_set]
    if {example.qid for example in train} & {example.qid for example in val}:
        raise AssertionError("qid leakage across train and validation")

    transition_counts = Counter(
        f"prefix{example.prefix_length}->query{example.target_query_number}"
        for example in examples
    )
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "source": {
            "path": str(source_path) if source_path is not None else None,
            "sha256": sha256_file(source_path)
            if source_path is not None and source_path.is_file()
            else None,
            "row_count": len(rows),
            "expected_r_source": SOURCE_NAME,
        },
        "source_filter": {
            "qrel_metrics_used_only_for_source_row_filtering": True,
            "qrels_or_answers_in_model_messages": False,
            "thresholds": {
                "min_fact_coverage": config.min_fact_coverage,
                "min_ndcg_at_20": config.min_ndcg_at_20,
                "min_doc_recall_at_20": config.min_doc_recall_at_20,
                "max_duplicate_doc_rate": config.max_duplicate_doc_rate,
            },
            "require_parse_success": config.require_parse_success,
            "require_no_tool_errors": config.require_no_tool_errors,
            "rows_kept": source_rows_kept,
            "row_outcomes": dict(sorted(filter_reasons.items())),
        },
        "example_recipe": {
            "target_transitions": [
                {
                    "prefix_query_count": prefix_length,
                    "target_query_number": target_query_number,
                }
                for prefix_length, target_query_number in TARGET_TRANSITIONS
            ],
            "target_tool": "search_fineweb_batch",
            "target_query_fields": ["query_1"],
            "hits": config.hits,
            "max_docs_per_prior_call": config.max_docs_per_call,
            "max_chars_per_passage": config.max_chars_per_passage,
            "messages_contain": [
                "question",
                "prior_queries",
                "compact_prior_retrieved_passages",
            ],
            "messages_exclude_source_fields": [
                "answer",
                "qrels",
                "gold",
                "source_scores",
            ],
        },
        "split": {
            "method": "seeded_group_split_by_qid",
            "seed": config.seed,
            "validation_fraction": config.val_fraction,
            "train_qids": train_qids,
            "validation_qids": val_qids,
            "train_qid_count": len(train_qids),
            "validation_qid_count": len(val_qids),
            "train_example_count": len(train),
            "validation_example_count": len(val),
        },
        "counts": {
            "source_qids": len(seen_qids),
            "qids_with_examples": len(qids_with_examples),
            "examples": len(examples),
            "examples_by_transition": dict(sorted(transition_counts.items())),
        },
        "config": asdict(config),
    }
    return train, val, manifest


def source_row_filter(
    row: dict[str, Any],
    config: BuildConfig,
) -> tuple[bool, str]:
    if str(row.get("r_source") or "") != SOURCE_NAME:
        return False, "wrong_source"
    if config.require_parse_success and not bool(row.get("parse_success")):
        return False, "parse_failure"
    if config.require_no_tool_errors and list(row.get("tool_errors") or []):
        return False, "tool_error"
    queries = normalized_queries(row)
    if len(queries) < TARGET_TRANSITIONS[0][1]:
        return False, "fewer_than_five_queries"
    if len({query.casefold() for query in queries}) != len(queries):
        return False, "duplicate_query"
    if metric(row, "qrels_fact_coverage") < config.min_fact_coverage:
        return False, "low_fact_coverage"
    if metric(row, "qrels_ndcg_at_20") < config.min_ndcg_at_20:
        return False, "low_ndcg_at_20"
    if metric(row, "qrels_doc_recall_at_20") < config.min_doc_recall_at_20:
        return False, "low_doc_recall_at_20"
    if (
        config.max_duplicate_doc_rate is not None
        and metric(row, "duplicate_doc_rate") > config.max_duplicate_doc_rate
    ):
        return False, "high_duplicate_doc_rate"
    return True, "kept"


def build_examples_for_row(
    row: dict[str, Any],
    *,
    config: BuildConfig,
) -> list[SFTExample]:
    qid = normalized_required_text(row.get("qid"), "qid")
    question = normalized_required_text(row.get("question"), f"qid={qid} question")
    queries = normalized_queries(row)
    per_call_docids = row.get("per_call_retrieved_docids")
    if not isinstance(per_call_docids, list):
        raise ValueError(f"qid={qid}: per_call_retrieved_docids must be a list")
    docs_by_id = retrieved_docs_by_id(row, qid=qid)
    examples: list[SFTExample] = []
    for prefix_length, target_query_number in TARGET_TRANSITIONS:
        target_index = target_query_number - 1
        if len(queries) <= target_index:
            continue
        if len(per_call_docids) < prefix_length:
            raise ValueError(
                f"qid={qid}: need {prefix_length} prior retrieval calls, "
                f"found {len(per_call_docids)}"
            )
        prior_queries = queries[:prefix_length]
        target_query = queries[target_index]
        if target_query.casefold() in {query.casefold() for query in prior_queries}:
            raise ValueError(
                f"qid={qid}: query {target_query_number} duplicates its prefix"
            )
        prior_passages = compact_prior_passages(
            per_call_docids[:prefix_length],
            docs_by_id=docs_by_id,
            max_docs_per_call=config.max_docs_per_call,
            max_chars_per_passage=config.max_chars_per_passage,
            qid=qid,
        )
        messages = [
            {
                "role": "user",
                "content": render_next_query_prompt(
                    question=question,
                    prior_queries=prior_queries,
                    prior_passages=prior_passages,
                    hits=config.hits,
                ),
            },
            {
                "role": "assistant",
                "content": render_one_query_tool_call(
                    target_query,
                    hits=config.hits,
                ),
            },
        ]
        examples.append(
            SFTExample(
                qid=qid,
                prefix_length=prefix_length,
                target_query_number=target_query_number,
                messages=messages,
            )
        )
    return examples


def render_next_query_prompt(
    *,
    question: str,
    prior_queries: list[str],
    prior_passages: list[list[tuple[str, str]]],
    hits: int,
) -> str:
    history: list[str] = []
    for call_index, (query, passages) in enumerate(
        zip(prior_queries, prior_passages, strict=True),
        start=1,
    ):
        history.append(f"Search {call_index}: {query}")
        if passages:
            history.extend(
                f"- [{docid}] {passage}" for docid, passage in passages
            )
        else:
            history.append("- No passage text was returned.")
    return (
        "You are the query-complement component of a retrieval agent. "
        "Use only the question and retrieval history below. Do not answer the "
        "question.\n\n"
        "Generate exactly one new BM25 search query that complements the prior "
        "queries and targets evidence still missing or uncertain. Do not repeat "
        "a prior query. Prefer a concise entity, title, alias, date, relation, or "
        "bridge query.\n\n"
        "Output exactly one JSON object wrapped in <tool_call> tags and no other "
        "text. It must call search_fineweb_batch with exactly query_1 and hits:\n"
        f'<tool_call>{{"name":"search_fineweb_batch","arguments":'
        f'{{"query_1":"NEW QUERY","hits":{hits}}}}}</tool_call>\n\n'
        f"Question:\n{question}\n\n"
        "Prior searches and retrieved passages:\n"
        + "\n".join(history)
        + "\n"
    )


def render_one_query_tool_call(query: str, *, hits: int) -> str:
    payload = {
        "name": "search_fineweb_batch",
        "arguments": {
            "query_1": normalize_whitespace(query),
            "hits": hits,
        },
    }
    return (
        "<tool_call>"
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        + "</tool_call>"
    )


def compact_prior_passages(
    per_call_docids: list[Any],
    *,
    docs_by_id: dict[str, str],
    max_docs_per_call: int,
    max_chars_per_passage: int,
    qid: str,
) -> list[list[tuple[str, str]]]:
    compacted: list[list[tuple[str, str]]] = []
    for call_index, raw_docids in enumerate(per_call_docids, start=1):
        if not isinstance(raw_docids, list):
            raise ValueError(
                f"qid={qid}: retrieved docids for call {call_index} must be a list"
            )
        call_passages: list[tuple[str, str]] = []
        for raw_docid in raw_docids[:max_docs_per_call]:
            docid = normalized_required_text(
                raw_docid,
                f"qid={qid} call={call_index} docid",
            )
            if docid not in docs_by_id:
                raise ValueError(
                    f"qid={qid}: passage text missing for retrieved docid={docid}"
                )
            call_passages.append(
                (
                    docid,
                    compact_text(
                        docs_by_id[docid],
                        max_chars=max_chars_per_passage,
                    ),
                )
            )
        compacted.append(call_passages)
    return compacted


def retrieved_docs_by_id(row: dict[str, Any], *, qid: str) -> dict[str, str]:
    raw_docs = row.get("retrieved_docs")
    if not isinstance(raw_docs, list):
        raise ValueError(f"qid={qid}: retrieved_docs must be a list")
    docs: dict[str, str] = {}
    for raw_doc in raw_docs:
        if not isinstance(raw_doc, dict):
            raise ValueError(f"qid={qid}: retrieved_docs entries must be objects")
        docid = normalized_required_text(raw_doc.get("docid"), f"qid={qid} docid")
        text = normalized_required_text(
            raw_doc.get("text"),
            f"qid={qid} docid={docid} text",
        )
        previous = docs.get(docid)
        if previous is not None and previous != text:
            raise ValueError(f"qid={qid}: conflicting text for docid={docid}")
        docs[docid] = text
    return docs


def grouped_split(
    qids: Iterable[str],
    *,
    val_fraction: float,
    seed: int,
) -> tuple[list[str], list[str]]:
    ordered = sorted(set(qids))
    shuffled = list(ordered)
    random.Random(seed).shuffle(shuffled)
    if not shuffled or val_fraction <= 0:
        val_count = 0
    elif len(shuffled) == 1:
        val_count = 0
    else:
        val_count = min(len(shuffled) - 1, max(1, round(len(shuffled) * val_fraction)))
    val_set = set(shuffled[:val_count])
    train = [qid for qid in ordered if qid not in val_set]
    val = [qid for qid in ordered if qid in val_set]
    return train, val


def normalized_queries(row: dict[str, Any]) -> list[str]:
    raw_queries = row.get("queries")
    if not isinstance(raw_queries, list):
        return []
    queries = [normalize_whitespace(str(query)) for query in raw_queries]
    if any(not query for query in queries):
        return []
    return queries


def metric(row: dict[str, Any], name: str) -> float:
    try:
        return float(row.get(name, 0.0) or 0.0)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid numeric field {name}: {row.get(name)!r}") from exc


def validate_config(config: BuildConfig) -> None:
    if not 0 <= config.val_fraction < 1:
        raise ValueError("val_fraction must be in [0, 1)")
    if config.hits < 1:
        raise ValueError("hits must be positive")
    if config.max_docs_per_call < 1:
        raise ValueError("max_docs_per_call must be positive")
    if config.max_chars_per_passage < 16:
        raise ValueError("max_chars_per_passage must be at least 16")
    for name in (
        "min_fact_coverage",
        "min_ndcg_at_20",
        "min_doc_recall_at_20",
    ):
        value = float(getattr(config, name))
        if not 0 <= value <= 1:
            raise ValueError(f"{name} must be in [0, 1]")
    if (
        config.max_duplicate_doc_rate is not None
        and not 0 <= config.max_duplicate_doc_rate <= 1
    ):
        raise ValueError("max_duplicate_doc_rate must be in [0, 1]")


def compact_text(text: str, *, max_chars: int) -> str:
    compacted = normalize_whitespace(text)
    if len(compacted) <= max_chars:
        return compacted
    clipped = compacted[: max_chars - 1].rstrip()
    if " " in clipped:
        clipped = clipped.rsplit(" ", 1)[0]
    return clipped.rstrip() + "…"


def normalize_whitespace(value: str) -> str:
    return " ".join(value.split())


def normalized_required_text(value: Any, label: str) -> str:
    normalized = normalize_whitespace(str(value or ""))
    if not normalized:
        raise ValueError(f"missing {label}")
    return normalized


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(row)
    return rows


def write_conversations(path: Path, examples: list[SFTExample]) -> None:
    payload = "".join(
        json.dumps(example.messages, ensure_ascii=False, separators=(",", ":"))
        + "\n"
        for example in examples
    )
    path.write_text(payload, encoding="utf-8")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def file_metadata(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "line_count": sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
