"""Union, reciprocal-rank-fusion, HotpotQA, and SQuAD retrieval metrics."""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class QuestionQrels:
    positive_docids: frozenset[str]
    doc_fact_ids: Mapping[str, frozenset[str]]

    @property
    def fact_ids(self) -> frozenset[str]:
        return frozenset(fact for facts in self.doc_fact_ids.values() for fact in facts)


def union_docids(searches: Sequence[Mapping[str, Any]]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for search in searches:
        hits = search.get("hits")
        if not isinstance(hits, list):
            raise TypeError("every search must contain a hit list")
        for hit in hits:
            if not isinstance(hit, Mapping):
                raise TypeError("every hit must be an object")
            docid = str(hit.get("docid") or "")
            if not docid:
                raise ValueError("every hit requires a docid")
            if docid not in seen:
                seen.add(docid)
                ordered.append(docid)
    return ordered


def rrf_docids(
    searches: Sequence[Mapping[str, Any]],
    *,
    rrf_k: int = 60,
    limit: int = 20,
) -> list[str]:
    if rrf_k < 0 or limit < 1:
        raise ValueError("rrf_k must be non-negative and limit must be positive")
    scores: dict[str, float] = {}
    first: dict[str, int] = {}
    seen_counter = 0
    for search in searches:
        hits = search.get("hits")
        if not isinstance(hits, list):
            raise TypeError("every search must contain a hit list")
        for rank, hit in enumerate(hits, start=1):
            if not isinstance(hit, Mapping):
                raise TypeError("every hit must be an object")
            docid = str(hit.get("docid") or "")
            if not docid:
                raise ValueError("every hit requires a docid")
            scores[docid] = scores.get(docid, 0.0) + 1.0 / (rrf_k + rank)
            if docid not in first:
                first[docid] = seen_counter
                seen_counter += 1
    return sorted(scores, key=lambda docid: (-scores[docid], first[docid], docid))[
        :limit
    ]


def _doc_recall(retrieved: Sequence[str], positives: frozenset[str]) -> float:
    return (
        len(set(retrieved).intersection(positives)) / len(positives)
        if positives
        else 0.0
    )


def _fact_coverage(retrieved: Sequence[str], qrels: QuestionQrels) -> float:
    facts = qrels.fact_ids
    covered = {
        fact
        for docid in retrieved
        for fact in qrels.doc_fact_ids.get(docid, frozenset())
    }
    return len(covered.intersection(facts)) / len(facts) if facts else 0.0


def hotpotqa_metrics(
    searches: Sequence[Mapping[str, Any]],
    qrels: QuestionQrels,
    *,
    rrf_k: int = 60,
    cutoff: int = 20,
) -> dict[str, float]:
    union = union_docids(searches)
    fused = rrf_docids(searches, rrf_k=rrf_k, limit=cutoff)
    return {
        "union_required_fact_coverage": _fact_coverage(union, qrels),
        "rrf_at_20_required_fact_coverage": _fact_coverage(fused, qrels),
        "union_positive_doc_recall": _doc_recall(union, qrels.positive_docids),
        "rrf_at_20_positive_doc_recall": _doc_recall(fused, qrels.positive_docids),
    }


def ndcg_at_k(
    ranking: Sequence[str], positives: frozenset[str], *, cutoff: int = 20
) -> float:
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, docid in enumerate(ranking[:cutoff], start=1)
        if docid in positives
    )
    ideal = sum(
        1.0 / math.log2(rank + 1) for rank in range(1, min(cutoff, len(positives)) + 1)
    )
    return dcg / ideal if ideal else 0.0


def squad_metrics(
    searches: Sequence[Mapping[str, Any]],
    qrels: QuestionQrels,
    *,
    rrf_k: int = 60,
    cutoff: int = 20,
) -> dict[str, float]:
    union = union_docids(searches)
    fused = rrf_docids(searches, rrf_k=rrf_k, limit=cutoff)
    positives = qrels.positive_docids
    return {
        "union_any_positive": float(bool(set(union).intersection(positives))),
        "rrf20_any_positive": float(bool(set(fused).intersection(positives))),
        "union_positive_doc_recall": _doc_recall(union, positives),
        "rrf20_positive_doc_recall": _doc_recall(fused, positives),
        "rrf20_ndcg": ndcg_at_k(fused, positives, cutoff=cutoff),
    }


def macro_average(rows: Sequence[Mapping[str, float]]) -> dict[str, float]:
    if not rows:
        raise ValueError("cannot aggregate an empty metric sequence")
    keys = set(rows[0])
    if any(set(row) != keys for row in rows):
        raise ValueError("metric rows have inconsistent fields")
    return {
        key: statistics.fmean(float(row[key]) for row in rows) for key in sorted(keys)
    }
