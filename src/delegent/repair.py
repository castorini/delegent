"""Qrel-blind deterministic repair for invalid or duplicate actor actions."""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from .actions import Action, normalize_text, normalized_key

STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "he",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "their",
        "this",
        "to",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "with",
    }
)
FALLBACK_SUFFIXES = ("relation", "biography", "history", "date", "source", "evidence")


def deterministic_repair(
    *,
    question: str,
    searches: Sequence[Mapping[str, Any]],
    action_index: int,
) -> Action:
    """Build a novel query from only the question and already retrieved text."""

    if action_index < 0:
        raise ValueError("action_index must be non-negative")
    prior = {normalized_key(search.get("query")) for search in searches}
    question_tokens = set(re.findall(r"[a-z0-9]+", question.casefold()))
    observed: list[str] = []
    for search in searches:
        hits = search.get("hits") or []
        if not isinstance(hits, Sequence):
            raise TypeError("search hits must be a sequence")
        for hit in hits:
            if isinstance(hit, Mapping):
                observed.extend(
                    re.findall(r"[a-z0-9]+", str(hit.get("text") or "").casefold())
                )
    counts = Counter(
        token
        for token in observed
        if len(token) >= 3 and token not in STOPWORDS and token not in question_tokens
    )
    first = {token: observed.index(token) for token in counts}
    novel = sorted(counts, key=lambda token: (-counts[token], first[token], token))[:4]
    base = normalize_text(question)
    if not base:
        raise ValueError("question must be non-empty")
    candidates: list[str] = []
    if novel:
        candidates.append(f"{base} {' '.join(novel)}")
    for offset in range(len(FALLBACK_SUFFIXES)):
        suffix = FALLBACK_SUFFIXES[(action_index + offset) % len(FALLBACK_SUFFIXES)]
        candidates.append(f"{base} {suffix}")
    for query in candidates:
        if normalized_key(query) not in prior:
            return Action(plan="deterministic qrel-blind evidence repair", query=query)
    digest = hashlib.sha256(
        f"{base}|{action_index}|{'|'.join(sorted(prior))}".encode()
    ).hexdigest()[:12]
    return Action(
        plan="deterministic qrel-blind degenerate repair",
        query=f"{base} evidence {digest}",
    )
