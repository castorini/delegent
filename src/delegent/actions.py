"""Strict public action schema for query-planning actors."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


class ActionValidationError(ValueError):
    """The actor response was not exactly a valid ``{plan, query}`` action."""


def normalize_text(value: object) -> str:
    return " ".join(str(value or "").split())


def normalized_key(value: object) -> str:
    return normalize_text(value).casefold()


@dataclass(frozen=True)
class Action:
    plan: str
    query: str

    def as_dict(self) -> dict[str, str]:
        return {"plan": self.plan, "query": self.query}


def parse_action(
    value: object,
    *,
    question: str,
    prior_queries: list[str] | tuple[str, ...] = (),
    max_plan_chars: int = 240,
    max_query_chars: int = 512,
) -> Action:
    """Parse an actor response with no markdown or extra JSON fields allowed."""

    if not isinstance(value, str):
        raise ActionValidationError("actor output must be a JSON string")

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ActionValidationError(
                    f"actor output contains duplicate JSON key: {key}"
                )
            result[key] = item
        return result

    try:
        payload: Any = json.loads(value, object_pairs_hook=reject_duplicate_keys)
    except json.JSONDecodeError as exc:
        raise ActionValidationError(
            f"actor output is not valid JSON: {exc.msg}"
        ) from exc
    if not isinstance(payload, dict) or set(payload) != {"plan", "query"}:
        raise ActionValidationError("actor output must contain exactly plan and query")
    if not isinstance(payload["plan"], str) or not isinstance(payload["query"], str):
        raise ActionValidationError("plan and query must both be strings")
    plan = normalize_text(payload["plan"])
    query = normalize_text(payload["query"])
    if not plan or len(plan) > max_plan_chars:
        raise ActionValidationError(f"plan must contain 1..{max_plan_chars} characters")
    if not query or len(query) > max_query_chars:
        raise ActionValidationError(
            f"query must contain 1..{max_query_chars} characters"
        )
    used = {normalized_key(item) for item in (question, *prior_queries)}
    if normalized_key(query) in used:
        raise ActionValidationError("query duplicates the question or a prior query")
    return Action(plan=plan, query=query)
