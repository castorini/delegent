"""Delegent: fixed-schedule model routing for retrieval trajectories."""

from .actions import Action, ActionValidationError, parse_action
from .bootstrap import paired_bootstrap
from .clients import ActorResult, HttpRetriever, OpenAICompatibleActor, SearchHit
from .costs import DEFAULT_COST_RATES, CostRates, summarize_frontier_cost
from .derive import derive_prefix_record
from .metrics import QuestionQrels, hotpotqa_metrics, rrf_docids, squad_metrics
from .repair import deterministic_repair
from .routing import FixedSchedule, UnknownSpendError, run_routed_record
from .split import salted_order

__all__ = [
    "DEFAULT_COST_RATES",
    "Action",
    "ActionValidationError",
    "ActorResult",
    "CostRates",
    "FixedSchedule",
    "HttpRetriever",
    "OpenAICompatibleActor",
    "QuestionQrels",
    "SearchHit",
    "UnknownSpendError",
    "derive_prefix_record",
    "deterministic_repair",
    "hotpotqa_metrics",
    "paired_bootstrap",
    "parse_action",
    "rrf_docids",
    "run_routed_record",
    "salted_order",
    "squad_metrics",
    "summarize_frontier_cost",
]
