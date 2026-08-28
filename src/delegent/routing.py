"""Fixed-schedule trajectory construction and routed-family orchestration."""

from __future__ import annotations

import copy
import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .actions import Action, ActionValidationError, normalize_text, parse_action
from .clients import Actor, HttpRetriever, OpenAICompatibleActor, Retriever
from .io import (
    DataValidationError,
    canonical_sha256,
    load_records,
    qid_filename,
    read_json,
    record_set_sha256,
    resolve_config_path,
    write_json,
)
from .prompts import compact_evidence, render_prompt
from .repair import deterministic_repair

RETRY_INSTRUCTION = (
    "\nThe preceding attempt was invalid or duplicated an earlier query. "
    "Return a different valid JSON action now.\n"
)
FORBIDDEN_GENERATION_KEYS = frozenset(
    {
        "answer",
        "answers",
        "gold",
        "qrel",
        "qrels",
        "positive_docids",
        "doc_fact_ids",
        "fact_ids",
        "required_fact_coverage",
        "qrels_fact_coverage",
        "qrels_doc_recall_at_20",
        "scores",
        "metrics",
        "raw_output",
        "raw_response",
        "raw_responses",
        "model_response",
        "response_text",
    }
)


@dataclass(frozen=True)
class FixedSchedule:
    cap: int
    frontier_actions: int
    frontier_actor: str = "frontier"
    delegate_actor: str = "delegate"

    def __post_init__(self) -> None:
        if self.cap < 1:
            raise ValueError("cap must be positive")
        if not 0 <= self.frontier_actions <= self.cap:
            raise ValueError("frontier_actions must be between zero and cap")
        if not self.frontier_actor or not self.delegate_actor:
            raise ValueError("actor labels must be non-empty")
        if self.frontier_actor == self.delegate_actor:
            raise ValueError("frontier and delegate actor labels must differ")

    @property
    def delegate_actions(self) -> int:
        return self.cap - self.frontier_actions

    @property
    def name(self) -> str:
        return f"frontier{self.frontier_actions}_delegate{self.delegate_actions}"


class UnknownSpendError(RuntimeError):
    """A model call may have been charged but cannot be safely replayed."""


@dataclass(frozen=True)
class PromptConfig:
    context_hits_per_search: int = 1
    passage_tokens: int = 160


@dataclass(frozen=True)
class GenerationConfig:
    max_attempts: int = 2
    deterministic_repair: bool = True


@dataclass(frozen=True)
class RoutedFamilyConfig:
    config_path: Path
    source_records: Path
    output_dir: Path
    cap: int
    frontier_actor: str
    delegate_actor: str
    frontier_prefixes: tuple[int, ...]
    prompt: PromptConfig
    generation: GenerationConfig
    retrieval: Mapping[str, Any] | None
    delegate: Mapping[str, Any] | None

    @property
    def schedules(self) -> tuple[FixedSchedule, ...]:
        return tuple(
            FixedSchedule(
                cap=self.cap,
                frontier_actions=prefix,
                frontier_actor=self.frontier_actor,
                delegate_actor=self.delegate_actor,
            )
            for prefix in self.frontier_prefixes
        )


class _AttemptLedger:
    """Durable, content-minimal model-call intents and receipts for one QID."""

    def __init__(
        self,
        root: Path,
        *,
        qid: str,
        arm: str,
        actor: str,
        source_record_sha256: str,
    ) -> None:
        self.root = root
        self.qid = qid
        self.arm = arm
        self.actor = actor
        self.source_record_sha256 = source_record_sha256

    def _paths(
        self, slot_one_indexed: int, attempt_one_indexed: int
    ) -> tuple[Path, Path]:
        stem = f"slot-{slot_one_indexed:03d}-attempt-{attempt_one_indexed:03d}"
        return self.root / f"{stem}.intent.json", self.root / f"{stem}.receipt.json"

    def _expected_identity(
        self, *, slot_one_indexed: int, attempt_one_indexed: int
    ) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "qid": self.qid,
            "actor": self.actor,
            "source_record_sha256": self.source_record_sha256,
            "slot_one_indexed": slot_one_indexed,
            "attempt_one_indexed": attempt_one_indexed,
        }

    @staticmethod
    def _sha256_text(value: object, *, field: str) -> str:
        if not isinstance(value, str) or len(value) != 64:
            raise DataValidationError(
                f"attempt ledger {field} must be a SHA-256 hex string"
            )
        try:
            int(value, 16)
        except ValueError as exc:
            raise DataValidationError(
                f"attempt ledger {field} must be a SHA-256 hex string"
            ) from exc
        return value

    def _validate_intent(
        self,
        value: object,
        *,
        slot_one_indexed: int,
        attempt_one_indexed: int,
    ) -> dict[str, Any]:
        if not isinstance(value, Mapping):
            raise DataValidationError("attempt intent must be an object")
        expected_keys = {
            "schema_version",
            "status",
            "arm",
            "qid",
            "actor",
            "source_record_sha256",
            "slot_one_indexed",
            "attempt_one_indexed",
            "prompt_sha256",
            "generation_reads_qrels",
            "generation_reads_answers",
        }
        if set(value) != expected_keys:
            raise DataValidationError("attempt intent fields drifted")
        expected = {
            "schema_version": "delegent-model-attempt-intent-v1",
            "status": "intent-recorded-before-call",
            **self._expected_identity(
                slot_one_indexed=slot_one_indexed,
                attempt_one_indexed=attempt_one_indexed,
            ),
            "generation_reads_qrels": False,
            "generation_reads_answers": False,
        }
        for key, expected_value in expected.items():
            if value.get(key) != expected_value:
                raise DataValidationError(f"attempt intent field drifted: {key}")
        self._sha256_text(value.get("prompt_sha256"), field="prompt_sha256")
        return dict(value)

    def _validate_receipt(
        self,
        value: object,
        *,
        intent: Mapping[str, Any],
        slot_one_indexed: int,
        attempt_one_indexed: int,
    ) -> dict[str, Any]:
        if not isinstance(value, Mapping):
            raise DataValidationError("attempt receipt must be an object")
        expected_keys = {
            "schema_version",
            "status",
            "arm",
            "qid",
            "actor",
            "source_record_sha256",
            "slot_one_indexed",
            "attempt_one_indexed",
            "intent_sha256",
            "validation",
            "output_sha256",
            "prompt_tokens",
            "generated_tokens",
            "elapsed_seconds",
            "error_type",
            "actor_call_may_have_been_charged",
            "raw_response_stored",
        }
        if set(value) != expected_keys:
            raise DataValidationError("attempt receipt fields drifted")
        expected = {
            "schema_version": "delegent-model-attempt-receipt-v1",
            **self._expected_identity(
                slot_one_indexed=slot_one_indexed,
                attempt_one_indexed=attempt_one_indexed,
            ),
            "intent_sha256": canonical_sha256(intent),
            "actor_call_may_have_been_charged": True,
            "raw_response_stored": False,
        }
        for key, expected_value in expected.items():
            if value.get(key) != expected_value:
                raise DataValidationError(f"attempt receipt field drifted: {key}")
        status = value.get("status")
        validation = value.get("validation")
        if status == "known-response":
            if validation not in {"accepted", "rejected"}:
                raise DataValidationError("known response validation status drifted")
            self._sha256_text(value.get("output_sha256"), field="output_sha256")
            error_type = value.get("error_type")
            if validation == "accepted" and error_type is not None:
                raise DataValidationError("accepted response cannot have an error type")
            if validation == "rejected" and error_type != "ActionValidationError":
                raise DataValidationError(
                    "only a known ActionValidationError may be retried"
                )
            for field in ("prompt_tokens", "generated_tokens"):
                counter = value.get(field)
                if (
                    isinstance(counter, bool)
                    or not isinstance(counter, int)
                    or counter < 0
                ):
                    raise DataValidationError(
                        f"attempt receipt {field} must be a non-negative integer"
                    )
            elapsed = value.get("elapsed_seconds")
            if (
                isinstance(elapsed, bool)
                or not isinstance(elapsed, (int, float))
                or not math.isfinite(float(elapsed))
                or elapsed < 0
            ):
                raise DataValidationError(
                    "attempt receipt elapsed_seconds must be finite and non-negative"
                )
        elif status == "unknown-spend":
            if validation != "unknown" or value.get("output_sha256") is not None:
                raise DataValidationError("unknown-spend receipt fields drifted")
            if not isinstance(value.get("error_type"), str) or not value["error_type"]:
                raise DataValidationError("unknown-spend receipt requires error_type")
            if any(
                value.get(field) is not None
                for field in ("prompt_tokens", "generated_tokens", "elapsed_seconds")
            ):
                raise DataValidationError(
                    "unknown-spend receipt cannot assert token or timing telemetry"
                )
        else:
            raise DataValidationError("attempt receipt status drifted")
        return dict(value)

    def _attempts_for_slot(self, slot_one_indexed: int) -> list[dict[str, Any]]:
        if not self.root.exists():
            return []
        prefix = f"slot-{slot_one_indexed:03d}-attempt-"
        intent_paths = sorted(self.root.glob(f"{prefix}*.intent.json"))
        receipt_paths = sorted(self.root.glob(f"{prefix}*.receipt.json"))
        intent_names = {path.name.removesuffix(".intent.json") for path in intent_paths}
        receipt_names = {
            path.name.removesuffix(".receipt.json") for path in receipt_paths
        }
        if receipt_names - intent_names:
            raise DataValidationError(
                "attempt ledger contains a receipt without an intent"
            )
        attempts: list[dict[str, Any]] = []
        for expected_attempt, intent_path in enumerate(intent_paths, start=1):
            stem = intent_path.name.removesuffix(".intent.json")
            expected_stem = f"{prefix}{expected_attempt:03d}"
            if stem != expected_stem:
                raise DataValidationError("attempt ledger indices must be contiguous")
            intent = self._validate_intent(
                read_json(intent_path),
                slot_one_indexed=slot_one_indexed,
                attempt_one_indexed=expected_attempt,
            )
            receipt_path = intent_path.with_name(f"{stem}.receipt.json")
            receipt = (
                self._validate_receipt(
                    read_json(receipt_path),
                    intent=intent,
                    slot_one_indexed=slot_one_indexed,
                    attempt_one_indexed=expected_attempt,
                )
                if receipt_path.is_file()
                else None
            )
            attempts.append(
                {
                    "intent_path": intent_path,
                    "receipt_path": receipt_path,
                    "intent": intent,
                    "receipt": receipt,
                }
            )
        return attempts

    def _all_slots(self) -> list[int]:
        if not self.root.exists():
            return []
        slots: set[int] = set()
        for path in self.root.glob("slot-*-attempt-*.json"):
            pieces = path.name.split("-")
            try:
                slots.add(int(pieces[1]))
            except (IndexError, ValueError) as exc:
                raise DataValidationError(
                    f"malformed attempt ledger filename: {path.name}"
                ) from exc
        return sorted(slots)

    def has_entries(self) -> bool:
        return self.root.exists() and any(self.root.iterdir())

    def assert_safe_for_resume(self, *, record_complete: bool) -> None:
        for slot_one_indexed in self._all_slots():
            for attempt in self._attempts_for_slot(slot_one_indexed):
                receipt = attempt["receipt"]
                if receipt is None:
                    raise UnknownSpendError(
                        "resume blocked by an unresolved model-call intent"
                    )
                if receipt["status"] == "unknown-spend":
                    raise UnknownSpendError(
                        "resume blocked by immutable unknown model spend"
                    )
                if not record_complete and receipt["validation"] == "accepted":
                    raise UnknownSpendError(
                        "resume blocked: an accepted response has no completed record"
                    )

    def prior_rejections(
        self,
        *,
        slot_one_indexed: int,
        allow_existing: bool,
        expected_prompt_sha256s: Sequence[str],
    ) -> list[dict[str, Any]]:
        attempts = self._attempts_for_slot(slot_one_indexed)
        if attempts and not allow_existing:
            raise FileExistsError(attempts[0]["intent_path"])
        if len(attempts) > len(expected_prompt_sha256s):
            raise DataValidationError(
                "attempt ledger exceeds the configured maximum attempt count"
            )
        receipts: list[dict[str, Any]] = []
        for attempt_index, attempt in enumerate(attempts):
            if (
                attempt["intent"]["prompt_sha256"]
                != expected_prompt_sha256s[attempt_index]
            ):
                raise DataValidationError(
                    "attempt prompt hash differs from the reconstructed prompt"
                )
            receipt = attempt["receipt"]
            if receipt is None:
                raise UnknownSpendError(
                    "model call has an unresolved intent and cannot be replayed"
                )
            if receipt["status"] == "unknown-spend":
                raise UnknownSpendError("unknown model spend cannot be replayed")
            if receipt["validation"] == "accepted":
                raise UnknownSpendError(
                    "accepted response cannot be reconstructed from a hash-only receipt"
                )
            receipts.append(receipt)
        return receipts

    def write_intent(
        self,
        *,
        slot_one_indexed: int,
        attempt_one_indexed: int,
        prompt_sha256: str,
    ) -> dict[str, Any]:
        intent_path, _ = self._paths(slot_one_indexed, attempt_one_indexed)
        intent = {
            "schema_version": "delegent-model-attempt-intent-v1",
            "status": "intent-recorded-before-call",
            **self._expected_identity(
                slot_one_indexed=slot_one_indexed,
                attempt_one_indexed=attempt_one_indexed,
            ),
            "prompt_sha256": prompt_sha256,
            "generation_reads_qrels": False,
            "generation_reads_answers": False,
        }
        write_json(intent_path, intent)
        return intent

    def write_receipt(
        self,
        *,
        intent: Mapping[str, Any],
        slot_one_indexed: int,
        attempt_one_indexed: int,
        status: str,
        validation: str,
        output_sha256: str | None,
        prompt_tokens: int | None,
        generated_tokens: int | None,
        elapsed_seconds: float | None,
        error_type: str | None,
    ) -> dict[str, Any]:
        _, receipt_path = self._paths(slot_one_indexed, attempt_one_indexed)
        receipt = {
            "schema_version": "delegent-model-attempt-receipt-v1",
            "status": status,
            **self._expected_identity(
                slot_one_indexed=slot_one_indexed,
                attempt_one_indexed=attempt_one_indexed,
            ),
            "intent_sha256": canonical_sha256(intent),
            "validation": validation,
            "output_sha256": output_sha256,
            "prompt_tokens": prompt_tokens,
            "generated_tokens": generated_tokens,
            "elapsed_seconds": elapsed_seconds,
            "error_type": error_type,
            "actor_call_may_have_been_charged": True,
            "raw_response_stored": False,
        }
        self._validate_receipt(
            receipt,
            intent=intent,
            slot_one_indexed=slot_one_indexed,
            attempt_one_indexed=attempt_one_indexed,
        )
        write_json(receipt_path, receipt)
        return receipt


def _check_no_forbidden_keys(value: object, *, location: str = "record") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).casefold()
            if normalized in FORBIDDEN_GENERATION_KEYS:
                raise DataValidationError(
                    f"generation input contains forbidden field {key!r} at {location}"
                )
            _check_no_forbidden_keys(child, location=f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _check_no_forbidden_keys(child, location=f"{location}[{index}]")


def _validate_hit(hit: object, *, location: str) -> None:
    if not isinstance(hit, Mapping):
        raise DataValidationError(f"{location} must be an object")
    if not str(hit.get("docid") or ""):
        raise DataValidationError(f"{location} requires docid")
    try:
        rank = int(hit.get("rank", 0))
        float(hit.get("score", 0.0))
    except (TypeError, ValueError) as exc:
        raise DataValidationError(f"{location} rank/score must be numeric") from exc
    if rank < 1:
        raise DataValidationError(f"{location} rank must be positive")
    if not isinstance(hit.get("text", ""), str):
        raise DataValidationError(f"{location} text must be a string")


def validate_source_record(
    source: Mapping[str, Any],
    *,
    cap: int,
    frontier_actor: str,
) -> None:
    _check_no_forbidden_keys(source)
    qid = str(source.get("qid") or "")
    question = normalize_text(source.get("question"))
    searches = source.get("searches")
    generations = source.get("generations")
    if not qid or not question:
        raise DataValidationError("source record requires non-empty qid and question")
    if not isinstance(searches, list) or len(searches) != cap + 1:
        raise DataValidationError(
            f"source searches must contain direct retrieval plus exactly {cap} actions"
        )
    if not isinstance(generations, list) or len(generations) != cap:
        raise DataValidationError(
            f"source generations must contain exactly {cap} actions"
        )
    for search_index, search in enumerate(searches):
        if not isinstance(search, Mapping):
            raise DataValidationError(f"search {search_index} must be an object")
        if not normalize_text(search.get("query")) or not isinstance(
            search.get("hits"), list
        ):
            raise DataValidationError(f"search {search_index} requires query and hits")
        if search_index and search.get("actor") != frontier_actor:
            raise DataValidationError(
                f"source generated search {search_index} actor is not {frontier_actor!r}"
            )
        for hit_index, hit in enumerate(search["hits"]):
            _validate_hit(hit, location=f"searches[{search_index}].hits[{hit_index}]")
    for action_index, generation in enumerate(generations):
        if (
            not isinstance(generation, Mapping)
            or generation.get("actor") != frontier_actor
        ):
            raise DataValidationError(
                f"source generation {action_index} actor is not {frontier_actor!r}"
            )


def validate_routed_record(
    record: Mapping[str, Any],
    *,
    source: Mapping[str, Any],
    schedule: FixedSchedule,
) -> None:
    """Deep-validate a routed record, including its exact copied prefix."""

    validate_source_record(
        source, cap=schedule.cap, frontier_actor=schedule.frontier_actor
    )
    _check_no_forbidden_keys(record)
    expected_scalars = {
        "schema_version": "delegent-routed-record-v1",
        "status": "complete",
        "qid": str(source["qid"]),
        "question": normalize_text(source["question"]),
        "arm": schedule.name,
        "cap": schedule.cap,
        "frontier_actor": schedule.frontier_actor,
        "delegate_actor": schedule.delegate_actor,
        "frontier_prefix_actions": schedule.frontier_actions,
        "delegate_actions": schedule.delegate_actions,
        "fixed_direct_searches": 1,
        "direct_retrieval_outside_generated_budget": True,
        "source_record_sha256": canonical_sha256(source),
        "prefix_reused_exactly": True,
        "canonical_future_frontier_actions_visible_to_delegate": False,
        "generation_reads_qrels": False,
        "generation_reads_answers": False,
        "scores_enter_prompts": False,
    }
    for key, expected in expected_scalars.items():
        if record.get(key) != expected:
            raise DataValidationError(f"routed record field drifted: {key}")
    searches = record.get("searches")
    generations = record.get("generations")
    if not isinstance(searches, list) or len(searches) != schedule.cap + 1:
        raise DataValidationError("routed record search length drifted")
    if not isinstance(generations, list) or len(generations) != schedule.cap:
        raise DataValidationError("routed record generation length drifted")
    prefix_end = schedule.frontier_actions + 1
    if searches[:prefix_end] != source["searches"][:prefix_end]:
        raise DataValidationError(
            "routed record search prefix is not an exact source copy"
        )
    if (
        generations[: schedule.frontier_actions]
        != source["generations"][: schedule.frontier_actions]
    ):
        raise DataValidationError(
            "routed record generation prefix is not an exact source copy"
        )
    for search_index, search in enumerate(searches):
        if not isinstance(search, Mapping):
            raise DataValidationError(f"routed search {search_index} is not an object")
        expected_actor = (
            None
            if search_index == 0
            else (
                schedule.frontier_actor
                if search_index <= schedule.frontier_actions
                else schedule.delegate_actor
            )
        )
        if expected_actor is not None and search.get("actor") != expected_actor:
            raise DataValidationError(f"routed search {search_index} actor drifted")
        if not normalize_text(search.get("query")) or not isinstance(
            search.get("hits"), list
        ):
            raise DataValidationError(
                f"routed search {search_index} query/hits drifted"
            )
        for hit_index, hit in enumerate(search["hits"]):
            _validate_hit(hit, location=f"searches[{search_index}].hits[{hit_index}]")
    for generation_index, generation in enumerate(generations):
        if not isinstance(generation, Mapping):
            raise DataValidationError(
                f"routed generation {generation_index} is not an object"
            )
        expected_actor = (
            schedule.frontier_actor
            if generation_index < schedule.frontier_actions
            else schedule.delegate_actor
        )
        if generation.get("actor") != expected_actor:
            raise DataValidationError(
                f"routed generation {generation_index} actor drifted"
            )
        if generation_index >= schedule.frontier_actions:
            if generation.get("slot_one_indexed") != generation_index + 1:
                raise DataValidationError(
                    f"delegate generation {generation_index} slot drifted"
                )
            if generation.get("future_frontier_actions_visible") is not False:
                raise DataValidationError(
                    f"delegate generation {generation_index} exposed a future action"
                )


def _generation_result(
    *,
    actor: Actor,
    question: str,
    searches: Sequence[Mapping[str, Any]],
    slot_one_indexed: int,
    actor_label: str,
    prompt_config: PromptConfig,
    generation_config: GenerationConfig,
    attempt_ledger: _AttemptLedger | None = None,
    resume_attempts: bool = False,
) -> tuple[Action, dict[str, Any]]:
    evidence = compact_evidence(
        searches,
        question=question,
        hits_per_search=prompt_config.context_hits_per_search,
        passage_tokens=prompt_config.passage_tokens,
    )
    base_prompt = render_prompt(question=question, evidence=evidence)
    prior_queries = [str(search["query"]) for search in searches]
    expected_prompt_sha256s = [
        hashlib.sha256(
            (
                base_prompt if attempt_index == 0 else base_prompt + RETRY_INSTRUCTION
            ).encode("utf-8")
        ).hexdigest()
        for attempt_index in range(generation_config.max_attempts)
    ]
    prior_receipts = (
        attempt_ledger.prior_rejections(
            slot_one_indexed=slot_one_indexed,
            allow_existing=resume_attempts,
            expected_prompt_sha256s=expected_prompt_sha256s,
        )
        if attempt_ledger is not None
        else []
    )
    errors: list[str] = [str(receipt["error_type"]) for receipt in prior_receipts]
    prompt_tokens = sum(int(receipt["prompt_tokens"]) for receipt in prior_receipts)
    generated_tokens = sum(
        int(receipt["generated_tokens"]) for receipt in prior_receipts
    )
    elapsed_seconds = sum(
        float(receipt["elapsed_seconds"]) for receipt in prior_receipts
    )
    output_sha256: str | None = (
        str(prior_receipts[-1]["output_sha256"]) if prior_receipts else None
    )
    action: Action | None = None
    attempts = len(prior_receipts)
    for attempt_index in range(attempts, generation_config.max_attempts):
        attempt_one_indexed = attempt_index + 1
        attempts = attempt_one_indexed
        prompt = base_prompt if attempt_index == 0 else base_prompt + RETRY_INSTRUCTION
        prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        intent = (
            attempt_ledger.write_intent(
                slot_one_indexed=slot_one_indexed,
                attempt_one_indexed=attempt_one_indexed,
                prompt_sha256=prompt_sha256,
            )
            if attempt_ledger is not None
            else None
        )
        try:
            result = actor.generate(prompt)
        except Exception as exc:
            if attempt_ledger is not None and intent is not None:
                attempt_ledger.write_receipt(
                    intent=intent,
                    slot_one_indexed=slot_one_indexed,
                    attempt_one_indexed=attempt_one_indexed,
                    status="unknown-spend",
                    validation="unknown",
                    output_sha256=None,
                    prompt_tokens=None,
                    generated_tokens=None,
                    elapsed_seconds=None,
                    error_type=type(exc).__name__,
                )
            raise UnknownSpendError(
                "model transport failed after intent; spend is unknown and was not retried"
            ) from exc
        current_prompt_tokens = int(result.prompt_tokens)
        current_generated_tokens = int(result.generated_tokens)
        current_elapsed_seconds = float(result.elapsed_seconds)
        current_output_sha256 = hashlib.sha256(result.text.encode("utf-8")).hexdigest()
        try:
            action = parse_action(
                result.text,
                question=question,
                prior_queries=prior_queries,
            )
        except ActionValidationError as exc:
            if attempt_ledger is not None and intent is not None:
                attempt_ledger.write_receipt(
                    intent=intent,
                    slot_one_indexed=slot_one_indexed,
                    attempt_one_indexed=attempt_one_indexed,
                    status="known-response",
                    validation="rejected",
                    output_sha256=current_output_sha256,
                    prompt_tokens=current_prompt_tokens,
                    generated_tokens=current_generated_tokens,
                    elapsed_seconds=round(current_elapsed_seconds, 6),
                    error_type=type(exc).__name__,
                )
            prompt_tokens += current_prompt_tokens
            generated_tokens += current_generated_tokens
            elapsed_seconds += current_elapsed_seconds
            output_sha256 = current_output_sha256
            errors.append(type(exc).__name__)
            continue
        if attempt_ledger is not None and intent is not None:
            attempt_ledger.write_receipt(
                intent=intent,
                slot_one_indexed=slot_one_indexed,
                attempt_one_indexed=attempt_one_indexed,
                status="known-response",
                validation="accepted",
                output_sha256=current_output_sha256,
                prompt_tokens=current_prompt_tokens,
                generated_tokens=current_generated_tokens,
                elapsed_seconds=round(current_elapsed_seconds, 6),
                error_type=None,
            )
        prompt_tokens += current_prompt_tokens
        generated_tokens += current_generated_tokens
        elapsed_seconds += current_elapsed_seconds
        output_sha256 = current_output_sha256
        break
    repaired = False
    if action is None:
        if not generation_config.deterministic_repair:
            raise ActionValidationError(
                "actor exhausted attempts and deterministic repair is disabled"
            )
        action = deterministic_repair(
            question=question,
            searches=searches,
            action_index=slot_one_indexed - 1,
        )
        repaired = True
    generation = {
        "actor": actor_label,
        "slot_one_indexed": slot_one_indexed,
        "attempts": attempts,
        "accepted_actor_action": not repaired,
        "deterministic_repair_used": repaired,
        "prompt_sha256": hashlib.sha256(base_prompt.encode("utf-8")).hexdigest(),
        "output_sha256": output_sha256,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "elapsed_seconds": round(elapsed_seconds, 6),
        "error_types": errors,
        "future_frontier_actions_visible": False,
    }
    return action, generation


def run_routed_record(
    source: Mapping[str, Any],
    *,
    schedule: FixedSchedule,
    actor: Actor,
    retriever: Retriever,
    retrieval_hits: int = 5,
    prompt_config: PromptConfig | None = None,
    generation_config: GenerationConfig | None = None,
    attempt_ledger: _AttemptLedger | None = None,
    resume_attempts: bool = False,
) -> dict[str, Any]:
    """Construct one routed trajectory without reading labels or future actions."""

    validate_source_record(
        source, cap=schedule.cap, frontier_actor=schedule.frontier_actor
    )
    if retrieval_hits < 1:
        raise ValueError("retrieval_hits must be positive")
    prompt_config = prompt_config or PromptConfig()
    generation_config = generation_config or GenerationConfig()
    question = normalize_text(source["question"])
    searches = copy.deepcopy(source["searches"][: schedule.frontier_actions + 1])
    generations = copy.deepcopy(source["generations"][: schedule.frontier_actions])
    prefix_searches = copy.deepcopy(searches)
    prefix_generations = copy.deepcopy(generations)
    for slot_zero_indexed in range(schedule.frontier_actions, schedule.cap):
        slot_one_indexed = slot_zero_indexed + 1
        action, generation = _generation_result(
            actor=actor,
            question=question,
            searches=searches,
            slot_one_indexed=slot_one_indexed,
            actor_label=schedule.delegate_actor,
            prompt_config=prompt_config,
            generation_config=generation_config,
            attempt_ledger=attempt_ledger,
            resume_attempts=resume_attempts,
        )
        hits = retriever.search(action.query, hits=retrieval_hits)
        searches.append(
            {
                "actor": schedule.delegate_actor,
                "query": action.query,
                "plan": action.plan,
                "hits": [hit.as_dict() for hit in hits],
            }
        )
        generations.append(generation)
    if searches[: schedule.frontier_actions + 1] != prefix_searches:
        raise AssertionError("frontier search prefix changed during routing")
    if generations[: schedule.frontier_actions] != prefix_generations:
        raise AssertionError("frontier generation prefix changed during routing")
    if len(searches) != schedule.cap + 1 or len(generations) != schedule.cap:
        raise AssertionError("routed trajectory length does not match cap")
    record = {
        "schema_version": "delegent-routed-record-v1",
        "status": "complete",
        "qid": str(source["qid"]),
        "question": question,
        "arm": schedule.name,
        "cap": schedule.cap,
        "frontier_actor": schedule.frontier_actor,
        "delegate_actor": schedule.delegate_actor,
        "frontier_prefix_actions": schedule.frontier_actions,
        "delegate_actions": schedule.delegate_actions,
        "fixed_direct_searches": 1,
        "direct_retrieval_outside_generated_budget": True,
        "source_record_sha256": canonical_sha256(source),
        "prefix_reused_exactly": True,
        "canonical_future_frontier_actions_visible_to_delegate": False,
        "generation_reads_qrels": False,
        "generation_reads_answers": False,
        "scores_enter_prompts": False,
        "searches": searches,
        "generations": generations,
    }
    validate_routed_record(record, source=source, schedule=schedule)
    return record


def _require_object(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DataValidationError(f"{field} must be an object")
    return value


def load_family_config(
    path: str | Path, *, dry_run: bool = False
) -> RoutedFamilyConfig:
    config_path = Path(path).resolve()
    value = read_json(config_path)
    config = _require_object(value, field="config")
    allowed = {
        "schema_version",
        "source_records",
        "output_dir",
        "cap",
        "frontier_actor",
        "delegate_actor",
        "frontier_prefixes",
        "retrieval",
        "delegate",
        "prompt",
        "generation",
    }
    if set(config) - allowed:
        raise DataValidationError(
            f"unknown config fields: {sorted(set(config) - allowed)}"
        )
    if config.get("schema_version") != "delegent-routed-family-config-v1":
        raise DataValidationError("unsupported config schema_version")
    cap = int(config.get("cap", 0))
    if cap < 1:
        raise DataValidationError("cap must be positive")
    prefixes_value = config.get("frontier_prefixes")
    if not isinstance(prefixes_value, list) or not prefixes_value:
        raise DataValidationError("frontier_prefixes must be a non-empty list")
    prefixes = tuple(int(item) for item in prefixes_value)
    if len(set(prefixes)) != len(prefixes) or any(
        not 0 <= item <= cap for item in prefixes
    ):
        raise DataValidationError(
            "frontier_prefixes must be unique integers between zero and cap"
        )
    frontier_actor = str(config.get("frontier_actor") or "")
    delegate_actor = str(config.get("delegate_actor") or "")
    # Constructing one schedule provides exact actor-label validation.
    FixedSchedule(cap, prefixes[0], frontier_actor, delegate_actor)
    prompt_value = _require_object(config.get("prompt", {}), field="prompt")
    if set(prompt_value) - {"context_hits_per_search", "passage_tokens"}:
        raise DataValidationError("unknown prompt config field")
    prompt = PromptConfig(
        context_hits_per_search=int(prompt_value.get("context_hits_per_search", 1)),
        passage_tokens=int(prompt_value.get("passage_tokens", 160)),
    )
    if min(prompt.context_hits_per_search, prompt.passage_tokens) < 1:
        raise DataValidationError("prompt bounds must be positive")
    generation_value = _require_object(config.get("generation", {}), field="generation")
    if set(generation_value) - {"max_attempts", "deterministic_repair"}:
        raise DataValidationError("unknown generation config field")
    generation = GenerationConfig(
        max_attempts=int(generation_value.get("max_attempts", 2)),
        deterministic_repair=bool(generation_value.get("deterministic_repair", True)),
    )
    if generation.max_attempts < 1:
        raise DataValidationError("generation.max_attempts must be positive")
    retrieval = config.get("retrieval")
    delegate = config.get("delegate")
    if not dry_run and (
        not isinstance(retrieval, Mapping) or not isinstance(delegate, Mapping)
    ):
        raise DataValidationError(
            "retrieval and delegate configs are required unless --dry-run is used"
        )
    if retrieval is not None:
        retrieval = _require_object(retrieval, field="retrieval")
        allowed_retrieval = {
            "base_url",
            "index",
            "hits",
            "timeout_seconds",
            "token_env",
        }
        if set(retrieval) - allowed_retrieval:
            raise DataValidationError("unknown retrieval config field")
    if delegate is not None:
        delegate = _require_object(delegate, field="delegate")
        allowed_delegate = {
            "base_url",
            "model",
            "max_tokens",
            "timeout_seconds",
            "api_key_env",
        }
        if set(delegate) - allowed_delegate:
            raise DataValidationError("unknown delegate config field")
    return RoutedFamilyConfig(
        config_path=config_path,
        source_records=resolve_config_path(
            config_path, config.get("source_records"), field="source_records"
        ),
        output_dir=resolve_config_path(
            config_path, config.get("output_dir"), field="output_dir"
        ),
        cap=cap,
        frontier_actor=frontier_actor,
        delegate_actor=delegate_actor,
        frontier_prefixes=prefixes,
        prompt=prompt,
        generation=generation,
        retrieval=retrieval,
        delegate=delegate,
    )


def dry_run_family(config: RoutedFamilyConfig) -> dict[str, Any]:
    records = load_records(config.source_records)
    arms: list[dict[str, Any]] = []
    for schedule in config.schedules:
        prompt_hashes: list[str] = []
        for source in records:
            validate_source_record(
                source, cap=config.cap, frontier_actor=config.frontier_actor
            )
            if schedule.delegate_actions:
                prefix = source["searches"][: schedule.frontier_actions + 1]
                evidence = compact_evidence(
                    prefix,
                    question=str(source["question"]),
                    hits_per_search=config.prompt.context_hits_per_search,
                    passage_tokens=config.prompt.passage_tokens,
                )
                prompt = render_prompt(
                    question=str(source["question"]), evidence=evidence
                )
                prompt_hashes.append(hashlib.sha256(prompt.encode("utf-8")).hexdigest())
        arms.append(
            {
                "arm": schedule.name,
                "frontier_prefix_actions": schedule.frontier_actions,
                "delegate_actions": schedule.delegate_actions,
                "first_delegate_prompt_sha256": prompt_hashes[0]
                if prompt_hashes
                else None,
            }
        )
    return {
        "schema_version": "delegent-routed-family-dry-run-v1",
        "status": "validated",
        "question_count": len(records),
        "cap": config.cap,
        "arms": arms,
        "external_calls": 0,
        "files_written": 0,
        "qrels_or_answers_opened": False,
    }


def _clients(
    config: RoutedFamilyConfig,
) -> tuple[OpenAICompatibleActor, HttpRetriever, int]:
    if config.delegate is None or config.retrieval is None:
        raise DataValidationError(
            "live generation requires delegate and retrieval configs"
        )
    delegate = config.delegate
    retrieval = config.retrieval
    actor = OpenAICompatibleActor(
        base_url=str(delegate.get("base_url") or ""),
        model=str(delegate.get("model") or ""),
        api_key_env=str(delegate["api_key_env"])
        if delegate.get("api_key_env")
        else None,
        max_tokens=int(delegate.get("max_tokens", 192)),
        timeout_seconds=float(delegate.get("timeout_seconds", 300)),
    )
    retriever = HttpRetriever(
        base_url=str(retrieval.get("base_url") or ""),
        index=str(retrieval.get("index") or ""),
        api_key_env=str(retrieval["token_env"]) if retrieval.get("token_env") else None,
        timeout_seconds=float(retrieval.get("timeout_seconds", 60)),
    )
    hits = int(retrieval.get("hits", 5))
    if hits < 1:
        raise DataValidationError("retrieval.hits must be positive")
    return actor, retriever, hits


def run_family(config: RoutedFamilyConfig, *, resume: bool = False) -> dict[str, Any]:
    records = load_records(config.source_records)
    actor, retriever, hits = _clients(config)
    completed = 0
    reused = 0
    source_set_sha = record_set_sha256(records)
    completion_identities: dict[str, str] = {}
    for schedule in config.schedules:
        records_dir = config.output_dir / schedule.name / "records"
        arm_records: list[dict[str, Any]] = []
        for source in records:
            qid = str(source["qid"])
            qid_file = qid_filename(qid)
            destination = records_dir / qid_file
            attempt_ledger = _AttemptLedger(
                config.output_dir
                / schedule.name
                / "attempts"
                / qid_file.removesuffix(".json"),
                qid=qid,
                arm=schedule.name,
                actor=schedule.delegate_actor,
                source_record_sha256=canonical_sha256(source),
            )
            if resume:
                attempt_ledger.assert_safe_for_resume(
                    record_complete=destination.is_file()
                )
            elif attempt_ledger.has_entries():
                raise FileExistsError(attempt_ledger.root)
            if destination.exists():
                if not resume:
                    raise FileExistsError(destination)
                existing = read_json(destination)
                if not isinstance(existing, Mapping):
                    raise DataValidationError(
                        f"existing record cannot be resumed: {destination}"
                    )
                validate_routed_record(existing, source=source, schedule=schedule)
                arm_records.append(dict(existing))
                reused += 1
                continue
            routed = run_routed_record(
                source,
                schedule=schedule,
                actor=actor,
                retriever=retriever,
                retrieval_hits=hits,
                prompt_config=config.prompt,
                generation_config=config.generation,
                attempt_ledger=attempt_ledger,
                resume_attempts=resume,
            )
            write_json(destination, routed)
            arm_records.append(routed)
            completed += 1
        if len(arm_records) != len(records):
            raise AssertionError("arm completion count drifted")
        completion = {
            "schema_version": "delegent-record-set-completion-v1",
            "status": "complete",
            "producer": "routed-family",
            "arm": schedule.name,
            "question_count": len(records),
            "cap": schedule.cap,
            "frontier_prefix_actions": schedule.frontier_actions,
            "delegate_actions": schedule.delegate_actions,
            "record_set_sha256": record_set_sha256(arm_records),
            "source_record_set_sha256": source_set_sha,
            "qrels_or_answers_opened": False,
        }
        completion_path = config.output_dir / schedule.name / "completion.json"
        if completion_path.exists():
            if not resume or read_json(completion_path) != completion:
                raise DataValidationError(
                    f"arm completion identity differs: {completion_path}"
                )
        else:
            write_json(completion_path, completion)
        completion_identities[schedule.name] = canonical_sha256(completion)
    summary = {
        "schema_version": "delegent-routed-family-summary-v1",
        "status": "complete",
        "question_count": len(records),
        "arm_count": len(config.schedules),
        "completed_records": completed,
        "reused_records": reused,
        "source_record_set_sha256": source_set_sha,
        "arm_completion_sha256": completion_identities,
        "generation_reads_qrels": False,
        "generation_reads_answers": False,
    }
    write_json(config.output_dir / "summary.json", summary, overwrite=resume)
    return summary
