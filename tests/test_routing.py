from __future__ import annotations

import copy
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from delegent.actions import ActionValidationError
from delegent.clients import ActorResult, SearchHit
from delegent.io import (
    DataValidationError,
    canonical_sha256,
    read_json,
    write_json,
    write_jsonl,
)
from delegent.routing import (
    FixedSchedule,
    GenerationConfig,
    PromptConfig,
    RoutedFamilyConfig,
    UnknownSpendError,
    run_family,
    run_routed_record,
    validate_routed_record,
)
from tests.helpers import demo_sources


class StubActor:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = outputs
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> ActorResult:
        self.prompts.append(prompt)
        if not self.outputs:
            raise RuntimeError("no stub output")
        return ActorResult(
            self.outputs.pop(0),
            prompt_tokens=5,
            generated_tokens=3,
            elapsed_seconds=0.1,
        )


class StubRetriever:
    def __init__(self) -> None:
        self.queries: list[tuple[str, int]] = []

    def search(self, query: str, *, hits: int) -> list[SearchHit]:
        self.queries.append((query, hits))
        return [
            SearchHit(
                docid=f"new-{len(self.queries)}", rank=1, score=1.0, text="new evidence"
            )
        ]


class TransportFailureActor:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> ActorResult:
        self.prompts.append(prompt)
        raise TimeoutError("response status is unknown")


class HardInterrupt(BaseException):
    pass


class InterruptedActor:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> ActorResult:
        self.prompts.append(prompt)
        raise HardInterrupt


class RoutingTests(unittest.TestCase):
    @staticmethod
    def _family_config(root: Path) -> RoutedFamilyConfig:
        source_path = root / "source.jsonl"
        write_jsonl(source_path, [demo_sources()[0]])
        return RoutedFamilyConfig(
            config_path=root / "config.json",
            source_records=source_path,
            output_dir=root / "output",
            cap=2,
            frontier_actor="frontier",
            delegate_actor="delegate",
            frontier_prefixes=(1,),
            prompt=PromptConfig(),
            generation=GenerationConfig(),
            retrieval={},
            delegate={},
        )

    def test_mixed_schedule_copies_exact_prefix_and_hides_future_action(self) -> None:
        source = demo_sources()[0]
        original = copy.deepcopy(source)
        actor = StubActor(
            ['{"plan":"check another source","query":"Lumen independent source"}']
        )
        retriever = StubRetriever()
        record = run_routed_record(
            source,
            schedule=FixedSchedule(2, 1),
            actor=actor,
            retriever=retriever,
        )
        self.assertEqual(source, original)
        self.assertEqual(record["searches"][:2], source["searches"][:2])
        self.assertEqual(record["generations"][:1], source["generations"][:1])
        self.assertEqual(record["searches"][2]["query"], "Lumen independent source")
        self.assertNotEqual(record["searches"][2], source["searches"][2])
        self.assertFalse(
            record["canonical_future_frontier_actions_visible_to_delegate"]
        )
        self.assertEqual(retriever.queries, [("Lumen independent source", 5)])

    def test_all_frontier_is_an_exact_zero_call_copy(self) -> None:
        source = demo_sources()[0]
        actor = StubActor([])
        record = run_routed_record(
            source,
            schedule=FixedSchedule(2, 2),
            actor=actor,
            retriever=StubRetriever(),
        )
        self.assertEqual(record["searches"], source["searches"])
        self.assertEqual(record["generations"], source["generations"])
        self.assertEqual(actor.prompts, [])

    def test_invalid_outputs_use_deterministic_qrel_blind_repair(self) -> None:
        source = demo_sources()[0]
        actor = StubActor(
            ["not-json", '{"plan":"p","query":"Project Lumen launch city"}']
        )
        retriever = StubRetriever()
        record = run_routed_record(
            source,
            schedule=FixedSchedule(2, 1),
            actor=actor,
            retriever=retriever,
            generation_config=GenerationConfig(
                max_attempts=2, deterministic_repair=True
            ),
        )
        generation = record["generations"][-1]
        self.assertTrue(generation["deterministic_repair_used"])
        self.assertEqual(
            generation["error_types"],
            ["ActionValidationError", "ActionValidationError"],
        )
        self.assertEqual(len(actor.prompts), 2)
        self.assertIn("preceding attempt was invalid", actor.prompts[1])

    def test_attempt_ledger_retries_only_known_schema_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self._family_config(root)
            actor = StubActor(["not-json", '{"plan":"p","query":"known valid query"}'])
            with patch(
                "delegent.routing._clients",
                return_value=(actor, StubRetriever(), 5),
            ):
                run_family(config)

            ledger = (
                root / "output" / "frontier1_delegate1" / "attempts" / "demo-route-001"
            )
            first_intent = read_json(ledger / "slot-002-attempt-001.intent.json")
            first_receipt = read_json(ledger / "slot-002-attempt-001.receipt.json")
            second_receipt = read_json(ledger / "slot-002-attempt-002.receipt.json")
            self.assertEqual(first_intent["status"], "intent-recorded-before-call")
            self.assertEqual(first_receipt["status"], "known-response")
            self.assertEqual(first_receipt["validation"], "rejected")
            self.assertEqual(first_receipt["error_type"], "ActionValidationError")
            self.assertEqual(second_receipt["validation"], "accepted")
            self.assertEqual(len(actor.prompts), 2)
            serialized = json.dumps([first_intent, first_receipt, second_receipt])
            self.assertNotIn("not-json", serialized)
            self.assertNotIn("known valid query", serialized)
            self.assertFalse(first_receipt["raw_response_stored"])

    def test_transport_failure_is_unknown_spend_and_never_retried(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self._family_config(root)
            actor = TransportFailureActor()
            with (
                patch(
                    "delegent.routing._clients",
                    return_value=(actor, StubRetriever(), 5),
                ),
                self.assertRaisesRegex(UnknownSpendError, "was not retried"),
            ):
                run_family(config)
            self.assertEqual(len(actor.prompts), 1)

            receipt_path = (
                root
                / "output"
                / "frontier1_delegate1"
                / "attempts"
                / "demo-route-001"
                / "slot-002-attempt-001.receipt.json"
            )
            receipt = read_json(receipt_path)
            self.assertEqual(receipt["status"], "unknown-spend")
            self.assertEqual(receipt["validation"], "unknown")
            self.assertEqual(receipt["error_type"], "TimeoutError")
            self.assertIsNone(receipt["output_sha256"])
            self.assertFalse(receipt["raw_response_stored"])

            replacement = StubActor(['{"plan":"p","query":"must never be replayed"}'])
            with (
                patch(
                    "delegent.routing._clients",
                    return_value=(replacement, StubRetriever(), 5),
                ),
                self.assertRaisesRegex(UnknownSpendError, "immutable unknown"),
            ):
                run_family(config, resume=True)
            self.assertEqual(replacement.prompts, [])

    def test_unresolved_intent_fail_closes_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self._family_config(root)
            interrupted = InterruptedActor()
            with (
                patch(
                    "delegent.routing._clients",
                    return_value=(interrupted, StubRetriever(), 5),
                ),
                self.assertRaises(HardInterrupt),
            ):
                run_family(config)
            self.assertEqual(len(interrupted.prompts), 1)

            ledger = (
                root / "output" / "frontier1_delegate1" / "attempts" / "demo-route-001"
            )
            self.assertTrue((ledger / "slot-002-attempt-001.intent.json").is_file())
            self.assertFalse((ledger / "slot-002-attempt-001.receipt.json").exists())

            replacement = StubActor(['{"plan":"p","query":"must never be replayed"}'])
            with (
                patch(
                    "delegent.routing._clients",
                    return_value=(replacement, StubRetriever(), 5),
                ),
                self.assertRaisesRegex(UnknownSpendError, "unresolved"),
            ):
                run_family(config, resume=True)
            self.assertEqual(replacement.prompts, [])

    def test_resume_rejects_tampered_known_rejection_prompt_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = replace(
                self._family_config(root),
                generation=GenerationConfig(max_attempts=1, deterministic_repair=False),
            )
            with (
                patch(
                    "delegent.routing._clients",
                    return_value=(StubActor(["not-json"]), StubRetriever(), 5),
                ),
                self.assertRaises(ActionValidationError),
            ):
                run_family(config)

            intent_path = (
                root
                / "output"
                / "frontier1_delegate1"
                / "attempts"
                / "demo-route-001"
                / "slot-002-attempt-001.intent.json"
            )
            intent = read_json(intent_path)
            intent["prompt_sha256"] = "0" * 64
            write_json(intent_path, intent, overwrite=True)
            receipt_path = intent_path.with_name("slot-002-attempt-001.receipt.json")
            receipt = read_json(receipt_path)
            receipt["intent_sha256"] = canonical_sha256(intent)
            write_json(receipt_path, receipt, overwrite=True)

            replacement = StubActor(
                ['{"plan":"p","query":"must never run after prompt drift"}']
            )
            with (
                patch(
                    "delegent.routing._clients",
                    return_value=(replacement, StubRetriever(), 5),
                ),
                self.assertRaisesRegex(DataValidationError, "reconstructed prompt"),
            ):
                run_family(config, resume=True)
            self.assertEqual(replacement.prompts, [])

    def test_generation_input_rejects_qrels(self) -> None:
        source = demo_sources()[0]
        source["qrels"] = {"d1": 1}
        with self.assertRaises(DataValidationError):
            run_routed_record(
                source,
                schedule=FixedSchedule(2, 2),
                actor=StubActor([]),
                retriever=StubRetriever(),
            )

    def test_deep_validation_rejects_tampered_prefix_and_raw_response(self) -> None:
        source = demo_sources()[0]
        schedule = FixedSchedule(2, 1)
        record = run_routed_record(
            source,
            schedule=schedule,
            actor=StubActor(['{"plan":"p","query":"novel query"}']),
            retriever=StubRetriever(),
        )
        record["searches"][1]["query"] = "tampered"
        with self.assertRaisesRegex(DataValidationError, "exact source copy"):
            validate_routed_record(record, source=source, schedule=schedule)
        record = run_routed_record(
            source,
            schedule=schedule,
            actor=StubActor(['{"plan":"p","query":"another novel query"}']),
            retriever=StubRetriever(),
        )
        record["generations"][-1]["raw_response"] = "must not be stored"
        with self.assertRaisesRegex(DataValidationError, "forbidden field"):
            validate_routed_record(record, source=source, schedule=schedule)

    def test_family_writes_completion_and_resume_deep_validates_record(self) -> None:
        source = demo_sources()[0]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_path = root / "source.jsonl"
            write_jsonl(source_path, [source])
            config = RoutedFamilyConfig(
                config_path=root / "config.json",
                source_records=source_path,
                output_dir=root / "output",
                cap=2,
                frontier_actor="frontier",
                delegate_actor="delegate",
                frontier_prefixes=(1,),
                prompt=PromptConfig(),
                generation=GenerationConfig(),
                retrieval={},
                delegate={},
            )
            with patch(
                "delegent.routing._clients",
                return_value=(
                    StubActor(['{"plan":"p","query":"completion query"}']),
                    StubRetriever(),
                    5,
                ),
            ):
                summary = run_family(config)
            completion = read_json(
                root / "output" / "frontier1_delegate1" / "completion.json"
            )
            self.assertEqual(
                completion["schema_version"], "delegent-record-set-completion-v1"
            )
            self.assertEqual(summary["arm_count"], 1)
            record_path = (
                root
                / "output"
                / "frontier1_delegate1"
                / "records"
                / "demo-route-001.json"
            )
            record = read_json(record_path)
            record["searches"][1]["query"] = "tampered"
            write_json(record_path, record, overwrite=True)
            with (
                patch(
                    "delegent.routing._clients",
                    return_value=(StubActor([]), StubRetriever(), 5),
                ),
                self.assertRaisesRegex(DataValidationError, "exact source copy"),
            ):
                run_family(config, resume=True)


if __name__ == "__main__":
    unittest.main()
