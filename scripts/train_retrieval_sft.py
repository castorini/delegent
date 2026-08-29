#!/usr/bin/env python3
"""Train or continue a LoRA retrieval policy from chat-message JSONL files.

Each non-empty input line is either a list of chat messages or an object with
``messages`` and optional ``metadata`` fields.  Only the final assistant turn
contributes to the loss.  Rows longer than ``max_length`` are rejected rather
than truncated so that a search state is never silently changed.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train-files", nargs="+", type=Path, required=True)
    parser.add_argument("--validation-files", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--input-adapter",
        type=Path,
        help="Continue this PEFT adapter; omit to initialize a fresh LoRA.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path}: expected a JSON object")
    return value


def load_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if isinstance(value, list):
                    messages = value
                    metadata: dict[str, Any] = {}
                elif isinstance(value, dict) and isinstance(value.get("messages"), list):
                    messages = value["messages"]
                    metadata = value.get("metadata") or {}
                else:
                    raise TypeError(
                        f"{path}:{line_number}: expected messages or an object containing messages"
                    )
                if not messages or not all(isinstance(message, dict) for message in messages):
                    raise ValueError(f"{path}:{line_number}: invalid message list")
                if str(messages[-1].get("role", "")).lower() != "assistant":
                    raise ValueError(f"{path}:{line_number}: final message must be assistant")
                rows.append({"messages": messages, "metadata": metadata})
    if not rows:
        raise ValueError("No training rows were loaded")
    return rows


def token_ids(tokenizer: Any, messages: list[dict[str, Any]], *, generation: bool) -> list[int]:
    value = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=generation,
        tokenize=True,
    )
    if isinstance(value, dict):
        value = value["input_ids"]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], list):
        value = value[0]
    return [int(item) for item in value]


def encode_row(
    row: dict[str, Any],
    *,
    tokenizer: Any,
    max_length: int,
) -> dict[str, list[int]]:
    messages = row["messages"]
    input_ids = token_ids(tokenizer, messages, generation=False)
    prefix_ids = token_ids(tokenizer, messages[:-1], generation=True)
    if len(input_ids) > max_length:
        raise ValueError(
            f"Training row exceeds max_length ({len(input_ids)} > {max_length}); refusing truncation"
        )
    if input_ids[: len(prefix_ids)] != prefix_ids:
        raise ValueError("Tokenizer chat template is not prefix-stable")
    labels = [-100] * len(prefix_ids) + input_ids[len(prefix_ids) :]
    if not labels or all(label == -100 for label in labels):
        raise ValueError("Training row has no assistant tokens to label")
    return {"input_ids": input_ids, "labels": labels}


class ChatDataset:
    def __init__(self, rows: list[dict[str, Any]], tokenizer: Any, max_length: int) -> None:
        self.examples = [
            encode_row(row, tokenizer=tokenizer, max_length=max_length) for row in rows
        ]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.examples[index]


@dataclass
class ChatCollator:
    pad_token_id: int

    def __call__(self, batch: list[dict[str, list[int]]]) -> dict[str, Any]:
        import torch

        width = max(len(item["input_ids"]) for item in batch)
        input_ids: list[list[int]] = []
        attention_mask: list[list[int]] = []
        labels: list[list[int]] = []
        for item in batch:
            padding = width - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [self.pad_token_id] * padding)
            attention_mask.append([1] * len(item["input_ids"]) + [0] * padding)
            labels.append(item["labels"] + [-100] * padding)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def resolve_text_targets(model: Any) -> list[str]:
    suffixes = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    targets = sorted(
        name
        for name, _module in model.named_modules()
        if "language_model.layers" in name and name.endswith(suffixes)
    )
    if not targets:
        raise RuntimeError("Could not resolve Gemma text-layer LoRA targets")
    return targets


def main() -> int:
    args = parse_args()
    config = load_json(args.config)
    required = {
        "base_model",
        "base_model_revision",
        "max_length",
        "max_steps",
        "batch_size",
        "gradient_accumulation_steps",
        "learning_rate",
        "warmup_ratio",
        "weight_decay",
        "seed",
        "lora_rank",
        "lora_alpha",
        "lora_dropout",
    }
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError(f"Training config is missing: {', '.join(missing)}")
    train_rows = load_rows(args.train_files)
    validation_rows = load_rows(args.validation_files)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "validated",
                    "train_rows": len(train_rows),
                    "validation_rows": len(validation_rows),
                    "config": str(args.config),
                    "model_or_adapter_loaded": False,
                },
                indent=2,
            )
        )
        return 0

    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    random.seed(int(config["seed"]))
    torch.manual_seed(int(config["seed"]))
    tokenizer_source = str(args.input_adapter or config["base_model"])
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        revision=None if args.input_adapter else config["base_model_revision"],
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        config["base_model"],
        revision=config["base_model_revision"],
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    if args.input_adapter:
        model = PeftModel.from_pretrained(model, str(args.input_adapter), is_trainable=True)
    else:
        model = get_peft_model(
            model,
            LoraConfig(
                task_type="CAUSAL_LM",
                r=int(config["lora_rank"]),
                lora_alpha=int(config["lora_alpha"]),
                lora_dropout=float(config["lora_dropout"]),
                target_modules=resolve_text_targets(model),
            ),
        )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    train_dataset = ChatDataset(train_rows, tokenizer, int(config["max_length"]))
    validation_dataset = ChatDataset(validation_rows, tokenizer, int(config["max_length"]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        max_steps=int(config["max_steps"]),
        per_device_train_batch_size=int(config["batch_size"]),
        per_device_eval_batch_size=int(config["batch_size"]),
        gradient_accumulation_steps=int(config["gradient_accumulation_steps"]),
        learning_rate=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
        warmup_ratio=float(config["warmup_ratio"]),
        lr_scheduler_type="linear",
        bf16=True,
        gradient_checkpointing=True,
        eval_strategy="steps",
        eval_steps=int(config["evaluation_interval_steps"]),
        save_strategy="steps",
        save_steps=int(config["checkpoint_interval_steps"]),
        save_total_limit=3,
        logging_steps=1,
        seed=int(config["seed"]),
        data_seed=int(config["seed"]),
        report_to=[],
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=ChatCollator(tokenizer.pad_token_id),
    )
    trainer.train()
    final_dir = args.output_dir / "final"
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
