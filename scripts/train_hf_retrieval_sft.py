#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standard HF chat SFT for retrieval tool-call traces")
    parser.add_argument("--model-id", default="LiquidAI/LFM2.5-1.2B-Instruct")
    parser.add_argument("--train-files", nargs="+", required=True)
    parser.add_argument("--val-files", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-train-examples", type=int, default=None)
    parser.add_argument("--max-val-examples", type=int, default=512)
    parser.add_argument("--max-steps", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--label-last-assistant-only",
        action="store_true",
        help="Only supervise the final assistant turn in each conversation; useful for OPD state->teacher-action data.",
    )
    parser.add_argument(
        "--lora-target-modules",
        nargs="+",
        default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    return parser.parse_args()


class ChatSFTDataset(Dataset):
    def __init__(
        self,
        conversations: list[list[dict]],
        *,
        tokenizer,
        max_length: int,
        label_last_assistant_only: bool = False,
    ) -> None:
        self.examples = []
        skipped = 0
        start = time.time()
        for messages in conversations:
            encoded = encode_chat_for_assistant_loss(
                messages,
                tokenizer=tokenizer,
                max_length=max_length,
                label_last_assistant_only=label_last_assistant_only,
            )
            if encoded is None:
                skipped += 1
                continue
            self.examples.append(encoded)
        print(
            f"Encoded {len(self.examples)} conversations "
            f"(skipped {skipped}) in {time.time() - start:.1f}s",
            flush=True,
        )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        return self.examples[idx]


@dataclass
class ChatSFTCollator:
    pad_token_id: int

    def __call__(self, batch: list[dict]) -> dict:
        max_len = max(len(item["input_ids"]) for item in batch)
        input_ids = []
        attention_mask = []
        labels = []
        for item in batch:
            pad_len = max_len - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [self.pad_token_id] * pad_len)
            attention_mask.append([1] * len(item["input_ids"]) + [0] * pad_len)
            labels.append(item["labels"] + [-100] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if not getattr(tokenizer, "chat_template", None):
        raise RuntimeError(f"{args.model_id} does not expose a tokenizer chat_template")

    train_conversations = load_conversations([Path(path) for path in args.train_files])
    val_conversations = load_conversations([Path(path) for path in args.val_files])
    random.shuffle(train_conversations)
    random.shuffle(val_conversations)
    if args.max_train_examples is not None:
        train_conversations = train_conversations[:args.max_train_examples]
    if args.max_val_examples is not None:
        val_conversations = val_conversations[:args.max_val_examples]

    print(f"Train conversations: {len(train_conversations)}")
    print(f"Val conversations: {len(val_conversations)}")

    train_dataset = ChatSFTDataset(
        train_conversations,
        tokenizer=tokenizer,
        max_length=args.max_length,
        label_last_assistant_only=args.label_last_assistant_only,
    )
    val_dataset = ChatSFTDataset(
        val_conversations,
        tokenizer=tokenizer,
        max_length=args.max_length,
        label_last_assistant_only=args.label_last_assistant_only,
    )
    if not train_dataset:
        raise RuntimeError("No train examples after encoding")

    dtype = getattr(torch, args.torch_dtype)
    model, loaded_trainable_adapter = load_causal_lm_for_sft(
        args.model_id,
        torch_dtype=dtype,
        trust_remote_code=args.trust_remote_code,
    )
    if len(tokenizer) > model.get_input_embeddings().num_embeddings:
        model.resize_token_embeddings(len(tokenizer))
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
    if args.use_lora and loaded_trainable_adapter:
        print("Continuing trainable PEFT adapter from --model-id; not adding a new LoRA adapter.")
        if hasattr(model, "print_trainable_parameters"):
            model.print_trainable_parameters()
    elif args.use_lora:
        from peft import LoraConfig, TaskType, get_peft_model

        if args.gradient_checkpointing and hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        target_modules = resolve_lora_target_modules(model, args.lora_target_modules)
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=target_modules,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    model.train()

    collator = ChatSFTCollator(pad_token_id=tokenizer.pad_token_id)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        drop_last=False,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        fused=torch.cuda.is_available(),
    )
    warmup_steps = max(1, int(args.max_steps * args.warmup_ratio))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=args.max_steps,
    )

    step = 0
    optimizer.zero_grad(set_to_none=True)
    running_loss = 0.0
    running_items = 0
    device = next(model.parameters()).device
    train_iter = infinite_loader(train_loader)

    while step < args.max_steps:
        for _micro in range(args.grad_accum):
            batch = {key: value.to(device) for key, value in next(train_iter).items()}
            outputs = model(**batch)
            loss = outputs.loss / args.grad_accum
            loss.backward()
            running_loss += float(loss.detach().cpu()) * args.grad_accum
            running_items += 1
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1

        if step == 1 or step % 10 == 0:
            avg = running_loss / max(1, running_items)
            print(f"step {step:05d}/{args.max_steps} train_loss={avg:.4f} lr={scheduler.get_last_lr()[0]:.3e}", flush=True)
            running_loss = 0.0
            running_items = 0
        if args.eval_every and step % args.eval_every == 0:
            val_loss = evaluate_loss(model, val_loader, device=device)
            print(f"step {step:05d} val_loss={val_loss:.4f}", flush=True)
        if args.save_every and step % args.save_every == 0:
            save_dir = args.output_dir / f"checkpoint-{step:05d}"
            save_model(model, tokenizer, save_dir, args)

    final_dir = args.output_dir / "final"
    save_model(model, tokenizer, final_dir, args)
    print(f"Saved final model to {final_dir}")


def load_conversations(paths: list[Path]) -> list[list[dict]]:
    conversations: list[list[dict]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                messages = json.loads(line)
                if not isinstance(messages, list):
                    raise ValueError(f"Expected list of messages in {path}:{line_no}")
                conversations.append(messages)
    return conversations


def resolve_lora_target_modules(model, target_modules: list[str]) -> list[str] | str:
    if target_modules == ["gemma4_text"]:
        suffixes = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
        resolved = [
            name
            for name, _module in model.named_modules()
            if "language_model.layers" in name and name.endswith(suffixes)
        ]
        if not resolved:
            raise RuntimeError("Could not resolve Gemma4 language-model LoRA target modules")
        return resolved
    if len(target_modules) == 1 and target_modules[0].startswith("regex:"):
        return target_modules[0].removeprefix("regex:")
    return target_modules


def load_causal_lm_for_sft(model_id: str, *, torch_dtype, trust_remote_code: bool):
    from transformers import AutoModelForCausalLM

    adapter_path = Path(model_id)
    if adapter_path.exists() and (adapter_path / "adapter_config.json").exists():
        from peft import PeftConfig, PeftModel

        config = PeftConfig.from_pretrained(str(adapter_path))
        base_model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
        )
        return PeftModel.from_pretrained(base_model, str(adapter_path), is_trainable=True), True
    return AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
    ), False


def encode_chat_for_assistant_loss(
    messages: list[dict],
    *,
    tokenizer,
    max_length: int,
    label_last_assistant_only: bool = False,
) -> dict | None:
    input_ids = chat_template_ids(
        tokenizer,
        messages,
        add_generation_prompt=False,
    )
    labels = [-100] * len(input_ids)
    last_assistant_idx = None
    if label_last_assistant_only:
        for idx, message in enumerate(messages):
            if str(message.get("role", "")).lower() == "assistant":
                last_assistant_idx = idx
    for idx, message in enumerate(messages):
        if str(message.get("role", "")).lower() != "assistant":
            continue
        if label_last_assistant_only and idx != last_assistant_idx:
            continue
        before = chat_template_ids(tokenizer, messages[:idx], add_generation_prompt=True) if idx else []
        after = chat_template_ids(
            tokenizer,
            messages[:idx + 1],
            add_generation_prompt=False,
        )
        start = len(before)
        end = len(after)
        for pos in range(start, min(end, len(labels))):
            labels[pos] = input_ids[pos]

    if len(input_ids) > max_length:
        input_ids = input_ids[-max_length:]
        labels = labels[-max_length:]
    if all(label == -100 for label in labels):
        return None
    return {"input_ids": input_ids, "labels": labels}


def chat_template_ids(tokenizer, messages: list[dict], *, add_generation_prompt: bool) -> list[int]:
    rendered = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=add_generation_prompt,
        tokenize=True,
    )
    return coerce_token_ids(rendered)


def coerce_token_ids(rendered) -> list[int]:
    if hasattr(rendered, "keys") and "input_ids" in rendered:
        return coerce_token_ids(rendered["input_ids"])
    if isinstance(rendered, dict) and "input_ids" in rendered:
        return coerce_token_ids(rendered["input_ids"])
    if hasattr(rendered, "ids"):
        return list(rendered.ids)
    if isinstance(rendered, list):
        if len(rendered) == 1 and hasattr(rendered[0], "ids"):
            return list(rendered[0].ids)
        if rendered and all(hasattr(item, "ids") for item in rendered):
            ids: list[int] = []
            for item in rendered:
                ids.extend(item.ids)
            return ids
        return [int(item) for item in rendered]
    if hasattr(rendered, "tolist"):
        values = rendered.tolist()
        if values and isinstance(values[0], list):
            return [int(item) for item in values[0]]
        return [int(item) for item in values]
    return list(rendered)


def infinite_loader(loader: DataLoader):
    while True:
        for batch in loader:
            yield batch


@torch.no_grad()
def evaluate_loss(model, loader: DataLoader, *, device) -> float:
    if not loader.dataset:
        return math.nan
    model.eval()
    losses = []
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        losses.append(float(model(**batch).loss.detach().cpu()))
    model.train()
    return sum(losses) / max(1, len(losses))


def save_model(model, tokenizer, output_dir: Path, args: argparse.Namespace) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    (output_dir / "training_args.json").write_text(
        json.dumps(vars(args), indent=2, default=str),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
