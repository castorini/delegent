#!/usr/bin/env python3
"""Resumable, no-truncation Hugging Face chat SFT for retrieval policies.

This entry point is intentionally restricted to a Slurm GPU allocation.  It
supports both plain chat-message JSONL rows and rows of the form
``{"messages": [...], "metadata": {...}}``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import shutil
import signal
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup


CHECKPOINT_RE = re.compile(r"^checkpoint-(\d{6})$")
SCHEMA_VERSION = "hf-retrieval-sft-resumable-v1"
PREEMPT_EXIT_CODE = 99
STOP_REQUESTED = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument(
        "--base-model-revision",
        required=True,
        help="Immutable Hugging Face commit revision for the base model.",
    )
    parser.add_argument("--train-files", nargs="+", type=Path, required=True)
    parser.add_argument("--val-files", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--max-train-examples", type=int)
    parser.add_argument("--max-val-examples", type=int, default=512)
    parser.add_argument("--max-steps", type=int, default=240)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-6)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--eval-every", type=int, default=40)
    parser.add_argument("--save-every", type=int, default=40)
    parser.add_argument("--seed", type=int, default=56)
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--label-last-assistant-only", action="store_true")
    parser.add_argument(
        "--overlength-policy",
        choices=("skip", "error"),
        default="skip",
        help="Never truncate. Either skip and record an overlength row or fail.",
    )
    parser.add_argument(
        "--max-overlength-fraction",
        type=float,
        default=0.20,
        help="Fail when a larger fraction of either split is skipped.",
    )
    parser.add_argument(
        "--resume",
        default="auto",
        help="'auto', 'none', or a checkpoint directory.",
    )
    parser.add_argument(
        "--lora-target-modules",
        nargs="+",
        default=["gemma4_text"],
    )
    return parser.parse_args()


def request_stop(_signum: int, _frame: Any) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True


def require_slurm_gpu() -> None:
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Training is permitted only inside a Slurm allocation")
    if not torch.cuda.is_available():
        raise RuntimeError("The Slurm allocation does not expose a CUDA GPU")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, default=str) + "\n").encode("utf-8")


def atomic_write(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_bytes(payload)
    temporary.replace(path)


def load_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
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
                    if not isinstance(metadata, dict):
                        raise ValueError(f"{path}:{line_number}: metadata must be an object")
                else:
                    raise ValueError(
                        f"{path}:{line_number}: expected messages list or object with messages"
                    )
                if not messages or not all(isinstance(item, dict) for item in messages):
                    raise ValueError(f"{path}:{line_number}: invalid messages")
                rows.append(
                    {
                        "messages": messages,
                        "metadata": metadata,
                        "source_file": str(path.resolve()),
                        "source_line": line_number,
                    }
                )
    return rows


def coerce_token_ids(rendered: Any) -> list[int]:
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


def chat_template_ids(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    *,
    add_generation_prompt: bool,
) -> list[int]:
    return coerce_token_ids(
        tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=add_generation_prompt,
            tokenize=True,
        )
    )


def encode_chat(
    messages: list[dict[str, Any]],
    *,
    tokenizer: Any,
    label_last_assistant_only: bool,
) -> dict[str, list[int]] | None:
    input_ids = chat_template_ids(tokenizer, messages, add_generation_prompt=False)
    labels = [-100] * len(input_ids)
    last_assistant_index = None
    if label_last_assistant_only:
        for index, message in enumerate(messages):
            if str(message.get("role", "")).lower() == "assistant":
                last_assistant_index = index
    for index, message in enumerate(messages):
        if str(message.get("role", "")).lower() != "assistant":
            continue
        if label_last_assistant_only and index != last_assistant_index:
            continue
        before = (
            chat_template_ids(
                tokenizer,
                messages[:index],
                add_generation_prompt=True,
            )
            if index
            else []
        )
        after = chat_template_ids(
            tokenizer,
            messages[: index + 1],
            add_generation_prompt=False,
        )
        for position in range(len(before), min(len(after), len(labels))):
            labels[position] = input_ids[position]
    if all(label == -100 for label in labels):
        return None
    return {"input_ids": input_ids, "labels": labels}


@dataclass
class EncodingReport:
    input_rows: int
    kept_rows: int
    overlength_rows: int
    no_assistant_loss_rows: int
    max_observed_tokens: int
    overlength_examples: list[dict[str, Any]]


class ChatDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        *,
        tokenizer: Any,
        max_length: int,
        label_last_assistant_only: bool,
        overlength_policy: str,
    ) -> None:
        self.examples: list[dict[str, list[int]]] = []
        overlength_examples: list[dict[str, Any]] = []
        no_loss = 0
        max_observed = 0
        started = time.time()
        for row in rows:
            encoded = encode_chat(
                row["messages"],
                tokenizer=tokenizer,
                label_last_assistant_only=label_last_assistant_only,
            )
            if encoded is None:
                no_loss += 1
                continue
            token_count = len(encoded["input_ids"])
            max_observed = max(max_observed, token_count)
            if token_count > max_length:
                detail = {
                    "source_file": row["source_file"],
                    "source_line": row["source_line"],
                    "qid": row["metadata"].get("qid"),
                    "tokens": token_count,
                }
                if overlength_policy == "error":
                    raise ValueError(
                        f"Overlength training row ({token_count}>{max_length}): {detail}"
                    )
                if len(overlength_examples) < 100:
                    overlength_examples.append(detail)
                continue
            self.examples.append(encoded)
        overlength_count = len(rows) - len(self.examples) - no_loss
        self.report = EncodingReport(
            input_rows=len(rows),
            kept_rows=len(self.examples),
            overlength_rows=overlength_count,
            no_assistant_loss_rows=no_loss,
            max_observed_tokens=max_observed,
            overlength_examples=overlength_examples,
        )
        print(
            f"Encoded {len(self.examples)}/{len(rows)} rows in {time.time() - started:.1f}s; "
            f"overlength={overlength_count}, no_assistant_loss={no_loss}",
            flush=True,
        )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.examples[index]


@dataclass
class ChatCollator:
    pad_token_id: int

    def __call__(self, batch: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        max_length = max(len(item["input_ids"]) for item in batch)
        inputs: list[list[int]] = []
        masks: list[list[int]] = []
        labels: list[list[int]] = []
        for item in batch:
            padding = max_length - len(item["input_ids"])
            inputs.append(item["input_ids"] + [self.pad_token_id] * padding)
            masks.append([1] * len(item["input_ids"]) + [0] * padding)
            labels.append(item["labels"] + [-100] * padding)
        return {
            "input_ids": torch.tensor(inputs, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class CyclingLoader:
    def __init__(self, loader: DataLoader, cursor: int = 0) -> None:
        if len(loader) == 0:
            raise ValueError("Empty training loader")
        self.loader = loader
        self.iterator = iter(loader)
        self.cursor = 0
        for _ in range(cursor % len(loader)):
            self._next_raw()
        self.cursor = cursor

    def _next_raw(self) -> dict[str, torch.Tensor]:
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            return next(self.iterator)

    def next(self) -> dict[str, torch.Tensor]:
        batch = self._next_raw()
        self.cursor += 1
        return batch


def resolve_lora_target_modules(model: Any, target_modules: list[str]) -> list[str] | str:
    if target_modules == ["gemma4_text"]:
        suffixes = (
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        )
        resolved = [
            name
            for name, _module in model.named_modules()
            if "language_model.layers" in name and name.endswith(suffixes)
        ]
        if not resolved:
            raise RuntimeError("Could not resolve Gemma-4 language-model LoRA modules")
        return resolved
    if len(target_modules) == 1 and target_modules[0].startswith("regex:"):
        return target_modules[0].removeprefix("regex:")
    return target_modules


def load_model(
    model_id: str,
    *,
    base_model_revision: str,
    torch_dtype: torch.dtype,
    trust_remote_code: bool,
) -> tuple[Any, bool]:
    adapter_path = Path(model_id)
    if adapter_path.exists() and (adapter_path / "adapter_config.json").exists():
        from peft import PeftConfig, PeftModel

        config = PeftConfig.from_pretrained(str(adapter_path))
        base_model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path,
            revision=base_model_revision,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
        )
        model = PeftModel.from_pretrained(
            base_model,
            str(adapter_path),
            is_trainable=True,
        )
        return model, True
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=base_model_revision,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
    )
    return model, False


def find_resume_checkpoint(output_dir: Path, resume: str) -> Path | None:
    if resume == "none":
        return None
    if resume != "auto":
        path = Path(resume).resolve()
        if not (path / "training_state.pt").exists():
            raise FileNotFoundError(f"Resume checkpoint lacks training_state.pt: {path}")
        return path
    candidates: list[tuple[int, Path]] = []
    if output_dir.exists():
        for path in output_dir.iterdir():
            match = CHECKPOINT_RE.match(path.name)
            if match and (path / "training_state.pt").exists():
                candidates.append((int(match.group(1)), path))
    return max(candidates)[1] if candidates else None


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def save_checkpoint(
    *,
    model: Any,
    tokenizer: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    output_dir: Path,
    step: int,
    batch_cursor: int,
    args: argparse.Namespace,
    input_manifest: dict[str, Any],
    final: bool = False,
) -> Path:
    name = "final" if final else f"checkpoint-{step:06d}"
    destination = output_dir / name
    temporary = output_dir / f".{name}.tmp-{os.getpid()}"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    model.save_pretrained(temporary)
    tokenizer.save_pretrained(temporary)
    torch.save(
        {
            "schema_version": SCHEMA_VERSION,
            "step": step,
            "batch_cursor": batch_cursor,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "rng": capture_rng_state(),
            "input_manifest": input_manifest,
        },
        temporary / "training_state.pt",
    )
    (temporary / "training_args.json").write_bytes(json_bytes(vars(args)))
    if destination.exists():
        shutil.rmtree(destination)
    temporary.replace(destination)
    atomic_write(
        output_dir / "latest_checkpoint.json",
        json_bytes({"path": str(destination.resolve()), "step": step, "final": final}),
    )
    print(f"Saved {'final' if final else 'resumable'} checkpoint: {destination}", flush=True)
    return destination


@torch.no_grad()
def evaluate_loss(model: Any, loader: DataLoader, device: torch.device) -> float:
    if len(loader) == 0:
        return math.nan
    model.eval()
    losses: list[float] = []
    for batch in loader:
        moved = {key: value.to(device) for key, value in batch.items()}
        losses.append(float(model(**moved).loss.detach().cpu()))
    model.train()
    return sum(losses) / len(losses)


def assert_overlength_rate(report: EncodingReport, maximum: float, label: str) -> None:
    denominator = max(1, report.input_rows)
    rate = report.overlength_rows / denominator
    if rate > maximum:
        raise RuntimeError(
            f"{label} overlength rate {rate:.3%} exceeds allowed {maximum:.3%}"
        )


def main() -> int:
    args = parse_args()
    require_slurm_gpu()
    if not re.fullmatch(r"[0-9a-f]{40,64}", args.base_model_revision):
        raise ValueError(
            "--base-model-revision must be an immutable lowercase commit SHA"
        )
    if not 0.0 <= args.max_overlength_fraction <= 1.0:
        raise ValueError("--max-overlength-fraction must be between 0 and 1")
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGUSR1, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    input_manifest = {
        "schema_version": SCHEMA_VERSION,
        "train_files": [
            {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for path in args.train_files
        ],
        "val_files": [
            {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for path in args.val_files
        ],
    }
    atomic_write(args.output_dir / "input_manifest.json", json_bytes(input_manifest))
    run_manifest = {
        "schema_version": SCHEMA_VERSION,
        "started_unix_seconds": time.time(),
        "resume_argument": args.resume,
        "slurm": {
            "job_id": os.environ["SLURM_JOB_ID"],
            "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "node_list": os.environ.get("SLURM_JOB_NODELIST"),
            "account": os.environ.get("SLURM_JOB_ACCOUNT"),
            "partition": os.environ.get("SLURM_JOB_PARTITION"),
        },
    }
    atomic_write(
        args.output_dir / f"run-{os.environ['SLURM_JOB_ID']}.json",
        json_bytes(run_manifest),
    )

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    resume_checkpoint = find_resume_checkpoint(args.output_dir, args.resume)
    load_id = str(resume_checkpoint) if resume_checkpoint else args.model_id
    tokenizer = AutoTokenizer.from_pretrained(
        load_id,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if not getattr(tokenizer, "chat_template", None):
        raise RuntimeError(f"{load_id} has no tokenizer chat template")

    train_rows = load_rows(args.train_files)
    val_rows = load_rows(args.val_files)
    random.Random(args.seed).shuffle(train_rows)
    random.Random(args.seed + 1).shuffle(val_rows)
    if args.max_train_examples is not None:
        train_rows = train_rows[: args.max_train_examples]
    if args.max_val_examples is not None:
        val_rows = val_rows[: args.max_val_examples]
    train_dataset = ChatDataset(
        train_rows,
        tokenizer=tokenizer,
        max_length=args.max_length,
        label_last_assistant_only=args.label_last_assistant_only,
        overlength_policy=args.overlength_policy,
    )
    val_dataset = ChatDataset(
        val_rows,
        tokenizer=tokenizer,
        max_length=args.max_length,
        label_last_assistant_only=args.label_last_assistant_only,
        overlength_policy=args.overlength_policy,
    )
    atomic_write(
        args.output_dir / "encoding_report.json",
        json_bytes(
            {
                "train": asdict(train_dataset.report),
                "validation": asdict(val_dataset.report),
            }
        ),
    )
    assert_overlength_rate(
        train_dataset.report,
        args.max_overlength_fraction,
        "training",
    )
    assert_overlength_rate(
        val_dataset.report,
        args.max_overlength_fraction,
        "validation",
    )
    if not train_dataset:
        raise RuntimeError("No training examples survived exact encoding")

    dtype = getattr(torch, args.torch_dtype)
    model, loaded_adapter = load_model(
        load_id,
        base_model_revision=args.base_model_revision,
        torch_dtype=dtype,
        trust_remote_code=args.trust_remote_code,
    )
    if len(tokenizer) > model.get_input_embeddings().num_embeddings:
        model.resize_token_embeddings(len(tokenizer))
    device = torch.device("cuda")
    model.to(device)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    if args.use_lora and not loaded_adapter:
        from peft import LoraConfig, TaskType, get_peft_model

        targets = resolve_lora_target_modules(model, args.lora_target_modules)
        model = get_peft_model(
            model,
            LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
                target_modules=targets,
            ),
        )
    elif not args.use_lora and loaded_adapter:
        raise ValueError("Adapter model supplied without --use-lora")
    if args.use_lora and hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()
    model.train()

    collator = ChatCollator(tokenizer.pad_token_id)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
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
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        fused=True,
    )
    warmup_steps = max(1, int(args.max_steps * args.warmup_ratio))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=args.max_steps,
    )

    step = 0
    batch_cursor = 0
    if resume_checkpoint:
        saved = torch.load(
            resume_checkpoint / "training_state.pt",
            map_location="cpu",
            weights_only=False,
        )
        if saved.get("input_manifest") != input_manifest:
            raise RuntimeError("Refusing resume: input manifest changed")
        step = int(saved["step"])
        batch_cursor = int(saved["batch_cursor"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        restore_rng_state(saved["rng"])
        print(f"Resumed {resume_checkpoint} at optimizer step {step}", flush=True)

    cycling = CyclingLoader(train_loader, batch_cursor)
    optimizer.zero_grad(set_to_none=True)
    running_loss = 0.0
    running_items = 0
    while step < args.max_steps:
        for _micro_step in range(args.grad_accum):
            batch = {key: value.to(device) for key, value in cycling.next().items()}
            loss = model(**batch).loss / args.grad_accum
            loss.backward()
            running_loss += float(loss.detach().cpu()) * args.grad_accum
            running_items += 1
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1

        if step == 1 or step % 10 == 0:
            average = running_loss / max(1, running_items)
            print(
                f"step {step:06d}/{args.max_steps} train_loss={average:.4f} "
                f"lr={scheduler.get_last_lr()[0]:.3e}",
                flush=True,
            )
            running_loss = 0.0
            running_items = 0
        if args.eval_every and step % args.eval_every == 0:
            validation_loss = evaluate_loss(model, val_loader, device)
            print(f"step {step:06d} val_loss={validation_loss:.4f}", flush=True)
        if args.save_every and step % args.save_every == 0:
            save_checkpoint(
                model=model,
                tokenizer=tokenizer,
                optimizer=optimizer,
                scheduler=scheduler,
                output_dir=args.output_dir,
                step=step,
                batch_cursor=cycling.cursor,
                args=args,
                input_manifest=input_manifest,
            )
        if STOP_REQUESTED:
            save_checkpoint(
                model=model,
                tokenizer=tokenizer,
                optimizer=optimizer,
                scheduler=scheduler,
                output_dir=args.output_dir,
                step=step,
                batch_cursor=cycling.cursor,
                args=args,
                input_manifest=input_manifest,
            )
            print("Checkpointed after scheduler signal; requesting requeue", flush=True)
            return PREEMPT_EXIT_CODE

    save_checkpoint(
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        scheduler=scheduler,
        output_dir=args.output_dir,
        step=step,
        batch_cursor=cycling.cursor,
        args=args,
        input_manifest=input_manifest,
        final=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
