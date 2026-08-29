# Post-training protocol

The reported local actor is a rank-16 LoRA on the frozen
`google/gemma-4-E4B-it` base. It is a cumulative adapter: three external
retrieval-supervision stages precede the final BrowseComp-Plus continuation.
The final stage mixes 3,694 BrowseComp-Plus training-prefix examples with 436
external long-prefix examples and validates on 76 external examples.

The BrowseComp-Plus source uses a frozen correlation-group-aware split with
580 training, 125 development, and 125 test questions. The split seed string is
`bcp-plus-prospective-holdout-v1-2026-07-25`; only the training partition
contributes final-stage training prefixes.

[`configs/gemma4-sft.json`](../configs/gemma4-sft.json) records the effective
hyperparameters of the reported final stage. [`results/training_lineage.json`](../results/training_lineage.json)
binds the released description to the reported adapter and training artifacts
without exposing questions, qrels, or model responses.

## Message format

Each JSONL row contains a `messages` list and optional `metadata`. The final
assistant turn is the structured next-search target. Only that turn contributes
to the loss. The historical final-stage configuration skipped overlength rows
but observed none; the released implementation instead rejects an overlength
row so that a search state is never silently changed.
[`examples/training_example.jsonl`](../examples/training_example.jsonl) is
synthetic and can be used for a no-model validation:

```bash
python scripts/train_retrieval_sft.py \
  --config configs/gemma4-sft.json \
  --train-files examples/training_example.jsonl \
  --validation-files examples/training_example.jsonl \
  --output-dir /tmp/delegent-sft-smoke \
  --dry-run
```

For GPU training, run `uv sync --extra train --frozen` to install the locked
environment, then remove `--dry-run`. To
continue an existing LoRA, pass `--input-adapter`. A from-base run omits that
flag and creates a fresh LoRA with the rank, alpha, and dropout in the config.
The reported final-stage environment used Python 3.11.15, PyTorch 2.13.0,
Transformers 5.14.1, PEFT 0.19.1, Accelerate 1.14.0, and Safetensors 0.8.0;
the `train` extra pins these package versions.

The reported final stage ran for 28 minutes 45 seconds on one NVIDIA RTX 6000
Ada Generation GPU (46,068 MiB), with 12 CPU cores and 150 GiB RAM allocated;
peak process memory recorded by the scheduler was approximately 19.8 GiB. This
is the final continuation stage only, not total project compute.

## Historical training code

The repository also preserves the exact Python sources used by the reported
training lineage:

- [`scripts/train_hf_retrieval_sft.py`](../scripts/train_hf_retrieval_sft.py)
  trained stages 0--2; SHA-256
  `f579f117e2a1e7631148e776788a1bc0d2dccf1a6cb89e1e4ae0e91907e7fe29`.
- [`scripts/train_hf_retrieval_sft_resumable.py`](../scripts/train_hf_retrieval_sft_resumable.py)
  trained the final BrowseComp-Plus continuation; SHA-256
  `c289c8d6063d74d547d838addc1e9da3f0fb6c9c98f4673e62450566d8f6e989`.

These files are byte-for-byte reconstructions of the sources bound by the
archived training manifests and final-stage job log. The resumable trainer
requires a Slurm GPU allocation and records input hashes, checkpoints,
optimizer state, and preemption-safe resume state. The smaller
[`scripts/train_retrieval_sft.py`](../scripts/train_retrieval_sft.py) remains a
reviewed convenience implementation for validating the disclosed message
format and objective without model loading.

The exact stage-1 data builder is also released as
[`scripts/build_sol_next_query_sft.py`](../scripts/build_sol_next_query_sft.py),
SHA-256
`1c035000abfbbf3f7cae2ec4bb2450aca9e78567dd169c146d4e2504cca2f7c5`.
It implements the grouped-by-question split, source-row filters, prompt
construction, and prefix-to-next-query targets for that stage.

## Four-stage settings

Stages 0--2 use the exact `f579...` trainer with fused AdamW, linear warmup and
decay, gradient-norm clipping at 1.0, bfloat16, gradient checkpointing,
final-assistant-only loss, microbatch size 1, gradient accumulation 8, and the
Gemma text q/k/v/o and gate/up/down projections as LoRA targets. Stage 0 creates
the rank-16, alpha-32, dropout-0.05 adapter; stages 1--3 continue it.

| Stage | Data construction and split | Maximum length | Steps / learning rate | Evaluation / save | Seed |
|---|---|---:|---:|---:|---:|
| 0 | 88 HotpotQA-style source trajectories; 68 pass training-partition thresholds; one query bundle each; seeded 58/10 row split; 5 hits, at most 6 calls, 80 characters per document | 2,048 | 50 / `1e-5` | 25 / 50 | 13 |
| 1 | 88 source rows; 53 pass the same thresholds plus parse-success and no-tool-error gates; prefix 4 to query 5 and prefix 5 to query 6; grouped 45/8 question split gives 87/16 rows; 5 hits, at most 2 documents per prior call, 240 characters per passage | 2,048 | 80 / `5e-6` | 20 / 20 | 13 |
| 2 | 128 qrel-blind 12-search trajectories; four targets at prefixes 8--11; grouped 109/19 question split gives 436/76 rows; at most 1 document per prior call and 160 whitespace tokens per passage | 6,144 | 120 / `3e-6` | 20 / 20 | 56 |
| 3 | 3,694 BrowseComp-Plus training-prefix rows plus 436 stage-2 rows; 4,130/76 rows | 16,384 | 240 / `2e-6` | 40 / 40 | 56 |

All stages use weight decay 0 and warmup ratio 0.03. Stages 0 and 1 selected
source rows using training-partition qrel metrics, but qrels and answers were
never placed in model messages. Stage 2 selected step 120 by external
assistant-only validation loss subject to strict structured-output and
novel-query checks. The historical stages 0--2 trainer left-truncated rows over
the configured maximum; the stage-2 preflight observed a 5,325-token maximum,
below its 6,144-token limit. Exact stage-0 and stage-1 overlength counts were not
archived. The final stage skipped overlength rows and observed none.

The sanitized aggregate hashes and stage-level settings are recorded in
[`results/training_lineage.json`](../results/training_lineage.json). Historical
raw manifests are not published because they contain machine paths and, for one
stage, question identifiers.

## Reproduction boundary

The exact historical stage trainers and a reviewed convenience implementation
are released. The final adapter still cannot be reconstructed from this
repository alone: the cumulative input adapter and raw teacher trajectories are
not distributed.
The first two HotpotQA-source stages selected rows using qrel metrics from the
training partition; qrels and answers were not placed in model messages. No
evaluation question was used as a training or validation row.
