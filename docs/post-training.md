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

## Reproduction boundary

The released trainer is a reviewed implementation of the disclosed training
objective and effective configuration. It is not the byte-identical historical
trainer and does not guarantee the historical optimizer, data-ordering, or
resume behavior. The final adapter cannot be reconstructed from this repository alone:
the cumulative input adapter and raw teacher trajectories are not distributed.
The first two HotpotQA-source stages selected rows using qrel metrics from the
training partition; qrels and answers were not placed in model messages. No
evaluation question was used as a training or validation row.
