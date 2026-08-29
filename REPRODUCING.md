# Reproducing Delegent

This guide describes the public, staged reproduction contract. Generation and scoring are deliberately separate: model actors never receive qrels or reference answers, and scoring begins only after the expected record set is complete.

## 1. Install and test

From the root of a clone or downloaded ZIP of this repository:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
python -m pytest -q
```

Paths inside a routed-family JSON configuration resolve relative to that configuration file. Other command-line paths resolve relative to the current shell directory.

## 2. Validate the routed-family configuration

```bash
python scripts/run_routed_family.py \
  --config examples/demo_config.json \
  --dry-run
```

Dry-run mode validates the synthetic source records and schedules and hashes the first delegate prompt for each schedule. It does not require endpoints, open credentials, make requests, or write outputs.

For a live run, copy [`configs/hotpotqa-c10.json`](configs/hotpotqa-c10.json) to a local ignored file and add the deployment-specific `retrieval` and `delegate` blocks. Keep tokens in the environment variables named by `token_env` and `api_key_env`; never put credential values in JSON. See [`configs/README.md`](configs/README.md) for the complete schema.

## 3. Prepare canonical Sol-Only records

The public routed runner consumes a completed canonical frontier-only record set. Each JSON record must contain:

- `qid` and `question`;
- one direct retrieval followed by exactly `cap` generated searches;
- exactly `cap` generation records;
- for each hit: `docid`, `rank`, `score`, and `text`.

The HotpotQA configuration uses `cap=10`. The direct retrieval is outside this generated-call cap.

Do not place qrels, reference answers, per-question scores, or future hidden actions in the generation input. Before routing, freeze the canonical record directory and preserve its content hash outside the public repository.

## 4. Run the routed HotpotQA family

```bash
python scripts/run_routed_family.py \
  --config configs/hotpotqa-c10.local.json
```

The configured frontier prefixes generate the fixed schedules 1+9 through 9+1. For prefix length `n`, the runner copies the direct retrieval and first `n` canonical Sol actions exactly, then calls the delegate only for slots `n+1` through `10`.

A run is admissible only when:

1. the copied prefix equals the canonical prefix exactly;
2. every generated action has a valid actor and slot;
3. the record contains no qrels, answers, or scores;
4. completed records are not silently overwritten;
5. retries and deterministic repairs are explicitly counted.

Before each delegate call, the runner durably records a content-minimal intent. A successful response produces a receipt containing only hashes, token/timing telemetry, and validation status; transport uncertainty produces an immutable unknown-spend receipt and stops that question without retry. On `--resume`, unresolved or unknown-spend attempts fail closed. An accepted response whose final routed record was never committed also fails closed because raw responses are deliberately not stored.

## 5. Derive zero-call Sol-Only prefix views

Sol-Only prefixes with one through nine calls are views of the completed ten-call Sol-Only trajectory and require no new model calls. The example paths retain the historical `allsol` directory name.

```bash
python scripts/derive_prefixes.py \
  --source-records data/hotpotqa/allsol10/records \
  --output-dir runs/hotpotqa/allsol-prefixes \
  --caps 1 2 3 4 5 6 7 8 9 \
  --frontier-actor gpt-5.6-sol
```

The derivation must preserve the direct retrieval and take only the first `k` generation/search slots. It must not open qrels.

## 6. Freeze the SQuAD evaluation sample

Exclude the independent pilot before sampling. The `--questions` input must therefore contain exactly the 7,482 supported non-pilot rows, with only `qid` and `question` fields. The split is deterministic but pseudorandom: rank the remaining QIDs by a frozen salted SHA-256 digest, use the first 992 for the released evaluation sample, and reserve the remainder.

```bash
python scripts/freeze_split.py \
  --questions data/squad/nonpilot-questions.jsonl \
  --salt "${DELEGENT_SPLIT_SALT:?set the frozen public protocol salt}" \
  --selection-count 992 \
  --selected runs/squad/split/sample992.jsonl \
  --remainder runs/squad/split/remainder.jsonl \
  --manifest runs/squad/split/manifest.json
```

Use the exact public protocol salt rather than choosing one after inspecting any labels. The manifest records that salt plus the input and output hashes. Freeze those artifacts before generating or scoring sample trajectories. Do not inspect qrels while choosing the sample.

## 7. Score only after completion

Generation must finish and the record set must be frozen before either scorer is given a qrels path.

```bash
python scripts/score_hotpotqa.py \
  --records runs/hotpotqa/c10/frontier5_delegate5/records \
  --completion runs/hotpotqa/c10/frontier5_delegate5/completion.json \
  --qrels data/hotpotqa/qrels.jsonl \
  --output runs/hotpotqa/aggregate-score.json \
  --reference-records data/hotpotqa/allsol10/records \
  --reference-completion data/hotpotqa/allsol10/completion.json \
  --bootstrap-repetitions 10000

python scripts/score_squad.py \
  --records runs/squad/allsol-prefixes/frontier5/records \
  --completion runs/squad/allsol-prefixes/frontier5/completion.json \
  --qrels data/squad/sample992-qrels.jsonl \
  --output runs/squad/aggregate-score.json
```

The routed-family and prefix-derivation commands write `completion.json` beside each arm's `records/` directory. A completion manifest binds the arm, question count, and record-set SHA-256. The scorers require this manifest and validate it before opening the qrels file. When `records/` follows the standard layout, `--completion` may be omitted and the adjacent manifest is inferred; explicit paths are shown above. Paired scoring independently requires and validates `--reference-completion` in the same way.

Each qrels row contains `qid`, `positive_docids`, and, for HotpotQA, `doc_fact_ids`, which maps positive document IDs to required-fact IDs. A scorer requires the record and qrels QID sets to match exactly; prepare the SQuAD sample records and sample-only qrels before invoking it. Scorers emit aggregate metrics only. Public results must not contain QID lists, per-question scores, retrieved passage text, raw responses, or qrels.

## 8. Summarize frontier cost after completion

Cost accounting is a separate, aggregate-only command. Like scoring, it refuses to inspect the records until their completion manifest matches the arm, question count, and record-set hash.

```bash
python scripts/summarize_cost.py \
  --records runs/hotpotqa/c10/frontier5_delegate5/records \
  --completion runs/hotpotqa/c10/frontier5_delegate5/completion.json \
  --frontier-actor gpt-5.6-sol \
  --uncached-input-rate 4.00 \
  --cached-input-rate 0.40 \
  --output-rate 20.00 \
  --output runs/hotpotqa/frontier5_delegate5-cost.json
```

Rates are USD per million tokens. Total input includes cached input; cached tokens receive the cached rate and the remainder receives the uncached rate. Reasoning tokens are already a subset of output tokens and are never billed twice. The summary contains aggregate token counts, realized frontier attempts, total and mean USD cost, rates, and content hashes—but no QIDs or per-question values. It does not estimate Gemma serving, GPU amortization, latency, energy, failed unrecorded requests, or post-training cost.

As with scoring, `--completion` may be omitted under the standard `arm/records` plus `arm/completion.json` layout. To summarize a derived SQuAD prefix, use `--records runs/squad/allsol-prefixes/frontier5/records` with its adjacent completion manifest.

## 9. Validate the post-training release

The release includes a dependency-free dry-run over a synthetic training row:

```bash
python scripts/train_retrieval_sft.py \
  --config configs/gemma4-sft.json \
  --train-files examples/training_example.jsonl \
  --validation-files examples/training_example.jsonl \
  --output-dir /tmp/delegent-sft-smoke \
  --dry-run
```

For GPU training, install the exact locked environment with
`uv sync --extra train --frozen` before removing `--dry-run`.

The effective final-stage hyperparameters and cumulative lineage are documented
in [`docs/post-training.md`](docs/post-training.md). The released trainer is a
reviewed reproduction implementation, not the byte-identical historical
trainer. The cumulative input adapter, raw teacher trajectories, and final
adapter are not included, so the published checkpoint cannot be reconstructed
from this repository alone.

## 10. Metrics

HotpotQA reports:

- macro union required-fact coverage;
- RRF@20 required-fact coverage with `k=60`;
- positive-document recall as a secondary diagnostic.

SQuAD reports union-any positive-document recall on the frozen 992-question sample. The reported result is not untouched confirmation evidence.

The public scorers compute retrieval metrics and optional paired bootstrap comparisons. Dollar cost comes from the separate completion-gated `summarize_cost.py` command under the price rules in [the evaluation protocol](docs/evaluation-protocol.md). The training implementation is released, while cumulative training inputs and checkpoint distribution are not.

## 11. Contamination and privacy checklist

Before publishing a run, verify all of the following:

- generation inputs contain questions and retrieval history only;
- qrels and reference answers are scorer-only;
- the evaluation questions were excluded from post-training data;
- no SQuAD evaluation question was used for Gemma post-training;
- copied Sol prefixes are exact and require zero new Sol calls;
- unknown-spend calls are never replayed automatically;
- endpoints, usernames, absolute machine paths, tokens, OAuth files, and private checkpoint paths are absent from committed files;
- only aggregate result tables are committed.

## 12. Released aggregate tables

The values displayed in the README are available as:

- [`results/hotpotqa.csv`](results/hotpotqa.csv)
- [`results/squad.csv`](results/squad.csv)

These are compact reporting artifacts, not generation inputs.
