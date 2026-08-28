# Released aggregate results

This directory contains only aggregate reporting tables used by the README.

- `hotpotqa.csv`: 366-question HotpotQA direct, routed, frontier-only, and Gemma-only results.
- `squad.csv`: frozen 3,741-question SQuAD selection results.

In `hotpotqa.csv`, the frontier/delegate columns describe the fixed schedule, while `sol_cost_usd` reports realized paid Sol usage. Some evaluation slots allowed a charged Sol recovery after repeated invalid Gemma actions, so the schedule columns are not a call-telemetry ledger.

The public `scripts/summarize_cost.py` command recreates aggregate frontier token and USD summaries from a completed record set. It validates the completion manifest first and emits no QIDs or per-question values.

The repository does not release QID lists, per-question scores, qrels, retrieved passages, raw model responses, or private execution receipts here.
