# Released aggregate results

This directory contains only aggregate reporting tables used by the README.

- `hotpotqa.csv`: 366-question HotpotQA direct, routed, Sol-Only endpoint, and Gemma-only results, including both post-trained and official Gemma checkpoints.
- `hotpotqa_sol_only.csv`: complete 0--10 Sol-call union-fact curves with 20- and 50-document direct retrieval.
- `squad.csv`: complete 0--5 Sol-call union-recall curves on the same randomly sampled 992 questions, with 20- and 50-document direct retrieval.
- `PROVENANCE.md`: SHA-256 bindings for the aggregate score artifacts and released figures.

In `hotpotqa.csv`, the frontier/delegate columns describe the fixed schedule, while `sol_cost_usd` reports realized paid Sol usage. Some evaluation slots allowed a charged Sol recovery after repeated invalid Gemma actions, so the schedule columns are not a call-telemetry ledger. Savings use ten-call Sol-Only with 20-document direct retrieval as the reference.

The SQuAD table intentionally omits cost: it reports retrieval quality on the shared 992-question sample, whereas the completed generation campaign covered a different number of trajectories. Mixing those denominators would be misleading.

The public `scripts/summarize_cost.py` command recreates aggregate frontier token and USD summaries from a completed record set. It validates the completion manifest first and emits no QIDs or per-question values.

The repository does not release QID lists, per-question scores, qrels, retrieved passages, raw model responses, or private execution receipts here.
