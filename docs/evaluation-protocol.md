# Delegent Evaluation Protocol

## Scope

Delegent measures retrieval quality under deterministic, fixed-slot model routing. It does not claim end-to-end answer accuracy, and a reduction in Sol actions is not automatically a reduction in latency, energy, FLOPs, or total deployment cost.

## Retrieval loop

Each question begins with one direct BM25 retrieval. This retrieval is shared across systems and is outside the generated-call cap. Each generated query returns up to five passages from the same frozen corpus.

For cap `p`, the schedule

```text
[Sol × n, Gemma × m], where n + m = p
```

assigns the early slots to GPT-5.6 Sol and the suffix to Gemma. HotpotQA uses `p=10`. The direct retrieval, prompt, schema, search index, hit counts, context windows, and fusion rule are identical across arms.

## Matched-prefix construction

Routed HotpotQA arms are derived from a completed ten-call Sol-Only trajectory:

1. copy the direct retrieval exactly;
2. copy the first `n` Sol generation and search records exactly;
3. reveal only that copied history to Gemma;
4. generate slots `n+1` through `10` with Gemma;
5. preserve retry, repair, and token telemetry.

Sol-Only prefixes with one through nine calls are zero-call views of the completed ten-call Sol-Only trajectory. This prevents stochastic regeneration or prompt drift from being mistaken for a routing effect.

## Qrel-blind generation

Generation inputs may contain:

- the question;
- prior generated queries;
- bounded retrieved evidence available at that slot;
- fixed model and retrieval configuration.

Generation inputs must not contain:

- qrels or required facts;
- reference answers;
- future Sol queries;
- per-question scores or aggregate metrics;
- evaluation feedback.

The generation command does not take a qrels path. Each finished arm receives a completion manifest binding its arm label, question count, and record-set SHA-256. The separate scorer validates every requested candidate and reference completion manifest before it opens the qrels path.

## Dataset separation

The post-trained Gemma uses qrel-blind Sol search trajectories from training questions isolated from the HotpotQA evaluation set and randomly sampled BrowseComp-Plus questions. No SQuAD evaluation question is used for training. Dataset overlap checks use normalized question text in addition to IDs.

## HotpotQA metrics

Let `F_q` be the required fact set for question `q`, and let `F_q(R)` be the required facts supported by retrieved documents in ranking or union `R`.

```text
fact_coverage(q, R) = |F_q(R)| / |F_q|
macro_fact_coverage = mean_q fact_coverage(q, R)
```

Union fact coverage deduplicates documents across the direct and generated-query rankings. RRF@20 first combines rankings with reciprocal-rank fusion,

```text
RRF(d) = sum_r 1 / (60 + rank_r(d)),
```

then evaluates the top 20 fused documents.

## SQuAD metric and sample

SQuAD union recall is the fraction of questions for which the deduplicated retrieval union contains at least one positive document.

An independent eight-question pilot is excluded first. The remaining supported questions are ranked by a frozen salted SHA-256 digest of each QID, and the first 992 form the fixed random evaluation sample used by both direct-retrieval depths and every Sol-call prefix. The reported sample should not be interpreted as untouched confirmation evidence.

## Cost accounting

Sol cost converts recorded tokens using the frozen evaluation prices:

- $4.00 per million uncached input tokens;
- $0.40 per million cached input tokens;
- $20.00 per million output tokens.

Reasoning tokens are included in output tokens and are not counted twice. Local Gemma inference is accounted for separately. GPU amortization, latency, energy, failed attempts, and one-time post-training are not silently folded into the Sol-cost axis.

The public `summarize_cost.py` command validates the arm's completion manifest before reading token telemetry and emits aggregate-only counts, rates, and costs without QIDs. It treats cached input as a subset of total input, charges the remaining input at the uncached rate, and reports reasoning tokens as a diagnostic subset of already billed output tokens.

## Failure and resume rules

- Completed records are immutable.
- The live routed-family runner fsyncs a content-minimal intent before every model call and writes a hash-and-usage receipt after a known response; raw responses are not retained in this ledger.
- An unresolved intent or transport failure is treated as unknown spend and is never automatically replayed.
- A known accepted response without its completed routed record also blocks automatic resume, because the public hash-only ledger cannot reconstruct that action safely.
- A retry or deterministic repair is explicit in telemetry.
- A routed record is rejected if its copied prefix differs from the canonical source.
- Scoring is aggregate-only and begins after completion.

The released table reports the realized paid Sol cost from the evaluation runs. Some published routed slots allowed a charged canonical Sol recovery after two invalid Gemma outputs, so scheduled and realized actor counts can differ. The clean public runner instead offers an explicitly logged, qrel-blind deterministic repair built only from the question and already retrieved text.
