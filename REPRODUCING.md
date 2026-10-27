# Reproducing the paper

This file lists the settings of the paper's experiments and how this release differs from the research code that ran
them. The research code was tied to our cluster (job queues, container images, SSH tunnels); this release keeps the
harness itself and drops the cluster plumbing.

## Models

| Role | Model | Serving |
|---|---|---|
| Frontier agent | GPT-6.1-Sol, high reasoning effort | OpenAI Responses API |
| Frontier agent (ablation) | Claude Opus 5.5 | Anthropic Messages API |
| SLM | Qwen3.8-27B-FP8 | vLLM, one GPU per replica, 131,072-token window |
| SLM | MiniCPM5-2B (coding only) | vLLM, one GPU per replica, 131,072-token window |

SLM sampling on coding: temperature 1.0, top-p 0.95; Qwen3.8-27B-FP8 thinks with reasoning effort `medium`. On search,
Qwen3.8-27B-FP8 answers without thinking, with temperature 0.7 and top-p 0.8.

## Coding

- Claude Code 2.1.232 with its default tools except WebSearch and WebFetch, `--max-turns 100`,
  `--permission-mode bypassPermissions`, `--print --output-format stream-json`.
- Context: up to 1M tokens for the frontier model (`CLAUDE_CODE_MAX_CONTEXT_TOKENS=1000000`, the default of
  `delegent claude`) and 131,072 tokens for the SLM; at most 32,768 output tokens per call.
- Tool time: 60 minutes per task (90 on DeepSWE), not counting time spent waiting for a model.
- The coding instruction ([coding.md](src/delegent/instructions/coding.md)) is the user-level `CLAUDE.md`; the frontier
  agent alone runs without it.
- Prompts: SWE-bench Verified gets "Please fix the following issue in the repository at {path}. Inspect the code,
  implement the fix, and run relevant tests. The repository and dependencies are available locally." followed by the
  problem statement; SWE-bench Pro adds the task's requirements and interface; DeepSWE uses the task's own
  instruction unchanged.
- Each task ran in its own container from the task's official image, without network access and without the
  repository's `.git`, and was graded with the benchmark's official harness from the final patch.
- Tasks: [samples/coding](samples/coding), drawn with `random.Random(20261020)` from the tasks that every compared
  system finished without an infrastructure failure.

## Agentic search

- The frontier agent has `search` (top 10 BM25 hits with the best-matching passage) and `get_document` (8,000
  characters per page), at most 100 turns, then one turn without tools for the final answer.
- With delegation it also has `Delegate` and the [search instruction](src/delegent/instructions/search.md). Each call
  runs one SLM subagent of at most 80 steps, with no retries.
- DeepSearchQA questions have several answers and run in two modes, `--multi-answer one` and `--multi-answer all`.
- Answers are graded against the gold answer by GPT-6.1-Sol at low effort with the official BrowseComp-Plus grading
  prompt, the same judge for every system.
- Questions: [samples/search](samples/search). The 200 BrowseComp-Plus questions are searched over the BrowseComp-Plus
  corpus; the other sets come from CMASS and are searched over ClimbMix.

## Differences from the research code

- **Frontier access.** The research code reached the Responses API through a job queue; this release calls the API
  directly with `OPENAI_API_KEY`. The request translation and the prompt-cache key are the same.
- **One relay process.** The research code started a relay per task inside the task container's network namespace;
  `delegent claude` starts one relay per Claude Code session.
- **Replica pinning.** The research code placed subagent conversations on replicas with a load-aware sticky proxy;
  this release pins each conversation to a replica by a hash of its opening.
- **Frontier tools on the BrowseComp-Plus corpus.** In the paper, the frontier agent searched the BrowseComp-Plus
  corpus with Claude Code's Bash and Read tools (ripgrep over the files), while its SLM subagent used BM25. This release
  gives the frontier agent the same `search` and `get_document` tools on every corpus.
- **Subagent prompt on ClimbMix.** The research code told the SLM subagent to cite `/corpus/` paths on both corpora;
  this release names ClimbMix documents by docid.
- **Not included.** Container images, task graders and the answer judge; use each benchmark's official harness.
