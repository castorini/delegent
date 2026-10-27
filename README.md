# Delegent

Delegent is an agent harness in which a frontier agent stays in charge of every task but hands focused subtasks to a
small language model (SLM) that runs on hardware you control. The SLM never answers on its own: it does the legwork,
such as reading code, running tests or searching a corpus, and the frontier agent reads its short report and decides
what to believe. Nothing is trained, so either model can be swapped.

With GPT-6.1-Sol as the frontier agent and Qwen3.8-27B-FP8 as the SLM, Delegent cuts the frontier cost of long-horizon
coding on DeepSWE to 33% and of agentic search to 41%, with no loss in task performance. Even the 2B MiniCPM5-2B cuts
the coding cost to a third.

| Task set | Solved, frontier alone | Solved, + Qwen3.8-27B-FP8 | Frontier cost, + Qwen3.8-27B-FP8 |
|---|---|---|---|
| SWE-bench Verified (50) | 46 | 46 | 44% |
| SWE-bench Pro (75) | 63 | 63 | 37% |
| DeepSWE v1.1 (50) | 29 | 29 | 33% |
| Agentic search, 6 evaluations (694) | 663 | 663 | 41% |

Frontier cost is what the frontier model is charged, with prompt caching, relative to the frontier agent alone in the
same harness. [REPRODUCING.md](REPRODUCING.md) gives the settings behind these numbers.

## How it works

```
                 frontier agent  (metered API: OpenAI Responses or Anthropic Messages)
                   |         ^
   Delegate(task)  |         |  report (+ automatic evidence check on search)
                   v         |
              Delegent relay / router  ---- one usage ledger for both models
                   |         ^
                   v         |
          SLM subagents, in parallel, each in its own tool loop and fresh context
                         (vLLM replicas on your GPU)
```

Delegent is one harness with one routing rule: calls from the agent in charge go to the frontier model, and every
subtask runs on the SLM. Only a subtask's final report reaches the frontier agent, never its tool output, so the
frontier agent's context stays short and its prompt cache keeps hitting.

- **Coding.** Claude Code runs unmodified against the relay (`delegent claude`). Its main agent uses the frontier
  model; its subagents (the built-in Agent tool) and its small auxiliary model use the SLM. A short instruction,
  supplied as a user-level `CLAUDE.md` ([coding.md](src/delegent/instructions/coding.md)), asks the frontier agent to
  delegate investigation and verification and to make every code change itself.
- **Agentic search.** The frontier agent searches a corpus with `search` and `get_document` and gets one more tool,
  `Delegate`, together with the [search instruction](src/delegent/instructions/search.md). A Delegate call starts an
  SLM subagent that searches the same corpus and returns an answer with evidence and a confidence; the harness then
  looks up every evidence line in the cited document and reports whether it was found verbatim.

Frontier calls carry a stable prompt-cache key per conversation, and each SLM subagent is pinned to one vLLM replica
so that its growing prefix stays cached there.

## Install

```bash
pip install git+https://github.com/castorini/Delegent.git
```

Python 3.10 or newer. For coding, install [Claude Code](https://docs.anthropic.com/en/docs/claude-code) as well.

## Serve the SLM

Any OpenAI- and Anthropic-compatible vLLM server works. The paper served each SLM on one GPU per replica:

```bash
vllm serve Qwen/Qwen3.8-27B-FP8 --served-model-name Qwen3.8-27B-FP8 --max-model-len 131072 \
  --kv-cache-dtype fp8 --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3
```

```bash
vllm serve openbmb/MiniCPM5-2B --served-model-name openbmb/MiniCPM5-2B --max-model-len 131072 \
  --dtype bfloat16 --enable-auto-tool-choice --tool-call-parser minicpm5 --reasoning-parser deepseek_r1
```

Several replicas of one model are given as a comma-separated `--slm` list.

## Coding with Claude Code

```bash
export OPENAI_API_KEY=...        # or ANTHROPIC_API_KEY with --frontier claude-...
delegent claude --frontier gpt-6.1-sol --slm http://localhost:8000 --slm-model Qwen3.8-27B-FP8
```

Everything after `--` goes to Claude Code, for example a headless run in a repository:

```bash
delegent claude --frontier gpt-6.1-sol --slm http://localhost:8000 --slm-model Qwen3.8-27B-FP8 -- \
  -p "Fix the failing test in tests/test_parser.py" --permission-mode bypassPermissions --max-turns 100
```

`delegent claude` starts the relay, writes the coding instruction to a fresh `CLAUDE_CONFIG_DIR`, points Claude Code at
the relay, and prints the frontier cost when Claude Code exits. Every model call is logged to `delegent-usage.jsonl`.
To drive another Anthropic-API agent, run the relay on its own with `delegent relay`, which prints the environment to
set.

## Agentic search

Questions are JSON lines with `id` and `question`. On the ClimbMix corpus (BM25 search API):

```bash
export CLIMBMIX_TOKEN=...
delegent search --frontier gpt-6.1-sol --slm http://localhost:8000 --slm-model Qwen3.8-27B-FP8 \
  --corpus climbmix --questions questions.jsonl --out runs/climbmix-delegent
```

On a local corpus of text files, build a BM25 index first:

```bash
delegent index --corpus corpus/ --out index/
delegent search --frontier gpt-6.1-sol --slm http://localhost:8000 --slm-model Qwen3.8-27B-FP8 \
  --corpus local --corpus-root corpus/ --index index/ --questions questions.jsonl --out runs/local-delegent
```

`--no-delegate` runs the frontier agent alone, the baseline. Each question's answer, frontier usage and delegations land
in `<out>/questions/<id>.json`, and every model call in `<out>/usage.jsonl`.

## Cost

```bash
delegent cost delegent-usage.jsonl --prices gpt-6.1-sol=2,0.10,10
```

Frontier cost is $2 per million uncached input tokens, $0.10 per million cached input tokens and $10 per million
output tokens for GPT-6.1-Sol (reasoning tokens count as output). `--prices model=uncached,cached,output` sets other
prices.

## Repository layout

| Path | Contents |
|---|---|
| `src/delegent/relay.py` | the relay: Anthropic Messages API in, routing by role, one usage ledger |
| `src/delegent/frontier.py` | frontier backends: OpenAI Responses (with the Anthropic translation) and Anthropic |
| `src/delegent/slm.py` | SLM backend: vLLM replicas, conversation pinning, window clamping |
| `src/delegent/delegate.py` | the Delegate tool, the SLM search subagent and the evidence check |
| `src/delegent/agent.py` | the search agent loop |
| `src/delegent/corpus.py`, `bm25.py` | ClimbMix API client, local corpus with a BM25 index |
| `src/delegent/instructions/` | the coding and search instructions |
| `samples/` | the task and question ids of the paper's evaluation |
| `tests/` | offline tests with fake model servers |

The code used for an earlier, unrelated routing study lives under the tag `legacy/fixed-schedule-routing`.

## License

Apache 2.0. The SLM search subagent uses the prompts and tool schemas of
[AREX-Turbo](https://huggingface.co/BAAI/AREX-Turbo) (Apache 2.0); see [NOTICE](NOTICE).
