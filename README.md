<h1 align="center">Delegent</h1>

<p align="center">
  <b>A multi-agent harness that reduces cost without sacrificing performance.</b><br>
  Built on Claude Code. The frontier model leads; a small language model on your own GPU does the legwork.
</p>

<p align="center">
  <a href="LICENSE"><img alt="License: Apache 2.0" src="https://img.shields.io/badge/license-Apache%202.0-blue.svg"></a>
  <img alt="Python 3.10+" src="https://img.shields.io/badge/python-3.10%2B-blue.svg">
  <img alt="Built on Claude Code" src="https://img.shields.io/badge/built%20on-Claude%20Code-orange.svg">
  <img alt="SLM on vLLM" src="https://img.shields.io/badge/SLM-vLLM-green.svg">
</p>

<table align="center">
  <tr>
    <td align="center"><h2>33%</h2>frontier cost on<br>long-horizon coding<br><sub>DeepSWE, 27B FP8 SLM</sub></td>
    <td align="center"><h2>41%</h2>frontier cost on<br>agentic search<br><sub>694 questions, 27B FP8 SLM</sub></td>
    <td align="center"><h2>34%</h2>with a 2B SLM<br>that fits on a laptop<br><sub>DeepSWE, MiniCPM5-2B</sub></td>
    <td align="center"><h2>No loss</h2>in solved tasks<br>or correct answers<br><sub>nothing is trained</sub></td>
  </tr>
</table>

Agents built on frontier models are expensive, because at every step the model re-reads its entire context. Much of
that work is legwork: locating code, reproducing a bug, running tests, issuing searches, reading documents.
**Delegent** keeps the frontier model in charge of every task and hands this legwork to a small language model (SLM)
running on a GPU you control. The SLM never answers on its own. The frontier model reads its short report, checks what
matters, and writes the final patch or answer. Nothing is trained, so either model can be swapped.

## How it works

<p align="center">
  <img src="assets/overview.png" alt="The user talks to the main loop, where the frontier model leads the task and spawns stacked SLM subagents through the Agent and Delegate tools; both loops act on the environment (a repository or a corpus) through tools such as search, Bash, Read and Edit, plus get_document for the main loop and visit for the subagents" width="640">
  <br>
  <sub>Delegent starts from stock Claude Code, and everything in orange is what it adds. The user talks to the main
  loop, where the frontier model leads the task and spawns subagents. The loops share most tools, and each has a few of
  its own.</sub>
</p>

Delegent is one harness with one rule: **calls from the main loop go to the frontier model, and every subtask runs on
the SLM.** Subtasks issued in the same turn run in parallel, each in its own tool loop with a fresh context, and only
their final reports reach the frontier model, never their tool output. On top of stock Claude Code, Delegent adds:

- **A relay that routes model calls by role.** Claude Code's main loop talks to the frontier model, while its subagents
  and its small auxiliary model talk to the SLM. The frontier model can be an OpenAI or an Anthropic model.
- **One short instruction.** The [instruction](src/delegent/instructions/delegent.md) tells the frontier model what to
  delegate and to keep its own tool use to a few targeted checks. For coding, it is installed as a user-level
  `CLAUDE.md` and asks the frontier model to delegate investigation and verification and to make every code change
  itself.
- **Search tools and a `Delegate` tool.** For agentic search, the frontier model gets `search`, `get_document` and
  `Delegate`. A delegated SLM subagent searches the same corpus with `search` and `visit` and reports an answer with
  evidence.
- **A quote check.** Before a search report comes back, Delegent looks up every evidence line in the cited document
  and notes whether it was found verbatim.
- **Caching on both sides.** Frontier calls carry a stable prompt-cache key per conversation, and each SLM subagent is
  pinned to one vLLM replica, so its growing prefix stays cached there.

## Results

All numbers are from our paper, with GPT-6.1-Sol at high reasoning effort as the frontier model, run in the same
harness with and without delegation. Cost is the frontier model's cost at API prices with prompt caching.

### Long-horizon coding

| Task set | Frontier model alone | + Qwen3.8-27B-FP8 | + MiniCPM5-2B |
|:--|:-:|:-:|:-:|
| SWE-bench Verified (50 tasks) | 46 solved, $0.244 | 46 solved, **44%** cost | 47 solved, **53%** cost |
| SWE-bench Pro (75 tasks) | 63 solved, $0.617 | 63 solved, **37%** cost | 62 solved, **44%** cost |
| DeepSWE v1.1 (50 tasks) | 29 solved, $2.110 | 29 solved, **33%** cost | 30 solved, **34%** cost |

<sub>Dollar amounts are frontier cost per task. On DeepSWE it falls from $2.11 to $0.70 with Qwen3.8-27B-FP8 and to
$0.71 with MiniCPM5-2B. Even with Qwen3.8-27B-FP8 paid for at API prices, a DeepSWE task costs 38% of the frontier
model alone.</sub>

### Agentic search

| Dataset | Frontier model alone | + Qwen3.8-27B-FP8 |
|:--|:-:|:-:|
| BrowseComp-Plus, own corpus (200) | 193 correct, $0.114 | 192 correct, **63%** cost |
| BrowseComp-Plus (80) | 72 correct, $0.211 | 71 correct, **31%** cost |
| Total Recall QA (68) | 64 correct, $0.048 | 67 correct, **31%** cost |
| HotpotQA (204) | 198 correct, $0.033 | 198 correct, **25%** cost |
| DeepSearchQA, find one (71) | 70 correct, $0.067 | 70 correct, **29%** cost |
| DeepSearchQA, find all (71) | 66 correct, $0.121 | 65 correct, **26%** cost |
| **All (694)** | **663 correct, $0.091** | **663 correct, 41% cost** |

<sub>Dollar amounts are frontier cost per question. All datasets except the first are from the CMASS suite over the
ClimbMix corpus, searched with BM25.</sub>

### The savings grow with the legwork

The longer the frontier model would work on a task by itself, the more of that work moves to the SLM, down to about a
quarter of the cost on tasks that take it more than 200 calls alone.

<p align="center">
  <img src="assets/savings-vs-legwork.png" alt="Median relative frontier cost per coding task falls from 0.62 for tasks with fewer than 25 calls alone to 0.26 for tasks with more than 200" width="620">
  <br>
  <sub>Median frontier cost with Delegent relative to the frontier model alone, per coding task, by how many calls the
  frontier model makes on the task alone (350 task pairs, three coding sets, both SLMs).</sub>
</p>

### Two frontier models

The frontier model behind Delegent can change freely. On DeepSWE, we also ran Claude Opus 5.5 at xhigh effort.

| Per DeepSWE task (50 tasks) | GPT-6.1-Sol | Claude Opus 5.5 |
|:--|:-:|:-:|
| How it works alone | long loop of edit, test, fix | plans first, then acts |
| Solved alone | 29 | 34 |
| Solved with an SLM | 29.5 | 38 |
| Frontier cost with an SLM | **34%** | **69%** |

<sub>With an SLM, values average the two SLMs. Delegent saves the most for a frontier model that would otherwise do
much of the legwork itself.</sub>

## Quick start

**1. Install.**

```bash
pip install git+https://github.com/castorini/Delegent.git
# or, from a downloaded copy of this repository
pip install .
```

For coding, also install [Claude Code](https://code.claude.com/docs). Delegent builds its coding harness around your
Claude Code installation: the relay takes Claude Code's model calls (through Claude Code's documented
`ANTHROPIC_BASE_URL` setting) and sends the main loop's calls to the frontier model and the subagents' calls to the
SLM, and the delegation instruction is added as a user-level `CLAUDE.md` in a separate config directory, so your own
Claude Code settings are untouched. Delegent does not bundle Claude Code, and your use of Claude Code
remains subject to Anthropic's terms.

**2. Serve an SLM** with vLLM on your GPU.

```bash
vllm serve Qwen/Qwen3.8-27B-FP8 --served-model-name Qwen3.8-27B-FP8 --max-model-len 131072 \
  --kv-cache-dtype fp8 --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3
```

<details>
<summary>MiniCPM5-2B instead (fits on a laptop GPU)</summary>

```bash
vllm serve openbmb/MiniCPM5-2B --served-model-name openbmb/MiniCPM5-2B --max-model-len 131072 \
  --dtype bfloat16 --enable-auto-tool-choice --tool-call-parser minicpm5 --reasoning-parser deepseek_r1
```

</details>

**3. Code with Claude Code**, with the frontier model in charge and the SLM doing the legwork.

```bash
export OPENAI_API_KEY=...
delegent claude --frontier gpt-6.1-sol --slm http://localhost:8000 --slm-model Qwen3.8-27B-FP8
```

When Claude Code exits, Delegent prints the frontier cost of the session. For a Claude model in charge, pass
`--frontier claude-...` and set `ANTHROPIC_API_KEY` instead.

## Usage

### Coding

Everything after `--` goes to Claude Code. A headless run on one issue:

```bash
delegent claude --frontier gpt-6.1-sol --slm http://localhost:8000 --slm-model Qwen3.8-27B-FP8 -- \
  -p "Fix the failing test in tests/test_parser.py" --permission-mode bypassPermissions --max-turns 100
```

Several SLM replicas go in one comma-separated list, `--slm http://gpu0:8000,http://gpu1:8000`. To drive another
agent that speaks the Anthropic Messages API, run `delegent relay` on its own; it prints the environment to set.

### Agentic search

Questions are JSON lines with an `id` and a `question`.

```bash
# over the ClimbMix corpus (BM25 search API)
export CLIMBMIX_API=...     # base URL of your ClimbMix search endpoint
export CLIMBMIX_TOKEN=...
delegent search --frontier gpt-6.1-sol --slm http://localhost:8000 --slm-model Qwen3.8-27B-FP8 \
  --corpus climbmix --questions questions.jsonl --out runs/climbmix

# over your own documents
delegent index --corpus docs/ --out index/
delegent search --frontier gpt-6.1-sol --slm http://localhost:8000 --slm-model Qwen3.8-27B-FP8 \
  --corpus local --corpus-root docs/ --index index/ --questions questions.jsonl --out runs/local
```

Add `--no-delegate` to run the frontier model alone as a baseline. Every answer lands in `<out>/questions/<id>.json`
with its frontier usage and delegations.

### Measuring cost

Every model call, frontier and SLM alike, is written to a usage ledger (`delegent-usage.jsonl`, or
`<out>/usage.jsonl` for search).

```bash
delegent cost delegent-usage.jsonl --prices gpt-6.1-sol=2,0.10,10
```

Prices are dollars per million uncached input, cached input and output tokens; reasoning tokens count as output.

### Optional extras

Neither is needed to reproduce the paper, which runs coding and search in separate loops as above.

#### One command that picks the loop

`delegent run` is a small convenience on top of `delegent claude` and `delegent search`. By default it picks the loop
from what the task comes with: a repository means coding, a corpus means search, and both mean one session in the
merged mode below. With `--pick-by model`, the frontier
model reads the task first and says whether it is a coding task or a search question, at the cost of one short call.

```bash
delegent run --frontier gpt-6.1-sol --slm http://localhost:8000 --slm-model Qwen3.8-27B-FP8 \
  --task "Fix the failing test in tests/test_parser.py" --repo path/to/repo
delegent run --frontier gpt-6.1-sol --slm http://localhost:8000 --slm-model Qwen3.8-27B-FP8 \
  --task "Which 1990s film ...?" --corpus climbmix --pick-by model
```

#### One session for coding and search

By default the two uses run in separate loops. For a session that alternates between coding and search, or a task that
needs both, add `--corpus` to `delegent claude`: the same Claude Code session then also gets `search`, `get_document`
and `Delegate` over that corpus (served to Claude Code over MCP), and its instruction covers both kinds of delegation.

```bash
delegent claude --frontier gpt-6.1-sol --slm http://localhost:8000 --slm-model Qwen3.8-27B-FP8 \
  --corpus local --corpus-root docs/ --index index/
```

On pure search, the separate search loop is cheaper. On 80 BrowseComp-Plus questions over ClimbMix, with GPT-6.1-Sol and
Qwen3.8-27B-FP8:

| Main loop | Correct | Frontier cost | Frontier calls per question |
|:--|:-:|:-:|:-:|
| Search loop, frontier model alone | 73 / 72 (two runs) | 100% | 15.0 / 13.9 |
| Search loop, with `Delegate` (default) | 71 | **33%** | 6.2 |
| Claude Code, frontier model alone | 73 | 180% | 27.2 |
| Claude Code, with `Delegate` (merged mode) | 71 | 108% | 14.4 |

<sub>Cost is relative to the search loop with the frontier model alone. The merged mode keeps accuracy and still roughly
halves Claude Code's own cost on search, but Claude Code's main loop makes about twice as many calls as the search loop,
and delegated searches in it took longer (median 747 s per question).</sub>

## Reproducing the paper

[REPRODUCING.md](REPRODUCING.md) lists the models, limits and prompts behind the results above, and
[samples/](samples) holds the exact task and question ids.

<details>
<summary>Repository layout</summary>

| Path | Contents |
|:--|:--|
| `src/delegent/relay.py` | the relay: Anthropic Messages API in, routing by role, one usage ledger |
| `src/delegent/frontier.py` | frontier backends: OpenAI Responses (with the Anthropic translation) and Anthropic |
| `src/delegent/slm.py` | SLM backend: vLLM replicas, conversation pinning, window clamping |
| `src/delegent/delegate.py` | the `Delegate` tool, the SLM search subagent and the evidence check |
| `src/delegent/agent.py` | the search agent loop |
| `src/delegent/mcp.py` | optional merged mode: corpus tools and `Delegate` for Claude Code over MCP |
| `src/delegent/corpus.py`, `bm25.py` | ClimbMix client and local BM25 index |
| `src/delegent/instructions/delegent.md` | the instruction for coding and search |
| `samples/` | task and question ids of the paper's evaluation |
| `tests/` | offline tests with fake model servers (`pytest`) |

</details>

## License

Apache 2.0. The SLM search subagent reuses the prompts and tool definitions of
[AREX-Turbo](https://huggingface.co/BAAI/AREX-Turbo) (Apache 2.0); see [NOTICE](NOTICE).
