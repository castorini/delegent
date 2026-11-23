# Guide for coding agents

Delegent is a small Python package (`src/delegent`, under 3,000 lines, dependencies `requests` and `numpy`) that pairs
a frontier model with a local small language model (SLM). Read [README.md](README.md) for what it does and
[REPRODUCING.md](REPRODUCING.md) for the paper's settings.

## Set up and test

```bash
pip install -e ".[test]"
pytest
```

The tests are offline and take a few seconds. They use fake model servers (`tests/test_relay.py`,
`tests/test_install.py`), so they need no GPU, API key or network. Run them after every change, and add a test next to
the code you change.

## Where things are

| Path | Contents |
|:--|:--|
| `src/delegent/cli.py` | every command; start here |
| `src/delegent/relay.py` | routes model calls: by role for the launcher, by model name when attached to an installed agent |
| `src/delegent/frontier.py` | frontier backends (OpenAI Responses with Anthropic translation, Anthropic) |
| `src/delegent/slm.py` | vLLM client: replica pinning, token counting, window clamping |
| `src/delegent/agent.py`, `delegate.py` | the search loop, the `Delegate` tool, the SLM search subagent, the evidence check |
| `src/delegent/mcp.py` | the search tools over MCP |
| `src/delegent/install.py` | `delegent install claude\|codex` |
| `src/delegent/instructions/delegent.md` | the instruction given to the frontier model |
| `src/delegent/prompts/arex.py` | vendored AREX-Turbo prompts and tool schemas |

## Rules

- **Do not reword the paper's prompts.** The `Agent` and `Delegate` parts of `instructions/delegent.md`, `SYSTEM` in
  `agent.py`, `PICK_PROMPT` in `pick.py` and everything in `prompts/arex.py` are the texts the paper's results were
  measured with. Change them only when the user asks, and say so in `REPRODUCING.md`.
- **Do not change result numbers** in `README.md`; they come from the paper.
- **Keep tests offline.** No test may call a real model API or need a GPU.
- **Never run `delegent install` or `delegent uninstall` against the user's real configuration** to try something
  out. Pass `--config-dir <scratch directory>`.
- **No new dependencies** without asking; the package deliberately uses only the standard library, `requests` and
  `numpy`.
- **Style:** plain functions, single quotes, lines up to 120 characters, a docstring that says what a module or
  function is for. Match the surrounding code.
- **Keep the docs in step.** A new command or option belongs in the README's command table, and anything that differs
  from the paper's setup belongs in `REPRODUCING.md`.
