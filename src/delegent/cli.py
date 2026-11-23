"""delegent: a frontier model that hands focused subtasks to a local SLM.

  delegent claude  --frontier gpt-6.1-sol --slm http://localhost:8000 --slm-model Qwen3.8-27B-FP8 [-- claude args]
                   [--corpus climbmix]   (optional: the same session can also search the corpus and delegate search)
  delegent relay   --frontier gpt-6.1-sol --slm http://localhost:8000 --slm-model Qwen3.8-27B-FP8 --port 8787
  delegent search  --frontier gpt-6.1-sol --slm http://localhost:8000 --slm-model Qwen3.8-27B-FP8 \\
                   --corpus climbmix --questions questions.jsonl --out runs/search
  delegent index   --corpus corpus/ --out index/
  delegent cost    runs/search/usage.jsonl

Optional, inside the agent you already use (your own Claude Code or Codex, your own frontier model and login):
  delegent install claude|codex --slm http://localhost:8000 --slm-model Qwen3.8-27B-FP8 [--corpus climbmix]
  delegent serve   --slm http://localhost:8000 --slm-model Qwen3.8-27B-FP8      (the relay both of them use)
  delegent uninstall claude|codex
"""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import traceback

import requests

from . import frontier as frontier_backends
from .ledger import Ledger, cost, load
from .relay import Relay, claude_env
from .slm import SLM


def add_models(ap, frontier=True):
    if frontier:
        ap.add_argument('--frontier', default='gpt-6.1-sol', help='frontier model (Claude models use the Anthropic API, '
                        'others the OpenAI Responses API)')
        ap.add_argument('--effort', default='high', help='frontier reasoning effort')
    ap.add_argument('--slm', required=True, help='SLM endpoint(s), comma-separated vLLM replicas of one model')
    ap.add_argument('--slm-model', required=True, help='model name served by the SLM endpoints')
    ap.add_argument('--slm-context', type=int, default=131072, help='SLM context window in tokens')


def make_slm(args):
    return SLM(args.slm, args.slm_model, context_tokens=args.slm_context)


def cmd_relay(args):
    ledger = Ledger(args.ledger)
    relay = Relay(frontier_backends.make(args.frontier, args.effort), make_slm(args), args.frontier,
                  host=args.host, port=args.port, ledger=ledger)
    print(f'relay on {relay.url} (frontier {args.frontier} -> calls for this model; everything else -> {args.slm_model})')
    print('Point an Anthropic-API agent at it:')
    for k, v in claude_env(relay, args.slm_model, args.frontier_context).items():
        print(f'  export {k}={v}')
    try:
        relay.serve_forever()
    except KeyboardInterrupt:
        pass


def cmd_claude(args):
    claude = args.claude_bin or shutil.which('claude')
    if not claude:
        sys.exit('Claude Code (claude) not found; install it or pass --claude-bin')
    ledger = Ledger(args.ledger)
    relay = Relay(frontier_backends.make(args.frontier, args.effort), make_slm(args), args.frontier, ledger=ledger)
    relay.start()
    config = Path(args.config_dir) if args.config_dir else Path(tempfile.mkdtemp(prefix='delegent-claude-'))
    config.mkdir(parents=True, exist_ok=True)
    tools, extra = {'Agent'}, []
    if getattr(args, 'corpus', None):   # optional merged mode: corpus tools and Delegate over MCP (delegent/mcp.py)
        from .corpus import open_corpus
        from .delegate import Delegate
        from .mcp import CorpusTools
        corpus = open_corpus(args.corpus, args.corpus_root, args.index)
        server = CorpusTools(corpus, Delegate(make_slm(args), corpus, max_steps=args.slm_max_steps),
                             ledger=ledger).start()
        (config / 'mcp.json').write_text(json.dumps(server.config()))
        tools, extra = {'Agent', 'Delegate'}, ['--mcp-config', str(config / 'mcp.json')]
    if not args.no_instruction:   # the instruction as a user-level CLAUDE.md
        from .agent import instruction
        (config / 'CLAUDE.md').write_text(instruction(tools))
    env = {**os.environ, **claude_env(relay, args.slm_model, args.frontier_context), 'CLAUDE_CONFIG_DIR': str(config)}
    passthrough = args.claude_args[1:] if args.claude_args[:1] == ['--'] else args.claude_args
    proc = subprocess.Popen([claude, '--model', args.frontier, *extra, *passthrough], env=env)
    try:
        code = proc.wait()
    except KeyboardInterrupt:
        proc.send_signal(signal.SIGINT)
        code = proc.wait()
    relay.shutdown()
    print('\ndelegent: ' + summary(ledger.rows, args.prices), file=sys.stderr)
    sys.exit(code)


def serve_args(args):
    """The `delegent serve` arguments that reproduce this relay."""
    out = ['--slm', args.slm, '--slm-model', args.slm_model, '--slm-context', str(args.slm_context),
           '--host', args.host, '--port', str(args.port)]
    for flag in ('anthropic_upstream', 'openai_upstream', 'ledger', 'corpus'):
        if getattr(args, flag, None):
            out += ['--' + flag.replace('_', '-'), str(getattr(args, flag))]
    if getattr(args, 'corpus', None):   # the relay may be started from any directory (the hook): absolute paths
        for flag in ('corpus_root', 'index'):
            if getattr(args, flag, None):
                out += ['--' + flag.replace('_', '-'), str(Path(getattr(args, flag)).resolve())]
        out += ['--slm-max-steps', str(args.slm_max_steps)]
    return out


def search_tools(args, ledger):
    """The corpus tools and Delegate for the attached relay (the same objects as merged mode), if a corpus is given."""
    if not getattr(args, 'corpus', None):
        return None
    from .corpus import open_corpus
    from .delegate import Delegate
    from .mcp import Toolset
    corpus = open_corpus(args.corpus, args.corpus_root, args.index)
    return Toolset(corpus, Delegate(make_slm(args), corpus, max_steps=args.slm_max_steps), ledger=ledger)


def relay_health(url):
    try:
        got = requests.get(url + '/health', timeout=2).json()
        return got if got.get('ok') else None
    except (requests.RequestException, ValueError):
        return None


def ensure_relay(args):
    """Start the attached relay in the background unless it is already up."""
    url, state = f'http://{args.host}:{args.port}', Path.home() / '.delegent'
    if relay_health(url):
        return
    state.mkdir(parents=True, exist_ok=True)
    with open(state / 'serve.log', 'ab') as log:
        subprocess.Popen([sys.executable, '-m', 'delegent.cli', 'serve', *serve_args(args)],
                         stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
    for _ in range(150):
        if relay_health(url):
            return
        time.sleep(0.1)
    sys.exit(f'delegent: the relay did not start on {url}; see {state / "serve.log"}')


def cmd_mcp(args):
    """The search tools for an agent, over stdio: every MCP message is handed to the relay's /mcp and its answer is
    written back. An agent starts its MCP servers before its first model call, so this also starts the relay when it
    is not running; tool calls of one turn run in parallel."""
    import threading
    ensure_relay(args)
    url, lock = f'http://{args.host}:{args.port}/mcp', threading.Lock()

    def handle(line):
        try:
            request = json.loads(line)
        except ValueError:
            return
        try:
            got = requests.post(url, json=request, timeout=(20, 7200))
            answer = got.json() if got.content else None
        except (requests.RequestException, ValueError) as exc:
            answer = ({'jsonrpc': '2.0', 'id': request['id'], 'error': {'code': -32603, 'message': str(exc)[:300]}}
                      if isinstance(request, dict) and 'id' in request else None)
        if answer is not None:
            with lock:
                sys.stdout.write(json.dumps(answer) + '\n')
                sys.stdout.flush()
    workers = []
    for line in sys.stdin:
        if line.strip():
            workers.append(threading.Thread(target=handle, args=(line,)))
            workers[-1].start()
    for worker in workers:
        worker.join()


def cmd_serve(args):
    """The attached relay, for an agent that was set up with `delegent install`: calls that name the SLM are served by
    the SLM, every other request passes through to the agent's own API with the agent's own credentials."""
    url = f'http://{args.host}:{args.port}'
    state = Path.home() / '.delegent'
    if args.ensure:   # prints nothing (it runs as a hook)
        return ensure_relay(args)
    ledger = Ledger(args.ledger or state / 'usage.jsonl')
    try:
        tools = search_tools(args, ledger)
    except (ValueError, RuntimeError, OSError) as exc:
        sys.exit(f'delegent: cannot open the corpus: {exc}')
    try:
        relay = Relay(None, make_slm(args), None, host=args.host, port=args.port, ledger=ledger,
                      upstream=args.anthropic_upstream, openai_upstream=args.openai_upstream, tools=tools)
    except OSError as exc:
        running = relay_health(url)
        if running:
            sys.exit(f'delegent: a relay is already serving {running.get("slm_model")} on {url}')
        sys.exit(f'delegent: cannot listen on {url}: {exc}')
    print(f'delegent relay on {relay.url}: calls for {args.slm_model} -> {args.slm}; everything else passes through '
          f'(Claude Code -> {relay.upstream}, Codex -> {relay.openai_upstream})'
          + (f'; search tools over the {args.corpus} corpus at {relay.url}/mcp' if tools else ''), flush=True)
    try:
        relay.serve_forever()
    except KeyboardInterrupt:
        pass
    print('\ndelegent: ' + summary(ledger.rows, args.prices), file=sys.stderr)


def cmd_install(args):
    from . import install
    url = f'http://{args.host}:{args.port}'
    try:
        if args.agent == 'claude':
            hook = None if args.no_autostart else install.serve_command(serve_args(args))
            done = install.install_claude(url, args.slm_model, hook, args.config_dir, args.force,
                                          install.tools_command(serve_args(args)) if args.corpus else None,
                                          args.claude_bin)
            after = ('Run `claude` as usual. The relay starts with the session'
                     if hook else 'Start the relay with `delegent serve ...`, then run `claude` as usual')
        else:
            done = install.install_codex(url, args.slm_model, args.slm_context, args.config_dir, args.codex_bin,
                                         args.env_key, args.openai_auth,
                                         install.tools_command(serve_args(args)) if args.corpus else None)
            after = (f'Run Codex with the profile: codex -p {install.PROFILE}\nThe relay starts with the session'
                     if args.corpus else
                     'Start the relay: delegent serve ' + ' '.join(a for a in serve_args(args)
                                                                   if a not in ('--host', '127.0.0.1'))
                     + f'\nThen run Codex with the profile: codex -p {install.PROFILE}')
    except install.InstallError as exc:
        sys.exit(f'delegent: {exc}')
    for line in done:
        print('  ' + line)
    print(after + f'.\nUndo with: delegent uninstall {args.agent}')


def cmd_uninstall(args):
    from . import install
    try:
        done = (install.uninstall_claude(args.config_dir, args.claude_bin) if args.agent == 'claude'
                else install.uninstall_codex(args.config_dir))
    except install.InstallError as exc:
        sys.exit(f'delegent: {exc}')
    for line in done or ['nothing to remove']:
        print('  ' + line)


def prices_from(items):
    out = {}
    for item in items or []:
        model, _, values = item.partition('=')
        out[model] = tuple(float(v) for v in values.split(','))
    return out


def summary(rows, prices):
    frontier_rows = [r for r in rows if r['role'] == 'frontier']
    text = f'{len(frontier_rows)} frontier calls, {len(rows) - len(frontier_rows)} SLM calls'
    try:
        return text + f', frontier cost ${cost(rows, prices_from(prices)):.3f}'
    except KeyError:   # no price known for the frontier model
        return text


def load_questions(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            rows.append({'id': str(row.get('id', row.get('qid', len(rows)))), 'question': row['question']})
    return rows


def cmd_search(args):
    from .agent import answer
    from .corpus import open_corpus
    from .delegate import Delegate
    out = Path(args.out)
    (out / 'questions').mkdir(parents=True, exist_ok=True)
    ledger = Ledger(out / 'usage.jsonl')
    corpus = open_corpus(args.corpus, args.corpus_root, args.index)
    frontier = frontier_backends.make(args.frontier, args.effort)
    delegate = None if args.no_delegate else Delegate(make_slm(args), corpus, max_steps=args.slm_max_steps,
                                                      thinking=args.slm_thinking)
    rows = load_questions(args.questions)[:args.limit or None]
    note = agent_notes(args.multi_answer)

    def one(row):
        path = out / 'questions' / f"{row['id']}.json"
        if path.exists():
            return
        try:
            result = {'id': row['id'], 'question': row['question'],
                      **answer(row['question'] + note, frontier, corpus, delegate, max_steps=args.max_steps, ledger=ledger,
                               frontier_model=args.frontier)}
        except Exception:   # noqa: BLE001
            result = {'id': row['id'], 'error': traceback.format_exc()[-3000:]}
        path.write_text(json.dumps(result, indent=1))
        print(row['id'], 'error' if 'error' in result else f"{result['frontier_calls']} frontier calls, "
              f"{len(result['delegations'])} delegations", flush=True)
    with concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(one, rows))
    print(f'{len(rows)} questions: ' + summary(ledger.rows, args.prices))


def agent_notes(mode):
    from .agent import MULTI_ANSWER
    return MULTI_ANSWER[mode] if mode else ''


def cmd_run(args):
    """One entry point: pick the loop for the task (by default from what it comes with; optionally, the frontier model
    picks), then run it exactly as `claude` or `search` would."""
    from . import pick
    task = Path(args.task_file).read_text() if args.task_file else args.task
    if not task:
        sys.exit('give --task or --task-file')
    if args.repo and args.corpus:   # both: one Claude Code session with the corpus tools too (optional merged mode)
        kind = 'coding'
    elif args.pick_by == 'environment':
        kind = pick.by_environment(args.repo, args.corpus)
    else:
        frontier = frontier_backends.make(args.frontier, 'low')
        kind = pick.by_model(task, frontier, Ledger(args.ledger), model=args.frontier)
        if kind is None and (args.repo or args.corpus):   # the model did not name a kind: decide by the environment
            kind = pick.by_environment(args.repo, args.corpus)
    print(f'delegent: {kind} task', file=sys.stderr)
    if kind == 'coding':
        if not args.repo:
            sys.exit('the task looks like coding, but no --repo was given')
        if args.corpus:
            print('delegent: with the corpus tools (merged mode)', file=sys.stderr)
        os.chdir(args.repo)
        rest = args.claude_args[1:] if args.claude_args[:1] == ['--'] else args.claude_args
        args.claude_args = ['-p', task, *rest]
        return cmd_claude(args)
    if kind == 'search':
        if not args.corpus:
            sys.exit('the task looks like search, but no --corpus was given')
        from .agent import answer
        from .corpus import open_corpus
        from .delegate import Delegate
        ledger = Ledger(args.ledger)
        corpus = open_corpus(args.corpus, args.corpus_root, args.index)
        delegate = Delegate(make_slm(args), corpus, max_steps=args.slm_max_steps)
        result = answer(task, frontier_backends.make(args.frontier, args.effort), corpus, delegate, ledger=ledger,
                        frontier_model=args.frontier)
        print(result['answer'])
        print('delegent: ' + summary(ledger.rows, args.prices), file=sys.stderr)
        return
    sys.exit('could not tell whether the task is coding or search; give --repo or --corpus')


def cmd_index(args):
    from .bm25 import build
    build(args.corpus, args.out, args.workers)


def cmd_cost(args):
    print(summary(load(args.ledger), args.prices))


def main(argv=None):
    ap = argparse.ArgumentParser(prog='delegent', description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    price_help = 'model=uncached,cached,output in $ per million tokens (repeatable)'

    p = sub.add_parser('relay', help='serve the relay for any Anthropic-API agent')
    add_models(p)
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=8787)
    p.add_argument('--frontier-context', type=int, default=1_000_000)
    p.add_argument('--ledger', default='delegent-usage.jsonl')
    p.set_defaults(func=cmd_relay)

    p = sub.add_parser('claude', help='run Claude Code through the relay, with the coding instruction')
    add_models(p)
    p.add_argument('--frontier-context', type=int, default=1_000_000)
    p.add_argument('--no-instruction', action='store_true', help='leave out the instruction (CLAUDE.md)')
    p.add_argument('--config-dir', help='CLAUDE_CONFIG_DIR to use (default: a fresh temporary one)')
    p.add_argument('--claude-bin')
    p.add_argument('--corpus', choices=['climbmix', 'local'],
                   help='optional merged mode: also give the session search, get_document and Delegate over this corpus')
    p.add_argument('--corpus-root', help='local corpus directory (with --corpus local)')
    p.add_argument('--index', help='local BM25 index (with --corpus local)')
    p.add_argument('--slm-max-steps', type=int, default=80, help='step cap of a delegated search subagent')
    p.add_argument('--ledger', default='delegent-usage.jsonl')
    p.add_argument('--prices', action='append', help=price_help)
    p.add_argument('claude_args', nargs=argparse.REMAINDER, help='arguments passed to claude (after --)')
    p.set_defaults(func=cmd_claude)

    p = sub.add_parser('search', help='answer questions with the search agent')
    add_models(p)
    p.add_argument('--corpus', choices=['climbmix', 'local'], required=True)
    p.add_argument('--corpus-root', help='local corpus directory')
    p.add_argument('--index', help='local BM25 index (delegent index)')
    p.add_argument('--questions', required=True, help='JSONL with id and question')
    p.add_argument('--out', required=True)
    p.add_argument('--no-delegate', action='store_true', help='the frontier model alone (baseline)')
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--max-steps', type=int, default=100)
    p.add_argument('--slm-max-steps', type=int, default=80)
    p.add_argument('--slm-thinking', action='store_true', help='let the SLM subagent think (off in the paper)')
    p.add_argument('--multi-answer', choices=['one', 'all'], help='questions with several answers: find one or all')
    p.add_argument('--prices', action='append', help=price_help)
    p.set_defaults(func=cmd_search)

    p = sub.add_parser('run', help='one entry point for coding or search (optional; see claude and search)')
    add_models(p)
    p.add_argument('--task', help='the task or question')
    p.add_argument('--task-file')
    p.add_argument('--repo', help='repository to work in (coding)')
    p.add_argument('--corpus', choices=['climbmix', 'local'], help='corpus to search (search)')
    p.add_argument('--corpus-root')
    p.add_argument('--index')
    p.add_argument('--pick-by', choices=['environment', 'model'], default='environment',
                   help='who picks the loop: what the task comes with (default) or, optionally, the frontier model')
    p.add_argument('--frontier-context', type=int, default=1_000_000)
    p.add_argument('--no-instruction', action='store_true')
    p.add_argument('--config-dir')
    p.add_argument('--claude-bin')
    p.add_argument('--slm-max-steps', type=int, default=80)
    p.add_argument('--ledger', default='delegent-usage.jsonl')
    p.add_argument('--prices', action='append', help=price_help)
    p.add_argument('claude_args', nargs=argparse.REMAINDER, help='extra arguments for claude (after --)')
    p.set_defaults(func=cmd_run)

    def add_attached(p):
        add_models(p, frontier=False)
        p.add_argument('--host', default='127.0.0.1')
        p.add_argument('--port', type=int, default=8787)
        p.add_argument('--anthropic-upstream', help='where Claude Code\'s own calls go (default: the Anthropic API)')
        p.add_argument('--openai-upstream', help='where Codex\'s own calls go (default: https://api.openai.com/v1)')
        p.add_argument('--ledger', help='usage ledger (default: ~/.delegent/usage.jsonl)')
        p.add_argument('--corpus', choices=['climbmix', 'local'],
                       help='also give the agent search, get_document and Delegate over this corpus')
        p.add_argument('--corpus-root', help='local corpus directory (with --corpus local)')
        p.add_argument('--index', help='local BM25 index (with --corpus local)')
        p.add_argument('--slm-max-steps', type=int, default=80, help='step cap of a delegated search subagent')

    p = sub.add_parser('serve', help='optional: the relay for an agent set up with `delegent install`')
    add_attached(p)
    p.add_argument('--ensure', action='store_true', help='start it in the background unless it is already running')
    p.add_argument('--prices', action='append', help=price_help)
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser('mcp', help='the search tools of the relay over stdio (set up by `delegent install --corpus`)')
    add_attached(p)
    p.set_defaults(func=cmd_mcp)

    p = sub.add_parser('install', help='optional: use delegent inside your own Claude Code or Codex')
    p.add_argument('agent', choices=['claude', 'codex'])
    add_attached(p)
    p.add_argument('--config-dir', help='the agent\'s config directory (default: ~/.claude or ~/.codex)')
    p.add_argument('--no-autostart', action='store_true', help='claude: do not add the hook that starts the relay')
    p.add_argument('--force', action='store_true', help='claude: replace an ANTHROPIC_BASE_URL that is already set')
    p.add_argument('--codex-bin')
    p.add_argument('--claude-bin')
    p.add_argument('--env-key', default='OPENAI_API_KEY', help='codex: environment variable with your API key')
    p.add_argument('--openai-auth', action='store_true',
                   help='codex: use Codex\'s own OpenAI login instead of an API key variable (untested)')
    p.set_defaults(func=cmd_install)

    p = sub.add_parser('uninstall', help='remove what `delegent install` added')
    p.add_argument('agent', choices=['claude', 'codex'])
    p.add_argument('--config-dir')
    p.add_argument('--claude-bin')
    p.set_defaults(func=cmd_uninstall)

    p = sub.add_parser('index', help='build a BM25 index over a local corpus')
    p.add_argument('--corpus', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--workers', type=int, default=os.cpu_count())
    p.set_defaults(func=cmd_index)

    p = sub.add_parser('cost', help='frontier cost of a usage ledger')
    p.add_argument('ledger')
    p.add_argument('--prices', action='append', help=price_help)
    p.set_defaults(func=cmd_cost)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == '__main__':
    main()
