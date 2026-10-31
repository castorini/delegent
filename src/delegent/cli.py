"""delegent: a frontier model that hands focused subtasks to a local SLM.

  delegent claude  --frontier gpt-6.1-sol --slm http://localhost:8000 --slm-model Qwen3.8-27B-FP8 [-- claude args]
  delegent relay   --frontier gpt-6.1-sol --slm http://localhost:8000 --slm-model Qwen3.8-27B-FP8 --port 8787
  delegent search  --frontier gpt-6.1-sol --slm http://localhost:8000 --slm-model Qwen3.8-27B-FP8 \\
                   --corpus climbmix --questions questions.jsonl --out runs/search
  delegent index   --corpus corpus/ --out index/
  delegent cost    runs/search/usage.jsonl
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
import traceback

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
    if not args.no_instruction:   # the coding instruction as a user-level CLAUDE.md
        from .agent import instruction
        (config / 'CLAUDE.md').write_text(instruction({'Agent'}))
    env = {**os.environ, **claude_env(relay, args.slm_model, args.frontier_context), 'CLAUDE_CONFIG_DIR': str(config)}
    passthrough = args.claude_args[1:] if args.claude_args[:1] == ['--'] else args.claude_args
    proc = subprocess.Popen([claude, '--model', args.frontier, *passthrough], env=env)
    try:
        code = proc.wait()
    except KeyboardInterrupt:
        proc.send_signal(signal.SIGINT)
        code = proc.wait()
    relay.shutdown()
    print('\ndelegent: ' + summary(ledger.rows, args.prices), file=sys.stderr)
    sys.exit(code)


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
    if args.pick_by == 'environment':
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
    p.add_argument('--no-instruction', action='store_true', help='leave out the coding instruction (CLAUDE.md)')
    p.add_argument('--config-dir', help='CLAUDE_CONFIG_DIR to use (default: a fresh temporary one)')
    p.add_argument('--claude-bin')
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
