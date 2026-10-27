"""Which agent loop a task needs: coding (Claude Code on a repository) or search (Delegent's search loop on a corpus).

By default the frontier model decides from the task text, with one short call before the task starts; `environment`
decides from what the task comes with instead (a repository means coding, a corpus means search)."""
import re
import time

from .sse import collect_message

PICK_PROMPT = ('Here is a task.\n\n<task>\n{task}\n</task>\n\nDo not do the task. Decide which kind of task it is and '
               'answer with exactly one word: coding if it asks you to change code in a software repository, or search '
               'if it asks you to find the answer to a question.')
KINDS = ('coding', 'search')


def parse(text):
    found = re.findall(r'\b(coding|search)\b', (text or '').lower())
    return found[0] if found else None


def by_model(task, frontier, ledger=None, model=None):
    payload = {'model': model or getattr(frontier, 'model', 'frontier'), 'max_tokens': 1024,
               'messages': [{'role': 'user', 'content': [{'type': 'text', 'text': PICK_PROMPT.format(task=task)}]}]}
    usage, started = {}, time.time()
    message = collect_message(frontier.stream(payload, usage.update))
    if ledger is not None:
        ledger.add('frontier', payload['model'], usage, time.time() - started, pick=True)
    return parse(''.join(b.get('text', '') for b in message['content'] if b.get('type') == 'text'))


def by_environment(repo=None, corpus=None):
    if bool(repo) == bool(corpus):
        raise ValueError('give either a repository (coding) or a corpus (search)')
    return 'coding' if repo else 'search'
