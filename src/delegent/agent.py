"""The search agent: the frontier model in one continuous conversation (so its prompt cache keeps hitting) with two
corpus tools, search and get_document, and, when delegation is on, the Delegate tool and the search instruction.
Tool calls issued in the same turn run in parallel, so independent subtasks are delegated together."""
import concurrent.futures
from importlib import resources
import time

from .delegate import DELEGATE_TOOL
from .bm25 import best_window
from .sse import collect_message

SYSTEM = """You are a research agent. Answer the user's question using the search tools over a fixed web corpus. \
The answer is contained in the corpus; do not rely on your memory without checking the corpus.

Tools:
- search(query): BM25 keyword search over the corpus; returns the top 10 documents (docid, score, best-matching passage).
- get_document(docid, offset): reads a document, 8000 characters at a time starting at offset.

Search with distinctive keywords, open the promising documents, and verify every condition of the question before you \
answer. When you are done, reply without calling a tool, in this format:
Explanation: <how the evidence supports the answer, citing docids>
Exact Answer: <the final answer, concise>
Confidence: <0-100%>"""
TOOLS = [
    {'name': 'search', 'description': 'BM25 keyword search over the corpus. Returns the top 10 documents with docid, '
     'score and the passage that best matches the query.',
     'input_schema': {'type': 'object', 'properties': {'query': {'type': 'string', 'description': 'keyword query'}},
                      'required': ['query']}},
    {'name': 'get_document', 'description': 'Read a document by docid: 8000 characters starting at offset (default 0); '
     'the reply says how long the document is.',
     'input_schema': {'type': 'object', 'properties': {'docid': {'type': 'string'},
                                                        'offset': {'type': 'integer', 'description': 'character offset'}},
                      'required': ['docid']}},
]
DOC_CHARS, SNIPPET = 8000, 400
FINAL = 'Stop searching now and give your final answer in the required format.'
# for questions with several correct answers (DeepSearchQA): find one, or find all
MULTI_ANSWER = {
    'one': ('\n\nSeveral answers may satisfy this question. Give ONE answer that satisfies it (any single correct answer '
            'is enough); put just that one answer after "Exact Answer:".'),
    'all': ('\n\nThis question may have several correct answers. Find ALL answers that satisfy it and give the complete '
            'list after "Exact Answer:" (comma-separated); do not include items that fail any condition.'),
}


def instruction(tools):
    """The delegent instruction for an agent with these tools: one part per subtask tool (Claude Code's Agent tool,
    the Delegate tool), each given only to an agent that has that tool."""
    text = resources.files('delegent.instructions').joinpath('delegent.md').read_text()
    parts, name = {}, None
    for line in text.splitlines(keepends=True):
        if line.startswith('<!-- part: '):
            name = line[len('<!-- part: '):].split(' -->')[0]
            parts[name] = ''
        elif name:
            parts[name] += line
    return ''.join(parts[n] for n in parts if n in set(tools))


def run_tool(corpus, name, inp):
    if name == 'search':
        query = str(inp.get('query') or '').strip()
        if not query:
            return 'error: empty query'
        hits = corpus.search(query)
        if not hits:
            return 'No results.'
        return '\n\n'.join(f'[{i}] docid: {docid} (score {score or 0:.2f}, {len(text)} chars)\n'
                           + ' '.join(best_window(text, query, SNIPPET).split())
                           for i, (docid, score, text) in enumerate(hits, 1))
    if name == 'get_document':
        docid = str(inp.get('docid') or '').strip()
        try:
            offset = max(0, int(inp.get('offset') or 0))
        except (TypeError, ValueError):
            offset = 0
        text = corpus.doc(docid)
        if not text:
            return f'error: no document {docid}'
        page = text[offset:offset + DOC_CHARS]
        end = offset + len(page)
        more = (f'\n[characters {offset}-{end} of {len(text)}; call get_document with offset {end} for more]'
                if end < len(text) else f'\n[end of document; {len(text)} characters]')
        return f'docid: {docid}\n{page}{more}'
    return f'error: unknown tool {name}'


def answer(question, frontier, corpus, delegate=None, max_steps=100, ledger=None, frontier_model=None):
    """Run the frontier model on one question -> result row (answer, frontier usage, actions, delegations)."""
    system = SYSTEM + ('\n\n' + instruction({'Delegate'}).rstrip('\n') if delegate else '')
    payload = {'model': frontier_model or getattr(frontier, 'model', 'frontier'), 'system': system,
               'tools': TOOLS + ([DELEGATE_TOOL] if delegate else []), 'max_tokens': 32768,
               'messages': [{'role': 'user', 'content': [{'type': 'text', 'text': 'QUESTION:\n' + question}]}]}
    actions, delegations, started, final = [], [], time.time(), ''

    def step(body):
        usage, t0 = {}, time.time()
        message = collect_message(frontier.stream(body, usage.update))
        if ledger is not None:
            ledger.add('frontier', payload['model'], usage, time.time() - t0)
        return message['content'], usage

    for n in range(1, max_steps + 1):
        content, usage = step(payload)
        payload['messages'].append({'role': 'assistant', 'content': content})
        calls = [b for b in content if b.get('type') == 'tool_use']
        text = '\n'.join(b.get('text', '') for b in content if b.get('type') == 'text').strip()

        def run(call, n=n):
            try:
                if call['name'] == 'Delegate' and delegate:
                    out, entry = delegate(question, call.get('input') or {})
                    delegations.append({**entry, 'step': n})
                    if ledger is not None:
                        ledger.add('slm', getattr(delegate.slm, 'model', 'slm'),
                                   {'input_tokens': entry['slm_input_tokens'],
                                    'output_tokens': entry['slm_output_tokens']}, entry['seconds'], subtask=True)
                    return out
                return run_tool(corpus, call['name'], call.get('input') or {})
            except Exception as exc:   # noqa: BLE001 -- the model sees the error and can retry
                return f'error: {type(exc).__name__}: {str(exc)[:200]}'
        with concurrent.futures.ThreadPoolExecutor(max(1, len(calls))) as pool:
            outs = list(pool.map(run, calls))
        actions.append({'step': n, 'usage': usage, 'text': text[:1500], 'result_chars': [len(o) for o in outs],
                        'calls': [{'name': c['name'], 'input': {k: str(v)[:300] for k, v in (c.get('input') or {}).items()}}
                                  for c in calls]})
        if not calls:
            final = text
            break
        payload['messages'].append({'role': 'user', 'content': [
            {'type': 'tool_result', 'tool_use_id': c['id'], 'content': o} for c, o in zip(calls, outs)]})
    else:   # step cap: one last turn without tools
        payload['messages'].append({'role': 'user', 'content': [{'type': 'text', 'text': FINAL}]})
        content, usage = step(dict(payload, tools=[]))
        final = '\n'.join(b.get('text', '') for b in content if b.get('type') == 'text').strip()
        actions.append({'step': max_steps + 1, 'usage': usage, 'calls': [], 'forced': True})
    total = lambda key: sum((a['usage'].get(key) or 0) for a in actions)  # noqa: E731
    return {'answer': final, 'frontier_calls': len(actions), 'frontier_input_tokens': total('input_tokens'),
            'frontier_cached_tokens': total('cached_input_tokens'), 'frontier_output_tokens': total('output_tokens'),
            'seconds': round(time.time() - started, 1), 'delegations': delegations,
            'slm_calls': sum(d['slm_steps'] for d in delegations), 'actions': actions}
