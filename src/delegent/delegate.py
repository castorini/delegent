"""Delegation to the SLM.

A delegated subtask runs in a fresh context on the SLM, and only its final report returns to the frontier agent. For
coding, the subtask is a Claude Code subagent (Claude Code's own Agent tool, routed to the SLM by the relay). For
search, the frontier agent calls the `Delegate` tool defined here: the SLM subagent searches the corpus in its own tool
loop and returns a report with evidence and a confidence, and the harness looks up every evidence line in the cited
document before the report reaches the frontier agent.

The search subagent uses the research loop and tool protocol of AREX-Turbo (search, visit, update_context, finish),
backed by the corpus; its prompts are vendored in delegent/prompts/arex.py.
"""
import json
import re
import time

from .bm25 import best_window, chunk_hits
from .prompts import arex
from .slm import ContextOverflow

DELEGATE_TOOL = {
    'name': 'Delegate',
    'description': ('Hand a focused search subtask to a research assistant: a small, fast model that searches the same '
                    'corpus with its own search engine and returns a short report with evidence, docids and its '
                    'confidence. Delegating is much cheaper than searching yourself. Reports can be incomplete or wrong. '
                    'Call Delegate several times in one turn to run independent subtasks in parallel.'),
    'input_schema': {'type': 'object', 'required': ['task', 'deliverable'], 'properties': {
        'task': {'type': 'string', 'description': 'what to find: self-contained, concrete and narrow, with distinctive '
                                                  'search phrases (the assistant sees only this and the question)'},
        'deliverable': {'type': 'string', 'description': 'what the report must contain'}}}}

TOOLS = [arex._SEARCH_TOOL, arex._VISIT_TOOL, arex._UPDATE_CONTEXT_TOOL, arex._FINISH_TOOL]
# AREX's instructions without its "# Tools" section, which the chat template renders from the tools parameter.
SYSTEM = arex._SYSTEM_PROMPT_TEMPLATE.split('# Tools')[0].rstrip()
USER = arex._USER_PROMPT_TEMPLATE
SOURCES = {
    'local': ('All sources are documents in a local corpus: `search` searches it and returns /corpus/... paths; pass '
              'such a path as the `url` of `visit`, and cite /corpus/... paths as the `url` of your evidences.'),
    'climbmix': ('All sources are documents in a fixed web corpus: `search` searches it and returns documents named by '
                 'docid (shard_...); pass a docid as the `url` of `visit`, and cite docids as the `url` of your '
                 'evidences.'),
}
TASK = '''{instruction}

What to deliver: {deliverable}

(This is one step of a larger research question, for context only: {question})
{sources} This step was planned from a hypothesis that can be wrong: if a thorough search shows the corpus does not
support it, call `finish` with what you did find and a low confidence.'''
THINK = re.compile(r'<think>.*?</think>', re.S)
EVIDENCE = re.compile(r'^\s*-\s*(.+?)\s*\(source:\s*([^)]*)\)\s*$', re.M)
CLEARED = '[cleared: old tool output]'


def corpus_kind(corpus):
    return 'climbmix' if type(corpus).__name__ == 'ClimbMix' else 'local'


def as_list(value):
    if isinstance(value, list):
        return [str(v) for v in value if str(v).strip()]
    value = str(value or '').strip()
    if value.startswith('['):
        try:
            got = json.loads(value)
            return [str(x) for x in got] if isinstance(got, list) else [value]
        except ValueError:
            pass
    return [v.strip() for v in value.split('\n') if v.strip()] or ([value] if value else [])


def arguments(call):
    try:
        got = json.loads(call['function'].get('arguments') or '{}')
        return got if isinstance(got, dict) else {}
    except ValueError:
        return {}


def title(text, docid, kind):
    m = re.match(r'---\s*\ntitle:\s*(.+)', text)
    if m:
        return m.group(1).strip()[:150]
    return docid.rsplit('/', 1)[-1][:150] if kind == 'local' else ' '.join(text[:100].split())


def tool_search(params, corpus, kind):
    out = []
    for q in as_list(params.get('query'))[:5]:
        try:
            hits = corpus.search(q, 10)
        except RuntimeError as exc:   # e.g. a query the search service rejects: the model sees the error
            out.append(f'## Results for: {q}\n(search failed: {str(exc)[:120]})')
            continue
        out.append(f'## Results for: {q}')
        if not hits:
            out.append('(no results)')
        for i, (docid, _, text) in enumerate(hits, 1):
            snippet = ' '.join(best_window(text, q, 360).split())
            out.append(f'{i}. {title(text, docid, kind)}\n   url: {docid}\n   {snippet}')
    return '\n'.join(out) or 'No query given.'


def tool_visit(params, corpus, kind, budget=12000):
    """A document whole if short, else its first chunk and the chunks that best match the goal, in document order."""
    urls = as_list(params.get('url'))[:3]
    goal = str(params.get('goal') or '')
    if not urls:
        return 'No url given.'
    each, out = budget // len(urls), []
    for url in urls:
        docid = corpus.canonical(url)
        text = corpus.doc(docid) if docid else ''
        if not text:
            hint = 'a /corpus/... path' if kind == 'local' else 'a docid (shard_...)'
            out.append(f'[{url}] not found. Use {hint} from the search results as the url.')
            continue
        head = f'[{docid}] {title(text, docid, kind)}' if kind == 'local' else f'[{docid}]'
        if len(text) <= each:
            out.append(f'{head}\n{text}')
            continue
        chunks = [text[i:i + 1200] for i in range(0, len(text), 1200)]
        hits = chunk_hits(text, goal, 1200)
        keep, used = [], 0
        for i in sorted(range(len(chunks)), key=lambda i: (-hits[i], i)):
            if used + len(chunks[i]) > each:
                break
            keep.append(i)
            used += len(chunks[i])
        parts, last = [], -2
        for i in sorted(set(keep) | {0}):
            parts.append(('' if i == last + 1 else '\n[...]\n') + chunks[i])
            last = i
        out.append(f'{head} (long document: the passages most relevant to the goal)\n' + ''.join(parts))
    return '\n\n'.join(out)


def report_of(params):
    lines = [f"Answer: {str(params.get('answer', '')).strip()}",
             f"Confidence: {str(params.get('confidence', '')).strip()}"]
    ev = params.get('evidences') or ''
    items = ev if isinstance(ev, list) else None
    if items is None:
        try:
            items = json.loads(ev)
        except (TypeError, ValueError):
            items = None
    if isinstance(items, list):
        lines.append('Evidence:')
        for it in items[:12]:
            if isinstance(it, dict):
                lines.append(f"- {str(it.get('evidence', '')).strip()} (source: {str(it.get('url', '')).strip()})")
    elif ev:
        lines.append('Evidence:\n' + str(ev)[:2500])
    return '\n'.join(lines)


def sampling(model, thinking):
    """AREX's own sampling; Qwen's recommended settings for other models (thinking / non-thinking)."""
    if 'arex' in (model or '').lower():
        return {'temperature': 1.0, 'top_p': 0.95}
    return {'temperature': 0.6, 'top_p': 0.95} if thinking else {'temperature': 0.7, 'top_p': 0.8}


def run_subagent(question, task, deliverable, slm, corpus, max_steps=80, max_chars=280000, thinking=False):
    """One SLM subagent run on a delegated subtask -> (report, steps, usage, finished)."""
    kind = corpus_kind(corpus)
    user = USER.format(question=TASK.format(instruction=task, deliverable=deliverable, question=question,
                                            sources=SOURCES[kind]))
    base = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': user}]
    messages, steps, usage, report, finished = list(base), [], [], '', False
    params = sampling(slm.model, thinking)

    def chat(force_finish):
        choice = {'type': 'function', 'function': {'name': 'finish'}} if force_finish else 'auto'
        return slm.chat(messages, tools=TOOLS, tool_choice=choice, thinking=thinking, sampling=params)

    for step in range(max_steps + 1):
        wrap = step == max_steps
        if wrap:   # forced finish: a real report with an honest confidence, not salvaged notes
            messages.append({'role': 'user', 'content': 'You have used your research budget. Call `finish` now with '
                                                        'your best answer, the evidences you have and an honest '
                                                        'confidence.'})
        elif step == max_steps - 8:
            messages.append({'role': 'user', 'content': 'Budget: 8 research steps left. Complete your verification and '
                                                        'call `finish`.'})
        try:
            msg, u = chat(wrap)
        except ContextOverflow:   # keep the task and the last user turns
            messages = base + [m for m in messages[-3:] if m['role'] == 'user']
            msg, u = chat(wrap)
        usage.append(u)
        content = THINK.sub('', msg.get('content') or '').strip()
        calls = [c for c in (msg.get('tool_calls') or []) if c.get('function')]
        steps.append({'calls': [{'name': c['function']['name'],
                                 'input': {k: str(v)[:300] for k, v in arguments(c).items()}} for c in calls],
                      'text': content[:1500]})
        done = next((c for c in calls if c['function']['name'] == 'finish'), None)
        if done is not None:
            report, finished = report_of(arguments(done)), True
            break
        if wrap:
            break
        update = next((c for c in calls if c['function']['name'] == 'update_context'), None)
        if update is not None:
            messages = base[:1] + [{'role': 'user', 'content': user + '\n\nYour saved research context (from '
                                    'update_context):\n' + str(arguments(update).get('context', ''))}]
            continue
        if not calls:
            messages += [{'role': 'assistant', 'content': content or '(no output)'},
                         {'role': 'user', 'content': 'No tool call found. Call exactly one tool: search, visit, '
                                                     'update_context or finish.'}]
            continue
        messages.append({'role': 'assistant', 'content': content, 'tool_calls': calls})
        for c in calls:
            name, args = c['function']['name'], arguments(c)
            if name == 'search':
                result = tool_search(args, corpus, kind)
            elif name == 'visit':
                result = tool_visit(args, corpus, kind)
            else:
                result = f'Unknown tool {name!r}: use search, visit, update_context or finish.'
            messages.append({'role': 'tool', 'tool_call_id': c.get('id', ''), 'content': result})
        if sum(len(str(m.get('content') or '')) for m in messages) > max_chars:   # drop old tool output
            for m in messages[2:-6]:
                if m['role'] == 'tool':
                    m['content'] = CLEARED
    if not report:   # no finish: the last visible text, marked as such
        tail = next((s['text'] for s in reversed(steps) if s['text']), '')
        report = 'The assistant did not finish its research. Its last notes:\n' + tail[:2500]
    return report, steps, usage, finished


def evidence_check(report, corpus, budget=1800):
    """Each '- evidence (source: docid)' line of a report -> the best-matching passage of the cited document, quoted
    verbatim, or a note that it was not found there."""
    out, used = ['Evidence anchors (automatic lookup in the cited documents; verbatim passages):'], 0
    for text, source in EVIDENCE.findall(report or '')[:8]:
        docid = corpus.canonical(source) or corpus.canonical(text)
        if not docid:
            entry = f'- NO DOCID cited for: "{text[:90]}"'
        else:
            try:
                doc = corpus.doc(docid)
            except Exception:   # noqa: BLE001
                doc = ''
            words = set(re.findall(r'\w{4,}', text.lower()))
            lines = [ln for ln in re.split(r'(?<=[.!?])\s+|\n', doc) if ln.strip()]
            best = max(lines, key=lambda ln: sum(w in ln.lower() for w in words), default='')
            hit = sum(w in best.lower() for w in words)
            entry = (f'- {docid}: {" ".join(best.split())[:260]}' if best and hit >= max(2, len(words) // 3)
                     else f'- not located in {docid}: "{text[:90]}"')
        if used + len(entry) > budget:
            break
        out.append(entry)
        used += len(entry)
    return '\n'.join(out) if len(out) > 1 else ''


class Delegate:
    """The Delegate tool of the search agent: one SLM subagent run per call, no automatic retries."""

    def __init__(self, slm, corpus, max_steps=80, thinking=False):
        self.slm, self.corpus, self.max_steps, self.thinking = slm, corpus, max_steps, thinking

    def __call__(self, question, inp):
        task, deliverable = str(inp.get('task', '')), str(inp.get('deliverable', ''))
        started = time.time()
        report, steps, usage, finished = run_subagent(question, task, deliverable, self.slm, self.corpus,
                                                      max_steps=self.max_steps, thinking=self.thinking)
        found = evidence_check(report, self.corpus)
        text = report.strip()[:6000] + ('\n\n' + found if found else '')
        return text, {'task': task[:600], 'deliverable': deliverable[:300], 'slm_steps': len(steps),
                      'finished': finished, 'seconds': round(time.time() - started, 1), 'report': report[:3000],
                      'slm_input_tokens': sum(u.get('prompt_tokens') or 0 for u in usage),
                      'slm_output_tokens': sum(u.get('completion_tokens') or 0 for u in usage)}
