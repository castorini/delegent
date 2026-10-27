import json
import threading
import time

from delegent import agent
from delegent.bm25 import build
from delegent.corpus import LocalCorpus
from delegent.delegate import Delegate, evidence_check, run_subagent
from delegent.ledger import Ledger
from delegent.sse import message_stream


def make_corpus(tmp_path):
    root = tmp_path / 'corpus'
    root.mkdir()
    (root / 'mars.txt').write_text('---\ntitle: Mars\n---\nMars has two moons, Phobos and Deimos. It is red.')
    (root / 'venus.txt').write_text('Venus has no moons. It rotates very slowly, retrograde.')
    build(root, tmp_path / 'index', workers=1, buckets=1 << 12)
    return LocalCorpus(root, tmp_path / 'index')


def test_bm25_local_corpus(tmp_path):
    corpus = make_corpus(tmp_path)
    hits = corpus.search('moons Phobos')
    assert hits[0][0] == '/corpus/mars.txt'
    assert 'Deimos' in corpus.doc('/corpus/mars.txt') and corpus.doc('mars.txt')
    assert corpus.canonical('/corpus/nope.txt') is None


def test_evidence_check(tmp_path):
    corpus = make_corpus(tmp_path)
    report = ('Answer: Phobos and Deimos\nConfidence: 90%\nEvidence:\n'
              '- Mars has two moons, Phobos and Deimos (source: /corpus/mars.txt)\n'
              '- Mars has rings (source: /corpus/venus.txt)')
    out = evidence_check(report, corpus)
    assert '- /corpus/mars.txt: Mars has two moons, Phobos and Deimos.' in out
    assert 'not located in /corpus/venus.txt' in out


class ScriptedSLM:
    model = 'Qwen3.8-27B-FP8'

    def __init__(self):
        self.calls = 0

    def chat(self, messages, tools=None, tool_choice='auto', thinking=False, sampling=None, **kw):
        self.calls += 1
        if self.calls == 1:
            call = {'id': 'c1', 'function': {'name': 'search', 'arguments': json.dumps({'query': ['moons of Mars']})}}
            return {'content': '', 'tool_calls': [call]}, {'prompt_tokens': 100, 'completion_tokens': 10}
        assert messages[-1]['role'] == 'tool' and '/corpus/mars.txt' in messages[-1]['content']
        finish = {'id': 'c2', 'function': {'name': 'finish', 'arguments': json.dumps({
            'answer': 'Phobos and Deimos', 'confidence': '90%',
            'evidences': [{'evidence': 'Mars has two moons, Phobos and Deimos', 'url': '/corpus/mars.txt'}]})}}
        return {'content': '', 'tool_calls': [finish]}, {'prompt_tokens': 300, 'completion_tokens': 30}


def test_subagent_report(tmp_path):
    corpus = make_corpus(tmp_path)
    report, steps, usage, finished = run_subagent('Q?', 'find the moons of Mars', 'names', ScriptedSLM(), corpus)
    assert finished and 'Answer: Phobos and Deimos' in report and '(source: /corpus/mars.txt)' in report
    assert len(steps) == 2


class ScriptedFrontier:
    """Turn 1: two Delegate calls in one turn; turn 2: the final answer."""
    model = 'gpt-x'

    def __init__(self):
        self.payloads = []

    def stream(self, payload, usage_sink, **kw):
        self.payloads.append(json.loads(json.dumps(payload)))
        usage_sink({'input_tokens': 500, 'cached_input_tokens': 400, 'output_tokens': 20})
        if len(self.payloads) == 1:
            content = [{'type': 'tool_use', 'id': f't{i}', 'name': 'Delegate',
                        'input': {'task': f'subtask {i}', 'deliverable': 'facts'}} for i in (1, 2)]
        else:
            content = [{'type': 'text', 'text': 'Exact Answer: Phobos and Deimos'}]
        yield message_stream(content, 'gpt-x')


def test_agent_runs_delegations_in_parallel(tmp_path):
    corpus = make_corpus(tmp_path)
    running, peak, lock = [0], [0], threading.Lock()

    class SlowDelegate(Delegate):
        def __call__(self, question, inp):
            with lock:
                running[0] += 1
                peak[0] = max(peak[0], running[0])
            time.sleep(0.2)
            with lock:
                running[0] -= 1
            return f"report for {inp['task']}", {'task': inp['task'], 'deliverable': '', 'slm_steps': 3,
                                                 'finished': True, 'seconds': 0.2, 'report': '',
                                                 'slm_input_tokens': 10, 'slm_output_tokens': 1}

    frontier = ScriptedFrontier()
    ledger = Ledger()
    result = agent.answer('Which moons?', frontier, corpus, SlowDelegate(ScriptedSLM(), corpus), ledger=ledger)
    assert result['answer'] == 'Exact Answer: Phobos and Deimos'
    assert peak[0] == 2 and result['slm_calls'] == 6 and len(result['delegations']) == 2
    second = frontier.payloads[1]
    assert 'Delegate' in [t['name'] for t in second['tools']] and 'research assistant' in second['system']
    results = second['messages'][-1]['content']
    assert [r['content'] for r in results] == ['report for subtask 1', 'report for subtask 2']
    assert [r['role'] for r in ledger.rows].count('frontier') == 2


def test_agent_alone_has_no_delegate(tmp_path):
    corpus = make_corpus(tmp_path)

    class Answering:
        model = 'gpt-x'
        payloads = []

        def stream(self, payload, usage_sink, **kw):
            Answering.payloads.append(payload)
            usage_sink({'input_tokens': 1, 'output_tokens': 1})
            yield message_stream([{'type': 'text', 'text': 'Exact Answer: none'}], 'gpt-x')

    result = agent.answer('Q?', Answering(), corpus, None)
    assert result['answer'] == 'Exact Answer: none' and result['delegations'] == []
    assert [t['name'] for t in Answering.payloads[0]['tools']] == ['search', 'get_document']


def test_one_instruction_parts_by_tool():
    coding, search = agent.instruction({'Agent'}), agent.instruction({'Delegate'})
    assert coding.startswith('# Working with subagents') and 'Delegate tool' not in coding
    assert search.startswith('You have a research assistant') and 'Agent tool' not in search
    assert agent.instruction({'Agent', 'Delegate'}) == coding + search
    assert agent.instruction(set()) == '' and '<!--' not in coding + search
