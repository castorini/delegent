import json

import requests

from delegent.delegate import Delegate
from delegent.ledger import Ledger
from delegent.mcp import CorpusTools

from test_search import ScriptedSLM, make_corpus


def rpc(url, method, params=None, id_=1):
    body = {'jsonrpc': '2.0', 'method': method, 'params': params or {}}
    if id_ is not None:
        body['id'] = id_
    return requests.post(url, json=body, timeout=30)


def test_corpus_tools_over_mcp(tmp_path):
    corpus = make_corpus(tmp_path)
    ledger = Ledger()
    server = CorpusTools(corpus, Delegate(ScriptedSLM(), corpus), ledger=ledger).start()
    try:
        assert server.config() == {'mcpServers': {'corpus': {'type': 'http', 'url': server.url}}}
        init = rpc(server.url, 'initialize', {'protocolVersion': '2025-06-18'}).json()['result']
        assert init['serverInfo']['name'] == 'corpus' and init['protocolVersion'] == '2025-06-18'
        assert rpc(server.url, 'notifications/initialized', id_=None).status_code == 202

        names = [t['name'] for t in rpc(server.url, 'tools/list').json()['result']['tools']]
        assert names == ['search', 'get_document', 'Delegate']

        hits = rpc(server.url, 'tools/call', {'name': 'search', 'arguments': {'query': 'moons Phobos'}}).json()
        assert '/corpus/mars.txt' in hits['result']['content'][0]['text'] and not hits['result']['isError']
        doc = rpc(server.url, 'tools/call', {'name': 'get_document', 'arguments': {'docid': '/corpus/venus.txt'}})
        assert 'retrograde' in doc.json()['result']['content'][0]['text']

        report = rpc(server.url, 'tools/call', {'name': 'Delegate', 'arguments': {
            'task': 'How many moons does Mars have?', 'deliverable': 'the number, with evidence'}}).json()
        assert 'Phobos' in report['result']['content'][0]['text']
        assert [r['role'] for r in ledger.rows] == ['slm']   # the subagent's SLM usage is on the ledger

        missing = rpc(server.url, 'tools/call', {'name': 'get_document', 'arguments': {'docid': 'nope'}}).json()
        assert missing['result']['isError']
        assert 'error' in rpc(server.url, 'resources/list').json()
    finally:
        server.shutdown()


def test_search_tools_only_without_delegate(tmp_path):
    server = CorpusTools(make_corpus(tmp_path)).start()
    try:
        names = [t['name'] for t in rpc(server.url, 'tools/list').json()['result']['tools']]
        assert names == ['search', 'get_document']
    finally:
        server.shutdown()


def test_claude_gets_mcp_config_and_both_instruction_parts(tmp_path, monkeypatch):
    """`delegent claude --corpus local` starts the tools, passes --mcp-config and installs both instruction parts."""
    import delegent.cli as cli

    corpus_dir = tmp_path / 'corpus'
    make_corpus(tmp_path)
    seen = {}

    class FakePopen:
        def __init__(self, cmd, env):
            seen['cmd'], seen['env'] = cmd, env

        def wait(self):
            config = json.loads(open(seen['cmd'][seen['cmd'].index('--mcp-config') + 1]).read())
            seen['tools'] = [t['name'] for t in rpc(config['mcpServers']['corpus']['url'], 'tools/list')
                             .json()['result']['tools']]
            return 0

    monkeypatch.setattr(cli.subprocess, 'Popen', FakePopen)
    monkeypatch.setattr(cli.frontier_backends, 'make', lambda *a, **k: object())
    config_dir = tmp_path / 'cfg'
    try:
        cli.main(['claude', '--slm', 'http://127.0.0.1:9', '--slm-model', 'm', '--claude-bin', '/bin/true',
                  '--config-dir', str(config_dir), '--corpus', 'local', '--corpus-root', str(corpus_dir),
                  '--index', str(tmp_path / 'index'), '--ledger', str(tmp_path / 'u.jsonl')])
    except SystemExit as exc:
        assert exc.code == 0
    assert seen['tools'] == ['search', 'get_document', 'Delegate']
    claude_md = (config_dir / 'CLAUDE.md').read_text()
    assert 'Agent tool' in claude_md and 'Delegate tool' in claude_md
