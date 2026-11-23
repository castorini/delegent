"""The optional integrations: the attached relay (`delegent serve`) and `delegent install claude|codex`."""
import json
from http.server import BaseHTTPRequestHandler

import pytest
import requests

from delegent import install
from delegent.ledger import Ledger
from delegent.relay import Relay
from delegent.slm import SLM
from delegent.sse import collect_message, message_stream

from test_relay import FakeVLLM, serve


class Upstream(BaseHTTPRequestHandler):
    """Stands in for the Anthropic API and the OpenAI API: records what arrives, answers with a fixed stream."""
    seen = []

    def log_message(self, *args):
        pass

    def reply(self, data, kind):
        self.send_response(200)
        self.send_header('Content-Type', kind)
        self.send_header('request-id', 'req_1')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        Upstream.seen.append(('GET', self.path, dict(self.headers), None))
        self.reply(b'{"data": ["frontier-model"]}', 'application/json')

    def do_POST(self):
        raw = self.rfile.read(int(self.headers['Content-Length']))
        Upstream.seen.append(('POST', self.path, dict(self.headers), raw))
        if self.path.startswith('/v1/messages'):
            stream = message_stream([{'type': 'text', 'text': 'frontier says hi'}], 'claude-x', 0, 7)
            usage = b'"usage":{"input_tokens":20,"cache_read_input_tokens":800,"cache_creation_input_tokens":100,'
            return self.reply(stream.replace(b'"usage":{"input_tokens":0,', usage, 1), 'text/event-stream')
        done = {'type': 'response.completed', 'response': {'usage': {
            'input_tokens': 1000, 'input_tokens_details': {'cached_tokens': 900}, 'output_tokens': 30}}}
        self.reply(f'event: response.completed\ndata: {json.dumps(done)}\n\n'.encode(), 'text/event-stream')


class VLLMResponses(BaseHTTPRequestHandler):
    """vLLM's OpenAI Responses endpoint."""
    seen = []

    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        VLLMResponses.seen.append((self.path, dict(self.headers), body))
        done = {'type': 'response.completed', 'response': {'usage': {'input_tokens': 400, 'output_tokens': 12}}}
        data = f'event: response.completed\ndata: {json.dumps(done)}\n\n'.encode()
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def test_attached_relay_serves_only_the_slm(tmp_path):
    """Claude Code behind the attached relay: its own calls reach the Anthropic API untouched, with its own
    credentials; the calls that name the SLM are served by the SLM exactly as in the standalone harness."""
    upstream, upstream_url = serve(Upstream)
    vllm, vllm_url = serve(FakeVLLM)
    ledger = Ledger(tmp_path / 'usage.jsonl')
    relay = Relay(None, SLM(vllm_url, 'Qwen3.8-27B-FP8'), None, ledger=ledger, upstream=upstream_url)
    relay.start()
    body = {'model': 'claude-x', 'max_tokens': 100, 'stream': True, 'messages': [{'role': 'user', 'content': 'hi'}]}
    own = {'Authorization': 'Bearer oauth-token', 'anthropic-beta': 'oauth-2025', 'anthropic-version': '2023-06-01'}
    try:
        assert requests.get(relay.url + '/health', timeout=30).json()['attached'] is True
        r = requests.post(relay.url + '/v1/messages?beta=true', json=body, headers=own, timeout=30)
        assert r.status_code == 200 and r.headers['request-id'] == 'req_1'
        assert collect_message([r.content])['content'][0]['text'] == 'frontier says hi'
        r = requests.post(relay.url + '/v1/messages?beta=true', json=dict(body, model='Qwen3.8-27B-FP8'),
                          headers=own, timeout=30)   # no relay token is needed: the agent keeps its own credentials
        assert collect_message([r.content])['content'][0]['text'] == 'slm report'
        assert requests.get(relay.url + '/v1/models', headers=own, timeout=30).json() == {'data': ['frontier-model']}
    finally:
        relay.shutdown()
        upstream.shutdown()
        vllm.shutdown()
    method, path, headers, raw = [s for s in Upstream.seen if s[0] == 'POST'][-1]
    assert path == '/v1/messages?beta=true' and json.loads(raw) == body   # the request went on as it was
    assert headers['Authorization'] == 'Bearer oauth-token' and headers['anthropic-beta'] == 'oauth-2025'
    assert all('Qwen' not in (s[3] or b'').decode() for s in Upstream.seen)   # no SLM call reached the frontier API
    rows = [(r['role'], r['model'], r['input_tokens'], r['cached_input_tokens'], r['output_tokens'])
            for r in ledger.rows]
    assert rows == [('frontier', 'claude-x', 920, 800, 7), ('slm', 'Qwen3.8-27B-FP8', 130000, None, 12)]
    assert ledger.rows[0]['cache_write_tokens'] == 100


def test_attached_relay_routes_codex_by_model(tmp_path):
    """Codex behind the attached relay (/codex): the main agent's calls reach the OpenAI API untouched, a subagent's
    call for the SLM goes to vLLM with the SLM's key and only the tools vLLM can run."""
    upstream, upstream_url = serve(Upstream)
    vllm, vllm_url = serve(VLLMResponses)
    ledger = Ledger()
    relay = Relay(None, SLM(vllm_url, 'slm-x'), None, ledger=ledger, openai_upstream=upstream_url + '/v1')
    relay.start()
    body = {'model': 'gpt-x', 'stream': True, 'instructions': 'be brief', 'input': [{'role': 'user', 'content': 'hi'}],
            'tools': [{'type': 'function', 'name': 'shell'}, {'type': 'web_search'}]}
    own = {'Authorization': 'Bearer sk-user'}
    try:
        r = requests.post(relay.url + '/codex/responses', json=body, headers=own, timeout=30)
        assert r.status_code == 200 and b'response.completed' in r.content
        r = requests.post(relay.url + '/codex/responses', json=dict(body, model='slm-x'), headers=own, timeout=30)
        assert r.status_code == 200
    finally:
        relay.shutdown()
        upstream.shutdown()
        vllm.shutdown()
    method, path, headers, raw = Upstream.seen[-1]
    assert path == '/v1/responses' and json.loads(raw) == body and headers['Authorization'] == 'Bearer sk-user'
    path, headers, sent = VLLMResponses.seen[-1]
    assert path == '/v1/responses' and sent['model'] == 'slm-x' and headers['Authorization'] == 'Bearer EMPTY'
    assert sent['tools'] == [{'type': 'function', 'name': 'shell'}] and sent['input'] == body['input']
    rows = [(r['role'], r['model'], r['input_tokens'], r['cached_input_tokens'], r['output_tokens'])
            for r in ledger.rows]
    assert rows == [('frontier', 'gpt-x', 1000, 900, 30), ('slm', 'slm-x', 400, None, 12)]


def test_install_claude_and_uninstall(tmp_path):
    settings = {'model': 'opus', 'env': {'FOO': '1'},
                'hooks': {'SessionStart': [{'hooks': [{'type': 'command', 'command': 'echo mine'}]}]}}
    (tmp_path / 'settings.json').write_text(json.dumps(settings))
    (tmp_path / 'CLAUDE.md').write_text('# My rules\n\nBe terse.\n')
    hook = install.serve_command(['--slm', 'http://gpu:8000', '--slm-model', 'slm-x'])
    for _ in range(2):   # installing twice changes nothing more
        install.install_claude('http://127.0.0.1:8787', 'slm-x', hook, tmp_path)
    got = json.loads((tmp_path / 'settings.json').read_text())
    assert got['model'] == 'opus' and got['env'] == {
        'FOO': '1', 'ANTHROPIC_BASE_URL': 'http://127.0.0.1:8787', 'CLAUDE_CODE_SUBAGENT_MODEL': 'slm-x',
        'ANTHROPIC_DEFAULT_HAIKU_MODEL': 'slm-x'}   # the roles `delegent claude` gives the SLM
    commands = [h['command'] for e in got['hooks']['SessionStart'] for h in e['hooks']]
    assert commands == ['echo mine', hook] and 'serve --ensure' in hook
    memory = (tmp_path / 'CLAUDE.md').read_text()
    assert memory.startswith('# My rules\n\nBe terse.\n') and memory.count(install.BEGIN) == 1
    assert install.instruction({'Agent'}).strip() in memory   # the same text the standalone harness installs
    assert json.loads((tmp_path / 'settings.json.delegent-backup').read_text()) == settings

    install.uninstall_claude(tmp_path)
    assert json.loads((tmp_path / 'settings.json').read_text()) == settings
    assert (tmp_path / 'CLAUDE.md').read_text() == '# My rules\n\nBe terse.\n'


def test_install_claude_keeps_another_gateway(tmp_path):
    (tmp_path / 'settings.json').write_text(json.dumps({'env': {'ANTHROPIC_BASE_URL': 'https://gateway.example'}}))
    with pytest.raises(install.InstallError):
        install.install_claude('http://127.0.0.1:8787', 'slm-x', None, tmp_path)
    install.install_claude('http://127.0.0.1:8787', 'slm-x', None, tmp_path, force=True)
    assert 'hooks' not in json.loads((tmp_path / 'settings.json').read_text())


def test_codex_profile_and_catalog(tmp_path):
    text = install.codex_profile('http://127.0.0.1:8787', 'slm-x', tmp_path / 'delegent.models.json')
    try:
        import tomllib
    except ImportError:   # Python 3.10
        assert 'default_subagent_model = "slm-x"' in text
    else:
        profile = tomllib.loads(text)
        assert profile['model_provider'] == 'delegent' and profile['agents'] == {'default_subagent_model': 'slm-x'}
        assert profile['model_providers']['delegent'] == {
            'name': 'Delegent relay', 'base_url': 'http://127.0.0.1:8787/codex', 'env_key': 'OPENAI_API_KEY',
            'wire_api': 'responses', 'supports_websockets': False}
        assert profile['developer_instructions'].strip() == install.instruction({'Codex'}).strip()
    models = [{'slug': 'big', 'tool_mode': 'code_mode_only', 'multi_agent_version': 'v2', 'context_window': 372000},
              {'slug': 'old', 'context_window': 272000, 'default_verbosity': 'low', 'apply_patch_tool_type': 'freeform'}]
    entry = install.slm_catalog_entry(models, 'slm-x', 131072)
    assert entry['slug'] == 'slm-x' and entry['context_window'] == 131072 and entry['apply_patch_tool_type'] is None
    assert 'default_verbosity' not in entry and models[1]['slug'] == 'old'   # built from the plainest entry, a copy
    assert install.plain_subagent_tasks(models[0])['multi_agent_version'] == 'v1'
    assert 'multi_agent_version' not in install.plain_subagent_tasks(models[1])


def test_attached_relay_serves_the_search_tools(tmp_path):
    """With a corpus, the attached relay gives the agent search, get_document and Delegate over MCP: the same tools,
    subagent loop and evidence check as the search harness."""
    from delegent.delegate import Delegate
    from delegent.mcp import Toolset
    from test_mcp import rpc
    from test_search import ScriptedSLM, make_corpus
    corpus, ledger = make_corpus(tmp_path), Ledger()
    relay = Relay(None, SLM('http://unused', 'slm-x'), None, ledger=ledger,
                  tools=Toolset(corpus, Delegate(ScriptedSLM(), corpus), ledger=ledger))
    plain = Relay(None, SLM('http://unused', 'slm-x'), None)
    relay.start()
    plain.start()
    try:
        url = relay.url + '/mcp'
        assert requests.get(relay.url + '/health', timeout=30).json()['search'] is True
        assert rpc(url, 'initialize', {'protocolVersion': '2025-06-18'}).json()['result']['serverInfo']['name'] == 'corpus'
        assert rpc(url, 'notifications/initialized', id_=None).status_code == 202
        assert [t['name'] for t in rpc(url, 'tools/list').json()['result']['tools']] == [
            'search', 'get_document', 'Delegate']
        report = rpc(url, 'tools/call', {'name': 'Delegate', 'arguments': {
            'task': 'How many moons does Mars have?', 'deliverable': 'the number, with evidence'}}).json()['result']
        text = report['content'][0]['text']
        assert 'Answer: Phobos and Deimos' in text and 'Evidence anchors' in text and not report['isError']
        assert [(r['role'], r.get('subtask')) for r in ledger.rows] == [('slm', True)]
        assert requests.get(url, timeout=30).status_code == 405 and requests.delete(url, timeout=30).status_code == 200
        assert rpc(plain.url + '/mcp', 'tools/list').status_code == 404   # no corpus, no search tools
    finally:
        relay.shutdown()
        plain.shutdown()


def test_search_tools_over_stdio(tmp_path):
    """`delegent mcp`, the server an agent starts: MCP over stdio, answered by the relay's /mcp."""
    import os
    import subprocess
    import sys
    from pathlib import Path
    from delegent.mcp import Toolset
    from test_search import make_corpus
    relay = Relay(None, SLM('http://unused', 'slm-x'), None, tools=Toolset(make_corpus(tmp_path)))
    relay.start()
    lines = [{'jsonrpc': '2.0', 'id': 1, 'method': 'initialize', 'params': {}},
             {'jsonrpc': '2.0', 'method': 'notifications/initialized'},
             {'jsonrpc': '2.0', 'id': 2, 'method': 'tools/call',
              'params': {'name': 'search', 'arguments': {'query': 'moons Phobos'}}}]
    env = {**os.environ, 'PYTHONPATH': str(Path(install.__file__).parents[1])}
    try:
        got = subprocess.run([sys.executable, '-m', 'delegent.cli', 'mcp', '--slm', 'http://unused', '--slm-model',
                              'slm-x', '--port', str(relay.server_address[1])], env=env, capture_output=True, text=True,
                             input=''.join(json.dumps(line) + '\n' for line in lines), timeout=60)
    finally:
        relay.shutdown()
    answers = {a['id']: a for a in map(json.loads, got.stdout.splitlines())}
    assert sorted(answers) == [1, 2]   # the notification gets no answer
    assert '/corpus/mars.txt' in answers[2]['result']['content'][0]['text']


def test_install_with_search(tmp_path):
    """--corpus adds the search part of the instruction and the search tools, as `delegent claude --corpus` does."""
    log = tmp_path / 'claude.log'
    fake = tmp_path / 'claude'
    fake.write_text(f'#!/bin/sh\necho "$@" >> {log}\n[ "$2" = get ] && exit 1\nexit 0\n')
    fake.chmod(0o755)
    home = tmp_path / 'home'
    command = install.tools_command(['--slm', 'http://gpu:8000', '--slm-model', 'slm-x'])
    install.install_claude('http://127.0.0.1:8787', 'slm-x', None, home, search=command, claude_bin=str(fake))
    assert f'mcp add --scope user corpus -- {" ".join(command)}' in log.read_text()
    assert 'delegent.cli mcp' in ' '.join(command)
    memory = (home / 'CLAUDE.md').read_text()
    assert install.instruction({'Agent', 'Delegate'}).strip() in memory and 'the Delegate tool' in memory

    text = install.codex_profile('http://127.0.0.1:8787', 'slm-x', tmp_path / 'm.json', search=command)
    assert 'the Delegate tool' in text and 'spawn_agent' in text
    try:
        import tomllib
    except ImportError:   # Python 3.10
        return
    server = tomllib.loads(text)['mcp_servers']['corpus']
    assert [server['command'], *server['args']] == command and server['tool_timeout_sec'] == 3600
