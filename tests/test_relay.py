import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import requests

from delegent.ledger import Ledger, cost
from delegent.relay import Relay, claude_env
from delegent.slm import SLM, single_system
from delegent.sse import collect_message, message_stream


class FakeFrontier:
    model = 'gpt-x'

    def __init__(self):
        self.calls = []

    def stream(self, payload, usage_sink, reported_input_tokens=0, headers=None):
        self.calls.append(payload)
        usage_sink({'input_tokens': 1000, 'cached_input_tokens': 800, 'output_tokens': 100})
        yield message_stream([{'type': 'text', 'text': 'frontier says hi'}], 'gpt-x')


class FakeVLLM(BaseHTTPRequestHandler):
    """vLLM's Anthropic endpoints: count_tokens and a streamed /v1/messages."""
    seen = []

    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        FakeVLLM.seen.append((self.path, body))
        if self.path == '/v1/messages/count_tokens':
            data = json.dumps({'input_tokens': 130000}).encode()
            self.send_response(200)
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        data = message_stream([{'type': 'text', 'text': 'slm report'}], body['model'], 130000, 12)
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def serve(handler):
    server = ThreadingHTTPServer(('127.0.0.1', 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f'http://127.0.0.1:{server.server_address[1]}'


def post(relay, model, path='/v1/messages', token=None):
    payload = {'model': model, 'max_tokens': 32768, 'system': [{'type': 'text', 'text': 'sys'}],
               'messages': [{'role': 'user', 'content': 'hi'}, {'role': 'system', 'content': 'note'}],
               'output_config': {'effort': 'high'}}
    return requests.post(relay.url + path, json=payload, headers={'x-api-key': token or relay.token}, timeout=30)


def test_relay_routes_by_role(tmp_path):
    vllm, url = serve(FakeVLLM)
    ledger = Ledger(tmp_path / 'usage.jsonl')
    fake = FakeFrontier()
    relay = Relay(fake, SLM(url, 'Qwen3.8-27B-FP8'), 'gpt-x', ledger=ledger)
    relay.start()
    try:
        assert post(relay, 'gpt-x', token='wrong').status_code == 401
        r = post(relay, 'gpt-x')
        assert collect_message([r.content])['content'][0]['text'] == 'frontier says hi'
        r = post(relay, 'Qwen3.8-27B-FP8')
        assert collect_message([r.content])['content'][0]['text'] == 'slm report'
        assert post(relay, 'Qwen3.8-27B-FP8', '/v1/messages/count_tokens').json() == {'input_tokens': 130000}
    finally:
        relay.shutdown()
        vllm.shutdown()
    assert len(fake.calls) == 1
    roles = [row['role'] for row in ledger.rows]
    assert roles == ['frontier', 'slm']
    generate = [b for p, b in FakeVLLM.seen if p == '/v1/messages'][-1]
    assert generate['max_tokens'] == 131072 - 130000   # clamped to what is left of the SLM window
    assert all(m['role'] != 'system' for m in generate['messages'])   # Qwen: system turns become user turns
    assert generate['chat_template_kwargs']['enable_thinking'] is True
    assert abs(cost(ledger.rows, {'gpt-x': (2.0, 0.1, 10.0)}) - (200 * 2 + 800 * 0.1 + 100 * 10) / 1e6) < 1e-12


def test_claude_env_assigns_roles():
    relay = Relay(FakeFrontier(), SLM('http://x', 's'), 'gpt-x')
    env = claude_env(relay, 'slm-model')
    relay.server_close()
    assert env['ANTHROPIC_MODEL'] == 'gpt-x' and env['CLAUDE_CODE_SUBAGENT_MODEL'] == 'slm-model'
    assert env['ANTHROPIC_DEFAULT_HAIKU_MODEL'] == 'slm-model' and env['ANTHROPIC_API_KEY'] == relay.token


def test_single_system_only_for_qwen():
    payload = {'messages': [{'role': 'system', 'content': 'x'}], 'output_config': {'effort': 'high'}}
    assert single_system(payload, 'openbmb/MiniCPM5-2B') is payload
    out = single_system(payload, 'Qwen3.8-27B-FP8')
    assert out['messages'][0]['role'] == 'user' and 'output_config' not in out


class TooLongVLLM(FakeVLLM):
    def do_POST(self):
        self.rfile.read(int(self.headers['Content-Length']))
        data = b'{"error": "This model\'s maximum context length is 131072 tokens"}'
        self.send_response(400)
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def test_overflow_becomes_prompt_too_long():
    vllm, url = serve(TooLongVLLM)
    relay = Relay(FakeFrontier(), SLM(url, 'Qwen3.8-27B-FP8'), 'gpt-x')
    relay.start()
    try:
        assert post(relay, 'Qwen3.8-27B-FP8', '/v1/messages/count_tokens').json() == {'input_tokens': 131073}
        r = post(relay, 'Qwen3.8-27B-FP8')
        assert r.status_code == 400 and 'prompt is too long' in r.json()['error']['message']
    finally:
        relay.shutdown()
        vllm.shutdown()
