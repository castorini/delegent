"""The relay between an agent and its two models.

It speaks the Anthropic Messages API, so Claude Code talks to it as it would to Anthropic. Each call is routed by
the model it names: calls for the frontier model go to the frontier backend, every other call (subagents, the
small/fast model) goes to the SLM. Claude Code is pointed at the relay with ANTHROPIC_BASE_URL and told which model each role uses
(`claude_env`), so the routing follows Claude Code's own roles.
"""
import json
import queue
import secrets
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import requests

from .ledger import Ledger
from .slm import ContextOverflow
from .sse import event

PING = b'event: ping\ndata: {"type":"ping"}\n\n'


def estimate_tokens(payload):
    """Rough token count (~3.5 characters per token) when no tokenizer can count a request."""
    return int(len(json.dumps({k: payload.get(k) for k in ('system', 'messages', 'tools')})) / 3.5)


class Relay(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, frontier, slm, frontier_model, host='127.0.0.1', port=0, token=None, ledger=None):
        super().__init__((host, port), Handler)
        self.frontier, self.slm, self.frontier_model = frontier, slm, frontier_model
        self.token = token or secrets.token_hex(16)
        self.ledger = ledger or Ledger()

    @property
    def url(self):
        host, port = self.server_address[:2]
        return f'http://{host}:{port}'

    def role(self, model):
        return 'frontier' if model == self.frontier_model else 'slm'

    def start(self):
        thread = threading.Thread(target=self.serve_forever, daemon=True)
        thread.start()
        return thread


class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self, *args):
        pass

    def send_json(self, code, value):
        body = json.dumps(value).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def error(self, code, message, kind='api_error'):
        self.close_connection = True
        self.send_json(code, {'type': 'error', 'error': {'type': kind, 'message': message}})

    def chunk(self, data):
        self.wfile.write(f'{len(data):X}\r\n'.encode() + data + b'\r\n')
        self.wfile.flush()

    def do_GET(self):
        if self.path.split('?')[0] == '/health':
            return self.send_json(200, {'ok': True})
        return self.error(404, 'Unsupported endpoint')

    def do_POST(self):
        relay = self.server
        if relay.token not in (self.headers.get('x-api-key'), (self.headers.get('Authorization') or '')[7:]):
            return self.error(401, 'Invalid relay token')
        n = int(self.headers.get('Content-Length', '0'))
        if not 0 < n <= 64 * 1024 * 1024:
            return self.error(413, 'Invalid request length')
        try:
            payload = json.loads(self.rfile.read(n))
        except ValueError:
            return self.error(400, 'Invalid JSON')
        path, role = self.path.split('?')[0], relay.role(payload.get('model'))
        if path == '/v1/messages/count_tokens':
            return self.count(payload)
        if path != '/v1/messages':
            return self.error(404, 'Unsupported endpoint')
        self.generate(payload, role)

    def count(self, payload):
        slm = self.server.slm
        try:
            tokens = slm.count_tokens(payload)
        except ContextOverflow:   # longer than the SLM window: report it as full, so the agent compacts
            tokens = slm.context_tokens + 1
        except (requests.RequestException, KeyError, ValueError):   # e.g. image blocks the SLM template cannot render
            tokens = estimate_tokens(payload)
        self.send_json(200, {'input_tokens': tokens})

    def generate(self, payload, role):
        """Stream the routed model's answer. Headers go out at once and SSE pings keep the stream alive until the first
        byte, so a slow frontier call never looks like a dead connection. An error before any byte becomes an HTTP
        error; after that, an SSE error event. Nothing is replayed."""
        relay, started, usage = self.server, time.time(), {}
        forward = {k: self.headers[k] for k in ('anthropic-version', 'anthropic-beta') if self.headers.get(k)}
        if role == 'frontier':
            source = relay.frontier.stream(payload, usage.update, reported_input_tokens=estimate_tokens(payload),
                                           headers=forward)
        else:
            source = relay.slm.messages(payload, usage.update, headers=forward)
        parts, done = queue.Queue(maxsize=64), object()

        def pump():
            try:
                for part in source:
                    parts.put(part)
                parts.put(done)
            except Exception as exc:   # noqa: BLE001 -- reported to the client below
                parts.put(exc)
        threading.Thread(target=pump, daemon=True).start()
        first = None
        try:   # a quick failure (e.g. a rejected request) still becomes an HTTP error
            first = parts.get(timeout=10)
        except queue.Empty:
            pass
        if isinstance(first, Exception):
            self.record(role, payload, usage, started, error=repr(first))
            if isinstance(first, ContextOverflow):   # Claude Code compacts on this error
                return self.error(400, f'prompt is too long: {first}', 'invalid_request_error')
            return self.error(502, f'{role} model failed: {str(first)[:500]}')
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.send_header('Transfer-Encoding', 'chunked')
        self.send_header('Cache-Control', 'no-cache')
        self.end_headers()
        error = None
        try:
            item = first
            while True:
                while item is None:   # still waiting for the model
                    self.chunk(PING)
                    try:
                        item = parts.get(timeout=10)
                    except queue.Empty:
                        item = None
                if item is done:
                    break
                if isinstance(item, Exception):
                    error = repr(item)
                    self.chunk(event('error', {'type': 'error', 'error': {'type': 'api_error',
                                                                          'message': f'{role} model failed'}}))
                    break
                self.chunk(item)
                item = parts.get()
            self.wfile.write(b'0\r\n\r\n')
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):   # the client closed the stream itself
            error = 'cancelled_by_client'
            self.close_connection = True
        self.record(role, payload, usage, started, error=error)

    def record(self, role, payload, usage, started, error=None):
        relay = self.server
        model = relay.frontier_model if role == 'frontier' else relay.slm.model
        extra = {'error': error} if error else {}
        relay.ledger.add(role, model, usage, time.time() - started, requested_model=payload.get('model'), **extra)


def claude_env(relay, slm_model, frontier_context_tokens=1_000_000, max_output_tokens=32768):
    """Environment that points Claude Code at the relay and assigns its roles: the main agent uses the frontier model,
    subagents and the small/fast model use the SLM."""
    return {'ANTHROPIC_BASE_URL': relay.url, 'ANTHROPIC_API_KEY': relay.token,
            'ANTHROPIC_MODEL': relay.frontier_model, 'ANTHROPIC_DEFAULT_OPUS_MODEL': relay.frontier_model,
            'ANTHROPIC_DEFAULT_SONNET_MODEL': relay.frontier_model, 'ANTHROPIC_DEFAULT_HAIKU_MODEL': slm_model,
            'CLAUDE_CODE_SUBAGENT_MODEL': slm_model,
            'CLAUDE_CODE_MAX_CONTEXT_TOKENS': str(frontier_context_tokens),
            'CLAUDE_CODE_MAX_OUTPUT_TOKENS': str(max_output_tokens),
            'CLAUDE_CODE_MAX_RETRIES': '0', 'CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC': '1', 'DISABLE_AUTOUPDATER': '1',
            'API_TIMEOUT_MS': '1800000'}
