"""The relay between an agent and its two models.

It speaks the Anthropic Messages API, so Claude Code talks to it as it would to Anthropic. Each call is routed by
the model it names: calls for the frontier model go to the frontier backend, every other call (subagents, the
small/fast model) goes to the SLM. Claude Code is pointed at the relay with ANTHROPIC_BASE_URL and told which model each role uses
(`claude_env`), so the routing follows Claude Code's own roles.

Attached (`Relay(None, slm, None)`, used by `delegent serve`), the relay sits behind an agent the user already runs.
Only calls that name the SLM are served by the SLM; every other request goes to the Anthropic API exactly as the agent
sent it, with the agent's own credentials, and the answer is returned byte for byte. The SLM path is the same code in
both modes. Codex is served the same way under `/codex`, over the OpenAI API: calls that name the SLM go to the
replica their conversation is pinned to, every other request goes to the OpenAI API unchanged. With a corpus, it also
serves the search tools and `Delegate` to either agent over MCP at `/mcp` (mcp.py).
"""
import json
import os
import queue
import secrets
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import requests

from .frontier import ANTHROPIC_URL
from .ledger import Ledger
from .slm import ContextOverflow, conversation_key
from .sse import event

PING = b'event: ping\ndata: {"type":"ping"}\n\n'
# headers that belong to one connection and are not forwarded by the attached relay
HOP = {'host', 'content-length', 'connection', 'keep-alive', 'transfer-encoding', 'accept-encoding', 'content-encoding',
       'proxy-connection', 'te', 'trailer', 'upgrade'}
MCP_PATH = '/mcp'
CODEX_PREFIX = '/codex'   # the attached relay serves Codex (the OpenAI API) under this path
OPENAI_URL = os.environ.get('DELEGENT_OPENAI_API', 'https://api.openai.com/v1')


def normalize_usage(usage):
    """Ledger usage from an Anthropic or an OpenAI usage object: input tokens include the cached ones."""
    if 'cache_read_input_tokens' in usage or 'cache_creation_input_tokens' in usage:   # Anthropic
        read, write = usage.get('cache_read_input_tokens') or 0, usage.get('cache_creation_input_tokens') or 0
        return {'input_tokens': (usage.get('input_tokens') or 0) + read + write, 'cached_input_tokens': read,
                'cache_write_tokens': write, 'output_tokens': usage.get('output_tokens')}
    details = usage.get('input_tokens_details') or usage.get('prompt_tokens_details') or {}
    return {'input_tokens': usage.get('input_tokens', usage.get('prompt_tokens')),
            'cached_input_tokens': details.get('cached_tokens'),
            'output_tokens': usage.get('output_tokens', usage.get('completion_tokens'))}


def estimate_tokens(payload):
    """Rough token count (~3.5 characters per token) when no tokenizer can count a request."""
    return int(len(json.dumps({k: payload.get(k) for k in ('system', 'messages', 'tools')})) / 3.5)


class Relay(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, frontier, slm, frontier_model, host='127.0.0.1', port=0, token=None, ledger=None,
                 upstream=None, openai_upstream=None, tools=None):
        super().__init__((host, port), Handler)
        self.frontier, self.slm, self.frontier_model = frontier, slm, frontier_model
        self.token = token or secrets.token_hex(16)
        self.ledger = ledger or Ledger()
        self.upstream = (upstream or ANTHROPIC_URL.split('/v1/')[0]).rstrip('/')
        self.openai_upstream = (openai_upstream or OPENAI_URL).rstrip('/')
        self.tools = tools   # attached, optional: the corpus tools and Delegate (mcp.Toolset), served at /mcp

    @property
    def attached(self):
        """No frontier backend of its own: the agent's frontier calls pass through to the Anthropic API."""
        return self.frontier is None

    @property
    def url(self):
        host, port = self.server_address[:2]
        return f'http://{host}:{port}'

    def role(self, model):
        if self.attached:   # the agent chooses its own frontier model; only calls for the SLM are ours
            return 'slm' if model == self.slm.model else 'frontier'
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
        relay = self.server
        if self.path.split('?')[0] == '/health':
            return self.send_json(200, {'ok': True, 'attached': relay.attached, 'slm_model': relay.slm.model,
                                        'search': relay.tools is not None})
        if self.path.split('?')[0] == MCP_PATH:   # MCP over HTTP without a server-initiated stream
            return self.error(405, 'no stream')
        if relay.attached:
            return self.codex(b'', {}) if self.path.startswith(CODEX_PREFIX + '/') else self.passthrough(b'', {})
        return self.error(404, 'Unsupported endpoint')

    def do_DELETE(self):
        if self.path.split('?')[0] == MCP_PATH:   # the MCP client closing its session
            return self.send_json(200, {})
        return self.error(404, 'Unsupported endpoint')

    def do_POST(self):
        relay = self.server
        if self.path.split('?')[0] == MCP_PATH:
            return self.mcp()
        if not relay.attached and relay.token not in (self.headers.get('x-api-key'),
                                                      (self.headers.get('Authorization') or '')[7:]):
            return self.error(401, 'Invalid relay token')
        n = int(self.headers.get('Content-Length', '0'))
        if not 0 < n <= 64 * 1024 * 1024:
            return self.error(413, 'Invalid request length')
        body = self.rfile.read(n)
        try:
            payload = json.loads(body)
        except ValueError:
            payload = None
        if relay.attached and self.path.startswith(CODEX_PREFIX + '/'):
            return self.codex(body, payload if isinstance(payload, dict) else {})
        if not isinstance(payload, dict):
            return self.passthrough(body, {}) if relay.attached else self.error(400, 'Invalid JSON')
        path, role = self.path.split('?')[0], relay.role(payload.get('model'))
        if relay.attached and (role == 'frontier' or path not in ('/v1/messages', '/v1/messages/count_tokens')):
            return self.passthrough(body, payload)
        if path == '/v1/messages/count_tokens':
            return self.count(payload)
        if path != '/v1/messages':
            return self.error(404, 'Unsupported endpoint')
        self.generate(payload, role)

    def mcp(self):
        """The search tools for the agent's main loop: search, get_document and Delegate, whose subagent searches in
        its own loop on the SLM and returns a report with checked evidence (the same objects as the search harness)."""
        relay = self.server
        if relay.tools is None:
            return self.error(404, 'no corpus: start `delegent serve` with --corpus')
        try:
            req = json.loads(self.rfile.read(int(self.headers.get('Content-Length') or 0)) or b'{}')
        except ValueError:
            return self.error(400, 'Invalid JSON')
        code, answer = relay.tools.respond(req)
        if answer is None:
            self.send_response(code)
            self.send_header('Content-Length', '0')
            self.end_headers()
            return
        self.send_json(code, answer)

    def proxy(self, url, headers, body, role, model, requested=None):
        """Attached: send a request on as it is and return the answer as it comes, byte for byte. Only the usage is
        read on the way, for the ledger (Anthropic and OpenAI answers, streamed or not)."""
        relay, started = self.server, time.time()
        try:
            up = requests.request(self.command, url, headers={**headers, 'Accept-Encoding': 'identity'},
                                  data=body or None, stream=True, timeout=(30, 1800))
        except requests.RequestException as exc:
            if model:
                relay.ledger.add(role, model, {}, time.time() - started, error=repr(exc)[:300])
            return self.error(502, f'{role} model unreachable: {str(exc)[:300]}')
        usage, buffer, error = {}, b'', None

        def read(line):
            if line.startswith(b'data:'):
                line = line[5:]
            try:
                ev = json.loads(line)
            except ValueError:
                return
            if isinstance(ev, dict):
                for holder in (ev, ev.get('message'), ev.get('response')):
                    if isinstance(holder, dict) and isinstance(holder.get('usage'), dict):
                        usage.update(holder['usage'])
        try:
            self.send_response(up.status_code)
            for k, v in up.headers.items():
                if k.lower() not in HOP:
                    self.send_header(k, v)
            self.send_header('Transfer-Encoding', 'chunked')
            self.end_headers()
            for part in up.iter_content(chunk_size=None):
                if not part:
                    continue
                self.chunk(part)
                if model:
                    *lines, buffer = (buffer + part).split(b'\n')
                    for line in lines:
                        read(line)
            self.wfile.write(b'0\r\n\r\n')
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            error = 'cancelled_by_client'
            self.close_connection = True
        except requests.RequestException as exc:   # the upstream stream broke: the agent sees a cut stream and retries
            error = repr(exc)[:300]
            self.close_connection = True
        finally:
            up.close()
        if not model:   # not a model call (e.g. a model listing): nothing to record
            return
        read(buffer)   # an answer that was not streamed is one JSON body
        if up.status_code != 200:
            error = error or f'HTTP {up.status_code}'
        extra = {'error': error} if error else {}
        if requested and requested != model:
            extra['requested_model'] = requested
        relay.ledger.add(role, model, normalize_usage(usage), time.time() - started, **extra)

    def client_headers(self):
        return {k: v for k, v in self.headers.items() if k.lower() not in HOP}

    def passthrough(self, body, payload):
        """Attached, Claude Code: a request that is not for the SLM goes to the Anthropic API unchanged, with the
        agent's own credentials and headers."""
        metered = self.path.split('?')[0] == '/v1/messages'
        self.proxy(self.server.upstream + self.path, self.client_headers(), body, 'frontier',
                   payload.get('model') if metered else None)

    def codex(self, body, payload):
        """Attached, Codex: requests under /codex are OpenAI API calls. One that names the SLM (a subagent) goes to the
        replica its conversation is pinned to; every other one goes to the OpenAI API unchanged, with the agent's own
        credentials."""
        relay = self.server
        rest = self.path[len(CODEX_PREFIX):]
        path, slm = rest.split('?')[0], relay.slm
        generates = path in ('/responses', '/chat/completions')
        if generates and relay.role(payload.get('model')) == 'slm':
            opening = payload.get('input') if path == '/responses' else payload.get('messages')
            key = conversation_key(payload.get('instructions'), opening[:1] if isinstance(opening, list) else opening)
            sent = dict(payload, model=slm.model)
            if isinstance(sent.get('tools'), list):   # hosted tools (e.g. web_search) exist only at OpenAI
                sent['tools'] = [t for t in sent['tools'] if t.get('type') == 'function']
            if sent.get('stream') and path == '/chat/completions':
                sent['stream_options'] = {**(sent.get('stream_options') or {}), 'include_usage': True}
            headers = {'Authorization': 'Bearer ' + slm.api_key, 'Content-Type': 'application/json'}
            return self.proxy(slm.endpoint_for(key) + '/v1' + rest, headers, json.dumps(sent).encode(), 'slm',
                              slm.model, payload.get('model'))
        self.proxy(relay.openai_upstream + rest, self.client_headers(), body, 'frontier',
                   payload.get('model') if generates else None)

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
