"""Optional merged mode: one Claude Code session that both codes and searches.

By default Delegent keeps the two uses apart: `delegent claude` for coding and `delegent search` for search, whose own
loop is cheaper on pure search. With `delegent claude --corpus ...`, Delegent also serves the search agent's corpus tools
(search, get_document) and the Delegate tool to Claude Code over MCP (streamable HTTP, JSON responses), so tasks that
mix coding and search, or a session that alternates between them, can stay in one main loop.
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .agent import TOOLS, run_tool
from .delegate import DELEGATE_TOOL

SERVER_NAME = 'corpus'   # Claude Code names the tools mcp__corpus__search, mcp__corpus__Delegate, ...


class CorpusTools(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, corpus, delegate=None, ledger=None, host='127.0.0.1', port=0):
        super().__init__((host, port), Handler)
        self.corpus, self.delegate, self.ledger = corpus, delegate, ledger

    @property
    def url(self):
        return f'http://{self.server_address[0]}:{self.server_address[1]}/mcp'

    def start(self):
        threading.Thread(target=self.serve_forever, daemon=True).start()
        return self

    def config(self):
        """The --mcp-config file content for Claude Code."""
        return {'mcpServers': {SERVER_NAME: {'type': 'http', 'url': self.url}}}

    def tools(self):
        return [{'name': t['name'], 'description': t['description'], 'inputSchema': t['input_schema'],
                 'annotations': {'readOnlyHint': True}}
                for t in TOOLS + ([DELEGATE_TOOL] if self.delegate else [])]

    def call(self, method, params):
        if method == 'initialize':
            return {'protocolVersion': params.get('protocolVersion') or '2025-06-18', 'capabilities': {'tools': {}},
                    'serverInfo': {'name': SERVER_NAME, 'version': '1'}}
        if method == 'ping':
            return {}
        if method == 'tools/list':
            return {'tools': self.tools()}
        if method == 'tools/call':
            name, inp = params.get('name'), params.get('arguments') or {}
            try:
                if name == 'Delegate' and self.delegate:
                    # a session has no single question; the task the frontier model wrote is the subagent's context
                    out, entry = self.delegate(str(inp.get('task', '')), inp)
                    if self.ledger is not None:
                        self.ledger.add('slm', getattr(self.delegate.slm, 'model', 'slm'),
                                        {'input_tokens': entry['slm_input_tokens'],
                                         'output_tokens': entry['slm_output_tokens']}, entry['seconds'], subtask=True)
                else:
                    out = run_tool(self.corpus, name, inp)
            except Exception as exc:   # noqa: BLE001 -- the model sees the error and can retry
                out = f'error: {type(exc).__name__}: {str(exc)[:200]}'
            return {'content': [{'type': 'text', 'text': out}], 'isError': out.startswith('error:')}
        raise ValueError(f'unknown method {method}')


class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self, *args):
        pass

    def reply(self, code, value=None):
        data = json.dumps(value).encode() if value is not None else b''
        self.send_response(code)
        if value is not None:
            self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):   # no server-initiated stream
        self.reply(405)

    def do_DELETE(self):
        self.reply(200)

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers.get('Content-Length') or 0)) or b'{}')
        if 'id' not in req:   # a notification
            return self.reply(202)
        try:
            result = self.server.call(req.get('method'), req.get('params') or {})
            self.reply(200, {'jsonrpc': '2.0', 'id': req['id'], 'result': result})
        except Exception as exc:   # noqa: BLE001
            self.reply(200, {'jsonrpc': '2.0', 'id': req['id'],
                             'error': {'code': -32603, 'message': f'{type(exc).__name__}: {exc}'[:500]}})
