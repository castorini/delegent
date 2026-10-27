"""Frontier model backends.

The harness speaks the Anthropic Messages API internally (it is what Claude Code sends). A frontier backend takes an
Anthropic request and streams back Anthropic SSE, so the same backend serves Claude Code through the relay and the
search agent loop directly.

- `OpenAIResponses` translates to the OpenAI Responses API (GPT models such as gpt-6.1-sol). The model's reasoning item
  is returned as an Anthropic `thinking` block whose signature carries the full Responses reasoning item (with its
  encrypted content). Claude Code sends it back unchanged, so later turns continue the same reasoning (store=false,
  stateless). Every request carries a stable prompt-cache key for its conversation.
- `Anthropic` forwards the request to the Anthropic API (Claude models), optionally fixing the reasoning effort.
"""
import base64
import hashlib
import json
import os
import time

import requests

from .sse import event

RESPONSES_URL = os.environ.get('DELEGENT_OPENAI_URL', 'https://api.openai.com/v1/responses')
ANTHROPIC_URL = os.environ.get('DELEGENT_ANTHROPIC_URL', 'https://api.anthropic.com/v1/messages')
SIGNATURE = 'delegent-reasoning-v1:'
RETRYABLE_STATUS = (408, 409, 429, 500, 502, 503, 504, 529)
RETRYABLE_EVENTS = ('server_is_overloaded', 'rate_limit', 'overloaded', 'temporarily_unavailable', 'server_error')
FIRST_OUTPUT = ('response.output_item.added', 'response.output_text.delta', 'response.reasoning_summary_text.delta',
                'response.function_call_arguments.delta', 'response.completed')


class FrontierError(RuntimeError):
    pass


def cache_key(*parts):
    """Stable prompt-cache key for a conversation, from its fixed beginning (model, instructions, first input)."""
    return 'dlg-' + hashlib.sha1('\x1f'.join(str(p) for p in parts).encode()).hexdigest()[:24]


def _text(content):
    if isinstance(content, str):
        return content
    parts = []
    for block in content or []:
        if block.get('type') == 'text':
            parts.append(block.get('text', ''))
        elif block.get('type') == 'image':
            parts.append('[image omitted]')
    return '\n'.join(parts)


def to_responses(payload, model, effort='high'):
    """Translate an Anthropic /v1/messages request body into an OpenAI Responses request body."""
    items = []
    for message in payload['messages']:
        role, content = message['role'], message['content']
        if role == 'system':
            items.append({'role': 'developer', 'content': [{'type': 'input_text', 'text': _text(content)}]})
            continue
        blocks = [{'type': 'text', 'text': content}] if isinstance(content, str) else content
        if role == 'user':
            texts = []

            def flush():
                if texts:
                    items.append({'role': 'user', 'content': [{'type': 'input_text', 'text': t} for t in texts]})
                    texts.clear()
            for block in blocks:
                if block['type'] == 'tool_result':
                    flush()
                    items.append({'type': 'function_call_output', 'call_id': block['tool_use_id'],
                                  'output': _text(block.get('content'))})
                elif block['type'] == 'text' and block.get('text'):
                    texts.append(block['text'])
            flush()
        elif role == 'assistant':
            for block in blocks:
                kind = block['type']
                if kind == 'thinking' and block.get('signature', '').startswith(SIGNATURE):
                    items.append(json.loads(base64.b64decode(block['signature'][len(SIGNATURE):])))
                elif kind == 'text' and block.get('text'):
                    items.append({'type': 'message', 'role': 'assistant',
                                  'content': [{'type': 'output_text', 'text': block['text']}]})
                elif kind == 'tool_use':
                    items.append({'type': 'function_call', 'call_id': block['id'], 'name': block['name'],
                                  'arguments': json.dumps(block.get('input') or {})})
                # thinking from another model (e.g. the SLM) is not shown to the frontier model
    tools = [{'type': 'function', 'name': t['name'], 'description': t.get('description', ''),
              'parameters': t.get('input_schema') or {'type': 'object', 'properties': {}}, 'strict': False}
             for t in payload.get('tools') or [] if t.get('name')]
    system = payload.get('system')
    body = {'model': model, 'instructions': _text(system) if system else '', 'input': items,
            'reasoning': {'effort': effort, 'summary': 'auto'}, 'include': ['reasoning.encrypted_content'],
            'store': False, 'stream': True}
    if tools:
        body.update(tools=tools, tool_choice='auto', parallel_tool_calls=True)
    return body


def strip_frontier_reasoning(payload):
    """Copy of an Anthropic request without the frontier model's signed reasoning (for the SLM)."""
    out = dict(payload)
    messages = []
    for message in payload.get('messages') or []:
        content = message['content']
        if message['role'] == 'assistant' and isinstance(content, list):
            kept = [b for b in content if not (b.get('type') == 'thinking'
                                               and b.get('signature', '').startswith(SIGNATURE))]
            message = dict(message, content=kept or [{'type': 'text', 'text': ''}])
        messages.append(message)
    out['messages'] = messages
    return out


def _events(response):
    for raw in response.iter_lines():
        if isinstance(raw, bytes):
            raw = raw.decode('utf-8')
        if raw and raw.startswith('data:'):
            data = raw[5:].strip()
            if data and data != '[DONE]':
                yield json.loads(data)


def stream_anthropic(events, request_model, reported_input_tokens, usage_sink):
    """Yield Anthropic SSE bytes for a stream of Responses events. usage_sink(dict) receives the frontier usage."""
    index, blocks, summaries, args_streamed = -1, {}, {}, set()
    has_call, started, final = False, False, None
    for ev in events:
        kind = ev.get('type', '')
        if not started:
            started = True
            yield event('message_start', {'type': 'message_start', 'message': {
                'id': 'msg_dlg_' + str(int(time.time() * 1000)), 'type': 'message', 'role': 'assistant',
                'content': [], 'model': request_model, 'stop_reason': None, 'stop_sequence': None,
                'usage': {'input_tokens': reported_input_tokens, 'output_tokens': 0}}})
        if kind == 'response.output_item.added':
            item, position = ev['item'], ev['output_index']
            if item['type'] == 'message':
                index += 1
                blocks[position] = index
                yield event('content_block_start', {'type': 'content_block_start', 'index': index,
                                                    'content_block': {'type': 'text', 'text': ''}})
            elif item['type'] == 'function_call':
                index += 1
                blocks[position] = index
                has_call = True
                yield event('content_block_start', {'type': 'content_block_start', 'index': index, 'content_block': {
                    'type': 'tool_use', 'id': item['call_id'], 'name': item['name'], 'input': {}}})
        elif kind == 'response.output_text.delta' and ev['output_index'] in blocks:
            yield event('content_block_delta', {'type': 'content_block_delta', 'index': blocks[ev['output_index']],
                                                'delta': {'type': 'text_delta', 'text': ev.get('delta', '')}})
        elif kind == 'response.function_call_arguments.delta' and ev['output_index'] in blocks:
            args_streamed.add(ev['output_index'])
            yield event('content_block_delta', {'type': 'content_block_delta', 'index': blocks[ev['output_index']],
                                                'delta': {'type': 'input_json_delta',
                                                          'partial_json': ev.get('delta', '')}})
        elif kind == 'response.reasoning_summary_text.delta':
            summaries[ev['output_index']] = summaries.get(ev['output_index'], '') + ev.get('delta', '')
        elif kind == 'response.output_item.done':
            item, position = ev['item'], ev['output_index']
            if item['type'] == 'reasoning':
                index += 1
                signature = SIGNATURE + base64.b64encode(json.dumps(item).encode()).decode()
                text = summaries.get(position) or '\n\n'.join(s.get('text', '') for s in item.get('summary') or [])
                yield event('content_block_start', {'type': 'content_block_start', 'index': index,
                                                    'content_block': {'type': 'thinking', 'thinking': ''}})
                if text:
                    yield event('content_block_delta', {'type': 'content_block_delta', 'index': index,
                                                        'delta': {'type': 'thinking_delta', 'thinking': text}})
                yield event('content_block_delta', {'type': 'content_block_delta', 'index': index,
                                                    'delta': {'type': 'signature_delta', 'signature': signature}})
                yield event('content_block_stop', {'type': 'content_block_stop', 'index': index})
            elif position in blocks:
                if item['type'] == 'function_call' and position not in args_streamed:
                    yield event('content_block_delta', {'type': 'content_block_delta', 'index': blocks[position],
                                                        'delta': {'type': 'input_json_delta',
                                                                  'partial_json': item.get('arguments') or '{}'}})
                yield event('content_block_stop', {'type': 'content_block_stop', 'index': blocks[position]})
        elif kind in ('response.completed', 'response.incomplete', 'response.failed', 'error'):
            final = ev
            break
    if final is None:
        raise FrontierError('stream ended without a terminal event')
    body = final.get('response') or {}
    usage = body.get('usage') or {}
    usage_sink({'input_tokens': usage.get('input_tokens'),
                'cached_input_tokens': (usage.get('input_tokens_details') or {}).get('cached_tokens'),
                'output_tokens': usage.get('output_tokens'),
                'reasoning_tokens': (usage.get('output_tokens_details') or {}).get('reasoning_tokens'),
                'model': body.get('model'), 'response_id': body.get('id'), 'status': final['type']})
    if final['type'] in ('response.failed', 'error'):
        raise FrontierError(json.dumps(final)[:2000])
    if final['type'] == 'response.incomplete':
        if (body.get('incomplete_details') or {}).get('reason') != 'max_output_tokens':
            raise FrontierError(json.dumps(final)[:2000])
        stop = 'max_tokens'
    else:
        stop = 'tool_use' if has_call else 'end_turn'
    yield event('message_delta', {'type': 'message_delta', 'delta': {'stop_reason': stop, 'stop_sequence': None},
                                  'usage': {'input_tokens': reported_input_tokens,
                                            'output_tokens': usage.get('output_tokens') or 0}})
    yield event('message_stop', {'type': 'message_stop'})


class OpenAIResponses:
    """A GPT model through the OpenAI Responses API."""

    def __init__(self, model, effort='high', api_key=None, url=RESPONSES_URL, attempts=6):
        self.model, self.effort, self.url, self.attempts = model, effort, url, attempts
        self.api_key = api_key or os.environ.get('OPENAI_API_KEY')
        if not self.api_key:
            raise FrontierError('OPENAI_API_KEY is not set')

    def _open(self, body, key, session):
        """POST and read until the first output event. Errors before any output are retried with backoff: nothing
        was generated, so a retry is not a replay."""
        headers = {'Authorization': 'Bearer ' + self.api_key, 'Content-Type': 'application/json',
                   'Accept': 'text/event-stream', 'session_id': key}
        body = dict(body, prompt_cache_key=key)
        problem = None
        for attempt in range(self.attempts):
            response = session.post(self.url, json=body, stream=True, timeout=(20, 1800), headers=headers)
            if response.status_code != 200:
                problem = f'HTTP {response.status_code}: {response.text[:1000]}'
                response.close()
                if response.status_code in RETRYABLE_STATUS:
                    time.sleep(min(90, 5 * 2 ** attempt))
                    continue
                raise FrontierError(problem)
            events, head = _events(response), []
            for ev in events:
                head.append(ev)
                kind = ev.get('type', '')
                if kind in FIRST_OUTPUT:
                    return response, head, events
                if kind in ('error', 'response.failed', 'response.incomplete'):
                    problem = json.dumps(ev)[:2000]
                    response.close()
                    if any(code in problem for code in RETRYABLE_EVENTS):
                        time.sleep(min(90, 10 * 2 ** attempt))
                        break
                    raise FrontierError(problem)
            else:
                response.close()
                raise FrontierError('stream ended before any output')
        raise FrontierError(f'frontier model unavailable after retries: {problem}')

    def stream(self, payload, usage_sink, reported_input_tokens=0, session=None, headers=None):
        """Anthropic request -> iterator of Anthropic SSE bytes."""
        body = to_responses(payload, self.model, self.effort)
        key = cache_key(self.model, body.get('instructions'), json.dumps(body['input'][:1]))
        session = session or requests.Session()
        response, head, rest = self._open(body, key, session)

        def chained():
            yield from head
            yield from rest
        try:
            yield from stream_anthropic(chained(), payload.get('model') or self.model, reported_input_tokens,
                                        usage_sink)
        finally:
            response.close()


class Anthropic:
    """A Claude model through the Anthropic Messages API."""

    def __init__(self, model, effort=None, api_key=None, url=ANTHROPIC_URL, attempts=12):
        self.model, self.effort, self.url, self.attempts = model, effort, url, attempts
        self.api_key = api_key or os.environ.get('ANTHROPIC_API_KEY')
        if not self.api_key:
            raise FrontierError('ANTHROPIC_API_KEY is not set')

    def stream(self, payload, usage_sink, reported_input_tokens=0, session=None, headers=None):
        body = dict(payload, model=self.model, stream=True)
        if self.effort:
            body['output_config'] = {**(body.get('output_config') or {}), 'effort': self.effort}
        sent = {'x-api-key': self.api_key, 'content-type': 'application/json',
                'anthropic-version': (headers or {}).get('anthropic-version') or '2023-06-01'}
        if (headers or {}).get('anthropic-beta'):
            sent['anthropic-beta'] = headers['anthropic-beta']
        session = session or requests.Session()
        problem = None
        for attempt in range(self.attempts):
            try:
                response = session.post(self.url, headers=sent, json=body, stream=True, timeout=(30, 900))
            except requests.RequestException as exc:
                response, problem = None, repr(exc)
            else:
                if response.status_code == 200:
                    break
                status = response.status_code
                problem = f'HTTP {status}: {response.text[:800]}'
                response.close()
                response = None
                if status not in RETRYABLE_STATUS:
                    raise FrontierError(problem)
            time.sleep(min(60, 5 * 2 ** attempt))
        if response is None:
            raise FrontierError(f'Anthropic API unavailable after retries: {problem}')
        usage, buffer = {}, b''
        try:
            for part in response.iter_content(chunk_size=None):
                if not part:
                    continue
                buffer += part
                *lines, buffer = buffer.split(b'\n')
                for line in lines:
                    if line.startswith(b'data: '):
                        try:
                            ev = json.loads(line[6:])
                        except ValueError:
                            continue
                        if ev.get('type') == 'message_start':
                            usage.update((ev.get('message') or {}).get('usage') or {})
                        elif ev.get('type') == 'message_delta':
                            usage.update(ev.get('usage') or {})
                yield part
        finally:
            response.close()
            usage_sink({'input_tokens': (usage.get('input_tokens') or 0) + (usage.get('cache_read_input_tokens') or 0)
                        + (usage.get('cache_creation_input_tokens') or 0),
                        'cached_input_tokens': usage.get('cache_read_input_tokens'),
                        'cache_write_tokens': usage.get('cache_creation_input_tokens'),
                        'output_tokens': usage.get('output_tokens'), 'model': self.model})


def make(model, effort='high'):
    """Frontier backend for a model name: Claude models through the Anthropic API, everything else through OpenAI."""
    if model.startswith('claude'):
        return Anthropic(model, effort=None if effort == 'default' else effort)
    return OpenAIResponses(model, effort=effort)
