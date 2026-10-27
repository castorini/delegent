"""The local SLM: one or more vLLM replicas of the same model.

Each subagent conversation re-sends its whole context at every step, so it is pinned to one replica (by its system
prompt and first message), where its prefix stays in vLLM's prefix cache.

- `messages()` serves Claude Code's Anthropic requests (vLLM's /v1/messages): the request is counted with the replica's
  own tokenizer and max_tokens is clamped to what is left of the SLM's context window.
- `chat()` serves the search subagent's OpenAI-style requests (vLLM's /v1/chat/completions).
"""
import hashlib
import json
import os
import time

import requests

from .frontier import strip_frontier_reasoning


class SLMError(RuntimeError):
    pass


class ContextOverflow(SLMError):
    pass


def single_system(payload, model, reasoning_effort=None):
    """Qwen3.x chat templates reject system messages after the first turn and take their own reasoning effort.
    Claude Code sends system-role messages inside `messages`; they become user turns wrapped in <system-reminder>, and
    Claude Code's output_config.effort becomes chat_template_kwargs (thinking on). Other models pass unchanged."""
    if not str(model).lower().startswith('qwen'):
        return payload
    payload = dict(payload)
    if isinstance(payload.get('system'), list):
        payload['system'] = '\n\n'.join(b.get('text', '') for b in payload['system']
                                        if isinstance(b, dict) and b.get('type') == 'text')
    messages = []
    for m in payload.get('messages') or []:
        if m.get('role') == 'system':
            content = m.get('content')
            text = content if isinstance(content, str) else '\n'.join(
                b.get('text', '') for b in content or [] if isinstance(b, dict) and b.get('type') == 'text')
            m = {'role': 'user', 'content': [{'type': 'text', 'text': f'<system-reminder>\n{text}\n</system-reminder>'}]}
        messages.append(m)
    payload['messages'] = messages
    output_config = dict(payload.get('output_config') or {})
    if 'effort' in output_config:
        output_config.pop('effort')
        if output_config:
            payload['output_config'] = output_config
        else:
            payload.pop('output_config', None)
        kwargs = dict(payload.get('chat_template_kwargs') or {})
        kwargs.setdefault('enable_thinking', True)
        kwargs.setdefault('reasoning_effort', reasoning_effort or os.environ.get('DELEGENT_QWEN_EFFORT', 'medium'))
        payload['chat_template_kwargs'] = kwargs
    return payload


def conversation_key(*parts):
    return hashlib.sha1(json.dumps(parts, sort_keys=True, default=str)[:20000].encode()).hexdigest()


class SLM:
    def __init__(self, endpoints, model, api_key=None, context_tokens=131072, max_output_tokens=32768,
                 temperature=1.0, top_p=0.95):
        if isinstance(endpoints, str):
            endpoints = [e for e in endpoints.split(',') if e]
        if not endpoints:
            raise SLMError('no SLM endpoint given')
        self.endpoints = [e.rstrip('/') for e in endpoints]
        self.model, self.api_key = model, api_key or os.environ.get('DELEGENT_SLM_API_KEY', 'EMPTY')
        self.context_tokens, self.max_output_tokens = context_tokens, max_output_tokens
        self.temperature, self.top_p = temperature, top_p

    def endpoint_for(self, key):
        """The replica a conversation is pinned to."""
        return self.endpoints[int(key[:8], 16) % len(self.endpoints)]

    def _headers(self, anthropic_version='2023-06-01'):
        return {'x-api-key': self.api_key, 'Authorization': 'Bearer ' + self.api_key,
                'anthropic-version': anthropic_version, 'Content-Type': 'application/json',
                'Accept-Encoding': 'identity'}

    def prepare(self, payload):
        """Claude Code's request as the SLM serves it."""
        payload = strip_frontier_reasoning(dict(payload, model=self.model))
        return single_system(payload, self.model)

    def count_tokens(self, payload, session=None):
        payload = self.prepare(payload)
        body = {k: payload[k] for k in ('model', 'system', 'messages', 'tools', 'tool_choice', 'thinking',
                                        'chat_template_kwargs') if k in payload}
        # vLLM maps output_config.effort to enable_thinking when generating but not when counting: count the same way.
        kwargs = dict(payload.get('chat_template_kwargs') or {})
        effort = (payload.get('output_config') or {}).get('effort')
        if effort is not None and 'enable_thinking' not in kwargs:
            kwargs['enable_thinking'] = effort != 'none'
        if kwargs:
            body['chat_template_kwargs'] = kwargs
        endpoint = self.endpoint_for(conversation_key(payload.get('system'), (payload.get('messages') or [None])[0]))
        r = (session or requests).post(endpoint + '/v1/messages/count_tokens', headers=self._headers(), json=body,
                                       timeout=(20, 120))
        if r.status_code == 400:   # vLLM refuses to count a request longer than its window
            raise ContextOverflow(f'the request exceeds the {self.context_tokens}-token window: {r.text[:300]}')
        r.raise_for_status()
        return r.json()['input_tokens']

    def messages(self, payload, usage_sink, headers=None, session=None):
        """Anthropic request -> iterator of Anthropic SSE bytes from the pinned replica."""
        session = session or requests.Session()
        payload = self.prepare(payload)
        count = self.count_tokens(payload, session)
        remaining = self.context_tokens - count
        if remaining < 64:
            raise ContextOverflow(f'{count} input tokens leave no room in the {self.context_tokens}-token window')
        payload['max_tokens'] = min(int(payload.get('max_tokens') or self.max_output_tokens), self.max_output_tokens,
                                    remaining)
        payload.update(temperature=self.temperature, top_p=self.top_p, stream=True)
        endpoint = self.endpoint_for(conversation_key(payload.get('system'), (payload.get('messages') or [None])[0]))
        sent = self._headers((headers or {}).get('anthropic-version') or '2023-06-01')
        response = None
        for wait in (0, 5, 15, 45):   # upstream 5xx before any byte was forwarded: the same request again
            time.sleep(wait)
            response = session.post(endpoint + '/v1/messages', headers=sent, json=payload, stream=True,
                                    timeout=(20, 1800))
            if response.status_code < 500:
                break
            response.close()
        if response.status_code != 200:
            text = response.text[:1000]
            response.close()
            raise SLMError(f'SLM rejected the request: HTTP {response.status_code}: {text}')
        usage = {'input_tokens': count}
        try:
            for part in response.iter_content(chunk_size=None):
                if part:
                    for line in part.split(b'\n'):
                        if line.startswith(b'data: ') and b'"output_tokens"' in line:
                            try:
                                ev = json.loads(line[6:])
                            except ValueError:
                                continue
                            got = (ev.get('usage') or (ev.get('message') or {}).get('usage') or {})
                            if got.get('output_tokens') is not None:
                                usage['output_tokens'] = got['output_tokens']
                    yield part
        finally:
            response.close()
            usage_sink(usage)

    def chat(self, messages, tools=None, tool_choice='auto', max_tokens=8192, thinking=False, sampling=None,
             attempts=4, session=None):
        """One OpenAI-style chat completion from the replica this conversation is pinned to."""
        body = {'model': self.model, 'messages': messages, 'max_tokens': max_tokens,
                **(sampling or {'temperature': self.temperature, 'top_p': self.top_p})}
        if tools:
            body.update(tools=tools, tool_choice=tool_choice)
        if not thinking:
            body['chat_template_kwargs'] = {'enable_thinking': False}
        endpoint = self.endpoint_for(conversation_key(messages[:2]))
        problem = None
        for attempt in range(attempts):
            try:
                r = (session or requests).post(endpoint + '/v1/chat/completions', headers=self._headers(), json=body,
                                               timeout=(20, 1800))
                if r.status_code == 200:
                    data = r.json()
                    return data['choices'][0]['message'], data.get('usage') or {}
                problem = f'{r.status_code}: {r.text[:300]}'
                if r.status_code == 400 and 'context' in r.text:
                    raise ContextOverflow(problem)
            except requests.RequestException as exc:
                problem = repr(exc)[:300]
            time.sleep(5 * (attempt + 1))
        raise SLMError(f'SLM chat failed: {problem}')
