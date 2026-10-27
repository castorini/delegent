"""Anthropic Messages streaming (SSE) helpers shared by the relay and the agent loop."""
import json
import uuid


def event(kind, data):
    """One server-sent event in the Anthropic Messages wire format."""
    return f'event: {kind}\ndata: {json.dumps(data, separators=(",", ":"))}\n\n'.encode()


def message_stream(content, model, input_tokens=0, output_tokens=0, stop_reason=None):
    """A complete Anthropic Messages stream for already finished content blocks."""
    out = [event('message_start', {'type': 'message_start', 'message': {
        'id': 'msg_' + uuid.uuid4().hex[:24], 'type': 'message', 'role': 'assistant', 'model': model, 'content': [],
        'stop_reason': None, 'stop_sequence': None, 'usage': {'input_tokens': input_tokens, 'output_tokens': 0}}})]
    for i, block in enumerate(content):
        kind = block.get('type')
        if kind == 'thinking':
            out.append(event('content_block_start', {'type': 'content_block_start', 'index': i,
                                                     'content_block': {'type': 'thinking', 'thinking': ''}}))
            if block.get('thinking'):
                out.append(event('content_block_delta', {'type': 'content_block_delta', 'index': i, 'delta': {
                    'type': 'thinking_delta', 'thinking': block['thinking']}}))
            out.append(event('content_block_delta', {'type': 'content_block_delta', 'index': i, 'delta': {
                'type': 'signature_delta', 'signature': block.get('signature', '')}}))
        elif kind == 'tool_use':
            out.append(event('content_block_start', {'type': 'content_block_start', 'index': i, 'content_block': {
                'type': 'tool_use', 'id': block['id'], 'name': block['name'], 'input': {}}}))
            out.append(event('content_block_delta', {'type': 'content_block_delta', 'index': i, 'delta': {
                'type': 'input_json_delta', 'partial_json': json.dumps(block.get('input') or {})}}))
        else:
            out.append(event('content_block_start', {'type': 'content_block_start', 'index': i,
                                                     'content_block': {'type': 'text', 'text': ''}}))
            out.append(event('content_block_delta', {'type': 'content_block_delta', 'index': i, 'delta': {
                'type': 'text_delta', 'text': block.get('text', '')}}))
        out.append(event('content_block_stop', {'type': 'content_block_stop', 'index': i}))
    if stop_reason is None:
        stop_reason = 'tool_use' if any(b.get('type') == 'tool_use' for b in content) else 'end_turn'
    out.append(event('message_delta', {'type': 'message_delta', 'delta': {'stop_reason': stop_reason,
                                                                           'stop_sequence': None},
                                       'usage': {'output_tokens': output_tokens}}))
    out.append(event('message_stop', {'type': 'message_stop'}))
    return b''.join(out)


def collect_message(chunks):
    """Rebuild the assistant message from Anthropic SSE bytes, as a client would."""
    message, blocks, partial, buffer = None, {}, {}, b''
    for chunk in chunks:
        buffer += chunk
        *parts, buffer = buffer.split(b'\n\n')
        for part in parts:
            data = [line[5:].strip() for line in part.decode('utf-8').split('\n') if line.startswith('data:')]
            if not data:
                continue
            ev = json.loads(data[0])
            kind = ev.get('type')
            if kind == 'message_start':
                message = ev['message']
            elif kind == 'content_block_start':
                blocks[ev['index']] = dict(ev['content_block'])
            elif kind == 'content_block_delta':
                delta, block = ev['delta'], blocks[ev['index']]
                if delta['type'] == 'text_delta':
                    block['text'] = block.get('text', '') + delta['text']
                elif delta['type'] == 'thinking_delta':
                    block['thinking'] = block.get('thinking', '') + delta['thinking']
                elif delta['type'] == 'signature_delta':
                    block['signature'] = delta['signature']
                elif delta['type'] == 'input_json_delta':
                    partial[ev['index']] = partial.get(ev['index'], '') + delta['partial_json']
            elif kind == 'content_block_stop' and ev['index'] in partial:
                blocks[ev['index']]['input'] = json.loads(partial.pop(ev['index']) or '{}')
            elif kind == 'message_delta':
                message['stop_reason'] = ev['delta'].get('stop_reason')
                message['usage'] = {**(message.get('usage') or {}), **(ev.get('usage') or {})}
            elif kind == 'error':
                raise RuntimeError(f'stream error: {ev.get("error")}')
    if message is None:
        raise RuntimeError('stream ended before message_start')
    message['content'] = [blocks[i] for i in sorted(blocks)]
    return message
