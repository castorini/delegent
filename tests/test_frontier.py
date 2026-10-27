import base64
import json

from delegent import frontier
from delegent.sse import collect_message


def responses_events():
    reasoning = {'type': 'reasoning', 'id': 'rs_1', 'summary': [{'type': 'summary_text', 'text': 'plan'}],
                 'encrypted_content': 'ENC'}
    return [
        {'type': 'response.output_item.added', 'output_index': 0, 'item': {'type': 'reasoning'}},
        {'type': 'response.output_item.done', 'output_index': 0, 'item': reasoning},
        {'type': 'response.output_item.added', 'output_index': 1, 'item': {'type': 'message'}},
        {'type': 'response.output_text.delta', 'output_index': 1, 'delta': 'Looking '},
        {'type': 'response.output_text.delta', 'output_index': 1, 'delta': 'now.'},
        {'type': 'response.output_item.done', 'output_index': 1, 'item': {'type': 'message'}},
        {'type': 'response.output_item.added', 'output_index': 2,
         'item': {'type': 'function_call', 'call_id': 'call_1', 'name': 'Delegate'}},
        {'type': 'response.function_call_arguments.delta', 'output_index': 2, 'delta': '{"task": "find X", '},
        {'type': 'response.function_call_arguments.delta', 'output_index': 2, 'delta': '"deliverable": "a quote"}'},
        {'type': 'response.output_item.done', 'output_index': 2, 'item': {'type': 'function_call'}},
        {'type': 'response.completed', 'response': {'id': 'resp_1', 'model': 'gpt-x', 'usage': {
            'input_tokens': 1000, 'input_tokens_details': {'cached_tokens': 900}, 'output_tokens': 50}}},
    ]


def test_stream_round_trip():
    usage = {}
    message = collect_message(frontier.stream_anthropic(iter(responses_events()), 'gpt-x', 7, usage.update))
    kinds = [b['type'] for b in message['content']]
    assert kinds == ['thinking', 'text', 'tool_use']
    assert message['content'][1]['text'] == 'Looking now.'
    assert message['content'][2]['input'] == {'task': 'find X', 'deliverable': 'a quote'}
    assert message['stop_reason'] == 'tool_use'
    assert usage['cached_input_tokens'] == 900 and usage['output_tokens'] == 50
    signature = message['content'][0]['signature']
    assert json.loads(base64.b64decode(signature[len(frontier.SIGNATURE):]))['encrypted_content'] == 'ENC'

    # the next request carries the assistant turn back: the reasoning item is restored unchanged
    payload = {'model': 'gpt-x', 'system': [{'type': 'text', 'text': 'You are an agent.'}], 'messages': [
        {'role': 'user', 'content': 'Q?'},
        {'role': 'assistant', 'content': message['content']},
        {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': 'call_1', 'content': 'report'},
                                     {'type': 'text', 'text': 'go on'}]},
        {'role': 'system', 'content': 'reminder'}],
        'tools': [{'name': 'Delegate', 'description': 'd', 'input_schema': {'type': 'object'}}]}
    body = frontier.to_responses(payload, 'gpt-x', 'high')
    types = [i.get('type') or i.get('role') for i in body['input']]
    assert types == ['user', 'reasoning', 'message', 'function_call', 'function_call_output', 'user', 'developer']
    assert body['input'][1]['encrypted_content'] == 'ENC'
    assert body['instructions'] == 'You are an agent.'
    assert body['store'] is False and body['tools'][0]['name'] == 'Delegate'


def test_strip_frontier_reasoning_keeps_other_blocks():
    signed = {'type': 'thinking', 'thinking': 't', 'signature': frontier.SIGNATURE + 'xx'}
    payload = {'messages': [{'role': 'assistant', 'content': [signed, {'type': 'text', 'text': 'hi'}]}]}
    out = frontier.strip_frontier_reasoning(payload)
    assert out['messages'][0]['content'] == [{'type': 'text', 'text': 'hi'}]
    assert payload['messages'][0]['content'][0] is signed   # the input is not modified


def test_cache_key_is_stable():
    assert frontier.cache_key('m', 'sys', 'first') == frontier.cache_key('m', 'sys', 'first')
    assert frontier.cache_key('m', 'sys', 'first') != frontier.cache_key('m', 'sys', 'other')
