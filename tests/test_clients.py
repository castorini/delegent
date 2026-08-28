from __future__ import annotations

import json
import os
import unittest
from unittest.mock import patch

from delegent.clients import HttpRetriever, OpenAICompatibleActor


class FakeResponse:
    def __init__(self, value: object) -> None:
        self.payload = json.dumps(value).encode("utf-8")

    def __enter__(self) -> FakeResponse:  # noqa: PYI034 -- typing.Self is Python 3.11+
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return self.payload


class ClientTests(unittest.TestCase):
    def test_actor_uses_openai_chat_contract_without_default_secret(self) -> None:
        captured = {}

        def fake_urlopen(request, timeout):  # type: ignore[no-untyped-def]
            captured["request"] = request
            captured["timeout"] = timeout
            return FakeResponse(
                {
                    "choices": [{"message": {"content": '{"plan":"p","query":"q"}'}}],
                    "usage": {"prompt_tokens": 4, "completion_tokens": 3},
                }
            )

        actor = OpenAICompatibleActor("https://actor.invalid", "demo-model")
        with patch("delegent.clients.urllib.request.urlopen", side_effect=fake_urlopen):
            result = actor.generate("prompt")
        self.assertEqual(result.prompt_tokens, 4)
        headers = {
            key.casefold(): value for key, value in captured["request"].header_items()
        }
        self.assertNotIn("authorization", headers)
        body = json.loads(captured["request"].data)
        self.assertEqual(body["temperature"], 0.0)
        self.assertEqual(body["model"], "demo-model")

    def test_actor_reads_only_named_environment_variable(self) -> None:
        actor = OpenAICompatibleActor(
            "https://actor.invalid", "demo-model", api_key_env="DELEGENT_TEST_TOKEN"
        )
        with patch.dict(os.environ, {}, clear=True), self.assertRaises(RuntimeError):
            actor.generate("prompt")

    def test_actor_accepts_base_url_that_already_ends_in_v1(self) -> None:
        captured = {}

        def fake_urlopen(request, timeout):  # type: ignore[no-untyped-def]
            captured["url"] = request.full_url
            return FakeResponse(
                {
                    "model": "demo-model",
                    "choices": [{"message": {"content": '{"plan":"p","query":"q"}'}}],
                    "usage": {},
                }
            )

        actor = OpenAICompatibleActor("https://actor.invalid/v1", "demo-model")
        with patch("delegent.clients.urllib.request.urlopen", side_effect=fake_urlopen):
            actor.generate("prompt")
        self.assertEqual(captured["url"], "https://actor.invalid/v1/chat/completions")

    def test_actor_rejects_mismatched_response_model_identity(self) -> None:
        actor = OpenAICompatibleActor("https://actor.invalid", "expected-model")
        response = FakeResponse(
            {
                "model": "different-model",
                "choices": [{"message": {"content": '{"plan":"p","query":"q"}'}}],
                "usage": {},
            }
        )
        with (
            patch("delegent.clients.urllib.request.urlopen", return_value=response),
            self.assertRaisesRegex(RuntimeError, "model identity"),
        ):
            actor.generate("prompt")

    def test_retriever_parses_pyserini_candidates(self) -> None:
        def fake_urlopen(request, timeout):  # type: ignore[no-untyped-def]
            self.assertIn("/v1/demo-index/search?", request.full_url)
            self.assertIn("query=lumen", request.full_url)
            return FakeResponse(
                {
                    "candidates": [
                        {
                            "docid": "d1",
                            "rank": 1,
                            "score": 2.5,
                            "doc": {"contents": "text"},
                        }
                    ]
                }
            )

        client = HttpRetriever("https://search.invalid", "demo-index")
        with patch("delegent.clients.urllib.request.urlopen", side_effect=fake_urlopen):
            hits = client.search("lumen", hits=5)
        self.assertEqual(
            hits[0].as_dict(), {"docid": "d1", "rank": 1, "score": 2.5, "text": "text"}
        )


if __name__ == "__main__":
    unittest.main()
